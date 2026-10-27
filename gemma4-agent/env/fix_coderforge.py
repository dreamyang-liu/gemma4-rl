"""Fix mliu444/coderforge-swegemma-skill-sft-v2 so it matches what swegemma 0.2.11 / ADK 1.36.1 actually send.

  A. Replace the stale `load_skill("openhands")` result (captured from an older editor-only skill that says
     "Use `run_command` for shell commands") with the real result for the shipped skill.
  B. Budget 64 tool calls / 10 minutes / no turn limit: rewrite the user-prompt budget block and budget mentions in
     compaction summaries and reasoning; keep only trajectories with <= MAX_CALLS tool calls.
  Then re-render (Gemma 4 chat template, thinking on) and re-tokenize, and drop trajectories whose compaction points
  would move under ADK's token-threshold rule (prompt >= 14336 before the recorded compaction, or not at it) or whose
  segments exceed the token cap.

usage (inside the gemma4-kaggle image, python -I):
  fix_coderforge.py verify --src DIR --model DIR              # reproduce tokenized/ bit-exactly from text/
  fix_coderforge.py calib  --src DIR --model DIR              # offset-based prompt counts vs exact renders
  fix_coderforge.py run    --src DIR --model DIR --out DIR [--workers N]
"""
import argparse
import collections
import glob
import json
import os
import re
import sys
from multiprocessing import Pool

import pyarrow as pa
import pyarrow.parquet as pq

THRESHOLD, MAX_SEG_TOKENS, MAX_CALLS = 14336, 18178, 64
MODEL_HEADER = '<|turn>model\n'
GEN_AFTER_TOOL = 3  # calibrated with `calib`: exact render(add_generation_prompt=True) - offset
OLD_BLOCK = '- Time allowance: 60.0 minutes\n- Tool calls allowance: 100 calls\n- Max loop iterations: 500 turns\n'
NEW_BLOCK = '- Time allowance: 10.0 minutes\n- Tool calls allowance: 64 calls\n'
TEXT_SCHEMA = pa.schema([('kind', pa.string()), ('trajectory_id', pa.string()), ('segment', pa.int64()),
                         ('repo', pa.string()), ('source', pa.string()), ('n_tokens', pa.int64()),
                         ('messages', pa.string())])
TOK_SCHEMA = pa.schema([('kind', pa.string()), ('trajectory_id', pa.string()), ('segment', pa.int32()),
                        ('n_tokens', pa.int32()), ('input_ids', pa.list_(pa.int32())),
                        ('labels', pa.list_(pa.int32()))])

# Budget phrasing in Haiku summaries / reasoning. Turn limits are dropped (new config has none), then numbers swapped.
_T = r'(?:loop )?(?:turns?|iterations?|loop iterations?)'
TURN_RULES = [
    (re.compile(r'(\b\d+(?:[- ]minutes?|-minute(?: time)? limit| minute(?: time)? limit| min)),? (\b\d+[- ]tool[- ]calls?),? and 500 ' + _T), r'\1 and \2'),
    (re.compile(r',? and 500[- ]' + _T + r'\b'), ''),
    (re.compile(r', 500[- ]' + _T + r'\b'), ''),
    (re.compile(r'\b500[- ]' + _T + r',? (?:and )?'), ''),
    (re.compile(r'; (?:[Ll]oop limit:?|[Mm]ax) 500 ' + _T + r'\b'), ''),
    (re.compile(r' (?:[Ll]oop limit:?|[Mm]ax) 500 ' + _T + r'\.'), ''),
    (re.compile(r' / 500 ' + _T + r'\b'), ''),
    (re.compile(r'(\b\d+[- ]min(?:ute)?s?), (\d+[- ]tool[- ]calls?),? or 500 ' + _T + r'\b'), r'\1 or \2'),
    (re.compile(r',? or 500 ' + _T + r'\b'), ''),
    (re.compile(r'; 500 ' + _T + r'\b'), ''),
]
NUM_RULES = [
    (re.compile(r'\b60(?:\.0)?([- ])(minutes?|min\b|minute)'), r'10\1\2'),
    (re.compile(r'\b100([- ])(tool[- ]calls?|calls?|tool[- ]call)\b'), r'64\1\2'),
    (re.compile(r'(?i)\b(tool[- ]call limit:?) 100\b'), r'\1 64'),
]
RESIDUAL = re.compile(r'\b60(?:\.0)?[- ]min|\b100[- ](?:tool[- ])?calls?\b|\b500[- ](?:loop )?(?:turns?|iterations?)|tool[- ]call limit:? 100\b', re.I)


def fix_budget_text(s):
    for rx, rep in TURN_RULES + NUM_RULES:
        s = rx.sub(rep, s)
    return s


class Renderer:
    def __init__(self, model_dir):
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(os.path.join(model_dir, 'tokenizer.json'))
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        self.tpl = env.from_string(open(os.path.join(model_dir, 'chat_template.jinja')).read())

    def render(self, messages, tools=None, gen=False):
        kw = {'tools': tools} if tools is not None else {}
        return self.tpl.render(messages=messages, bos_token='<bos>', enable_thinking=True,
                               add_generation_prompt=gen, **kw)


def agent_spans(text, messages):
    """Trained assistant spans (thought + tool call through <|tool_response>) and every assistant-turn start."""
    trained = [m.get('train', True) for m in messages if m['role'] == 'assistant']
    n_tool = sum(m['role'] == 'tool' for m in messages)
    if (text.count('<|turn>') != 3 or text.count(MODEL_HEADER) != 1
            or text.count('<|tool_response>') != len(trained) or text.count('<tool_response|>') != n_tool):
        raise ValueError('special_token_in_text')
    pos = text.index(MODEL_HEADER) + len(MODEL_HEADER)
    spans, starts = [], []
    for k, train in enumerate(trained):
        end = text.index('<|tool_response>', pos) + len('<|tool_response>')
        starts.append((pos, train))
        if train:
            spans.append((pos, end))
        if k < len(trained) - 1:
            pos = text.index('<tool_response|>', end) + len('<tool_response|>')
    return spans, starts


def summary_spans(text):
    if text.count(MODEL_HEADER) != 1 or text.count('<turn|>') != 3 or not text.endswith('<turn|>\n'):
        raise ValueError('special_token_in_text')
    return [(text.index(MODEL_HEADER) + len(MODEL_HEADER), len(text) - 1)]


def tokenize(tok, text, spans):
    enc = tok.encode(text, add_special_tokens=False)
    labels, k = [], 0
    for tid, (a, b) in zip(enc.ids, enc.offsets):
        while k < len(spans) and a >= spans[k][1]:
            k += 1
        labels.append(tid if k < len(spans) and a >= spans[k][0] and b <= spans[k][1] else -100)
    return enc.ids, labels, enc.offsets


def encode_row(R, kind, messages, tools):
    """-> ids, labels, request prompt token counts (agent rows: one per trained turn)."""
    if kind == 'agent':
        text = R.render(messages, tools)
        spans, starts = agent_spans(text, messages)
    else:
        text = R.render(messages)
        spans, starts = summary_spans(text), []
    ids, labels, offsets = tokenize(R.tok, text, spans)
    reqs = []
    if starts:
        tok_starts = [a for a, _ in offsets]
        import bisect
        first = starts[0][0]
        for pos, train in starts:
            if train:  # generation prompt adds 3 tokens when continuing after a tool response
                reqs.append(bisect.bisect_left(tok_starts, pos) + (0 if pos == first else GEN_AFTER_TOOL))
    return ids, labels, reqs


def transform(kind, messages, old_ls, new_ls, stats):
    out = []
    for m in messages:
        m = dict(m)
        c = m.get('content')
        if m['role'] == 'tool' and c == old_ls:
            m['content'] = new_ls
            stats['A:load_skill_replaced'] += 1
        elif isinstance(c, str):
            if OLD_BLOCK in c:
                c = c.replace(OLD_BLOCK, NEW_BLOCK)
                stats['B:prompt_block_replaced'] += 1
            if (kind == 'summary') or (m['role'] == 'user' and c.startswith('For context')):
                c2 = fix_budget_text(c)
                stats['B:summary_text_edited'] += c2 != c
                c = c2
            m['content'] = c
        elif isinstance(c, list):  # "For context:" summary turn in agent rows
            parts = []
            for p in c:
                if p.get('type') == 'text':
                    t2 = fix_budget_text(p['text'].replace(OLD_BLOCK, NEW_BLOCK))
                    stats['B:summary_text_edited'] += t2 != p['text']
                    p = {**p, 'text': t2}
                parts.append(p)
            m['content'] = parts
        if m.get('reasoning_content'):
            r2 = fix_budget_text(m['reasoning_content'])
            stats['B:reasoning_edited'] += r2 != m['reasoning_content']
            m['reasoning_content'] = r2
        out.append(m)
    return out


def residual_count(messages):
    n = 0
    for m in messages:
        if m['role'] == 'tool':
            continue
        c = m.get('content')
        c = c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
        n += len(RESIDUAL.findall(c or '')) + len(RESIDUAL.findall(m.get('reasoning_content') or ''))
    return n


def segment_ok(reqs, n_tokens, seg, last_seg):
    if n_tokens > MAX_SEG_TOKENS:
        return 'over_max_segment_tokens'
    if any(p >= THRESHOLD for p in reqs[:-1]):
        return 'compaction_would_trigger_earlier'
    if not last_seg and reqs and reqs[-1] < THRESHOLD:
        return 'compaction_would_not_trigger'
    if seg > 0 and reqs and reqs[0] >= THRESHOLD:
        return 'segment_starts_over_threshold'
    return None


def process_shard(job):
    path, out_dir, model_dir, old_ls, new_ls, tools = job
    R = Renderer(model_dir)
    stats = collections.Counter()
    t = pq.read_table(path).to_pylist()
    by_traj = collections.defaultdict(list)
    for r in t:
        by_traj[r['trajectory_id']].append(r)
    text_rows, tok_rows = [], []
    for tid, rows in by_traj.items():
        stats['traj_in'] += 1
        agents = sorted((r for r in rows if r['kind'] == 'agent'), key=lambda r: r['segment'])
        n_seg = len(agents)
        calls = sum(len(m.get('tool_calls') or []) for r in agents for m in json.loads(r['messages'])
                    if m['role'] == 'assistant' and m.get('train'))
        if calls > MAX_CALLS:
            stats['drop:calls>64'] += 1
            continue
        new_text, new_tok, reason = [], [], None
        for r in sorted(rows, key=lambda r: (r['kind'] != 'agent', r['segment'])):
            msgs = transform(r['kind'], json.loads(r['messages']), old_ls, new_ls, stats)
            stats['residual_budget_mentions'] += residual_count(msgs)
            try:
                ids, labels, reqs = encode_row(R, r['kind'], msgs, tools if r['kind'] == 'agent' else None)
            except ValueError as e:
                reason = str(e)
                break
            if r['kind'] == 'agent':
                reason = segment_ok(reqs, len(ids), r['segment'], r['segment'] == n_seg - 1)
                if reason:
                    break
            new_text.append({**r, 'n_tokens': len(ids), 'messages': json.dumps(msgs, ensure_ascii=False)})
            new_tok.append({'kind': r['kind'], 'trajectory_id': tid, 'segment': r['segment'], 'n_tokens': len(ids),
                            'input_ids': ids, 'labels': labels})
        if reason:
            stats['drop:' + reason] += 1
            continue
        stats['traj_kept'] += 1
        for r in new_tok:
            stats[f"{r['kind']}_rows"] += 1
            stats[f"{r['kind']}_tokens"] += r['n_tokens']
            stats[f"{r['kind']}_label_tokens"] += sum(x != -100 for x in r['labels'])
        text_rows += new_text
        tok_rows += new_tok
    name = os.path.basename(path)
    pq.write_table(pa.Table.from_pylist(text_rows, schema=pq.read_schema(path)), os.path.join(out_dir, 'text', name))
    pq.write_table(pa.Table.from_pylist(tok_rows, schema=TOK_SCHEMA), os.path.join(out_dir, 'tokenized', name))
    return dict(stats)


def real_load_skill(skill_dir):
    import asyncio
    import types
    from google.adk.models.lite_llm import _safe_json_serialize
    from google.adk.skills import load_skill_from_dir
    from google.adk.tools.skill_toolset import LoadSkillTool, SkillToolset
    tool = LoadSkillTool(SkillToolset(skills=[load_skill_from_dir(skill_dir)]))
    ctx = types.SimpleNamespace(invocation_id='x', agent_name='swe_agent', state={})
    return _safe_json_serialize(asyncio.run(tool.run_async(args={'skill_name': 'openhands'}, tool_context=ctx)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['verify', 'calib', 'run'])
    ap.add_argument('--src', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out')
    ap.add_argument('--workers', type=int, default=64)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    tools = json.load(open(os.path.join(a.src, 'tools.json')))
    old_ls = open(os.path.join(a.src, 'pipeline/prompts/load_skill_openhands.json')).read()

    if a.mode == 'verify':
        R = Renderer(a.model)
        for shard in ('filtered_reward1-00000-of-00224.parquet', 'filtered_reward1-00100-of-00224.parquet'):
            text = pq.read_table(os.path.join(a.src, 'text', shard)).to_pylist()
            tok = {(r['kind'], r['trajectory_id'], r['segment']): r
                   for r in pq.read_table(os.path.join(a.src, 'tokenized', shard)).to_pylist()}
            c = collections.Counter()
            for r in text:
                ids, labels, _ = encode_row(R, r['kind'], json.loads(r['messages']), tools if r['kind'] == 'agent' else None)
                ref = tok.get((r['kind'], r['trajectory_id'], r['segment']))
                c['rows'] += 1
                c['ids_equal'] += ref is not None and ids == ref['input_ids']
                c['labels_equal'] += ref is not None and labels == ref['labels']
                c['n_tokens_equal'] += len(ids) == r['n_tokens']
            print(shard, dict(c))
        return

    if a.mode == 'calib':
        R = Renderer(a.model)
        text = pq.read_table(os.path.join(a.src, 'text', 'filtered_reward1-00100-of-00224.parquet')).to_pylist()
        by = collections.defaultdict(list)
        for r in text:
            if r['kind'] == 'agent':
                by[r['trajectory_id']].append(r)
        diffs, viol = collections.Counter(), collections.Counter()
        for tid, rows in list(by.items())[:6]:
            rows.sort(key=lambda r: r['segment'])
            for r in rows:
                msgs = json.loads(r['messages'])
                _, _, reqs = encode_row(R, 'agent', msgs, tools)
                # exact: render prefix (system, base, ..., up to the tool response before trained turn i) + gen prompt
                trained_idx = [i for i, m in enumerate(msgs) if m['role'] == 'assistant' and m.get('train')]
                for k, i in enumerate(trained_idx[:8]):
                    exact = len(R.tok.encode(R.render(msgs[:i], tools, gen=True), add_special_tokens=False).ids)
                    diffs[exact - reqs[k]] += 1
                viol[segment_ok(reqs, r['n_tokens'], r['segment'], r['segment'] == len(rows) - 1) or 'ok'] += 1
        print('exact - offset prompt tokens:', dict(diffs))
        print('original data under segment rules:', dict(viol))
        return

    os.makedirs(os.path.join(a.out, 'text'), exist_ok=True)
    os.makedirs(os.path.join(a.out, 'tokenized'), exist_ok=True)
    new_ls = real_load_skill(os.path.join(a.src, 'skill/openhands'))
    shards = sorted(glob.glob(os.path.join(a.src, 'text', '*.parquet')))
    if a.limit:
        shards = shards[:a.limit]
    total = collections.Counter()
    with Pool(a.workers) as pool:
        for i, s in enumerate(pool.imap_unordered(process_shard, [(p, a.out, a.model, old_ls, new_ls, tools) for p in shards])):
            total.update(s)
            if i % 20 == 0:
                print(f'{i + 1}/{len(shards)} shards', flush=True)
    json.dump(dict(total), open(os.path.join(a.out, 'fix_stats.json'), 'w'), indent=1)
    print(json.dumps(dict(total), indent=1))


if __name__ == '__main__':
    main()
