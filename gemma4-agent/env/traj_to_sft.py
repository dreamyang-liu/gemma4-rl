"""Bedrock-sampled swegemma trajectories (proxy logs) -> tokenized SFT rows for env/sft_train.py.

Each runner worker log (runs/bedrock_logs/<worker>.jsonl) holds the chat-completion requests of its tasks in order.
A task starts with the swegemma task prompt; ADK compaction replaces that prompt by a "For context:" user message,
which starts a new segment, and the summarizer calls are single-message requests. Per segment, the last request
plus its response is the full conversation; it is rendered once with the model's chat template (as vLLM does:
tool-call arguments deserialized, enable_thinking from the request) and every model-generated span is labeled:
from the end of that step's generation prompt through its stop token (<|tool_response> after a tool call, <turn|>
after a text answer). The step prompts are exact prefixes of the segment render (checked against vLLM /tokenize),
so one row reproduces every step's context. The template drops the thoughts of assistant turns that precede a user
message, so a segment containing later user messages (e.g. after compaction) is split at each of them: one row per
span, labeling the assistant turns after that span's last user message. Steps whose response has no reasoning are not
labeled (at inference the generation prompt opens a thought channel that the render would not contain). Summarizer
calls become rows too.

usage (gemma4-kaggle image, python -I):
  traj_to_sft.py --logs 'runs/bedrock_logs/rbsh_r*.jsonl' --runs 'runs/rbsh_r*' --tasks TASKS.jsonl
                 --model models/gemma-4-31B-it-bf16 --out ext/sft_rebench/shell_v2 [--max-per-task 4]
  traj_to_sft.py --check ext/sft_rebench/vllm_tokenize_cases.json --model DIR   # renderer vs vLLM /tokenize
"""
import argparse
import collections
import copy
import glob
import json
import os
import re

import pyarrow as pa
import pyarrow.parquet as pq

TOK_SCHEMA = pa.schema([('kind', pa.string()), ('trajectory_id', pa.string()), ('segment', pa.int32()),
                        ('n_tokens', pa.int32()), ('input_ids', pa.list_(pa.int32())),
                        ('labels', pa.list_(pa.int32()))])
TASK_PREFIX = 'You are evaluating a software engineering task'
# Explicit "I remember the upstream fix" statements (same strict set as analysis/leak_scan.py); --exclude-recall drops
# trajectories with any such step, so the data does not teach Gemma to claim memory of a repository's history.
RECALL = [re.compile(p) for p in (
    r"(?i)\b(I (recall|remember) (the|that the|this) (fix|PR|commit|patch|change|upstream)|from memory|let me recall)\b",
    r"(?i)\bupstream (repo|repository|fix|patch|PR|pull request|commit|history|change|implementation|test)s?\b",
    r"(?i)\bthe (actual|real|official) (upstream )?(fix|patch|PR|commit)\b",
    r"(?i)hidden tests? (likely |probably |may |might |will )?(mirror|match|be like|be the same as|come from|is from|are from)")]


def fold_text_into_thought(messages):
    """Teacher models (Kimi, GLM) often write a visible note before a tool call. Gemma's template renders such text
    after the tool responses and closes the turn there, so step prompts are no longer prefixes of the segment render
    (and Gemma itself almost never does this). Move that note into the thought; text-only answers are kept."""
    out = []
    for m in messages:
        if m.get('role') == 'assistant' and m.get('tool_calls'):
            t = _first_text(m.get('content')).strip()
            if t:
                m = dict(m)
                r = (m.get('reasoning') or m.get('reasoning_content') or '').rstrip()
                m['reasoning'] = (r + '\n\n' + t) if r else t
                m.pop('reasoning_content', None)
                m['content'] = None
        out.append(m)
    return out


def recall_steps(task):
    n = 0
    for seg in task['segments']:
        for _, resp in seg:
            m = resp['message']
            t = (m.get('reasoning') or m.get('reasoning_content') or '') + '\n' + (m.get('content') or '')
            n += any(rx.search(t) for rx in RECALL)
    return n


class Renderer:
    def __init__(self, model_dir):
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(os.path.join(model_dir, 'tokenizer.json'))
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)

        def raise_exception(msg):
            raise ValueError(msg)
        env.globals['raise_exception'] = raise_exception
        self.tpl = env.from_string(open(os.path.join(model_dir, 'chat_template.jinja')).read())

    def render(self, messages, tools=None, gen=False, enable_thinking=True):
        kw = {'tools': tools} if tools else {}
        return self.tpl.render(messages=prepare(messages), bos_token='<bos>', enable_thinking=enable_thinking,
                               add_generation_prompt=gen, **kw)

    def encode(self, text):
        enc = self.tok.encode(text, add_special_tokens=False)
        return enc.ids, enc.offsets


def prepare(messages):
    """vLLM-side preprocessing: string content -> [{'type': 'text', 'text': ...}] (vLLM detects this template as
    OpenAI content format; its list branch differs from the string branch, e.g. a trailing space after system text),
    and tool-call arguments JSON string -> mapping."""
    out = []
    for m in messages:
        m = copy.deepcopy(m)
        if isinstance(m.get('content'), str):
            m['content'] = [{'type': 'text', 'text': m['content']}]
        for tc in m.get('tool_calls') or []:
            a = tc.get('function', {}).get('arguments')
            if isinstance(a, str):
                try:
                    tc['function']['arguments'] = json.loads(a) if a.strip() else {}
                except json.JSONDecodeError:
                    tc['function']['arguments'] = {}
        out.append(m)
    return out


def assistant_message(resp):
    m = {'role': 'assistant', 'content': resp.get('content')}
    r = resp.get('reasoning') or resp.get('reasoning_content')
    if r:
        m['reasoning'] = r
    if resp.get('tool_calls'):
        m['tool_calls'] = resp['tool_calls']
    return m


def encode_conversation(R, messages, tools, enable_thinking, first_label_idx):
    """Render; label each assistant message at index >= first_label_idx (that has reasoning when thinking is on)."""
    text = R.render(messages, tools, gen=False, enable_thinking=enable_thinking)
    spans, skipped = [], 0
    for i, m in enumerate(messages):
        if m['role'] != 'assistant' or i < first_label_idx:
            continue
        if enable_thinking and not (m.get('reasoning') or m.get('reasoning_content')):
            skipped += 1
            continue
        prefix = R.render(messages[:i], tools, gen=True, enable_thinking=enable_thinking)
        if not text.startswith(prefix):
            raise ValueError(f'step prompt is not a prefix of the render at message {i}')
        start = len(prefix)
        stop = '<|tool_response>' if m.get('tool_calls') else '<turn|>'
        end = text.index(stop, start) + len(stop)
        spans.append((start, end))
    ids, offsets = R.encode(text)
    labels, k = [], 0
    for tid, (a, b) in zip(ids, offsets):
        while k < len(spans) and a >= spans[k][1]:
            k += 1
        labels.append(tid if k < len(spans) and a >= spans[k][0] and b <= spans[k][1] else -100)
    return ids, labels, len(spans), skipped


def _first_text(content):
    if isinstance(content, str):
        return content
    return ''.join(p.get('text', '') for p in content or [] if isinstance(p, dict))


def split_tasks(rows):
    """Worker log rows -> list of tasks: {'prompt', 'segments': [[(req, resp)...]...], 'summaries': [(req, resp)]}.

    Requests of different tasks can interleave in one worker log (an agent that hit the session timeout keeps calling
    the model while the next task runs), so rows are linked by content, not order: a segment is identified by its
    first user message; an uncompacted segment starts with the task prompt; a summarizer call belongs to the segment
    whose first user text appears in its prompt; a compacted segment belongs to the summarizer whose summary appears
    in its "For context:" message.
    """
    tasks, seg_task, seg_index, summary_task = [], {}, {}, []
    by_prompt = {}
    for r in rows:
        if r.get('status') != 200:
            continue
        req = r['request']
        resp = r['chat_response']['choices'][0]
        msgs = req['messages']
        if len(msgs) == 1:
            prompt = _first_text(msgs[0].get('content'))
            owner = None
            for key, t in seg_task.items():
                snippet = key[:400]
                if snippet and snippet in prompt:
                    owner = t
            if owner is not None:
                owner['summaries'].append((req, resp))
                summary_task.append((_first_text(resp['message'].get('content')).strip()[:300], owner))
            continue
        first = _first_text(msgs[1].get('content'))
        if first not in seg_task:
            if isinstance(msgs[1].get('content'), str) and first.startswith(TASK_PREFIX):
                t = by_prompt.get(first)
                if t is None:
                    t = {'prompt': first, 'segments': [], 'summaries': []}
                    by_prompt[first] = t
                    tasks.append(t)
            else:
                t = next((o for summ, o in reversed(summary_task) if summ and summ in first), None)
                if t is None:
                    continue
            seg_task[first] = t
            seg_index[first] = len(t['segments'])
            t['segments'].append([])
        seg_task[first]['segments'][seg_index[first]].append((req, resp))
    for t in tasks:
        for seg in t['segments']:
            seg.sort(key=lambda x: len(x[0]['messages']))
    return tasks


def problem_statement(prompt):
    m = re.search(r'Problem Statement:\n(.*?)\n\n## Task Budget', prompt, re.S)
    return (m.group(1) if m else '').strip()


def check(args):
    R = Renderer(args.model)
    cases = json.load(open(args.check))
    bad = collections.Counter()
    for c in cases:
        q = c['request']
        ids, _ = R.encode(R.render(q['messages'], q.get('tools'), gen=True,
                                   enable_thinking=bool((q.get('chat_template_kwargs') or {}).get('enable_thinking'))))
        ok = ids == c['vllm_prompt_ids']
        bad[(c['kind'], ok)] += 1
    print('renderer vs vLLM /tokenize:', dict(bad))
    return all(ok for (_, ok) in bad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logs')
    ap.add_argument('--runs')
    ap.add_argument('--tasks')
    ap.add_argument('--model', required=True)
    ap.add_argument('--out')
    ap.add_argument('--max-per-task', type=int, default=4)
    ap.add_argument('--include-unresolved', action='store_true')
    ap.add_argument('--exclude-recall', action='store_true', help='skip trajectories with upstream-recall statements')
    ap.add_argument('--text-to-thought', action='store_true',
                    help='move visible text of tool-calling assistant turns into the thought (teacher-model data)')
    ap.add_argument('--check')
    args = ap.parse_args()
    if args.check:
        raise SystemExit(0 if check(args) else 1)

    R = Renderer(args.model)
    by_ps = {}
    for l in open(args.tasks):
        t = json.loads(l)
        by_ps[t['problem_statement'].strip()[:400]] = t['instance_id']
    results = {}
    for d in glob.glob(args.runs):
        p = os.path.join(d, 'results.jsonl')
        if os.path.exists(p):
            for l in open(p):
                r = json.loads(l)
                results[(os.path.basename(d), r['id'])] = r
    os.makedirs(args.out, exist_ok=True)
    stats = collections.Counter()
    per_task = collections.Counter()
    out_rows, meta = [], []
    for log in sorted(glob.glob(args.logs)):
        worker = os.path.basename(log)[:-len('.jsonl')]
        for t in split_tasks([json.loads(l) for l in open(log)]):
            stats['tasks_in_logs'] += 1
            iid = by_ps.get(problem_statement(t['prompt'])[:400])
            res = results.get((worker, iid))
            if iid is None or res is None:
                stats['skip:unmatched'] += 1
                continue
            if not res['resolved'] and not args.include_unresolved:
                stats['skip:unresolved'] += 1
                continue
            if per_task[iid] >= args.max_per_task:
                stats['skip:per_task_cap'] += 1
                continue
            if args.exclude_recall and recall_steps(t):
                stats['skip:recall'] += 1
                continue
            tid = f'{worker}:{iid}'
            rows, info = [], {'trajectory_id': tid, 'instance_id': iid, 'worker': worker, 'resolved': res['resolved'],
                              'tool_calls': res['tool_calls'], 'duration_s': res['duration_s'], 'segments': []}
            try:
                si = 0
                for seg in t['segments']:
                    req, resp = seg[-1]
                    stats['steps_hit_max_tokens'] += sum(r.get('finish_reason') == 'length' for _, r in seg)
                    msgs = req['messages'] + [assistant_message(resp['message'])]
                    if args.text_to_thought:
                        msgs = fold_text_into_thought(msgs)
                    think = bool((req.get('chat_template_kwargs') or {}).get('enable_thinking'))
                    users = [i for i, m in enumerate(msgs) if m['role'] == 'user'] + [len(msgs)]
                    for u, nxt in zip(users[:-1], users[1:]):
                        if not any(m['role'] == 'assistant' for m in msgs[u + 1:nxt]):
                            continue
                        ids, labels, n_lab, n_skip = encode_conversation(R, msgs[:nxt], req.get('tools'), think, u + 1)
                        stats['steps_labeled'] += n_lab
                        stats['steps_no_reasoning_masked'] += n_skip
                        if not n_lab:
                            continue
                        rows.append({'kind': 'agent', 'trajectory_id': tid, 'segment': si, 'n_tokens': len(ids),
                                     'input_ids': ids, 'labels': labels})
                        info['segments'].append({'n_tokens': len(ids), 'n_labeled': sum(x != -100 for x in labels),
                                                 'steps': n_lab})
                        si += 1
                for si, (req, resp) in enumerate(t['summaries']):
                    msgs = req['messages'] + [assistant_message(resp['message'])]
                    think = bool((req.get('chat_template_kwargs') or {}).get('enable_thinking'))
                    ids, labels, n_lab, _ = encode_conversation(R, msgs, None, think, 1)
                    if n_lab:
                        rows.append({'kind': 'summary', 'trajectory_id': tid, 'segment': si, 'n_tokens': len(ids),
                                     'input_ids': ids, 'labels': labels})
            except ValueError as e:
                stats['skip:render_error'] += 1
                print('render error', tid, e)
                continue
            per_task[iid] += 1
            stats['trajectories'] += 1
            stats['agent_rows'] += sum(r['kind'] == 'agent' for r in rows)
            stats['summary_rows'] += sum(r['kind'] == 'summary' for r in rows)
            stats['tokens'] += sum(r['n_tokens'] for r in rows)
            stats['label_tokens'] += sum(sum(x != -100 for x in r['labels']) for r in rows)
            out_rows += rows
            meta.append(info)
    os.makedirs(os.path.join(args.out, 'tokenized'), exist_ok=True)
    pq.write_table(pa.Table.from_pylist(out_rows, schema=TOK_SCHEMA), os.path.join(args.out, 'tokenized', 'part-0.parquet'))
    with open(os.path.join(args.out, 'trajectories.jsonl'), 'w') as f:
        for m in meta:
            f.write(json.dumps(m) + '\n')
    stats['distinct_tasks'] = len(per_task)
    json.dump(dict(stats), open(os.path.join(args.out, 'stats.json'), 'w'), indent=1)
    print(json.dumps(dict(stats), indent=1))


if __name__ == '__main__':
    main()
