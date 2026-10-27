"""Which model's trajectories are closest to Gemma 4 31B (QAT) served by vLLM, as SFT data for Gemma?

Reference: local vLLM sampling (runs/lsh_*, shell_v2 on rebench tasks). Candidates ran shell_v2 on the same tasks.
  A. trajectory and step behaviour on the shared tasks: outcome stats; per-step command classes (JS divergence vs
     the reference); action length in Gemma tokens; parallel tool calls; visible text; thinking.
  B. --score URL: Gemma's NLL of each source's actions. Each sampled step is rendered as vLLM would (Gemma chat
     template, all thoughts removed, enable_thinking off so the generation prompt has an empty thought channel), and
     the vLLM server at URL returns prompt logprobs; we average over the action's tokens (tool calls + text).
usage (gemma4-kaggle image): python -I policy_closeness.py ROOT [--score http://g4-score:8000] [--steps 150]
"""
import argparse
import collections
import glob
import json
import math
import os
import random
import re
import statistics as st
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument('root')
ap.add_argument('--score', default='')
ap.add_argument('--steps', type=int, default=150)
ap.add_argument('--tasks', default='runs/cand_tasks100.txt')
ap.add_argument('--out', default='analysis/policy_closeness.json')
args = ap.parse_args()
os.chdir(args.root)

SOURCES = {  # name: (results glob, model-call log glob)
    'gemma-vllm (reference)': ('runs/lsh_r*', 'runs/local_logs/lsh_*.jsonl'),
    'gemma-bedrock raw': ('runs/rbsh_r*', 'runs/bedrock_logs/rbsh_*.jsonl'),
    'gemma-bedrock adapted': ('runs/cand_gv_s*', 'runs/bedrock_logs_v/cand_gv_*.jsonl'),
    'qwen3-coder-480b': ('runs/cand_qc_s*', 'runs/cand_logs/g4-qcproxy/cand_qc_*.jsonl'),
    'glm-5': ('runs/cand_glm_s*', 'runs/cand_logs/g4-glmproxy/cand_glm_*.jsonl'),
    'kimi-k2.5': ('runs/cand_k25_s*', 'runs/cand_logs/g4-k25proxy/cand_k25_*.jsonl'),
}
TASKS = {l.strip() for l in open(args.tasks) if l.strip()}
task_by_prompt = {}
for l in open('ext/rebench_filter/rebench_tasks_usable.jsonl'):
    t = json.loads(l)
    task_by_prompt[(t['repo'], t['problem_statement'][:200])] = t['instance_id']
PROMPT_RE = re.compile(r'task for repository (\S+)\.\n\nProblem Statement:\n', re.S)


def task_of(user_text):
    m = PROMPT_RE.search(user_text)
    if not m:
        return None
    return task_by_prompt.get((m.group(1), user_text[m.end():m.end() + 200]))


def steps(log_glob):
    """Yield (task_id, request, message) for every tool-enabled model call, segmenting each worker log by task."""
    for f in sorted(glob.glob(log_glob)):
        cur = None
        for line in open(f):
            r = json.loads(line)
            req = r['request']
            msgs = req.get('messages', [])
            if len(msgs) == 2 and msgs[1]['role'] == 'user':
                c = msgs[1]['content']
                cur = task_of(c if isinstance(c, str) else json.dumps(c))
            if r.get('status') != 200 or not req.get('tools') or cur not in TASKS:
                continue
            resp = r.get('chat_response') or {}
            ch = (resp.get('choices') or [{}])[0]
            yield cur, req, ch.get('message') or {}


def cmd_class(name, a):
    if name != 'run_command':
        return name
    c = (a.get('command') or '').strip() if isinstance(a, dict) else ''
    if c in ('', ':'):
        return 'noop'
    if re.search(r"python3? - <<|write_text|sed -i|cat > |cat >>|\btee\b|perl -pi|git apply|\bpatch\b", c):
        return 'edit'
    if re.search(r'pytest|unittest|\btox\b', c):
        return 'test'
    if re.search(r'\b(grep|rg|find|ls|git grep|git ls-files)\b', c):
        return 'search'
    if re.search(r'\b(cat|sed -n|nl|head|tail)\b', c):
        return 'read'
    if re.search(r'\bgit (diff|status|log|show)', c):
        return 'git'
    if re.search(r'\bpython3? ', c):
        return 'run'
    return 'other'


def args_of(tc):
    a = tc.get('function', {}).get('arguments')
    if isinstance(a, str):
        try:
            return json.loads(a or '{}')
        except Exception:
            return {'_raw': a}
    return a or {}


from jinja2.sandbox import ImmutableSandboxedEnvironment  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

MD = 'models/gemma-4-31b-it-qat-w4a16-ct'
TOK = Tokenizer.from_file(f'{MD}/tokenizer.json')
TPL = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True).from_string(open(f'{MD}/chat_template.jinja').read())


def clean(m):
    m = {k: v for k, v in m.items() if k in ('role', 'content', 'tool_calls', 'tool_call_id', 'name')}
    if m.get('tool_calls'):
        m['tool_calls'] = [{'id': tc.get('id'), 'type': 'function',
                            'function': {'name': tc['function']['name'], 'arguments': args_of(tc)}} for tc in m['tool_calls']]
    if m.get('content') is None:
        m['content'] = ''
    return m


def render(msgs, tools, gen):
    return TPL.render(messages=msgs, tools=tools, bos_token='<bos>', enable_thinking=False, add_generation_prompt=gen)


def action_text(req, msg):
    """vLLM generation prompt for this context (thoughts removed, empty thought channel) and the action: the step's
    tool calls in Gemma's tool-call syntax. Visible text is left out because the template places an assistant's text
    after its tool responses, which differs between sources; text-only steps are not scored."""
    ctx = [clean(m) for m in req['messages']]
    act = clean({'role': 'assistant', 'content': '', 'tool_calls': msg.get('tool_calls')})
    assert act.get('tool_calls'), 'no tool call'
    mini = render([{'role': 'user', 'content': 'x'}, act], req['tools'], False)
    a, b = mini.index('<|tool_call>'), mini.rindex('<tool_call|>') + len('<tool_call|>')
    return render(ctx, req['tools'], True), mini[a:b]


# ---------------- A. behaviour ----------------
report = {}
ref_classes = None
rng = random.Random(0)
samples = {}
for name, (res_glob, log_glob) in SOURCES.items():
    att = [json.loads(l) for d in glob.glob(res_glob) if os.path.exists(f'{d}/results.jsonl')
           for l in open(f'{d}/results.jsonl')]
    att = [a for a in att if a['id'] in TASKS]
    crashes = sum('Sandbox execution error' in (a.get('error') or '') for a in att)  # harness aborted the attempt
    att = [a for a in att if 'Sandbox execution error' not in (a.get('error') or '')]
    per_task = collections.defaultdict(list)
    for a in att:
        per_task[a['id']].append(a['resolved'])
    cls, alen, par, vis, think, n = collections.Counter(), [], 0, [], [], 0
    pool = []
    for tid, req, msg in steps(log_glob):
        tcs = msg.get('tool_calls') or []
        n += 1
        par += len(tcs) > 1
        for tc in tcs:
            cls[cmd_class(tc.get('function', {}).get('name'), args_of(tc))] += 1
        if not tcs:
            cls['(no tool call)'] += 1
        vis.append(len(msg.get('content') or ''))
        think.append(len(msg.get('reasoning') or msg.get('reasoning_content') or ''))
        alen.append(len(TOK.encode(json.dumps([args_of(t) for t in tcs]) + (msg.get('content') or ''), add_special_tokens=False).ids))
        pool.append((tid, req, msg))
    tot = sum(cls.values()) or 1
    dist = {k: v / tot for k, v in cls.items()}
    if ref_classes is None:
        ref_classes = dist

    def jsd(p, q):
        keys = set(p) | set(q)
        m = {k: (p.get(k, 0) + q.get(k, 0)) / 2 for k in keys}
        kl = lambda a: sum(a.get(k, 0) * math.log2(a.get(k, 0) / m[k]) for k in keys if a.get(k, 0) > 0)
        return (kl(p) + kl(q)) / 2

    report[name] = {
        'tasks': len(per_task), 'attempts': len(att), 'harness_crashes_excluded': crashes,
        'pass@1': round(sum(sum(v) / len(v) for v in per_task.values()) / max(1, len(per_task)), 3),
        'median_tool_calls': st.median([a['tool_calls'] for a in att]) if att else None,
        'no_patch': round(sum(not a['patch_chars'] for a in att) / max(1, len(att)), 3),
        'steps': n, 'command_mix': {k: round(v, 3) for k, v in sorted(dist.items(), key=lambda x: -x[1])},
        'command_mix_JSD_vs_ref': round(jsd(dist, ref_classes), 4),
        'action_tokens_p50': st.median(alen) if alen else None,
        'parallel_tool_call_steps': round(par / max(1, n), 3),
        'visible_text_chars_p50': st.median(vis) if vis else None,
        'steps_with_thinking': round(sum(t > 0 for t in think) / max(1, n), 3),
    }
    samples[name] = rng.sample(pool, min(args.steps, len(pool)))

# ---------------- B. Gemma NLL of actions ----------------
if args.score:
    for name, pool in samples.items():
        nll, ntok, fails = [], 0, 0
        for tid, req, msg in pool:
            try:
                prompt, action = action_text(req, msg)
                p_ids = TOK.encode(prompt, add_special_tokens=False).ids
                a_ids = TOK.encode(prompt + action, add_special_tokens=False).ids
                if a_ids[:len(p_ids)] != p_ids or len(a_ids) > 32000:
                    fails += 1
                    continue
                body = {'model': 'gemma', 'prompt': a_ids, 'max_tokens': 1, 'temperature': 0, 'prompt_logprobs': 0}
                r = json.load(urllib.request.urlopen(urllib.request.Request(
                    args.score + '/v1/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'}), timeout=600))
                plp = r['choices'][0]['prompt_logprobs']
                lp = [list(plp[i].values())[0]['logprob'] if str(a_ids[i]) not in plp[i] else plp[i][str(a_ids[i])]['logprob']
                      for i in range(len(p_ids), len(a_ids))]
                nll.append(-sum(lp) / len(lp))
                ntok += len(lp)
            except Exception as e:
                fails += 1
                print(name, 'score error', repr(e)[:160])
        report[name]['gemma_action_nll_per_token_mean'] = round(sum(nll) / max(1, len(nll)), 4)
        report[name]['gemma_action_nll_per_token_p50'] = round(st.median(nll), 4) if nll else None
        report[name]['scored_steps'] = len(nll)
        report[name]['score_failures'] = fails
        print(f'{name}: scored {len(nll)} steps ({ntok} action tokens), mean NLL/token {report[name]["gemma_action_nll_per_token_mean"]}', flush=True)

json.dump(report, open(args.out, 'w'), indent=1)
for name, r in report.items():
    print(f'\n== {name}')
    for k, v in r.items():
        print(f'  {k}: {v}')
