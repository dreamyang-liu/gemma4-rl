"""Compaction behaviour per source on the same rebench tasks (shell_v2): how often ADK compacts, at which agent step,
the reported prompt tokens at the trigger, and for the adapted Bedrock proxy how often it injects thoughts that vLLM's
template would hide (assistant thoughts before the last user message, e.g. after a continuation nudge or a compaction
"For context" turn).
usage: python3 analysis/compaction_compare.py [ROOT]"""
import glob
import json
import os
import statistics as st
import sys

os.chdir(sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOURCES = {'gemma-vllm (reference)': 'runs/local_logs/lsh_*.jsonl',
           'gemma-bedrock raw': 'runs/bedrock_logs/rbsh_*.jsonl',
           'gemma-bedrock adapted': 'runs/bedrock_logs_v/col_gv_*.jsonl',
           'qwen3-coder (gemma counts)': 'runs/cand_logs/g4-qcproxy/col_qc*.jsonl',
           'glm-5 hard tasks (gemma counts)': 'runs/cand_logs/g4-glmproxy/col_glm_*.jsonl'}


def attempts(pattern):
    """Split each worker log into attempts: a new attempt starts with a [system, user] request."""
    for f in sorted(glob.glob(pattern)):
        cur = None
        for line in open(f):
            r = json.loads(line)
            msgs = r['request'].get('messages', [])
            if len(msgs) == 2 and msgs[1]['role'] == 'user':
                if cur:
                    yield cur
                cur = []
            if cur is not None and r.get('status') == 200:
                cur.append(r)
        if cur:
            yield cur


for name, pat in SOURCES.items():
    n_att, n_comp, first_step, trig_tokens, hidden_inj, adapted_reqs = 0, [], [], [], 0, 0
    at_step, n_steps = {}, []
    for att in attempts(pat):
        n_att += 1
        steps = comps = 0
        prev_prompt = None
        for r in att:
            req = r['request']
            if not req.get('tools'):  # summarizer call = one compaction
                comps += 1
                if comps == 1:
                    first_step.append(steps)
                if prev_prompt is not None:
                    trig_tokens.append(prev_prompt)
                continue
            steps += 1
            prev_prompt = (r['chat_response'].get('usage') or {}).get('prompt_tokens')
            if steps in (5, 10, 15, 20) and comps == 0:  # context size before any compaction
                at_step.setdefault(steps, []).append(prev_prompt or 0)
            if 'adapted' in name:
                msgs = req['messages']
                last_user = max((i for i, m in enumerate(msgs) if m['role'] == 'user'), default=-1)
                adapted_reqs += 1
                hidden_inj += any(m['role'] == 'assistant' and (m.get('reasoning') or m.get('reasoning_content'))
                                  for m in msgs[:last_user])
        n_comp.append(comps)
        n_steps.append(steps)
    print(f'== {name}: {n_att} attempts')
    print(f'  compactions per attempt: mean {st.mean(n_comp):.2f}; attempts with >=1: {sum(c > 0 for c in n_comp) / n_att:.0%}')
    if first_step:
        print(f'  agent step of first compaction: p25/p50/p75 {sorted(first_step)[len(first_step) // 4]} / '
              f'{st.median(first_step)} / {sorted(first_step)[3 * len(first_step) // 4]}')
    if trig_tokens:
        print(f'  reported prompt tokens at trigger: p50 {st.median(trig_tokens):.0f}, min {min(trig_tokens)}')
    print(f'  model calls per attempt p50 {st.median(n_steps)}; prompt tokens at step 5/10/15/20 (no compaction yet): '
          + ' / '.join(f'{st.median(at_step[k]):.0f}' for k in (5, 10, 15, 20) if at_step.get(k)))
    if adapted_reqs:
        print(f'  requests where thoughts are injected that vLLM would hide: {hidden_inj / adapted_reqs:.1%}')
