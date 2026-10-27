"""Manifest of successful rebench trajectories (shell_v2 agent) available as Gemma SFT data.

One row per resolved attempt: source, task, run dir, trace, model-call log, tool calls, duration, tokens, whether the
model's thoughts are present, and an estimated Kaggle time: (prompt + completion tokens) * 900.8 s/M (direct 4xL4
calibration, discussion 746250) + measured tool time. Prompt tokens are vLLM/Gemma counts for col_gv, col_glm and lsh;
raw Bedrock counts (rbsh) undercount by ~1.6x because Bedrock drops earlier thoughts, so their estimate is low.
usage: python3 analysis/build_sft_manifest.py [ROOT]   -> runs/sft_manifest.jsonl + summary
"""
import collections
import glob
import json
import os
import statistics as st
import sys

os.chdir(sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOURCES = {  # name: (run dir glob, proxy log dir, prompt tokens are Gemma/vLLM counts, model thoughts present)
    'gemma_bedrock_adapted': ('runs/col_gv_r*_s*', 'runs/bedrock_logs_v', True, True),
    'glm5_hard_tasks': ('runs/col_glm_*', 'runs/cand_logs/g4-glmproxy', True, False),
    # Qwen3-Coder writes a short note before ~93% of its tool calls; that text is the candidate thought for Gemma
    'qwen3_coder_480b': ('runs/col_qc*', 'runs/cand_logs/g4-qcproxy', True, False),
    # Converse with thinking, earlier thoughts replayed, Gemma-written compaction summaries
    'kimi_k25_converse': ('runs/col_k25[sv]_r*_s*', 'runs/cand_logs/g4-k25cproxy', True, True),
    'glm5_converse': ('runs/col_glm[5v]*_r*_s*', 'runs/cand_logs/g4-glmcproxy', True, True),
    'glm53_converse': ('runs/col_g53s_r*_s*', 'runs/cand_logs/g4-glm53proxy', True, True),
    'gemma_bedrock_raw': ('runs/rbsh_r*', 'runs/bedrock_logs', False, True),
    'gemma_local_vllm': ('runs/lsh_r*', 'runs/local_logs', True, True),
}
import re  # noqa: E402

# Explicit "I remember the upstream fix" statements (analysis/leak_scan.py, strict set). Gemma itself: ~1% of trajectories.
RECALL = [re.compile(p) for p in (
    r"(?i)\b(I (recall|remember) (the|that the|this) (fix|PR|commit|patch|change|upstream)|from memory|let me recall)\b",
    r"(?i)\bupstream (repo|repository|fix|patch|PR|pull request|commit|history|change|implementation|test)s?\b",
    r"(?i)\bthe (actual|real|official) (upstream )?(fix|patch|PR|commit)\b",
    r"(?i)hidden tests? (likely |probably |may |might |will )?(mirror|match|be like|be the same as|come from|is from|are from)")]


def recall_counts(log_path):
    """Per attempt (in the worker's order) the number of model calls with an explicit upstream-recall statement."""
    counts, cur = [], None
    if not os.path.exists(log_path):
        return counts
    for line in open(log_path):
        r = json.loads(line)
        msgs = r['request'].get('messages', [])
        if len(msgs) == 2 and msgs[1]['role'] == 'user':
            if cur is not None:
                counts.append(cur)
            cur = 0
        if cur is not None and r.get('status') == 200 and r['request'].get('tools'):
            m = r['chat_response']['choices'][0]['message']
            t = (m.get('reasoning') or m.get('reasoning_content') or '') + '\n' + (m.get('content') or '')
            cur += any(rx.search(t) for rx in RECALL)
    if cur is not None:
        counts.append(cur)
    return counts


rows = []
for src, (pat, logdir, vllm_counts, thoughts) in SOURCES.items():
    for d in sorted(glob.glob(pat)):
        f = os.path.join(d, 'results.jsonl')
        if not os.path.exists(f):
            continue
        rec = recall_counts(os.path.join(logdir, os.path.basename(d) + '.jsonl'))
        for k, line in enumerate(open(f)):
            r = json.loads(line)
            if not r.get('resolved'):
                continue
            tr = os.path.join(d, 'results', 'traces', f"trace_{r['id']}.json")
            P = C = tool_s = 0
            if os.path.exists(tr):
                t = json.load(open(tr))
                P, C = t['final_metrics']['total_prompt_tokens'], t['final_metrics']['total_completion_tokens']
                for s in t['steps']:
                    tcs, obs = s.get('tool_calls'), s.get('observation')
                    if isinstance(tcs, list) and tcs and isinstance(obs, dict):
                        a, b = tcs[0].get('extra', {}).get('timestamp'), obs.get('extra', {}).get('timestamp')
                        if a and b:
                            tool_s += max(0.0, b - a)
            rows.append({'source': src, 'task': r['id'], 'run_dir': d, 'trace': tr if os.path.exists(tr) else None,
                         'model_log': os.path.join(logdir, os.path.basename(d) + '.jsonl'),
                         'tool_calls': r['tool_calls'], 'duration_s': r['duration_s'],
                         'prompt_tokens': P, 'completion_tokens': C, 'prompt_tokens_are_vllm_counts': vllm_counts,
                         'has_model_thoughts': thoughts, 'kaggle_est_s': round((P + C) * 900.8e-6 + tool_s, 1),
                         'recall_signals': rec[k] if k < len(rec) else None})
with open('runs/sft_manifest.jsonl', 'w') as fo:
    for r in rows:
        fo.write(json.dumps(r) + '\n')

print(f'{len(rows)} successful trajectories -> runs/sft_manifest.jsonl')
print(f"{'source':<24}{'success':>8}{'tasks':>7}{'calls p50':>10}{'est s p50':>10}{'est s p90':>10}{'<=330s':>8}{'recall':>8}  thoughts")
by = collections.defaultdict(list)
for r in rows:
    by[r['source']].append(r)
for src, rs in by.items():
    est = [r['kaggle_est_s'] for r in rs]
    print(f"{src:<24}{len(rs):>8}{len({r['task'] for r in rs}):>7}{st.median(r['tool_calls'] for r in rs):>10}"
          f"{st.median(est):>10.0f}{sorted(est)[int(0.9 * (len(est) - 1))]:>10.0f}{sum(e <= 330 for e in est) / len(est):>8.0%}"
          f"{sum((r['recall_signals'] or 0) > 0 for r in rs) / len(rs):>8.0%}  {'yes' if rs[0]['has_model_thoughts'] else 'no'}")
allt = {r['task'] for r in rows}
clean = [r for r in rows if not r['recall_signals']]
print(f'tasks with at least one successful trajectory (any source): {len(allt)}; '
      f'without recall statements: {len(clean)} trajectories over {len({r["task"] for r in clean})} tasks')
