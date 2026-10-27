"""Distribution of successful sampled trajectories: tokens, tool calls, time split (LLM vs tool), Kaggle-time estimate.
Kaggle estimate: (prompt + completion tokens) * 900.8 s/M (Ayush Thakur's direct 4xL4 calibration, discussion 746250)
+ measured tool time."""
import glob, json, os, sys
import numpy as np

KAGGLE_S_PER_TOK = 900.8e-6


def attempts(run_glob):
    for d in glob.glob(run_glob):
        f = os.path.join(d, 'results.jsonl')
        if not os.path.exists(f):
            continue
        for l in open(f):
            r = json.loads(l)
            tr = os.path.join(d, 'results/traces', f"trace_{r['id']}.json")
            if os.path.exists(tr):
                yield d, r, json.load(open(tr))


def stats(r, t):
    steps = t['steps']
    tool_t, prompts, comps = 0.0, [], []
    for s in steps:
        m = s.get('metrics')
        if isinstance(m, dict):
            prompts.append(m.get('prompt_tokens', 0)); comps.append(m.get('completion_tokens', 0))
        obs = s.get('observation')
        tcs = s.get('tool_calls')
        if isinstance(obs, dict) and isinstance(tcs, list) and tcs:
            t0 = tcs[0].get('extra', {}).get('timestamp'); t1 = obs.get('extra', {}).get('timestamp')
            if t0 and t1:
                tool_t += max(0.0, t1 - t0)
    fm = t['final_metrics']
    P, C = fm['total_prompt_tokens'], fm['total_completion_tokens']
    return dict(tool_calls=r['tool_calls'], llm_calls=len(prompts), duration_s=r['duration_s'], tool_s=tool_t,
                llm_s=r['duration_s'] - tool_t, prompt_tok=P, completion_tok=C, total_tok=P + C,
                max_ctx=max(prompts or [0]), out_per_call=float(np.median(comps or [0])),
                kaggle_est_s=(P + C) * KAGGLE_S_PER_TOK + tool_t)


def table(name, rows):
    print(f'\n== {name}: {len(rows)} successful trajectories')
    keys = ['tool_calls', 'llm_calls', 'duration_s', 'llm_s', 'tool_s', 'prompt_tok', 'completion_tok', 'total_tok',
            'max_ctx', 'out_per_call', 'kaggle_est_s']
    print(f"{'metric':<15}" + ''.join(f'{h:>10}' for h in ['p10', 'p25', 'p50', 'p75', 'p90', 'max', 'mean']))
    for k in keys:
        v = np.array([r[k] for r in rows], float)
        q = np.percentile(v, [10, 25, 50, 75, 90])
        print(f'{k:<15}' + ''.join(f'{x:>10.0f}' for x in list(q) + [v.max(), v.mean()]))
    ks = np.array([r['kaggle_est_s'] for r in rows]) / 60
    edges = [0, 3, 4, 5, 5.5, 8, 1e9]
    print('kaggle_est (min): ' + '  '.join(f'{a}-{b if b < 1e8 else "inf"}: {int(((ks >= a) & (ks < b)).sum())}'
                                          for a, b in zip(edges, edges[1:])))
    tc = np.array([r['tool_calls'] for r in rows])
    print('tool calls: ' + '  '.join(f'{a}-{b}: {int(((tc >= a) & (tc <= b)).sum())}' for a, b in
                                     [(0, 10), (11, 15), (16, 20), (21, 25), (26, 28), (29, 99)]))


root = sys.argv[1] if len(sys.argv) > 1 else '.'
os.chdir(root)
groups = {'Gemma 31B (Bedrock, budget-aware v2, 8 min/28 calls)': ['runs/bs_r[0-7]_s*'],
          'Kimi K3 (Bedrock, 30 min/28 calls; Kimi tokenizer)': ['runs/k_fastapi_*', 'runs/k_rich_*', 'runs/k_requests_*', 'runs/k_r[12]_*']}
for name, globs in groups.items():
    rows = [stats(r, t) for g in globs for _, r, t in attempts(g) if r['resolved']]
    table(name, rows)
    if name.startswith('Gemma'):
        allrows = [stats(r, t) for g in globs for _, r, t in attempts(g)]
        table('Gemma 31B, all attempts (for reference)', allrows)
