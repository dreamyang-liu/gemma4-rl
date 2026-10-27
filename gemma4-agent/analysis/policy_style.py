"""Behavioral comparison of successful Gemma vs Kimi trajectories (traces): tool mix, reasoning length per call,
calls before the first edit, test runs, text-only (no tool call) turns."""
import collections, glob, json, os, re, sys
import numpy as np

root = sys.argv[1]
os.chdir(root)


def trajs(globs):
    for g in globs:
        for d in glob.glob(g):
            f = os.path.join(d, 'results.jsonl')
            if not os.path.exists(f):
                continue
            for l in open(f):
                r = json.loads(l)
                tr = os.path.join(d, 'results/traces', f"trace_{r['id']}.json")
                if r['resolved'] and os.path.exists(tr):
                    yield r, json.load(open(tr))


def feats(t):
    tools, first_edit, tests, think_chars, msg_chars, nocall = collections.Counter(), None, 0, [], [], 0
    k = 0
    for s in t['steps']:
        if s.get('source') != 'agent' or not isinstance(s.get('metrics'), dict):
            continue
        tcs = s.get('tool_calls') or []
        think_chars.append(len(s.get('reasoning_content') or s.get('message') or ''))
        if not tcs:
            nocall += 1
        for tc in tcs:
            k += 1
            name = tc.get('function_name')
            tools[name] += 1
            if name in ('edit_file', 'write_file') and first_edit is None:
                first_edit = k
            cmd = json.dumps(tc.get('arguments', {}))
            if name == 'run_command' and re.search(r'pytest|unittest', cmd):
                tests += 1
            if name == 'run_command' and re.search(r'sed -i|python3? - <<|cat >|perl -pi', cmd) and first_edit is None:
                first_edit = k
    return tools, first_edit, tests, think_chars, nocall


for name, globs in [('Gemma', ['runs/bs_r[0-7]_s*']),
                    ('Kimi', ['runs/k_fastapi_*', 'runs/k_rich_*', 'runs/k_r[12]_*'])]:
    T, FE, TE, TH, NC, N = collections.Counter(), [], [], [], [], 0
    for r, t in trajs(globs):
        tools, fe, te, th, nc = feats(t)
        T.update(tools); FE.append(fe if fe else 0); TE.append(te); TH += th; NC.append(nc); N += 1
    tot = sum(T.values())
    print(f'\n== {name}: {N} successful trajectories, {tot} tool calls')
    print('tool mix: ' + ', '.join(f'{k} {v / tot:.0%}' for k, v in T.most_common()))
    print(f'message/thought chars per call: p50 {np.median(TH):.0f} p90 {np.percentile(TH, 90):.0f}')
    print(f'first edit at call: p50 {np.median(FE):.0f}  | test runs per traj p50 {np.median(TE):.0f} | text-only turns p50 {np.median(NC):.0f}')
