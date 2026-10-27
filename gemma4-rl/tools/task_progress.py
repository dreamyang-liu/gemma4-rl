"""Per-task progress of an RL run: for tasks attempted at more than one step, compare the pass count (of 8) at the
first and the latest attempt. Reads the harness results.jsonl of every trajectory dir whose name starts with one of the
given run prefixes (steps are the s<k> index in the run names; the replayed steps keep the lost run's indices).

usage: python3 tools/task_progress.py dapo_colo12s_e15_10260006 dapo_colo12s_e15_10260423
"""
import collections
import glob
import json
import os
import re
import statistics as st
import sys

G4A = os.environ.get('G4A') or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'gemma4-agent')
RUNS = os.path.join(G4A, 'runs')
attempts = collections.defaultdict(list)
for pref in sys.argv[1:]:
    groups = collections.defaultdict(list)
    for d in glob.glob(os.path.join(RUNS, f'{pref}_s*w*_g*_k*')):
        f = os.path.join(d, 'results.jsonl')
        m = re.search(r'_s(\d+)w(\d+)_g(\d+)_k(\d+)$', d)
        if not os.path.exists(f) or not m:
            continue
        try:
            r = json.loads(open(f).readline())
        except Exception:
            continue
        groups[(int(m.group(1)), int(m.group(2)), int(m.group(3)))].append((r['id'], bool(r['resolved'])))
    for (s, w, g), rs in groups.items():
        if len(rs) >= 8:
            attempts[rs[0][0]].append((s, sum(ok for _, ok in rs), len(rs)))
multi = {t: sorted(a) for t, a in attempts.items() if len({s for s, _, _ in a}) >= 2}
print(f'tasks seen: {len(attempts)}; attempted at >= 2 steps: {len(multi)}')
if not multi:
    sys.exit(0)
first = [a[0][1] for a in multi.values()]
last = [a[-1][1] for a in multi.values()]
print(f'mean passes/8: first attempt {st.mean(first):.2f} -> latest {st.mean(last):.2f} '
      f'(mean step gap {st.mean(a[-1][0] - a[0][0] for a in multi.values()):.1f}); '
      f'improved {sum(l > f for f, l in zip(first, last))}, same {sum(l == f for f, l in zip(first, last))}, '
      f'worse {sum(l < f for f, l in zip(first, last))}')
buckets = collections.defaultdict(list)
for a in multi.values():
    f, l = a[0][1], a[-1][1]
    buckets['0/8 all-fail' if f == 0 else '8/8 all-pass (retried)' if f == 8 else '1-7/8 mixed'].append((f, l))
for k in ('0/8 all-fail', '1-7/8 mixed', '8/8 all-pass (retried)'):
    v = buckets.get(k)
    if v:
        print(f'  first {k:22s} n={len(v):3d}: mean {st.mean(x for x, _ in v):.2f} -> {st.mean(y for _, y in v):.2f}; '
              f'latest >0: {sum(y > 0 for _, y in v)}, latest 8/8: {sum(y == 8 for _, y in v)}')
print('note: re-drawn tasks skew to those that were not all-pass (the solved-task skip), and 8 samples per attempt '
      'carry ~±1.4 passes of binomial noise; judge trends over many tasks, not single ones.')
