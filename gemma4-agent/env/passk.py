"""pass@k over replicate runs: python3 env/passk.py 'runs/pk_r*_s*'  (dirs grouped by rep prefix pk_rNN)."""
import collections, glob, json, math, re, sys
pat = sys.argv[1] if len(sys.argv) > 1 else 'runs/pk_r*_s*'
res = collections.defaultdict(dict)  # task -> rep -> resolved
dur = []; tout = 0
for d in glob.glob(pat):
    rep = re.search(r'_r(\d+)_', d).group(1)
    try:
        for x in map(json.loads, open(f'{d}/results.jsonl')):
            res[x['id']][rep] = bool(x['resolved']); dur.append(x['duration_s']); tout += x['duration_s'] >= 595
    except FileNotFoundError:
        pass
def pass_at_k(n, c, k):
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)
ns = [len(v) for v in res.values()]
print(f'tasks with results: {len(res)}  attempts: {sum(ns)}  min/max attempts per task: {min(ns) if ns else 0}/{max(ns) if ns else 0}'
      f'  timeouts(>=595s): {tout}  mean_s: {sum(dur)/max(1,len(dur)):.0f}')
full = {t: v for t, v in res.items() if len(v) >= 1}
for k in (1, 2, 4, 8, 16):
    vals = [pass_at_k(len(v), sum(v.values()), k) for v in full.values() if len(v) >= k]
    if vals:
        print(f'pass@{k:<2} = {sum(vals)/len(vals):.3f}  (over {len(vals)} tasks with >= {k} attempts; x106 = {106*sum(vals)/len(vals):.1f})')
dist = collections.Counter(sum(v.values()) for v in full.values() if len(v) == 16)
if dist:
    print('solve-count distribution over 16 attempts (tasks with all 16):', dict(sorted(dist.items())))
    for p in ('fastapi', 'rich', 'requests'):
        ts = [t for t, v in full.items() if t.startswith(p) and len(v) == 16]
        if ts: print(f'  {p}: pass@1 {sum(sum(full[t].values())/16 for t in ts)/len(ts):.3f}  pass@16 {sum(any(full[t].values()) for t in ts)}/{len(ts)}')
