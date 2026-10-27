"""Merge LLM filter verdicts (ext/rebench_filter/out_*.jsonl) with the source tasks; write filtered subsets.
usage: python -I rebench_merge.py SRC.jsonl FILTER_DIR"""
import collections, glob, json, os, re, sys

src, d = sys.argv[1:3]
rows = {json.loads(l)['instance_id']: json.loads(l) for l in open(src)}
v = {}
bad = collections.Counter()
for f in sorted(glob.glob(os.path.join(d, 'out_*.jsonl'))):
    ids = open(os.path.join(d, 'md_' + f[-8:-6], 'ids.txt')).read().split()
    got = [json.loads(l) for l in open(f) if l.strip()]
    if [g['instance_id'] for g in got] != ids:
        bad['order_or_count_mismatch:' + os.path.basename(f)] += 1
    for g in got:
        s = [g.get(k) for k in ('similarity', 'solvable', 'spec_clear')]
        if not all(isinstance(x, int) and 0 <= x <= 3 for x in s):
            bad['bad_scores'] += 1
            continue
        rule = 'keep' if min(s) >= 2 else ('maybe' if sorted(s)[0] == 1 and sorted(s)[1] >= 2 else 'drop')
        bad['verdict_fixed'] += rule != g.get('verdict')
        g['verdict'] = rule
        v[g['instance_id']] = g
missing = set(rows) - set(v)
print('judged', len(v), '/', len(rows), 'missing', len(missing), dict(bad))

# same issue under several instance ids (backports): same repo and the same normalised problem statement, or gold
# patches whose changed lines overlap >= 80%; keep the best-scored one
norm = lambda s: re.sub(r'\W+', ' ', s.lower())[:400]
plines = lambda p: {l[1:].strip() for l in p.splitlines() if l[:1] in '+-' and not l.startswith(('+++', '---')) and l[1:].strip()}
seen, dup = collections.defaultdict(list), set()
for iid in sorted(v, key=lambda i: (-sum(v[i][k] for k in ('similarity', 'solvable', 'spec_clear')), i)):
    r = rows[iid]
    k, pl = norm(r['problem_statement']), plines(r['patch'])
    if any(k == k2 or (pl and len(pl & pl2) >= 0.8 * min(len(pl), len(pl2))) for k2, pl2 in seen[r['repo']]):
        dup.add(iid)
    else:
        seen[r['repo']].append((k, pl))
c = collections.Counter(g['verdict'] for g in v.values())
print('verdicts', dict(c), '| duplicate issues', len(dup), '| keep after dedup', sum(g['verdict'] == 'keep' and i not in dup for i, g in v.items()))
reasons = collections.Counter(r for g in v.values() if g['verdict'] != 'keep' for r in g.get('drop_reasons') or [])
print('reasons (non-keep)', reasons.most_common(8))
for k in ('similarity', 'solvable', 'spec_clear'):
    print(k, sorted(collections.Counter(g[k] for g in v.values()).items()))
keep = [i for i, g in v.items() if g['verdict'] == 'keep' and i not in dup]
krepo = collections.Counter(rows[i]['repo'] for i in keep)
print('keep repos', len(krepo), krepo.most_common(20))
print('keep sim=3', sum(v[i]['similarity'] == 3 for i in keep))
for name, ids in (('keep', keep), ('keep_maybe', [i for i, g in v.items() if g['verdict'] in ('keep', 'maybe') and i not in dup])):
    with open(os.path.join(d, f'rebench_{name}.jsonl'), 'w') as f:
        for i in ids:
            f.write(json.dumps({**rows[i], 'llm_filter': v[i]}, ensure_ascii=False) + '\n')
    print(name, len(ids))
with open(os.path.join(d, 'verdicts_all.jsonl'), 'w') as f:
    for i in rows:
        if i in v:
            f.write(json.dumps({**v[i], 'repo': rows[i]['repo'], 'duplicate_issue': i in dup}) + '\n')
