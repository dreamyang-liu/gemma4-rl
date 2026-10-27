"""Localization hints (files + enclosing functions/classes) from Kimi's successful patches -> tasks file with hints_text.
usage: python -I make_hints.py ROOT OUT_TASKS_JSONL"""
import glob, json, os, re, sys
import pyarrow.parquet as pq

root, out = sys.argv[1], sys.argv[2]
tasks = {json.loads(l)['instance_id']: json.loads(l) for l in open(os.path.join(root, 'dev_tasks.jsonl'))}
patches = {}
for d in sorted(glob.glob(os.path.join(root, 'runs/k_*'))):
    f = os.path.join(d, 'results.jsonl')
    if not os.path.exists(f):
        continue
    for l in open(f):
        r = json.loads(l)
        if r['resolved'] and r['id'] not in patches:
            t = pq.read_table(os.path.join(d, 'submission.parquet')).to_pylist()
            p = [x for x in t if x.get('id') == r['id']] or t
            patches[r['id']] = p[0].get('prediction') or ''


def locate(patch):
    files, cur = {}, None
    for line in patch.splitlines():
        m = re.match(r'^\+\+\+ b/(.+)$', line)
        if m:
            cur = m.group(1); files.setdefault(cur, [])
            continue
        m = re.match(r'^@@ [^@]+ @@\s*(.*)$', line)
        if m and cur:
            ctx = m.group(1)
            n = re.search(r'(?:def|class)\s+(\w+)', ctx)
            if n and n.group(1) not in files[cur]:
                files[cur].append(n.group(1))
        m = re.match(r'^\+\s*(?:async\s+)?(?:def|class)\s+(\w+)', line)
        if m and cur and m.group(1) not in files[cur]:
            files[cur].append(m.group(1))
    import keyword
    drop = r'(^|/)tests?/|test_[^/]*\.py$|conftest\.py$|(^|/)_?(repro|tmp|scratch)[^/]*$'
    return {f: [x for x in s if not keyword.iskeyword(x)] for f, s in files.items() if not re.search(drop, f)}


with open(out, 'w') as fo:
    for tid, p in sorted(patches.items()):
        loc = locate(p)
        parts = [f'`{f}`' + (f" ({', '.join(f'`{s}`' for s in syms[:4])})" if syms else '') for f, syms in loc.items()]
        hint = 'The change belongs in ' + '; '.join(parts) + '.'
        t = dict(tasks[tid]); t['hints_text'] = hint
        fo.write(json.dumps(t) + '\n')
        print(tid, '|', hint)
