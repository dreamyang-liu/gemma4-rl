"""Markdown task cards (multi-line, 10 per file) for LLM filtering; the Read tool truncates long JSON lines.
usage: python -I rebench_prep_md.py SRC.jsonl OUT_DIR [--shards 16 --per-file 10]"""
import json, os, re, sys

src, out = sys.argv[1:3]
arg = lambda k, d: int(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
n_shards, per_file = arg('--shards', 16), arg('--per-file', 10)


def pstats(p):
    files = re.findall(r'^\+\+\+ (?:b/)?(\S+)', p, re.M)
    ch = sum(1 for l in p.splitlines() if l[:1] in '+-' and not l.startswith(('+++', '---')))
    return files, ch


def clip(s, n):
    s = (s or '').replace('\r', '').strip()
    return s if len(s) <= n else s[:n] + f'\n...[{len(s) - n} more chars]'


def card(r):
    files, ch = pstats(r['patch'])
    tfiles, tch = pstats(r['test_patch'])
    lm = r.get('meta', {}).get('llm_metadata', {})
    f2p = r.get('FAIL_TO_PASS') or []
    return '\n'.join([
        f"## TASK {r['instance_id']}",
        f"repo: {r['repo']} | created: {(r.get('created_at') or '')[:10]} | publisher difficulty: {r.get('screening_difficulty')} "
        f"| pr_categories: {lm.get('pr_categories')}",
        f"gold patch: {len(files)} file(s), {ch} changed lines: {', '.join(files)}",
        f"test patch: {', '.join(tfiles)} ({tch} lines) | FAIL_TO_PASS: {len(f2p)} e.g. {f2p[:4]}",
        f"publisher LLM flags: detected_issues={lm.get('detected_issues')} test_alignment_issues={lm.get('test_alignment_issues')}",
        '### problem_statement (the ONLY thing the agent sees)', '~~~', clip(r['problem_statement'], 2000), '~~~',
        '### gold patch (excerpt)', '~~~diff', clip(r['patch'], 1300), '~~~',
        '### test patch (excerpt)', '~~~diff', clip(r['test_patch'], 1000), '~~~', ''])


rows = [json.loads(l) for l in open(src)]
per = -(-len(rows) // n_shards)
for s in range(n_shards):
    d = os.path.join(out, f'md_{s:02d}')
    os.makedirs(d, exist_ok=True)
    part = rows[s * per:(s + 1) * per]
    for j in range(0, len(part), per_file):
        with open(os.path.join(d, f'part_{j // per_file:02d}.md'), 'w') as f:
            f.write('\n'.join(card(r) for r in part[j:j + per_file]))
    with open(os.path.join(d, 'ids.txt'), 'w') as f:
        f.write('\n'.join(r['instance_id'] for r in part) + '\n')
print(len(rows), 'tasks;', n_shards, 'shards of', per)
