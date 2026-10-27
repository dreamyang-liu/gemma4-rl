"""Rebench rows -> swegemma tasks.jsonl (+ repo_dir, image) for env/rebench_run.py.

PASS_TO_PASS is restricted to nodes in the test_patch files: swegemma's Phase 2 runs pytest only on those files and
requires every listed node to pass in that run. Only tasks with a built snapshot are written.
usage: python -I rebench_tasks.py ROWS.jsonl SNAPSHOT_DIR OUT.jsonl"""
import json, os, re, sys

rows_path, snap_dir, out_path = sys.argv[1:4]
built = {}
for l in open(os.path.join(snap_dir, 'build_log.jsonl')):
    b = json.loads(l)
    if b['status'] in ('ok', 'exists') and os.path.exists(os.path.join(snap_dir, b['instance_id'] + '.tgz')):
        built.setdefault(b['instance_id'], b.get('repo_dir'))
n = dropped_p2p = 0
with open(out_path, 'w') as f:
    for l in open(rows_path):
        r = json.loads(l)
        iid = r['instance_id']
        if iid not in built:
            continue
        test_files = set(re.findall(r'^\+\+\+ b/(\S+)', r['test_patch'], re.M))
        p2p = [t for t in r.get('PASS_TO_PASS') or [] if t.split('::', 1)[0] in test_files]
        dropped_p2p += len(r.get('PASS_TO_PASS') or []) - len(p2p)
        f.write(json.dumps({
            'instance_id': iid, 'repo': r['repo'], 'base_commit': r['base_commit'],
            'problem_statement': r['problem_statement'], 'patch': r['patch'], 'test_patch': r['test_patch'],
            'hints_text': '', 'created_at': r.get('created_at', ''),
            'FAIL_TO_PASS': r.get('FAIL_TO_PASS') or [], 'PASS_TO_PASS': p2p,
            'repo_dir': built[iid], 'image': f'rebench/{iid.lower()}:latest',
            'llm_filter': r.get('llm_filter')}) + '\n')
        n += 1
print(f'{n} tasks written; PASS_TO_PASS nodes outside test_patch files dropped: {dropped_p2p}')
