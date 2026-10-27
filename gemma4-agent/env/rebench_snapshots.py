"""Build swegemma-style repo snapshots (<instance_id>.tgz) from the rebench task images.

Like the competition snapshots: the image's repo working tree (incl. untracked build artifacts) with a fresh .git
holding only base_commit's history (git fast-export | fast-import on branch main), so commits after base_commit and
unreachable objects in the image never reach the agent. fast-export strips signatures, so commit SHAs change; the
HEAD tree is checked against base_commit's tree. Where fast-import rejects old history (malformed idents) or the tree
drifts, a --no-local --single-branch clone of base_commit is used instead (same objects and SHAs).
usage: python -I rebench_snapshots.py TASKS.jsonl OUT_DIR [--workers 48]"""
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

tasks_path, out_dir = sys.argv[1:3]
workers = int(sys.argv[sys.argv.index('--workers') + 1]) if '--workers' in sys.argv else 48
os.makedirs(out_dir, exist_ok=True)

SCRIPT = r'''
set -euo pipefail
R="$1"; BC="$2"; OUT="$3"
cd "$R"
head=$(git rev-parse HEAD)
tree=$(git rev-parse "$BC^{tree}")
dirty=$(git status --porcelain --untracked-files=no | wc -l)
untracked=$(git status --porcelain --untracked-files=all | grep -c '^??' || true)
git branch -f _snap "$BC" >/dev/null
rm -rf /tmp/ng /tmp/ws && mkdir -p /tmp/ng /tmp/ws
method=fast-export
if ! { git -C /tmp/ng init -q &&
       git fast-export --signed-tags=strip --tag-of-filtered-object=drop --reencode=yes refs/heads/_snap |
         git -C /tmp/ng fast-import --quiet &&
       git -C /tmp/ng branch -m _snap main &&
       git -C /tmp/ng symbolic-ref HEAD refs/heads/main &&
       test "$(git -C /tmp/ng rev-parse 'main^{tree}')" = "$tree"; } 2>/tmp/fe.err; then
  # malformed idents or tree drift in old history: clone only base_commit's ancestry (original objects and SHAs)
  method=clone
  rm -rf /tmp/ng
  git clone -q --no-local --single-branch --no-tags --branch _snap "$R" /tmp/ng
  git -C /tmp/ng remote remove origin
  git -C /tmp/ng branch -m _snap main
  git -C /tmp/ng reflog expire --expire=now --all
  git -C /tmp/ng gc -q --prune=now
fi
tar --exclude=./.git -cf - . | tar -xf - -C /tmp/ws
mv /tmp/ng/.git /tmp/ws/.git
cd /tmp/ws
git reset -q
test "$(git rev-parse HEAD^{tree})" = "$tree"
n_commits=$(git rev-list --count HEAD)
tar czf "$OUT.tmp" -C /tmp/ws .
mv "$OUT.tmp" "$OUT"
echo "{\"head\": \"$head\", \"dirty\": $dirty, \"untracked\": $untracked, \"commits\": $n_commits, \"method\": \"$method\"}"
'''


def build(r):
    iid = r['instance_id']
    img = f'rebench/{iid.lower()}:latest'
    out = os.path.join(out_dir, f'{iid}.tgz')
    if os.path.exists(out):
        return {'instance_id': iid, 'status': 'exists'}
    wd = subprocess.run(['docker', 'image', 'inspect', '--format', '{{.Config.WorkingDir}}', img],
                        capture_output=True, text=True).stdout.strip()
    res = subprocess.run(['docker', 'run', '--rm', '--network', 'none', '-u', '0', '-v', f'{os.path.abspath(out_dir)}:/out',
                          '--entrypoint', 'bash', img, '-c', SCRIPT, '_', wd, r['base_commit'], f'/out/{iid}.tgz'],
                         capture_output=True, text=True, timeout=1800)
    if res.returncode != 0:
        return {'instance_id': iid, 'status': 'failed', 'repo_dir': wd, 'err': (res.stderr or res.stdout)[-600:]}
    info = json.loads(res.stdout.strip().splitlines()[-1])
    return {'instance_id': iid, 'status': 'ok', 'repo_dir': wd, 'head_is_base': info['head'] == r['base_commit'],
            **info, 'mb': round(os.path.getsize(out) / 1e6, 1)}


rows = [json.loads(l) for l in open(tasks_path)]
with ThreadPoolExecutor(workers) as ex, open(os.path.join(out_dir, 'build_log.jsonl'), 'a') as log:
    for i, res in enumerate(ex.map(build, rows)):
        log.write(json.dumps(res) + '\n')
        log.flush()
        if res['status'] == 'failed' or i % 50 == 0:
            print(i + 1, res, flush=True)
subprocess.run(['docker', 'run', '--rm', '-v', f'{os.path.abspath(out_dir)}:/o', 'alpine:3', 'chown', '-R',
                f'{os.getuid()}:{os.getgid()}', '/o'])
print('done', flush=True)
