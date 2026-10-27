"""Pre-submit self check.

usage: check.py [--tests "tests/test_x.py,tests/test_y.py::test_z"] [--repro "python3 repro.py"] [--no-auto]

1. Shows the diff stat and compiles every changed/new .py file (catches syntax errors).
2. Imports each touched top-level package.
3. Runs existing tests for the touched modules (auto-detected) plus --tests, via pytest.main.
   Failures are re-run with your changes stashed, to separate REGRESSIONS (new) from PRE-EXISTING ones.
4. Runs --repro and shows its exit code and output tail.
"""

import argparse
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import split_list, find_root, git_files, is_test  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--tests', default='')
ap.add_argument('--repro', default='')
ap.add_argument('--no-auto', action='store_true')
ap.add_argument('--timeout', type=int, default=100)
args, _ = ap.parse_known_args()

root = find_root()
PY = sys.executable or 'python3'


def sh(cmd, timeout=60, **kw):
    try:
        r = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=timeout, **kw)
        return r.returncode, (r.stdout or '') + (r.stderr or '')
    except subprocess.TimeoutExpired as e:
        return 124, f'TIMEOUT after {timeout}s\n' + ((e.stdout or b'').decode(errors='replace') if isinstance(e.stdout, bytes) else (e.stdout or ''))


def tail(text, n):
    return text if len(text) <= n else '...' + text[-n:]


_, stat = sh(['git', 'diff', '--stat', 'HEAD'])
_, untracked = sh(['git', 'ls-files', '--others', '--exclude-standard'])
changed = [l for l in sh(['git', 'diff', '--name-only', 'HEAD'])[1].splitlines() if l.strip()]
new_files = [l for l in untracked.splitlines() if l.strip()]
print('## Diff\n' + (stat.strip() or '(no tracked changes)'))
if new_files:
    print('untracked (NOT part of the patch unless source): ' + ', '.join(new_files[:15]))
src_changed = [f for f in changed + new_files if f.endswith('.py') and not is_test(f)
               and not re.search(r'(^|/)(repro|reproduce|check|debug|scratch)\w*\.py$', f)]
if not src_changed:
    print('WARNING: no source .py changes detected -- the patch would be empty or non-source.')

print('\n## Compile')
bad = 0
for f in src_changed:
    if (root / f).exists():
        rc, o = sh([PY, '-m', 'py_compile', f])
        if rc:
            bad += 1
            print(f'SYNTAX ERROR in {f}:\n{tail(o, 800)}')
print('ok' if not bad else f'{bad} file(s) failed to compile')

print('\n## Import')
mods = []
for f in src_changed:
    m = f[4:] if f.startswith('src/') else f
    m = m[:-3].replace('/', '.').removesuffix('.__init__')
    if (root / f).exists() and re.fullmatch(r'[A-Za-z_][\w.]*', m) and not m.startswith(('docs_src', 'scripts')):
        mods.append(m)
for m in mods[:6]:
    rc, o = sh([PY, '-c', f'import {m}'], timeout=60)
    print(f'import {m}: ' + ('ok' if rc == 0 else 'FAILED\n' + tail(o, 1200)))

tests = split_list(args.tests)
if not args.no_auto:
    all_tests = [f for f in git_files(root) if f.endswith('.py') and is_test(f) and os.path.basename(f).startswith('test')]
    for f in src_changed:
        stem = os.path.splitext(os.path.basename(f))[0].lstrip('_')
        if stem in ('__init__', 'init', ''):
            continue
        cands = [t for t in all_tests if re.fullmatch(rf'test_{re.escape(stem)}\w*\.py', os.path.basename(t))]
        tests += [c for c in sorted(cands, key=len)[:2] if c not in tests]
tests = tests[:5]

RUNNER = ('import sys, pytest; sys.exit(pytest.main(["-q", "--tb=short", "-rfE", "--no-header", '
          '"-p", "no:cacheprovider", "-o", "addopts="] + sys.argv[1:]))')


def run_tests(ids):
    rc, o = sh([PY, '-c', RUNNER, *ids], timeout=args.timeout)
    failed = sorted(set(re.findall(r'^(?:FAILED|ERROR) (\S+)', o, re.M)))
    summary = (re.findall(r'^=*\s*(\d+ (?:passed|failed|errors?|skipped)[^\n]*?)\s*=*$', o, re.M) or [''])[-1]
    return rc, o, failed, summary


if tests and bad:
    print('\n## Tests\nskipped: fix the syntax errors above first')
elif tests:
    print(f'\n## Tests: {" ".join(tests)}')
    rc, o, failed, summary = run_tests(tests)
    print(summary or f'(exit code {rc})')
    if rc not in (0, 5):
        print(tail(o, 3500))
    if failed and changed:
        # Which failures exist without our change? (only tracked edits are stashed)
        sh(['git', 'stash', 'push', '-q'])
        try:
            _, _, base_failed, _ = run_tests(failed[:30])
        finally:
            prc, po = sh(['git', 'stash', 'pop', '-q'])
            if prc:
                print('WARNING: git stash pop failed:', po)
        regress = [f for f in failed if f not in base_failed]
        pre = [f for f in failed if f in base_failed]
        print(f'\nREGRESSIONS caused by your change ({len(regress)}): ' + (', '.join(regress[:15]) or 'none'))
        print(f'pre-existing failures, ignore ({len(pre)}): ' + (', '.join(pre[:10]) or 'none'))
elif not args.no_auto:
    print('\n## Tests\n(no matching existing test files found; pass --tests explicitly)')

if args.repro:
    print(f'\n## Repro: {args.repro}')
    rc, o = sh(['/bin/bash', '-c', args.repro], timeout=args.timeout)
    print(f'exit code {rc}\n{tail(o, 2000)}')
