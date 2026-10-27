"""Print several code locations in one call (no 150-line read_file limit).

usage: show.py --targets "rich/console.py:Console.print, fastapi/routing.py:1480-1560, get_typed_signature"
  path:START-END    line range
  path:Symbol       class/def/assignment in that file (Class.method allowed)
  Symbol            search all source files for the definition
  path              whole file (capped)
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import split_list, Budget, classify, find_defs, find_root, git_files, node_span, numbered, read  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--targets', '--terms', '--target', dest='targets', required=True)
ap.add_argument('--max-chars', type=int, default=20000)
ap.add_argument('--max-lines', type=int, default=220, help='max lines per target')
args, _ = ap.parse_known_args()

root = find_root()
files = None
out = Budget(args.max_chars)


def show(path: str, start: int, end: int, label: str) -> None:
    lines = read(root, path).splitlines()
    if not lines:
        out.add(label, f'(cannot read {path})')
        return
    end = min(end, len(lines))
    stop = min(end, start + args.max_lines - 1)
    body = numbered(lines, start, stop)
    if stop < end:
        body += f'\n  ...[{end - stop} more lines: request {path}:{stop + 1}-{end}]'
    out.add(f'{label}  ({path}:{start}-{end} of {len(lines)})', body, min_keep=300)


for raw in split_list(args.targets):
    raw = raw.replace('/workspace/', '')
    path, _, spec = raw.partition(':')
    if not spec and not (root / path).is_file():
        path, spec = '', raw
    if path and not (root / path).is_file():
        out.add(raw, f'(no such file: {path})')
        continue
    m = re.fullmatch(r'(\d+)\s*-\s*(\d+)', spec)
    if path and m:
        show(path, int(m.group(1)), int(m.group(2)), raw)
    elif path and not spec:
        show(path, 1, 10**9, raw)
    else:
        if files is None:
            files = classify(git_files(root))[0]
        hits = find_defs(root, [path] if path else files, spec)
        if not hits and path:
            hits = find_defs(root, files, spec)
        if not hits:
            out.add(raw, f'(definition "{spec}" not found; try explore.py)')
        for f, node in hits[:3]:
            s, e = node_span(node)
            show(f, s, e, f'{spec}')
out.emit()
