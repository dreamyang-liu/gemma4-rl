"""One-shot repository exploration for an issue.

usage: explore.py --terms "Name1,func_name,Class.method,some error text" [--max-chars 14000]

Prints, for the given terms: where they occur in source/tests/docs, the full source of
matching definitions, related existing test files and their test names, the package's own
exception/warning classes (to reuse instead of inventing new ones), and the CHANGELOG head.
"""

import argparse
import ast
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import split_list, Budget, classify, find_defs, find_root, git_files, node_span, numbered, parse, read  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--terms', '--targets', '--query', dest='terms', required=True, help='comma-separated identifiers / strings from the issue')
ap.add_argument('--max-chars', type=int, default=14000)
ap.add_argument('--def-lines', type=int, default=70, help='max lines printed per definition')
args, _ = ap.parse_known_args()

terms = split_list(args.terms)
root = find_root()
files = git_files(root)
src, tests, docs = classify(files)
docs = [d for d in docs if not re.match(r'docs/(?!en/)[a-z]{2}(-[a-z]+)?/', d)]  # skip translations
texts = {f: read(root, f) for f in src + tests + docs}
out = Budget(args.max_chars)

pkgs = sorted({f.split('/')[0] if not f.startswith('src/') else '/'.join(f.split('/')[:2]) for f in src if '/' in f})
out.add('Repository', f'root={root}  source_py={len(src)}  test_py={len(tests)}  docs={len(docs)}  packages={pkgs[:8]}')


def grep(term: str, pool: list[str]):
    """Literal search; falls back to case-insensitive when nothing matches."""
    for flags in (0, re.I):
        pat = re.compile(re.escape(term), flags)
        hits = [(f, i + 1, line.strip()) for f in pool for i, line in enumerate(texts[f].splitlines()) if pat.search(line)]
        if hits:
            # definitions first, plain imports last
            hits.sort(key=lambda h: (0 if re.match(r'(async\s+)?(def|class)\s', h[2]) else
                                     2 if re.match(r'(from\s+\S+\s+)?import\s', h[2]) else 1))
            return hits, bool(flags)
    return [], False


def_nodes = {}
occ_cap = max(900, int(args.max_chars * 0.45 / max(1, len(terms))))  # keep room for definitions
for term in terms:
    lines = []
    s_hits, ci = grep(term, src)
    if s_hits:
        per = Counter(f for f, _, _ in s_hits)
        lines.append(f'source: {len(s_hits)} hits{" (case-insensitive)" if ci else ""} in ' +
                     ', '.join(f'{f}({n})' for f, n in per.most_common(8)))
        for f, ln, txt in s_hits[:14]:
            lines.append(f'  {f}:{ln}: {txt[:150]}')
    else:
        lines.append('source: no hits')
    t_hits, _ = grep(term, tests)
    if t_hits:
        per = Counter(f for f, _, _ in t_hits)
        lines.append(f'tests: {len(t_hits)} hits in ' + ', '.join(f'{f}({n})' for f, n in per.most_common(6)))
        for f, ln, txt in t_hits[:5]:
            lines.append(f'  {f}:{ln}: {txt[:150]}')
    d_hits, _ = grep(term, docs)
    if d_hits:
        per = Counter(f for f, _, _ in d_hits)
        lines.append(f'docs/scripts: {len(d_hits)} hits in ' + ', '.join(f'{f}({n})' for f, n in per.most_common(5)))
    body = '\n'.join(lines)
    if len(body) > occ_cap:
        body = body[:occ_cap].rsplit('\n', 1)[0] + '\n  ...[more hits; narrow the term]'
    out.add(f'Occurrences of "{term}"', body)
    if re.fullmatch(r'[A-Za-z_][\w.]*', term):
        def_nodes[term] = find_defs(root, src, term)[:3]

# Full source of matching definitions (replaces several read_file calls).
for term, hits in def_nodes.items():
    for f, node in hits:
        start, end = node_span(node)
        src_lines = texts[f].splitlines()
        if isinstance(node, ast.ClassDef) and end - start + 1 > args.def_lines:
            members = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            body = numbered(src_lines, start, min(end, start + 25)) + '\n  methods: ' + ', '.join(
                f'{m.name}@{m.lineno}' for m in members)
        else:
            stop = min(end, start + args.def_lines - 1)
            body = numbered(src_lines, start, stop) + (f'\n  ...[{end - stop} more lines, use show.py]' if end > stop else '')
        out.add(f'Definition {term}  ({f}:{start}-{end})', body, min_keep=600)

# Related existing tests: by file name or by importing modules where terms were defined/found.
mods = {f for hits in def_nodes.values() for f, _ in hits}
mods |= {f for t in terms for f, _, _ in grep(t, src)[0][:3]}
stems = {os.path.splitext(os.path.basename(m))[0].lstrip('_') for m in mods}
scored = Counter()
for t in tests:
    base = os.path.basename(t).lower()
    scored[t] += sum(3 for s in stems if s and s.lower() in base)
    scored[t] += sum(2 for term in terms if term.lower() in base)
    scored[t] += sum(1 for term in terms if term in texts[t])
rel = [t for t, sc in scored.most_common(6) if sc > 0]
if rel:
    body = []
    for t in rel[:4]:
        names = re.findall(r'^\s*(?:async\s+)?def (test_\w+)', texts[t], re.M)
        body.append(f'{t}: {len(names)} tests: ' + ', '.join(names[:25]) + (' ...' if len(names) > 25 else ''))
    body += [f'{t}' for t in rel[4:]]
    body.append(f'run with check.py --tests "{",".join(rel[:2])}"')
    out.add('Related existing tests', '\n'.join(body))

# Package's own exception / warning classes and message conventions.
exc = []
for f in src:
    tree = parse(root, f)
    if tree is None:
        continue
    for n in ast.walk(tree):
        if isinstance(n, ast.ClassDef):
            bases = [ast.unparse(b) for b in n.bases]
            if any(re.search(r'(Error|Exception|Warning)$', b) for b in bases) or re.search(r'(Error|Warning)$', n.name):
                exc.append(f'{n.name}({", ".join(bases)}) {f}:{n.lineno}')
if exc:
    out.add('Exception/warning classes defined in the package (reuse these)', '\n'.join(exc[:40]))

for cl in ('CHANGELOG.md', 'CHANGES.md', 'HISTORY.md', 'docs/en/docs/release-notes.md', 'NEWS.md'):
    if (root / cl).exists():
        head = read(root, cl).splitlines()[:22]
        out.add(f'{cl} (head)', '\n'.join(head), min_keep=200)
        break

out.emit()
