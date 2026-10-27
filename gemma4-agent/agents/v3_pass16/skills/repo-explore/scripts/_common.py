"""Shared helpers for repo-explore scripts (stdlib only)."""

import ast
import os
import shutil
import tempfile
import re
import subprocess
import sys
from pathlib import Path

DOC_DIRS = ('docs/', 'docs_src/', 'doc/', 'examples/')


def install_cli() -> None:
    """Copy the scripts to $TMPDIR/repo-explore so later calls can use run_command
    (`python3 /tmp/repo-explore/show.py ...`) instead of run_skill_script."""
    here = Path(os.path.dirname(os.path.abspath(__file__)))
    dst = Path(tempfile.gettempdir()) / 'repo-explore'
    try:
        dst.mkdir(parents=True, exist_ok=True)
        for f in here.glob('*.py'):
            shutil.copyfile(f, dst / f.name)
    except OSError:
        pass


def find_root() -> Path:
    """The skill runs from a temp dir; locate the repository workspace."""
    cands = [p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) if p]
    cands += [os.environ.get('PWD', ''), '/workspace', os.getcwd()]
    for c in cands:
        if c and (Path(c) / '.git').exists():
            install_cli()
            return Path(c).resolve()
    sys.exit('repo-explore: cannot locate repository root (no .git found)')


def git_files(root: Path) -> list[str]:
    out = subprocess.run(['git', 'ls-files', '-co', '--exclude-standard'], cwd=root,
                         capture_output=True, text=True).stdout
    return [f for f in out.splitlines() if f and (root / f).is_file()]


def is_test(path: str) -> bool:
    name = path.rsplit('/', 1)[-1]
    return bool(re.search(r'(^|/)(tests?|testing)/', path)) or name.startswith('test_') or name.endswith('_test.py') \
        or name == 'conftest.py'


def is_doc(path: str) -> bool:
    return path.startswith(DOC_DIRS) or path.endswith(('.md', '.rst', '.txt'))


def classify(files: list[str]) -> tuple[list[str], list[str], list[str]]:
    """-> (source .py, test .py, docs files)."""
    src, tests, docs = [], [], []
    for f in files:
        if f.endswith('.py'):
            if is_test(f):
                tests.append(f)
            elif is_doc(f) or f.startswith(('scripts/', 'benchmarks/', 'tools/')):
                docs.append(f)
            else:
                src.append(f)
        elif f.endswith(('.md', '.rst')):
            docs.append(f)
    return src, tests, docs


def read(root: Path, f: str) -> str:
    try:
        return (root / f).read_text(encoding='utf-8', errors='replace')
    except OSError:
        return ''


_AST_CACHE: dict[str, ast.Module | None] = {}


def parse(root: Path, f: str) -> ast.Module | None:
    if f not in _AST_CACHE:
        try:
            _AST_CACHE[f] = ast.parse(read(root, f))
        except (SyntaxError, ValueError):
            _AST_CACHE[f] = None
    return _AST_CACHE[f]


def find_defs(root: Path, files: list[str], dotted: str) -> list[tuple[str, ast.AST]]:
    """Find class/def (any depth) or module/class-level assignment for 'name' or 'Class.method'."""
    parts = dotted.split('.')
    hits = []
    for f in files:
        tree = parse(root, f)
        if tree is None:
            continue
        if len(parts) == 1:
            hits += [(f, n) for n in _named(tree) if _name_of(n) == parts[0]]
        else:
            for cls in ast.walk(tree):
                if isinstance(cls, ast.ClassDef) and cls.name == parts[-2]:
                    hits += [(f, n) for n in cls.body if _name_of(n) == parts[-1]]
    return hits


def _named(tree: ast.Module):
    """Defs at any depth; assignments only at module or class-body level."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield node
    for body in [tree.body] + [n.body for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        for node in body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                yield node


def _name_of(node: ast.AST) -> str | None:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    return None


def node_span(node: ast.AST) -> tuple[int, int]:
    start = min([node.lineno] + [d.lineno for d in getattr(node, 'decorator_list', [])])
    return start, getattr(node, 'end_lineno', node.lineno)


def numbered(text_lines: list[str], start: int, end: int) -> str:
    return '\n'.join(f'{i:5d}  {text_lines[i - 1]}' for i in range(start, min(end, len(text_lines)) + 1))


class Budget:
    """Collects output sections under a total character budget."""

    def __init__(self, limit: int):
        self.limit, self.parts, self.used, self.dropped = limit, [], 0, []

    def add(self, title: str, body: str, *, min_keep: int = 400) -> None:
        body = body.rstrip()
        if not body:
            return
        room = self.limit - self.used - len(title) - 8
        if room < min_keep:
            self.dropped.append(title)
            return
        if len(body) > room:
            body = body[:room].rsplit('\n', 1)[0] + '\n  ...[section truncated]'
        chunk = f'## {title}\n{body}\n'
        self.parts.append(chunk)
        self.used += len(chunk)

    def emit(self) -> None:
        out = '\n'.join(self.parts)
        if self.dropped:
            out += f'\n[omitted for space: {", ".join(self.dropped)}]\n'
        sys.stdout.write(out)


def split_list(raw: str) -> list[str]:
    """Accept 'a,b', '["a", "b"]' or "['a','b']" (models sometimes pass lists)."""
    raw = raw.strip()
    if raw.startswith('[') and raw.endswith(']'):
        raw = raw[1:-1]
    items = [t.strip().strip('`\'"').strip() for t in re.split(r'[,\n]', raw)]
    return [t for t in items if t]
