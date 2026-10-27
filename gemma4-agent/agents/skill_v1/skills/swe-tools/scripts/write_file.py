"""write_file: create or overwrite a file, creating parent directories (same format as the native write_file)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _budget import annotate, record  # noqa: E402
from _common import REPO, args, emit, error, ok, repo_path  # noqa: E402

a = args(['filepath', 'content'])
filepath, content = a.get('filepath', ''), a.get('content', '')
n, used = record(REPO, 'write_file', '%s (%d chars)' % (filepath, len(content)))
try:
    path = repo_path(filepath)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    res = ok(filepath=filepath, size=len(content))
except Exception as e:  # noqa: BLE001
    res = error('FileWriteError', str(e))
emit(annotate(res, REPO, n, used))
