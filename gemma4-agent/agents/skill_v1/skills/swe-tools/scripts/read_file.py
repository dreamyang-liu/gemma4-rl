"""read_file: read a file or a 1-indexed line range (same limits and format as the native read_file)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _budget import annotate, record  # noqa: E402
from _common import MAX_FILE_CHARS, MAX_FILE_LINES, REPO, args, emit, error, ok, repo_path, to_int  # noqa: E402

a = args(['filepath', 'start_line', 'end_line'])
filepath = a.get('filepath', '')
n, used = record(REPO, 'read_file', '%s:%s-%s' % (filepath, a.get('start_line', ''), a.get('end_line', '')))
try:
    start_req, end_req = to_int(a.get('start_line'), 'start_line'), to_int(a.get('end_line'), 'end_line')
    path = repo_path(filepath)
    if not os.path.isfile(path):
        raise FileNotFoundError("File '%s' not found in workspace." % filepath)
    with open(path, encoding='utf-8', errors='replace') as f:
        lines = f.read().splitlines()
    total = len(lines)
    if total == 0:
        res = ok(filepath=filepath, content='', start_line=1, end_line=0, total_lines=0, is_truncated=False)
    else:
        start = max(start_req if start_req is not None else 1, 1)
        end = min(end_req if end_req is not None else total, total)
        if start > end:
            res = error('ValueError', 'start_line (%d) cannot be greater than end_line (%d)' % (start, end), {})
        else:
            truncated = False
            if end - start + 1 > MAX_FILE_LINES:
                end, truncated = start + MAX_FILE_LINES - 1, True
            snippet = '\n'.join(lines[start - 1:end])
            if len(snippet) > MAX_FILE_CHARS:
                snippet, truncated = snippet[:MAX_FILE_CHARS], True
            res = ok(filepath=filepath, content=snippet, start_line=start, end_line=end, total_lines=total,
                     is_truncated=truncated)
except Exception as e:  # noqa: BLE001 - reported to the model like the native tool does
    res = error('FileReadError', str(e))
emit(annotate(res, REPO, n, used))
