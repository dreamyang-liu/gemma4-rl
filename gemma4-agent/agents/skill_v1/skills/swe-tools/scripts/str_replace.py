"""str_replace: replace a unique block of text in an existing file, using the harness's own matching engine
(exact, then whitespace-tolerant, then regex, then JSON-unescaped) - same behaviour as the native edit_file."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _budget import annotate, record  # noqa: E402
from _common import MAX_OUTPUT_CHARS, REPO, args, emit, error, ok, repo_path  # noqa: E402
from _edit import apply_replacement  # noqa: E402

a = args(['filepath', 'old_string', 'new_string', 'allow_multiple'])
filepath = a.get('filepath', '')
n, used = record(REPO, 'str_replace', '%s (%d -> %d chars)' % (filepath, len(a.get('old_string', '')),
                                                               len(a.get('new_string', ''))))
try:
    path = repo_path(filepath)
    if not os.path.isfile(path):
        raise FileNotFoundError("File '%s' not found in workspace." % filepath)
    with open(path, encoding='utf-8', errors='replace', newline='') as f:
        content = f.read()
    if not content:
        raise ValueError("File '%s' is empty. Use write_file to populate an empty or newly created file." % filepath)
    allow_multiple = str(a.get('allow_multiple', 'false')).strip().lower() in ('true', '1', 'yes')
    r = apply_replacement(content, a.get('old_string', ''), a.get('new_string', ''), filepath=filepath,
                          allow_multiple=allow_multiple)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write(r.new_content)
    diff = r.diff
    res = ok(filepath=filepath, occurrences=r.occurrences, strategy=r.strategy, diff=diff[:MAX_OUTPUT_CHARS],
             is_truncated=len(diff) > MAX_OUTPUT_CHARS)
except Exception as e:  # noqa: BLE001
    res = error('FileEditError', str(e))
emit(annotate(res, REPO, n, used))
