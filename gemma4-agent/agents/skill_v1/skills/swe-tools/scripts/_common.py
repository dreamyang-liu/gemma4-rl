"""Argument parsing, repository location and output for the swe-tools scripts (Python 3.7+)."""
import json
import os
import sys

MAX_OUTPUT_CHARS = 5000
MAX_FILE_LINES = 150
MAX_FILE_CHARS = 10000


def repo_dir():
    """The harness runs skill scripts from the repository directory and ADK's wrapper changes into a temp dir
    afterwards, keeping the original directory in its local `_orig_cwd`."""
    f = sys._getframe()
    while f is not None:
        d = f.f_locals.get('_orig_cwd')
        if isinstance(d, str) and os.path.isdir(os.path.join(d, '.git')):
            return d
        f = f.f_back
    return '/workspace'


REPO = repo_dir()


def args(keys):
    """run_skill_script passes {"k": v} as ['--k', 'v', ...]; a value may itself start with '--'."""
    out, argv, i = {}, sys.argv[1:], 0
    while i < len(argv):
        if argv[i].startswith('--') and argv[i][2:] in keys and i + 1 < len(argv):
            out[argv[i][2:]] = argv[i + 1]
            i += 2
        else:
            i += 1
    return out


def repo_path(filepath):
    """Path relative to the repository; a leading /workspace/ is accepted like the native tools do."""
    p = filepath.strip()
    for prefix in ('/workspace/', REPO + '/'):
        if p.startswith(prefix):
            p = p[len(prefix):]
    if p.startswith('/') or p in ('', '.', '/workspace'):
        raise ValueError("Invalid file path '%s': give a path inside the repository." % filepath)
    full = os.path.normpath(os.path.join(REPO, p))
    if not full.startswith(os.path.normpath(REPO) + os.sep):
        raise ValueError("Invalid file path '%s': give a path inside the repository." % filepath)
    return full


def to_int(val, name):
    if val is None or str(val).strip().lower() in ('', 'none', 'null'):
        return None
    try:
        return int(float(str(val).strip()))
    except ValueError:
        raise ValueError('Invalid integer for %s: %r' % (name, val))


def ok(**kw):
    d = {'status': 'ok'}
    d.update(kw)
    return d


def error(error_type, message, details=None):
    d = {'status': 'error', 'error_type': error_type, 'error_message': message}
    if details is not None:
        d['details'] = details
    return d


def emit(payload):
    """Plain text instead of JSON: ADK already wraps stdout in a JSON envelope, and a JSON string inside it gets
    double-escaped. The budget lines come first so they are seen before the output."""
    d = dict(payload)
    lines = []
    if 'budget_warning' in d:
        lines.append('!!! budget_warning: ' + d.pop('budget_warning'))
    if 'progress' in d:
        lines.append('progress: ' + d.pop('progress'))
    lines.append('status: ' + str(d.pop('status', '')))
    blocks = []
    for k, v in d.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                (blocks if isinstance(v2, str) and '\n' in v2 else lines).append((k2, v2))
            continue
        if isinstance(v, str) and ('\n' in v or k in ('stdout', 'stderr', 'content', 'diff')):
            blocks.append((k, v))
        else:
            lines.append('%s: %s' % (k, v))
    out = [x if isinstance(x, str) else '%s: %s' % x for x in lines]
    for k, v in blocks:
        out.append('----- %s -----' % k)
        out.append(v)
    print('\n'.join(out))
