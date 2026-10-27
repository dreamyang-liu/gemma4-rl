"""Call log and budget reminders for the swe-tools scripts.

Every script call is recorded in <repo>/.git/swe_tools_log.json (inside .git, so never part of the patch). The
response of each call gets a "progress" line and, when due, a "budget_warning":
  - no source file modified by call 7 (or 3 minutes): start editing;
  - still none by call 15 (or 5 minutes): MUST edit now;
  - with an edit in place, 5 calls or 90 seconds left: finish and submit;
  - 1 call or 30 seconds left: the next call MUST be submit_patch.
Time is measured from the first tool call (plus a small offset for the first model turn).
"""
import json
import os
import subprocess
import time

MAX_CALLS = 30       # = max_tool_calls in eval_config.yaml
TIME_LIMIT = 480     # = max_time_minutes * 60 in eval_config.yaml
FIRST_TURN_S = 10


def _log_path(repo):
    return os.path.join(repo, '.git', 'swe_tools_log.json')


def record(repo, tool, summary):
    """Append this call to the log; return (call number, seconds used)."""
    path = _log_path(repo)
    try:
        with open(path) as f:
            log = json.load(f)
    except (OSError, ValueError):
        log = {'start': time.time(), 'calls': []}
    log['calls'].append({'tool': tool, 'summary': summary[:200], 't': round(time.time() - log['start'], 1)})
    try:
        with open(path, 'w') as f:
            json.dump(log, f)
    except OSError:
        pass
    return len(log['calls']), time.time() - log['start'] + FIRST_TURN_S


def _is_scratch(path):
    name = os.path.basename(path)
    parts = path.split('/')
    return ('tests' in parts or 'test' in parts or name.startswith(('test_', 'repro', 'scratch', 'debug', 'tmp'))
            or name.endswith('_test.py'))


def source_edited(repo):
    """True once a non-test source file differs from the baseline commit (or a new source file exists)."""
    try:
        out = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=repo,
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True,
                             timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    for line in out.splitlines():
        path = line[3:].strip().strip('"').split(' -> ')[-1]
        if path and not _is_scratch(path) and (line[:2] != '??' or path.endswith('.py')):
            return True
    return False


def annotate(payload, repo, n, used):
    """Add progress and the due reminder to a response dict."""
    left_calls = MAX_CALLS - n
    left_s = max(0, int(TIME_LIMIT - used))
    payload['progress'] = 'tool call %d/%d, about %ds of %ds used' % (n, MAX_CALLS, used, TIME_LIMIT)
    edited = source_edited(repo)
    msg = None
    if not edited:
        if left_calls <= 1 or left_s <= 30:
            msg = ('YOU MUST EDIT NOW: this is your last chance. No source file has been modified. Your next call MUST '
                   'be str_replace with your best fix; an unedited tree scores zero.')
        elif n >= 15 or used >= 300:
            msg = ('YOU MUST EDIT NOW: %d/%d tool calls used and no source file has been modified. Your next call '
                   'MUST be str_replace (or write_file for a new file) with your best fix in the most likely '
                   'location. More reading cannot score.' % (n, MAX_CALLS))
        elif n >= 7 or used >= 180:
            msg = ('WARNING: %d/%d tool calls used and no source file has been modified yet. Start editing now: '
                   'make your best-guess fix with str_replace. You can refine it afterwards.' % (n, MAX_CALLS))
    elif left_calls <= 1 or left_s <= 30:
        msg = ('YOU MUST CALL submit_patch NOW: %d tool call(s) and about %ds left. Your next tool call MUST be '
               'submit_patch.' % (left_calls, left_s))
    elif left_calls <= 5 or left_s <= 90:
        msg = ('WARNING: only %d tool calls and about %ds left. Stop exploring, finish your check and call '
               'submit_patch.' % (left_calls, left_s))
    if msg:
        payload['budget_warning'] = msg
    return payload
