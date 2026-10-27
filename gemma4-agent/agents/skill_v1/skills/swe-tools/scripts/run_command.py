"""run_command: execute a shell command in the repository (same result format as the native run_command)."""
import os
import signal
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _budget import TIME_LIMIT, annotate, record  # noqa: E402
from _common import MAX_OUTPUT_CHARS, REPO, args, emit, error, ok  # noqa: E402

COMMAND_TIMEOUT = 230  # below the harness's 240 s per-call limit, so a timeout is reported instead of cut off

a = args(['command'])
command = a.get('command', '')
n, used = record(REPO, 'run_command', command)
if not command.strip():
    res = error('ValueError', 'command must be a non-empty string')
else:
    timeout = max(5, min(COMMAND_TIMEOUT, int(TIME_LIMIT - used) - 5))
    proc = subprocess.Popen(['/bin/bash', '-c', command], cwd=REPO, env=os.environ.copy(), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, universal_newlines=True, errors='replace', start_new_session=True)
    try:
        out, err = proc.communicate(timeout=timeout)
        out, err = out[:MAX_OUTPUT_CHARS], err[:MAX_OUTPUT_CHARS]
        if proc.returncode != 0:
            res = error('CommandError', err if err else out,
                        details={'stdout': out, 'stderr': err, 'exit_code': proc.returncode})
        else:
            res = ok(stdout=out, stderr=err, exit_code=0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
        proc.communicate()
        res = error('TimeoutExceeded', 'Command timed out after %d seconds' % timeout)
emit(annotate(res, REPO, n, used))
