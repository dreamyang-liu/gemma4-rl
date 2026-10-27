"""Health check for rebench sampling runs (budget_v2: runs/rbs_*, shell_v2: runs/rbsh_*, shell_v2 on local vLLM:
runs/lsh_*).

Prints per-prefix progress and resolve rate, harness-side errors, Bedrock proxy errors/throttling in the last
--window minutes, stale runners, orphan task containers and host load.
usage: python3 env/rebench_monitor.py [--window 20]
"""
import collections
import glob
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
WINDOW = float(sys.argv[sys.argv.index('--window') + 1]) if '--window' in sys.argv else 20.0
now = time.time()
os.chdir(ROOT)


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


names = sh("docker ps --format '{{.Names}} {{.Image}}'").split('\n')
running = collections.Counter()
task_containers = 0
for line in names:
    if not line.strip():
        continue
    n, img = line.split(' ', 1)
    m = re.match(r'g4-(rbsh?|lsh)_r\d', n)
    if m:
        running[m.group(1)] += 1
    if img.startswith('rebench/'):
        task_containers += 1
n_tasks = sum(1 for _ in open('ext/rebench_filter/rebench_tasks_usable.jsonl'))

for prefix, label in (('rbs', 'budget_v2'), ('rbsh', 'shell_v2'), ('lsh', 'shell_v2-local')):
    rows = []
    for p in glob.glob(f'runs/{prefix}_r*/results.jsonl'):
        rows += [json.loads(l) for l in open(p)]
    if not rows and not running[prefix]:
        continue
    by_rep = collections.defaultdict(dict)
    for p in glob.glob(f'runs/{prefix}_r*/results.jsonl'):
        rep = int(re.search(r'_r(\d+)_', p).group(1))
        for l in open(p):
            r = json.loads(l)
            by_rep[rep][r['id']] = r['resolved']
    done = sum(len(v) for v in by_rep.values())
    res = sum(sum(v.values()) for v in by_rep.values())
    harness_err = collections.Counter()
    for r in rows:
        e = r.get('error') or ''
        if e and not e.startswith(('Agent exceeded', 'Agent completed execution without')):
            harness_err[e[:90]] += 1
    print(f'[{label}] runners={running[prefix]} attempts={done}/{4 * n_tasks} resolved={res} '
          f'({res / max(1, done):.1%}) per-rep={[len(by_rep[r]) for r in sorted(by_rep)]}')
    if harness_err:
        print('   harness/eval errors:', harness_err.most_common(5))
    stale = []
    for log in glob.glob(f'runs/{prefix}_r*/log.txt'):
        name = log.split('/')[1]
        if any(name in l for l in names) and now - os.path.getmtime(log) > 20 * 60:
            stale.append(name)
    if stale:
        print('   STALE runners (log idle > 20 min):', stale[:10])
    tracebacks = []
    for log in glob.glob(f'runs/{prefix}_r*/log.txt'):
        if now - os.path.getmtime(log) > WINDOW * 60 + 600:
            continue
        txt = open(log, errors='replace').read()
        for m in re.finditer(r'^Traceback \(most recent call last\):\n(?:.*\n){0,40}?(\S.*(?:Error|Exception).*)$', txt, re.M):
            tracebacks.append((log.split('/')[1], m.group(1)[:120]))
    if tracebacks:
        print('   tracebacks in runner logs:', collections.Counter(t[1] for t in tracebacks).most_common(4),
              'e.g.', tracebacks[0][0])

# Bedrock proxy: per-request status codes and latency in the window (the proxy retries 429/5xx internally;
# a non-200 here means retries were exhausted)
codes, dts, n_req = collections.Counter(), [], 0
for p in glob.glob('runs/bedrock_logs/rbs*_r*.jsonl') + glob.glob('runs/local_logs/lsh_r*.jsonl'):
    if now - os.path.getmtime(p) > WINDOW * 60:
        continue
    with open(p, 'rb') as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 4_000_000))
        tail = f.read().decode(errors='replace').split('\n')[1:]
    for l in tail:
        if not l.strip():
            continue
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            continue
        if now - r['ts'] > WINDOW * 60:
            continue
        n_req += 1
        codes[r['status']] += 1
        dts.append(r['dt'])
dts.sort()
stats = ' '.join(f'{c}:' + sh(f"docker exec {c} python -c \"import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8000/stats').read().decode())\" 2>/dev/null").strip()
                 for c in ('g4-bproxy', 'g4-lproxy'))
print(f'[llm] last {WINDOW:.0f} min: requests={n_req} status={dict(codes)} '
      f'latency p50={dts[len(dts) // 2] if dts else 0:.1f}s p90={dts[int(len(dts) * .9)] if dts else 0:.1f}s | proxy totals {stats}')
print(f'[host] task containers={task_containers} runners={sum(running.values())} '
      f'load={os.getloadavg()[0]:.0f} mem_used_gb={sh("free -g | awk NR==2{print\\$3}").strip()} '
      f'disk_free={sh("df -h " + os.path.dirname(os.path.abspath(__file__)) + " | awk NR==2{print\\$4}").strip()}')
vq = [l.split()[0] for l in names if l.startswith('g4-vq')]
if vq:
    probe = ("import urllib.request,re\nfor k in %r:\n    try:\n        t = urllib.request.urlopen(f'http://{k}:8000/metrics', timeout=10).read().decode()\n"
             "        g = lambda n: sum(float(x) for x in re.findall(r'^vllm:' + n + r'(?:{[^}]*})? ([0-9.e+-]+)$', t, re.M))\n"
             "        print(k, 'running=%%d waiting=%%d kv=%%.0f%%%% preempt=%%d' %% (g('num_requests_running'), g('num_requests_waiting'), 100 * g('kv_cache_usage_perc'), g('num_preemptions_total')))\n"
             "    except Exception as e:\n        print(k, 'DOWN', e)\n") % (vq,)
    out = subprocess.run(['docker', 'exec', '-i', 'g4-lproxy', 'python', '-'], input=probe, capture_output=True, text=True)
    print('[vllm]', ' | '.join((out.stdout or out.stderr).strip().split('\n')))
