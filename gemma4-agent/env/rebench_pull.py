"""Pull the pinned docker images of the filtered rebench tasks, most dev-like first, under Docker Hub's anonymous
limit (100 pulls/h per IP). Verifies the image config digest and tags each image as rebench/<instance_id>:latest.
usage: python -I rebench_pull.py TASKS.jsonl LOG.jsonl [--per-hour 90 --workers 6]"""
import collections, json, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

tasks_path, log_path = sys.argv[1:3]
arg = lambda k, d: int(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
PER_HOUR, WORKERS = arg('--per-hour', 90), arg('--workers', 6)

rows = [json.loads(l) for l in open(tasks_path)]
score = lambda r: tuple(-r['llm_filter'][k] for k in ('similarity', 'spec_clear', 'solvable'))
rows.sort(key=score)
done = set()
try:
    done = {json.loads(l)['instance_id'] for l in open(log_path) if json.loads(l)['status'] == 'ok'}
except FileNotFoundError:
    pass
stamps, lock = collections.deque(), threading.Lock()


def budget():
    while True:
        with lock:
            now = time.time()
            while stamps and now - stamps[0] > 3600:
                stamps.popleft()
            if len(stamps) < PER_HOUR:
                stamps.append(now)
                return
            wait = 3600 - (now - stamps[0]) + 1
        time.sleep(wait)


def pull(r):
    p = r['rl_training_v2']['portable_runtime_profile']
    ref, want = p['pinned_image_ref'], p['image_config_digest']
    tag = f"rebench/{r['instance_id'].lower()}:latest"
    t0 = time.time()
    have = subprocess.run(['docker', 'image', 'inspect', '--format', '{{.Id}}', ref], capture_output=True, text=True)
    if have.returncode != 0:
        budget()
        for attempt in range(3):
            res = subprocess.run(['docker', 'pull', '-q', ref], capture_output=True, text=True)
            if res.returncode == 0:
                break
            if 'toomanyrequests' in res.stderr:
                time.sleep(600)
            else:
                time.sleep(30)
        if res.returncode != 0:
            return {'instance_id': r['instance_id'], 'status': 'pull_failed', 'err': res.stderr[-300:]}
    got = subprocess.run(['docker', 'image', 'inspect', '--format', '{{.Id}} {{.Size}}', ref], capture_output=True, text=True).stdout.split()
    if not got or got[0] != want:
        return {'instance_id': r['instance_id'], 'status': 'digest_mismatch', 'got': got[:1], 'want': want}
    subprocess.run(['docker', 'tag', ref, tag], check=True)
    return {'instance_id': r['instance_id'], 'status': 'ok', 'tag': tag, 'size_gb': round(int(got[1]) / 1e9, 2),
            'sim': r['llm_filter']['similarity'], 's': round(time.time() - t0, 1)}


todo = [r for r in rows if r['instance_id'] not in done]
print(f'{len(rows)} tasks, {len(done)} already pulled, {len(todo)} to go, {PER_HOUR}/h, {WORKERS} workers', flush=True)
with ThreadPoolExecutor(WORKERS) as ex, open(log_path, 'a') as log:
    for i, res in enumerate(ex.map(pull, todo)):
        log.write(json.dumps(res) + '\n')
        log.flush()
        if res['status'] != 'ok' or i % 20 == 0:
            print(i + 1, res, flush=True)
print('done', flush=True)
