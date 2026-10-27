"""Agent server: runs one swegemma trajectory per request and returns its reward and its exact training segments.

POST /run {"sample_id": "...", "instance_id": "..."}
  1. runs `gemma4-agent/rebench_run.sh <agent> <run name> --tasks <instance_id>` against the TITO proxy, i.e. the
     unchanged harness that made the collection logs, with the model behind http://<proxy>/w/<run name>/v1;
  2. reads the reward (`resolved` in runs/<run name>/results.jsonl, swegemma's Phase 2) and the proxy log of the
     run, and splits it into training segments (tito_samples.py: one per compaction segment, or one per call);
  3. returns {status, reward, segments: [{tokens, response_length, loss_mask, rollout_log_probs}], metrics}.
status is "failed" when the harness produced no result row or no model calls (an infrastructure failure: the sample
must be dropped, not scored 0). An agent that ran but did not resolve the task is "ok" with reward 0.

usage: python agent_server.py --proxy-url http://127.0.0.1:8300 --proxy-log-dir DIR --run-prefix rl_exp1 [--port 8400]
"""
import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

import tito_samples

ap = argparse.ArgumentParser()
ap.add_argument('--proxy-url', required=True, help='the TITO proxy as the harness containers reach it')
ap.add_argument('--proxy-log-dir', required=True, help="the proxy's --log-dir as seen from this host")
ap.add_argument('--run-prefix', required=True, help='prefix of the run names (runs/<prefix>_<sample>)')
ap.add_argument('--gemma4-agent', default=str(Path(__file__).resolve().parent.parent / 'gemma4-agent'),
                help='the harness tree (default: the sibling directory gemma4-agent)')
ap.add_argument('--agent', default='agents/shell_v2')
ap.add_argument('--tasks-file', default='ext/rebench_filter/rebench_tasks_usable.jsonl')
ap.add_argument('--model', default='gemma-4-31b-it', help="the proxy's served model name")
ap.add_argument('--net', default='host', help='docker network of the harness containers')
ap.add_argument('--max-concurrent', type=int, default=64)
ap.add_argument('--sample-mode', choices=['merged', 'per_call'], default='merged',
                help='merged: one sample per compaction segment (tito_samples.segments_merged); per_call: one per call')
ap.add_argument('--timeout', type=float, default=2700.0, help='seconds for one trajectory, setup and grading included')
ap.add_argument('--host', default='127.0.0.1')
ap.add_argument('--port', type=int, default=8400)
args = ap.parse_args()

ROOT = Path(args.gemma4_agent)
SEM = asyncio.Semaphore(args.max_concurrent)
STATS = {'running': 0, 'ok': 0, 'failed': 0, 'resolved': 0}
app = FastAPI()


def run_name(sample_id):
    name = f'{args.run_prefix}_{sample_id}'
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name):
        raise ValueError(f'bad run name {name!r}')
    return name


async def harness(name, instance_id):
    env = {**os.environ, 'NET_NAME': args.net, 'TASKS_FILE': args.tasks_file}
    cmd = ['./rebench_run.sh', args.agent, name, '--tasks', instance_id,
           '--external-endpoint', f'{args.proxy_url.rstrip("/")}/w/{name}/v1', '--external-model', args.model]
    proc = await asyncio.create_subprocess_exec(*cmd, cwd=ROOT, env=env, stdout=asyncio.subprocess.DEVNULL,
                                                stderr=asyncio.subprocess.DEVNULL)
    try:
        return await asyncio.wait_for(proc.wait(), args.timeout)
    except asyncio.TimeoutError:
        kill = await asyncio.create_subprocess_exec('docker', 'rm', '-f', f'g4-{name}',
                                                    stdout=asyncio.subprocess.DEVNULL,
                                                    stderr=asyncio.subprocess.DEVNULL)
        await kill.wait()
        proc.kill()
        return None


@app.get('/health')
async def health():
    return {'status': 'ok', **STATS}


@app.post('/run')
async def run(body: dict):
    sample_id, instance_id = str(body['sample_id']), str(body['instance_id'])
    name = run_name(sample_id)
    log = Path(args.proxy_log_dir) / f'{name}.jsonl'
    out = ROOT / 'runs' / name
    if log.exists() or out.exists():
        return JSONResponse({'status': 'failed', 'error': f'run {name} exists already'}, status_code=409)
    async with SEM:
        STATS['running'] += 1
        t0 = time.time()
        try:
            code = await harness(name, instance_id)
        finally:
            STATS['running'] -= 1
    rows = [json.loads(l) for l in open(out / 'results.jsonl')] if (out / 'results.jsonl').exists() else []
    calls = tito_samples.load_calls(log) if log.exists() else []
    metrics = {'harness_exit': code, 'wall_s': round(time.time() - t0, 1), 'calls': len(calls)}
    if not rows or not calls:
        STATS['failed'] += 1
        return {'status': 'failed', 'error': 'timeout' if code is None else 'no result row or no model calls',
                'reward': None, 'segments': [], 'metrics': metrics}
    row = rows[-1]
    segs = tito_samples.segments_merged(calls) if args.sample_mode == 'merged' else tito_samples.segments(calls)
    reward = 1.0 if row.get('resolved') else 0.0
    metrics.update(tool_calls=row.get('tool_calls'), agent_s=row.get('duration_s'), error=row.get('error'),
                   patch_chars=row.get('patch_chars'), segments=len(segs),
                   tokens_total=sum(len(s['tokens']) for s in segs),
                   tokens_trained=sum(sum(s['loss_mask']) for s in segs), sample_mode=args.sample_mode,
                   forced=sum(len(c.get('forced') or ()) for c in calls),
                   compactions=sum(1 for c in calls if c.get('summarizer')))
    STATS['ok'] += 1
    STATS['resolved'] += int(reward)
    return {'status': 'ok', 'reward': reward, 'resolved': bool(row.get('resolved')), 'metrics': metrics,
            'segments': [{k: s[k] for k in ('tokens', 'response_length', 'loss_mask', 'rollout_log_probs')}
                         for s in segs]}


if __name__ == '__main__':
    uvicorn.run(app, host=args.host, port=args.port, log_level='warning')
