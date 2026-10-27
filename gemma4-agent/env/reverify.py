"""Re-run Phase 2 on saved predictions (submission.parquet files) and keep full test output.
usage: python reverify.py OUT_JSONL CONC sub1.parquet [sub2.parquet ...]"""
import asyncio, json, sys, time
from pathlib import Path
import pandas as pd
from swegemma.config import EvalConfig
from swegemma.deduplication import resolve_task_snapshot_paths
from swegemma.evaluate import Evaluator
from swegemma.harness.verification import verify_task
from swegemma.models import load_tasks

DATA = Path('/kaggle/input/competitions/gemma-4-developer-agent')
out, conc, subs = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3:]
pred = pd.concat([pd.read_parquet(s) for s in subs]).set_index('id')['prediction'].to_dict()
cfg = EvalConfig(tasks_path=DATA / 'tasks.jsonl', snapshots_dir=DATA / 'snapshots', results_dir=out.parent / 'reverify_results',
                 submission_dir=None, models={}, sandbox='subprocess', verification_timeout_seconds=300,
                 wheels_dir=DATA / 'wheels', graph_dir=str(DATA / 'graphs'), embeddings_dir=str(DATA / 'embeddings'))
ev = Evaluator(cfg)
tasks = [t for t in load_tasks(cfg.tasks_path) if t.instance_id in pred]
sem = asyncio.Semaphore(conc)

async def one(t):
    p = pred[t.instance_id]
    if p == 'NO_PATCH':
        return dict(id=t.instance_id, resolved=False, exit_code=None, error='NO_PATCH', test_output='')
    async with sem:
        snap, base, pp = resolve_task_snapshot_paths(cfg.snapshots_dir, t.instance_id, t.repo)
        r = await verify_task(ev.docker, cfg, t, snap, base_snapshot_path=base, patch_path=pp, agent_patch=p,
                              start_time=time.perf_counter())
        return dict(id=t.instance_id, resolved=bool(r.resolved), exit_code=r.test_exit_code, error=r.error,
                    test_output=(r.test_output or '')[-6000:])

async def main():
    rows = await asyncio.gather(*(one(t) for t in tasks))
    out.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    print('REVERIFY', sum(r['resolved'] for r in rows), '/', len(rows))
asyncio.run(main())
