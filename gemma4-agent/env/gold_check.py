"""Sanity-check the replica: run Phase 2 (verify_task) with each task's gold patch. Expect ~100% resolved."""
import asyncio, json, os, sys, time
from pathlib import Path
from swegemma.config import EvalConfig
from swegemma.deduplication import resolve_task_snapshot_paths
from swegemma.evaluate import Evaluator
from swegemma.harness.verification import verify_task
from swegemma.models import load_tasks

DATA = Path('/kaggle/input/competitions/gemma-4-developer-agent')
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else '/kaggle/working/run')
CONC = int(sys.argv[2]) if len(sys.argv) > 2 else 16
cfg = EvalConfig(tasks_path=DATA / 'tasks.jsonl', snapshots_dir=DATA / 'snapshots', results_dir=OUT / 'results',
                 submission_dir=None, models={}, sandbox=os.environ.get('SANDBOX', 'subprocess'), verification_timeout_seconds=300,
                 wheels_dir=DATA / 'wheels', graph_dir=str(DATA / 'graphs'), embeddings_dir=str(DATA / 'embeddings'))
ev = Evaluator(cfg)
tasks = load_tasks(cfg.tasks_path)
sem = asyncio.Semaphore(CONC)

async def one(t):
    async with sem:
        snap, base, patch = resolve_task_snapshot_paths(cfg.snapshots_dir, t.instance_id, t.repo)
        t0 = time.perf_counter()
        r = await verify_task(ev.docker, cfg, t, snap, base_snapshot_path=base, patch_path=patch,
                              agent_patch=t.patch, start_time=t0)
        row = dict(id=t.instance_id, resolved=bool(r.resolved), exit_code=r.test_exit_code, error=r.error,
                   s=round(time.perf_counter() - t0, 1), tail=(r.test_output or '')[-1500:] if not r.resolved else '')
        print(json.dumps({k: v for k, v in row.items() if k != 'tail'}), flush=True)
        return row

async def main():
    rows = await asyncio.gather(*(one(t) for t in tasks))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'gold.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    print('GOLD', sum(r['resolved'] for r in rows), '/', len(rows))
asyncio.run(main())
