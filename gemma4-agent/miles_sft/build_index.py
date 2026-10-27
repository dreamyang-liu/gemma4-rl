"""Build the Miles prompt-data index for whole-trajectory SFT.

Each row is one SFT row (agent segment or compaction summary) as a pointer {src, row} into the pre-tokenized
coderforge-fixed parquet files. Rows are ordered by the whole-trajectory plan (train_plan.json from the FSDP run):
each step holds whole trajectories and is padded with fully masked rows to exactly --rows-per-step rows, so one
Miles rollout (rollout_batch_size == global_batch_size == rows-per-step) is one optimizer step.

usage: python -I build_index.py --plan train_plan.json --data /data --steps 100 --rows-per-step 24 --out index.parquet
"""
import argparse, collections, glob, os
import pyarrow as pa, pyarrow.parquet as pq
import json

ap = argparse.ArgumentParser()
ap.add_argument('--plan', required=True)
ap.add_argument('--data', required=True, help='dir of tokenized/*.parquet (as seen inside the training container)')
ap.add_argument('--data-host', default=None, help='same dir as seen by this script, if different')
ap.add_argument('--steps', type=int, default=100)
ap.add_argument('--rows-per-step', type=int, default=24)
ap.add_argument('--out', required=True)
a = ap.parse_args()

plan = json.load(open(a.plan))['steps'][:a.steps]
want = {t for s in plan for t in s}
host = a.data_host or a.data
by_traj = collections.defaultdict(list)
for f in sorted(glob.glob(os.path.join(host, '*.parquet'))):
    t = pq.read_table(f, columns=['trajectory_id', 'kind', 'segment', 'n_tokens'])
    for i, (tid, k, sg, n) in enumerate(zip(*[t[c].to_pylist() for c in ('trajectory_id', 'kind', 'segment', 'n_tokens')])):
        if tid in want:
            by_traj[tid].append((n, sg, os.path.join(a.data, os.path.basename(f)), i, k))
rows = []
for s, trajs in enumerate(plan):
    # within a step, sort by length (longest first; pads last) so consecutive rows, which Miles groups into
    # micro-batches of --micro-batch-size under bshd, have similar lengths and little padding
    step_rows = sorted((r for t in trajs for r in by_traj[t]), key=lambda r: -r[0])
    assert len(step_rows) <= a.rows_per_step, (s, len(step_rows))
    for n, sg, src, i, k in step_rows:
        rows.append({'id': f's{s}:{src}:{i}', 'metadata': {'src': src, 'row': i, 'step': s, 'pad': False, 'kind': k, 'n_tokens': n}})
    for p in range(a.rows_per_step - len(step_rows)):
        rows.append({'id': f's{s}:pad{p}', 'metadata': {'src': '', 'row': -1, 'step': s, 'pad': True, 'kind': 'pad', 'n_tokens': 2}})
pq.write_table(pa.Table.from_pylist(rows), a.out)
n_pad = sum(r['metadata']['pad'] for r in rows)
print(f'steps {len(plan)} trajectories {len(want)} rows {len(rows)} (real {len(rows) - n_pad}, pad {n_pad}) -> {a.out}')
