"""Mirror a run's LoRA checkpoints (and what a resume needs) to a private Hugging Face repo.

Uploads, per checkpoint, megatron/iter_NNNNNNN/adapter/ (per-rank adapter + optimizer shards, merged adapter_model.safetensors,
adapter_config.json) and megatron/rollout/<N>/data_source/ (data-source position), plus task_stats.json and DESIGN.md;
checkpoints already on the hub are skipped. --loop MIN re-checks every MIN minutes (for running jobs).

usage: python3 tools/hf_sync.py <run dir> <repo_id> [--loop 15]
"""
import argparse
import glob
import os
import re
import sys
import time

from huggingface_hub import HfApi, create_repo

ap = argparse.ArgumentParser()
ap.add_argument('run_dir')
ap.add_argument('repo_id')
ap.add_argument('--loop', type=float, default=0, help='minutes between re-checks; 0 = once')
args = ap.parse_args()
run = os.path.basename(os.path.normpath(args.run_dir))
api = HfApi()
create_repo(args.repo_id, private=True, exist_ok=True, repo_type='model')


def sync_once():
    have = set(api.list_repo_files(args.repo_id))
    done = {m.group(1) for f in have for m in [re.search(rf'^{run}/megatron/(iter_\d{{7}})/adapter/adapter_config\.json$', f)] if m}
    for it in sorted(glob.glob(os.path.join(args.run_dir, 'megatron', 'iter_*'))):
        name = os.path.basename(it)
        adapter = os.path.join(it, 'adapter')
        # a checkpoint is complete when every rank wrote its training state (8 ranks) and the merged adapter exists
        if name in done or not os.path.exists(os.path.join(adapter, 'adapter_model.safetensors')):
            continue
        if len(glob.glob(os.path.join(adapter, 'training_state_rank*.pt'))) < 8:
            continue
        t0 = time.time()
        api.upload_folder(folder_path=adapter, repo_id=args.repo_id, path_in_repo=f'{run}/megatron/{name}/adapter',
                          commit_message=f'{run}: {name} adapter + optimizer state')
        step = int(name.split('_')[1])
        rdir = os.path.join(args.run_dir, 'megatron', 'rollout', str(step), 'data_source')
        if os.path.isdir(rdir):
            api.upload_folder(folder_path=rdir, repo_id=args.repo_id, path_in_repo=f'{run}/megatron/rollout/{step}/data_source',
                              commit_message=f'{run}: rollout state {step}')
        print(f'{time.strftime("%H:%M:%S")} uploaded {name} ({time.time() - t0:.0f}s)', flush=True)
    for f in ('task_stats.json',):
        p = os.path.join(args.run_dir, f)
        if os.path.exists(p):
            api.upload_file(path_or_fileobj=p, path_in_repo=f'{run}/{f}', repo_id=args.repo_id, commit_message=f'{run}: {f}')
    design = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'DESIGN.md')
    if os.path.exists(design):
        api.upload_file(path_or_fileobj=design, path_in_repo='DESIGN.md', repo_id=args.repo_id, commit_message='DESIGN.md')


readme = f"""# gemma4-rl checkpoints (private)

LoRA checkpoints of the Gemma 4 31B DAPO runs in `~/project/gemma4-rl` (Miles + swegemma harness). Layout per run:

- `<run>/megatron/iter_NNNNNNN/adapter/`: `adapter_model.safetensors` + `adapter_config.json` (PEFT LoRA on
  `gemma-4-31b-it-qat-dequant-bf16`, rank 32, alpha 64, targets mlp + o_proj), `adapter_megatron_rank*.pt` and
  `training_state_rank*.pt` (per-rank Megatron shards and DistributedOptimizer state, needed to resume training).
- `<run>/megatron/rollout/<N>/data_source/`: the data source position at that checkpoint (resume draws the next tasks).
- `<run>/task_stats.json`: per-task pass history used by the solved-task skip.
- `DESIGN.md`: the design and findings log.

Resume: copy `<run>/megatron` back into `runs/<run>/`, then `RESUME=1 ./run_experiment.sh <run> ...` (see DESIGN.md).
"""
api.upload_file(path_or_fileobj=readme.encode(), path_in_repo='README.md', repo_id=args.repo_id, commit_message='README')
while True:
    try:
        sync_once()
    except Exception as e:  # keep the loop alive across transient network errors
        print(f'{time.strftime("%H:%M:%S")} sync error: {type(e).__name__}: {str(e)[:200]}', flush=True)
    if not args.loop:
        break
    time.sleep(args.loop * 60)
