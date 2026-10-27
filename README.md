# Gemma 4 31B RL on the Kaggle developer-agent harness

Reinforcement learning (DAPO / GRPO with [Miles](https://github.com/radixark/miles), LoRA on the dequantized QAT
Gemma 4 31B) for the Kaggle "Gemma 4 Developer Agent" task: a shell-only ADK agent (`shell_v2`) that fixes
SWE-rebench issues inside per-task docker sandboxes. The rollouts are produced by the unchanged competition harness,
so the training trajectories have exactly the format of the collection logs (same prompts, tool declarations,
compaction, time and tool-call budget messages).

Two trees, siblings of each other:

| tree | what it is | docs |
|---|---|---|
| `gemma4-agent/` | local replica of the Kaggle harness (swegemma + ADK in the Kaggle GPU image), the agents (`agents/shell_v2*`), the rebench task pipeline (`env/rebench_*.py`), SFT and teacher-sampling tooling | `gemma4-agent/README.md` |
| `gemma4-rl/` | the RL stack: TITO proxy (`tito_proxy.py`, vLLM's chat serving on top of SGLang), agent server, rollout function with dynamic sampling (`gemma4_rollout.py`), Miles / Megatron / SGLang patches (`miles_patches.py`), launch scripts, tools, tests | `gemma4-rl/DESIGN.md` (design notes and every finding) |

Tested on one node with 8x NVIDIA H200 (143 GB), Docker 25 with the NVIDIA runtime, Python 3.12 on the host.
Disk: about 1.3 TB for the 523 task images, 44 GB of repo snapshots, 80 GB of model weights, plus the run outputs.

## Setup on a new machine

```bash
git clone git@github.com:dreamyang-liu/gemma4-rl.git ~/project     # any path works; the two trees must stay siblings
cd ~/project
./setup.sh venv            # gemma4-rl/.venv: fastapi, uvicorn, huggingface_hub (agent server, tools, hf CLI)
gemma4-rl/.venv/bin/hf auth login   # the Gemma weights are gated; also used by tools/hf_sync.py
./setup.sh miles-image     # gemma4-miles:patched = radixark/miles:latest + gemma4-agent/miles_sft/image/Dockerfile,
                           # gemma4-miles:g4rl = that + miles_patches.py (gemma4-rl/docker/build_miles_image.sh)
./setup.sh kaggle-image    # gemma4-kaggle:v170 = gcr.io/kaggle-gpu-images/python:v170 + competition wheelhouse, and :v170-plus
./setup.sh model           # hf download google/gemma-4-31b-it-qat-w4a16-ct, then env/dequant_qat.py -> models/gemma-4-31b-it-qat-dequant-bf16
./setup.sh tasks           # 523 task images from docker.io/swerebenchv2 (pinned digests, ~6 h at Docker Hub's anonymous rate) + snapshots
./setup.sh check           # everything present?
```

Every step skips what already exists, so `./setup.sh all` or re-running a step is safe. Notes:

- `kaggle-image` needs the 41 wheels of the Kaggle dataset `metric/gemma-4-developer-agent-wheelhouse` in
  `gemma4-agent/wheelhouse/`; the step downloads them with the `kaggle` CLI (`pip install kaggle`, token in
  `~/.kaggle/kaggle.json`) when they are missing.
- `MILES_HUB_IMAGE=timemagic/gemma4-miles:g4rl ./setup.sh miles-image` pulls a pushed copy of the training image
  instead of building it. The copy is accepted only if its marker `/root/.g4rl_patched` equals the sha256 of
  `gemma4-rl/miles_patches.py`; `run_experiment.sh` makes the same check at every launch. `timemagic/gemma4-miles:g4rl-1027`
  is the build of Oct 27 2026 from this tree's `miles_patches.py` (marker `fb77f6e5...`); `:g4rl` follows the latest push.
  To publish a new build: `gemma4-rl/docker/build_miles_image.sh`, then `docker tag gemma4-miles:g4rl <hub repo>:<tag>` and `docker push`.
- Secrets never go into the repo: HF token at `~/.cache/huggingface/token`, W&B key in `~/wandb_key` (read by
  `run_experiment.sh` with `WANDB=1` and passed to docker by name only).
- The Kaggle competition data (`gemma4-agent/data/`) is needed only for the Kaggle-replica evaluation
  (`gemma4-agent/run.sh`), not for RL.

## Run

The production configuration (run `dapo_colo12s_e15`, started Oct 26 2026; colocated training and rollout on 8 GPUs,
Megatron TP4 x DP2, two TP4 SGLang engines, 16 tasks x 8 trajectories per step, DAPO dynamic sampling with 2x
oversampling, solved-task skip, 12-minute enforced trajectory budget, checkpoint every 2 steps):

```bash
cd ~/project/gemma4-rl
NAME=dapo_colo12s_e15; mkdir -p runs/$NAME
COLOCATE=1 ENGINE_TP=4 DAPO=1 OVERSAMPLE=2.0 SKIP_SOLVED=1 NUM_EPOCH=15 SAVE_INTERVAL=2 MAX_CONCURRENT=96 \
AGENT=agents/shell_v2_12m GLOBAL_D512=1 D512_MIN_SEQ=16384 WANDB=1 \
nohup ./run_experiment.sh $NAME 0 16 8 --sglang-schedule-conservativeness 0.3 --distributed-timeout-minutes 120 \
  > runs/$NAME/launcher.log 2>&1 &
```

Arguments: `<name> <num_rollout> <rollout_batch_size> <n_samples_per_prompt> [extra Miles args]`; with `NUM_EPOCH`
set, `num_rollout` is ignored. All knobs are documented at the top of `run_experiment.sh`. What the script starts:

1. the TITO proxy (`gemma4-kaggle:v170` container, host network, port 8300): the Kaggle vLLM serving layer on CPU,
   tokens from Miles' SGLang router, one log per trajectory in `runs/<name>/proxy_logs/`;
2. the agent server (host venv, port 8400): one `gemma4-agent/rebench_run.sh agents/shell_v2_12m <run> --tasks <id>`
   per trajectory, reward from swegemma's Phase 2, training segments from the proxy log (`tito_samples.py`);
3. Miles (`gemma4-miles:g4rl` container): `train.py` with `gemma4_rollout.generate_rollout`, LoRA rank 32 on the MLP
   and `o_proj`, lr 1e-5, clip-higher 0.28, token-level loss.

Outputs: `gemma4-rl/runs/<name>/` (train.log, launcher.log, agent_server.log, proxy_logs/, megatron/iter_N/adapter,
megatron/rollout/N, task_stats.json) and one harness run dir per trajectory under `gemma4-agent/runs/<name>_<MMDDHHMM>_s<step>w<wave>_g<group>_k<sample>/`.

Everyday operations (all from `~/project/gemma4-rl`):

```bash
./stop_experiment.sh $NAME                                   # agent server first, then Miles, proxy, harnesses, orphan sandboxes
RESUME=1 <same env as above> ./run_experiment.sh $NAME 0 16 8 <same extra args>   # from the latest megatron/iter_*/adapter
.venv/bin/python tools/export_replay.py runs/$NAME --steps 18 && REPLAY_STEPS=1 RESUME=1 ... ./run_experiment.sh ...
                                                             # re-train lost steps from their recorded trajectories
nohup .venv/bin/python tools/hf_sync.py runs/$NAME <hf-user>/<private-repo> --loop 15 > runs/$NAME/hf_sync.log 2>&1 &
                                                             # mirror every checkpoint to a private Hugging Face repo
.venv/bin/python tools/task_progress.py <run prefixes...>    # pass counts per task, first vs latest attempt
.venv/bin/python tools/render_trajectory.py <harness run name>   # one trajectory as a markdown transcript
grep -a -E "Timer (train|log_probs) end|informative groups|step [0-9]+: \{'train/loss" runs/$NAME/train.log | tail
```

Known behaviour worth knowing before a run: the enforced trajectory budget is 12 minutes (`agents/shell_v2_12m`)
while the system prompt still says 8 minutes, by decision; a step takes about 2 hours (rollout ~45 min, log-probs
~10 min, training ~60 min); `--distributed-timeout-minutes 120` is required with DP2, otherwise a long-sequence
straggler kills the job with a misleading NCCL error. Details, measurements and the reasons for every patch are in
`gemma4-rl/DESIGN.md`.

## Tests

```bash
cd ~/project/gemma4-rl
docker run --rm -v $PWD:/root/g4rl:ro -e PYTHONPATH=/root/miles:/root/g4rl --entrypoint python3 gemma4-miles:g4rl /root/g4rl/tests/test_rollout_filter.py
docker run --rm -v $PWD:/root/g4rl:ro -e PYTHONPATH=/root/miles:/root/g4rl --entrypoint python3 gemma4-miles:g4rl /root/g4rl/tests/test_rollout_replay.py runs/<run>/replay_steps '<expectations>'
```

`tests/parity_test.py` and `tests/format_diff.py` (proxy parity with a real vLLM server, trajectory format against the
collection logs) need the recorded cases in `tests/*.jsonl`, which are not in git (35 MB); `kernels/test_attn_d512.py`
checks the Triton d512 attention kernel on a GPU inside the Miles image.

## What is not in git, and where it comes from

| path | size | source |
|---|---|---|
| `gemma4-rl/runs/`, `gemma4-agent/runs/` | 10s of GB | run outputs; checkpoints of `dapo_colo12s_e15` are mirrored to the private HF repo `mliu444/gemma4-rl-dapo-colo12s` |
| `gemma4-agent/models/` | 80 GB | `./setup.sh model` (HF `google/gemma-4-31b-it-qat-w4a16-ct` + `env/dequant_qat.py`) |
| task images `rebench/<instance_id>:latest` | ~1.3 TB | `./setup.sh tasks` (`env/rebench_pull.py` from `ext/rebench_filter/rebench_usable_sources.jsonl`, pinned `docker.io/swerebenchv2` digests) |
| `gemma4-agent/ext/rebench_snapshots/` | 44 GB | `./setup.sh tasks` (`env/rebench_snapshots.py`: the repo at base_commit, exported from each task image) |
| `gemma4-agent/wheelhouse/` | 841 MB | Kaggle dataset `metric/gemma-4-developer-agent-wheelhouse` (`./setup.sh kaggle-image`) |
| `gemma4-agent/data/` | 21 GB | Kaggle competition data (needs the Kaggle token and rules acceptance); only for `run.sh` |
| `gemma4-agent/rebench_rl_train_portable_v3.jsonl`, `ext/rebench_filter/rebench_keep*.jsonl` | 53 + 43 MB | 1834 SWE-rebench-V2 Python tasks and their LLM-filtered subset (742); the tracked `rebench_tasks_usable.jsonl` (523 tasks whose gold patch passes) and `rebench_usable_sources.jsonl` are derived from them |
| `gemma4-agent/ckpt/`, `teacher/`, `ext/sft_*`, `ext/coderforge*` | 100s of GB | SFT checkpoints, teacher weights, SFT data (not needed for RL) |
| `gemma4-rl/dev_logs/`, `gemma4-rl/tests/*.jsonl`, `gemma4-rl/data/replay_smoke1_*.json` | 150 MB | development logs and recorded test cases |
