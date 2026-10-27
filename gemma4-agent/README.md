# Gemma 4 Developer Agent — local Kaggle replica

- `env/Dockerfile` — gcr.io/kaggle-gpu-images/python:v170 + competition wheelhouse (`--no-deps`, same as official notebook) → image `gemma4-kaggle:v170`
- `env/kaggle_run.py` — official Getting Started notebook cells 2/4/5 (vLLM server + swegemma Evaluator Phase 1+2, subprocess sandbox)
- `wheelhouse/` — Kaggle dataset metric/gemma-4-developer-agent-wheelhouse (41 wheels)
- `models/` — HF google/gemma-4-*-it-qat-w4a16-ct (mounted at Kaggle path .../other/<name>/2)
- `data/` — competition data (tasks.jsonl, snapshots/, graphs/, embeddings/, wheels/, sample_submission/)
- `agents/sample` — official sample_submission as-is; `agents/baseline` — sample minus dummy LoRAs, 5 min / 40 tool calls per task
- `dev_tasks.jsonl` — the 106/129 training tasks whose gold patch passes Phase 2 here (use this as the dev set)
- `env/gold_check.py` — runs Phase 2 with gold patches (harness sanity check, CPU only)
- `runs/<name>/` — log.txt, results.jsonl, submission.parquet, summary.json

```
./run.sh agents/baseline smoke --gpus 0,1,2,3 --first 2
./run.sh agents/baseline dev --tasks-file /kaggle/dev_tasks.jsonl          # dev set
IMAGE=gemma4-kaggle:v170 ./run.sh ...                                          # strict replica image
./run.sh agents/baseline full_a --gpus 0,1,2,3 --shard 0/2 &   # two 4-GPU "L4 boxes" in parallel
./run.sh agents/baseline full_b --gpus 4,5,6,7 --shard 1/2 --port 8001 &
```

Fidelity notes
- Same vLLM args as Kaggle (TP=4, max_model_len 32768, gemma4 tool/reasoning parsers) + `--disable-custom-all-reduce` (L4 has no P2P anyway).
- KV cache emulates 4x L4: gpu_memory_utilization = 0.9*23034MiB / H200 total, plus vLLM's <70GiB-GPU
  scheduler defaults (max_num_batched_tokens 2048, max_num_seqs 256) → 45.2k KV tokens for 31B (forum: ~46k on real L4).
  `--no-l4-emulation` disables.
- Default image `gemma4-kaggle:v170-plus` = replica + inline_snapshot/dirty_equals (test-only deps the Kaggle image lacks;
  without them only 75/129 gold patches pass). Phase 1 uses the subprocess sandbox like the Kaggle notebook: no editable
  install, host site-packages visible (so e.g. src-layout `requests` resolves to the image's installed requests).
- The official docker sandbox (`swebench-sandbox:latest`, built from data/docker/Dockerfile.public) is worse for
  public tasks (43/129 gold) because the public wheels miss deps like typing_inspection.
- `--network none` (Kaggle internet off). Tasks sequential like the notebook.
- NOT emulated: speed. H200 decodes much faster than L4, so wall-time / max_time_minutes behaviour
  and the 12h total budget are optimistic locally — check summary.json wall_min with that in mind.
