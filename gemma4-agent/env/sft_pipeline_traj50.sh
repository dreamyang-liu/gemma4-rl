#!/usr/bin/env bash
# After the pass@16 run: bf16 base eval -> full-weight SFT (500 steps) -> SFT eval. Logs to runs/sft_pipeline.log.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
LOG=$ROOT/runs/sft_pipeline.log; exec >> $LOG 2>&1
say() { echo "[$(date +%H:%M:%S)] $*"; }
SRV=g4-vserve
serve() {  # serve <host_model_dir> -> container $SRV, model name /served
  docker rm -f $SRV g4-v31b >/dev/null 2>&1; sleep 5
  docker run -d --init --name $SRV --network none --gpus all --ipc host --shm-size 64g -v "$1":/served:ro gemma4-kaggle:v170-plus \
    python3 -m vllm.entrypoints.openai.api_server --model /served --host 127.0.0.1 --port 8000 --max-model-len 32768 --dtype bfloat16 \
    --gpu-memory-utilization 0.90 --tensor-parallel-size 4 --data-parallel-size 2 --enable-auto-tool-choice --tool-call-parser gemma4 \
    --reasoning-parser gemma4 --reasoning-config '{"reasoning_start_str": "<|channel>", "reasoning_end_str": "<channel|>"}' \
    --no-scheduler-reserve-full-isl --disable-custom-all-reduce >/dev/null
  for i in $(seq 1 80); do sleep 15; docker logs $SRV 2>&1 | grep -q "Application startup complete" && { say "server up ($1)"; return 0; }
    docker ps --format '{{.Names}}' | grep -q $SRV || break; done
  say "SERVER FAILED for $1"; docker logs $SRV 2>&1 | tail -30; return 1
}
evaluate() {  # evaluate <tag>: 2 reps x 106 tasks, 48 workers
  for r in 0 1; do for s in $(seq 0 23); do
    (NETNS=$SRV ./run.sh agents/openhands_sft64 oh64_${1}_r${r}_s$s --gpus none --tasks-file /kaggle/dev_tasks.jsonl --shard $s/24 \
       --external-endpoint http://127.0.0.1:8000/v1 --external-model /served > /dev/null 2>&1 &)
  done; done
  sleep 60; while docker ps --format '{{.Names}}' | grep -q "g4-oh64_${1}_"; do sleep 60; done
  python3 env/passk.py "runs/oh64_${1}_r*_s*"
}
say "whole-trajectory SFT (50 steps, sdpa) + eval"
OUT=$ROOT/ckpt/sft_traj50; docker run --rm -v $ROOT/ckpt:/c alpine:3 rm -rf /c/sft_traj50; mkdir -p $OUT
say "SFT training start -> $OUT"
docker run --rm --name g4-sft --network none --gpus all --ipc host --shm-size 64g --ulimit memlock=-1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e NCCL_DEBUG=WARN \
  -v $ROOT/env:/e:ro -v $ROOT/models/gemma-4-31B-it-bf16:/m:ro -v $ROOT/ext/coderforge-fixed/tokenized:/data:ro -v $OUT:/out \
  gemma4-kaggle:v170 torchrun --nproc_per_node 8 /e/sft_train.py --model /m --data /data --out /out --steps 50 --rows-per-step 13 --save-every 0 --warmup 5 --attn sdpa
rc=$?; say "SFT training exit code $rc"; docker run --rm -v $OUT:/o alpine:3 chown -R $(id -u):$(id -g) /o
[ $rc -eq 0 ] && [ -f $OUT/config.json ] || { say "training failed"; exit 1; }
serve $OUT || exit 1
say "SFT eval start"; evaluate sft_traj50; say "SFT eval done"
say "PIPELINE DONE"
