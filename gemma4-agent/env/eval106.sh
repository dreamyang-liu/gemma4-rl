#!/usr/bin/env bash
# Evaluate a model on the 106 dev tasks with the Kaggle-replica harness (run.sh, subprocess sandbox).
# usage: env/eval106.sh <tag> <host_model_dir> <gpu_list e.g. 0,1,2,3>
#   env: REPS=4 WORKERS_PER_GPU=6 AGENT=agents/shell_v2
# Serves the model with one bf16 vLLM per GPU (Kaggle serving flags, TP1) on network g4int behind a logging proxy
# (g4-evp-<tag>, logs in runs/eval_logs/<tag>), runs REPS x 106 attempts over WORKERS_PER_GPU x n_gpus workers
# (runs/ev_<tag>_r{rep}_s{shard}), prints pass@k and stops the servers. Few agents per GPU keep per-call latency
# close to Kaggle's, so the 8-minute budget binds about as it does there.
set -uo pipefail
cd "$(dirname "$0")/.."
TAG=$1; MODEL=$(realpath "$2"); GPUS=$3
REPS=${REPS:-4}; WPG=${WORKERS_PER_GPU:-6}; AGENT=${AGENT:-agents/shell_v2}
IFS=, read -ra G <<< "$GPUS"
ups=()
for g in "${G[@]}"; do
  c=g4-ev-$TAG-$g; docker rm -f $c >/dev/null 2>&1
  docker run -d --init --name $c --network g4int --gpus "\"device=$g\"" --ipc host --shm-size 32g -v "$MODEL":/served:ro \
    gemma4-kaggle:v170-plus python3 -m vllm.entrypoints.openai.api_server --model /served --host 0.0.0.0 --port 8000 \
    --max-model-len 32768 --dtype bfloat16 --gpu-memory-utilization 0.90 --enable-auto-tool-choice --tool-call-parser gemma4 \
    --reasoning-parser gemma4 --reasoning-config '{"reasoning_start_str": "<|channel>", "reasoning_end_str": "<channel|>"}' \
    --no-scheduler-reserve-full-isl >/dev/null
  ups+=("http://$c:8000/v1")
done
for i in $(seq 1 80); do
  n=0; for g in "${G[@]}"; do docker logs g4-ev-$TAG-$g 2>&1 | grep -q "Application startup complete" && n=$((n + 1)); done
  [ $n -eq ${#G[@]} ] && break; sleep 15
done
echo "[$TAG] $n/${#G[@]} servers up for $MODEL"
[ $n -eq ${#G[@]} ] || { for g in "${G[@]}"; do docker logs --tail 20 g4-ev-$TAG-$g; done; exit 1; }
mkdir -p runs/eval_logs/$TAG
docker rm -f g4-evp-$TAG >/dev/null 2>&1
docker run -d --name g4-evp-$TAG --network g4int -v "$PWD/env/bedrock_proxy.py:/p.py:ro" -v "$PWD/runs/eval_logs/$TAG:/logs" \
  gemma4-kaggle:v170 python -I /p.py --port 8000 --log-dir /logs --backend openai --upstream "$(IFS=,; echo "${ups[*]}")" >/dev/null
sleep 3
SH=$((WPG * ${#G[@]} / REPS)); [ $SH -lt 1 ] && SH=1
for r in $(seq 0 $((REPS - 1))); do for s in $(seq 0 $((SH - 1))); do
  n=ev_${TAG}_r${r}_s$s
  (NET_NAME=g4int ./run.sh $AGENT $n --gpus none --tasks-file /kaggle/dev_tasks.jsonl --shard $s/$SH \
     --external-endpoint http://g4-evp-$TAG:8000/w/$n/v1 --external-model /served > /dev/null 2>&1 &)
  sleep 3
done; done
sleep 60; while docker ps --format '{{.Names}}' | grep -q "g4-ev_${TAG}_r"; do sleep 60; done
echo "[$TAG] $(python3 env/passk.py "runs/ev_${TAG}_r*_s*" | tail -3 | tr '\n' ' ')"
for g in "${G[@]}"; do docker stop -t 20 g4-ev-$TAG-$g >/dev/null; docker rm g4-ev-$TAG-$g >/dev/null; done
docker rm -f g4-evp-$TAG >/dev/null
