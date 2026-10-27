#!/usr/bin/env bash
# Run an agent submission dir through the Kaggle-replica harness.
# usage: ./run.sh <agent_dir> <run_name> [--gpus 0,1,2,3] [kaggle_run.py args...]
#   e.g. ./run.sh agents/baseline smoke --gpus 0,1,2,3 --first 2
# Internet is disabled (--network none), as in Kaggle scoring.
# NETNS=<container> joins that container's (offline) network namespace instead, e.g. to reach a teacher
# server on 127.0.0.1; --gpus none runs without GPUs (use with --external-endpoint).
# TASKS_JSONL=<file> mounts another tasks file at /kaggle/dev_tasks.jsonl (e.g. tasks with hints_text).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
AGENT=$(realpath "$1"); NAME=$2; shift 2
GPUS=0,1,2,3
if [[ "${1:-}" == --gpus ]]; then GPUS=$2; shift 2; fi
OUT="$ROOT/runs/$NAME"; mkdir -p "$OUT"

MOUNTS=()
for m in "$ROOT"/models/gemma-4-*; do
  MOUNTS+=(-v "$m:/kaggle/input/models/google/gemma-4/other/$(basename "$m")/2:ro")
done

NET=(--network none); [[ -n "${NETNS:-}" ]] && NET=(--network "container:$NETNS"); [[ -n "${NET_NAME:-}" ]] && NET=(--network "$NET_NAME")
GPU=(--gpus "\"device=$GPUS\""); [[ "$GPUS" == none ]] && GPU=()
docker run --rm --name "g4-$NAME" "${NET[@]}" "${GPU[@]}" \
  --ipc host --shm-size 32g --ulimit memlock=-1 \
  -v "$ROOT/data:/kaggle/input/competitions/gemma-4-developer-agent:ro" \
  -v "$AGENT:/kaggle/agent:ro" \
  -v "$OUT:/kaggle/working/run" \
  -v "$ROOT/env/kaggle_run.py:/opt/kaggle_run.py:ro" -v "${TASKS_JSONL:-$ROOT/dev_tasks.jsonl}:/kaggle/dev_tasks.jsonl:ro" \
  "${MOUNTS[@]}" \
  "${IMAGE:-gemma4-kaggle:v170-plus}" python /opt/kaggle_run.py --out /kaggle/working/run "$@" 2>&1 | tee "$OUT/log.txt"
docker run --rm -v "$OUT:/o" alpine:3 chown -R "$(id -u):$(id -g)" /o 2>/dev/null || true
