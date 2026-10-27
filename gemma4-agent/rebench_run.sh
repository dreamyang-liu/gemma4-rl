#!/usr/bin/env bash
# Run swegemma on rebench tasks (env/rebench_run.py) in the Kaggle-replica image, with per-task docker sandboxes.
# usage: ./rebench_run.sh <agent_dir> <run_name> [rebench_run.py args...]
#   gold check:  ./rebench_run.sh agents/budget_v2_t1 rb_gold_s0 --gold --shard 0/32
#   agent:       NET_NAME=g4int ./rebench_run.sh agents/budget_v2_t1 rb_r0_s0 --shard 0/32 \
#                  --external-endpoint http://g4-bproxy:8000/w/rb_r0_s0/v1 --external-model google.gemma-4-31b
# The harness container gets the host docker socket (it starts the task containers, which run with --network none).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
AGENT=$(realpath "$1"); NAME=$2; shift 2
OUT="$ROOT/runs/$NAME"; mkdir -p "$OUT"
NET=(--network none); [[ -n "${NET_NAME:-}" ]] && NET=(--network "$NET_NAME")
docker run --rm --name "g4-$NAME" "${NET[@]}" \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$AGENT:/kaggle/agent:ro" \
  -v "$OUT:/kaggle/working/run" \
  -v "$ROOT/env/rebench_run.py:/opt/rebench_run.py:ro" \
  -v "$ROOT/${TASKS_FILE:-ext/rebench_filter/rebench_tasks.jsonl}:/rebench/tasks.jsonl:ro" \
  -v "$ROOT/ext/rebench_snapshots:/rebench/snapshots:ro" \
  "${IMAGE:-gemma4-kaggle:v170-plus}" python /opt/rebench_run.py --out /kaggle/working/run \
  --tasks-file /rebench/tasks.jsonl --snapshots /rebench/snapshots "$@" 2>&1 | tee "$OUT/log.txt"
docker run --rm -v "$OUT:/o" alpine:3 chown -R "$(id -u):$(id -g)" /o 2>/dev/null || true
