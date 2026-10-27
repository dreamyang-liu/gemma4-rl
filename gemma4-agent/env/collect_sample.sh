#!/usr/bin/env bash
# SFT data collection: shell_v2 agent on the gold-checked rebench tasks with a chosen model, REPS attempts per task.
#   C=<tag> PROXY=<proxy container> MODEL=<label> REPS=2 WORKERS=64 env/collect_sample.sh
# Outputs runs/col_<tag>_r<rep>_s<shard>/ (results + traces); model calls are logged by the proxy (see its --log-dir).
# Tasks: TASKS_LIST (one instance id per line; default all usable rebench tasks). Dev tasks are never used here.
set -uo pipefail
cd "$(dirname "$0")/.."
: "${C:?tag}" "${PROXY:?proxy container}" "${MODEL:=model}"
REPS=${REPS:-2}; WORKERS=${WORKERS:-64}; AGENT=${AGENT:-agents/shell_v2}
TASKS_FILE=ext/rebench_filter/rebench_tasks_usable.jsonl
if [[ -n "${TASKS_LIST:-}" ]]; then mapfile -t T < "$TASKS_LIST"
else mapfile -t T < <(python3 -c "import json;[print(json.loads(l)['instance_id']) for l in open('$TASKS_FILE')]"); fi
for r in $(seq 0 $((REPS - 1))); do
  for s in $(seq 0 $((WORKERS - 1))); do
    mine=(); for i in "${!T[@]}"; do (( (i + r * 7) % WORKERS == s )) && mine+=("${T[$i]}"); done
    [[ ${#mine[@]} -eq 0 ]] && continue
    n=col_${C}_r${r}_s$s
    (NET_NAME=g4int TASKS_FILE=$TASKS_FILE ./rebench_run.sh "$AGENT" "$n" --tasks "${mine[@]}" \
       --external-endpoint "http://$PROXY:8000/w/$n/v1" --external-model "$MODEL" > /dev/null 2>&1 &)
    sleep 2
  done
done
sleep 60; while docker ps --format '{{.Names}}' | grep -q "g4-col_${C}_"; do sleep 60; done
echo DONE
