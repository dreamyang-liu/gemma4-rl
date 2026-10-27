#!/usr/bin/env bash
# Extra Bedrock reps (r4..r7) on the tasks never solved in r0..r3 (runs/bs_unsolved.txt).
set -uo pipefail
cd "$(dirname "$0")/.."
TASKS=$(cat runs/bs_unsolved.txt); SHARDS=16
for r in 4 5 6 7; do for s in $(seq 0 $((SHARDS - 1))); do
  n=bs_r${r}_s$s
  (NET_NAME=g4int ./run.sh agents/budget_v2_t1 $n --gpus none --tasks-file /kaggle/dev_tasks.jsonl --tasks $TASKS --shard $s/$SHARDS \
     --external-endpoint http://g4-bproxy:8000/w/$n/v1 --external-model google.gemma-4-31b > /dev/null 2>&1 &)
  sleep 5
done; done
sleep 60; while docker ps --format '{{.Names}}' | grep -q "g4-bs_r"; do sleep 60; done
echo DONE
