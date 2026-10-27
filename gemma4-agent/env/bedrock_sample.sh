#!/usr/bin/env bash
# Bedrock Gemma 4 31B sampling with the budget-aware v2 agent: REPS reps x SHARDS shards over the 106 dev tasks.
# Needs g4-bproxy on the internal network g4int. Outputs runs/$PREFIX_r{rep}_s{shard} (PREFIX default bs), proxy logs runs/bedrock_logs/$PREFIX_r*_s*.jsonl
set -uo pipefail
cd "$(dirname "$0")/.."
REPS=${REPS:-4}; SHARDS=${SHARDS:-16}; AGENT=${AGENT:-agents/budget_v2_t1}; PREFIX=${PREFIX:-bs}
for r in $(seq 0 $((REPS - 1))); do for s in $(seq 0 $((SHARDS - 1))); do
  n=${PREFIX}_r${r}_s$s
  (NET_NAME=g4int ./run.sh $AGENT $n --gpus none --tasks-file /kaggle/dev_tasks.jsonl --shard $s/$SHARDS \
     --external-endpoint http://g4-bproxy:8000/w/$n/v1 --external-model google.gemma-4-31b > /dev/null 2>&1 &)
  sleep 5
done; done
sleep 60; while docker ps --format '{{.Names}}' | grep -q "g4-${PREFIX}_r"; do sleep 60; done
python3 env/passk.py "runs/${PREFIX}_r*_s*"
