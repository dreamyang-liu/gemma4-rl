#!/usr/bin/env bash
# Bedrock Gemma 4 31B sampling on gold-checked rebench tasks: REPS reps x SHARDS shards, budget-aware v2 agent.
# Outputs runs/rbs_r{rep}_s{shard}/ (results.jsonl with patches, traces), proxy logs runs/bedrock_logs/rbs_r*_s*.jsonl
set -uo pipefail
cd "$(dirname "$0")/.."
REPS=${REPS:-4}; SHARDS=${SHARDS:-16}; AGENT=${AGENT:-agents/budget_v2_t1}
TASKS=${TASKS:-ext/rebench_filter/rebench_tasks_usable.jsonl}
for r in $(seq 0 $((REPS - 1))); do for s in $(seq 0 $((SHARDS - 1))); do
  n=rbs_r${r}_s$s
  (NET_NAME=g4int TASKS_FILE=$TASKS ./rebench_run.sh $AGENT $n --shard $s/$SHARDS \
     --external-endpoint http://g4-bproxy:8000/w/$n/v1 --external-model google.gemma-4-31b > /dev/null 2>&1 &)
  sleep 5
done; done
sleep 60; while docker ps --format '{{.Names}}' | grep -q "g4-rbs_r"; do sleep 60; done
echo DONE
