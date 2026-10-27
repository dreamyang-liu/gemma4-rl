#!/usr/bin/env bash
# Same 100 rebench tasks (runs/cand_tasks100.txt), shell_v2 agent, one attempt per candidate model, 25 workers each.
# Candidates: g4-bproxy-v (Gemma 31B on Bedrock, vLLM adaptation) and Bedrock chat proxies for other models.
set -uo pipefail
cd "$(dirname "$0")/.."
mapfile -t T < runs/cand_tasks100.txt
W=25
for spec in "gv:g4-bproxy-v:google.gemma-4-31b" "qc:g4-qcproxy:qwen3-coder-480b" "glm:g4-glmproxy:glm-5" "k25:g4-k25proxy:kimi-k2.5"; do
  IFS=: read -r c proxy model <<< "$spec"
  for s in $(seq 0 $((W - 1))); do
    mine=(); for i in "${!T[@]}"; do (( i % W == s )) && mine+=("${T[$i]}"); done
    n=cand_${c}_s$s
    (NET_NAME=g4int TASKS_FILE=ext/rebench_filter/rebench_tasks_usable.jsonl ./rebench_run.sh agents/shell_v2 $n --tasks "${mine[@]}" \
       --external-endpoint http://$proxy:8000/w/$n/v1 --external-model $model > /dev/null 2>&1 &)
    sleep 2
  done
done
sleep 60; while docker ps --format '{{.Names}}' | grep -q "g4-cand_"; do sleep 30; done
echo DONE
