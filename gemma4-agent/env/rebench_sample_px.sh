#!/usr/bin/env bash
# Rebench sampling with resume: per rep, the usable tasks without a result in any runs/{PREFIX}_r{rep}_* dir are split over
# WORKERS_PER_REP workers (runs/{PREFIX}_r{rep}_x{k}_{tag}). PREFIX=rbs is budget_v2, rbsh is shell_v2 (Bedrock), lsh is shell_v2 on local vLLM (ENDPOINT=http://g4-lproxy:8000 MODEL=/served).
set -uo pipefail
cd "$(dirname "$0")/.."
ENDPOINT=${ENDPOINT:-http://g4-bproxy:8000}; MODEL=${MODEL:-google.gemma-4-31b}
PREFIX=${PREFIX:-rbs}; REPS=${REPS:-4}; W=${WORKERS_PER_REP:-32}; AGENT=${AGENT:-agents/budget_v2_t1}; TAG=${TAG:-$(date +%H%M)}
TASKS=${TASKS:-ext/rebench_filter/rebench_tasks_usable.jsonl}
PLAN=/tmp/${PREFIX}_plan_$TAG.txt
python3 - "$TASKS" "$REPS" "$W" "$PREFIX" > $PLAN <<'PY'
import glob, json, sys
tasks_path, reps, w, prefix = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
ids = [json.loads(l)['instance_id'] for l in open(tasks_path)]
for r in range(reps):
    done = set()
    for p in glob.glob(f'runs/{prefix}_r{r}_*/results.jsonl'):
        done |= {json.loads(l)['id'] for l in open(p)}
    todo = [i for i in ids if i not in done]
    for k in range(w):
        part = todo[k::w]
        if part:
            print(r, k, ' '.join(part))
PY
wc -l < $PLAN
while read -r r k ids; do
  n=${PREFIX}_r${r}_x${k}_$TAG
  (NET_NAME=g4int TASKS_FILE=$TASKS ./rebench_run.sh $AGENT $n --tasks $ids \
     --external-endpoint $ENDPOINT/w/$n/v1 --external-model $MODEL > /dev/null 2>&1 &)
  sleep 3
done < $PLAN
sleep 60; while docker ps --format '{{.Names}}' | grep -q "g4-${PREFIX}_r"; do sleep 60; done
echo DONE
