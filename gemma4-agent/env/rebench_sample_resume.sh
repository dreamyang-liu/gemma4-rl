#!/usr/bin/env bash
# Resume rebench sampling: per rep, the usable tasks without a result in any runs/rbs_r{rep}_* dir are split over
# WORKERS_PER_REP workers (runs/rbs_r{rep}_x{k}_{tag}). Same agent / endpoint as env/rebench_sample.sh.
set -uo pipefail
cd "$(dirname "$0")/.."
REPS=${REPS:-4}; W=${WORKERS_PER_REP:-32}; AGENT=${AGENT:-agents/budget_v2_t1}; TAG=${TAG:-$(date +%H%M)}
TASKS=${TASKS:-ext/rebench_filter/rebench_tasks_usable.jsonl}
python3 - "$TASKS" "$REPS" "$W" > /tmp/rbs_plan.txt <<'PY'
import glob, json, sys
tasks_path, reps, w = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
ids = [json.loads(l)['instance_id'] for l in open(tasks_path)]
for r in range(reps):
    done = set()
    for p in glob.glob(f'runs/rbs_r{r}_*/results.jsonl'):
        done |= {json.loads(l)['id'] for l in open(p)}
    todo = [i for i in ids if i not in done]
    for k in range(w):
        part = todo[k::w]
        if part:
            print(r, k, ' '.join(part))
PY
wc -l < /tmp/rbs_plan.txt
while read -r r k ids; do
  n=rbs_r${r}_x${k}_$TAG
  (NET_NAME=g4int TASKS_FILE=$TASKS ./rebench_run.sh $AGENT $n --tasks $ids \
     --external-endpoint http://g4-bproxy:8000/w/$n/v1 --external-model google.gemma-4-31b > /dev/null 2>&1 &)
  sleep 3
done < /tmp/rbs_plan.txt
sleep 60; while docker ps --format '{{.Names}}' | grep -q "g4-rbs_r"; do sleep 60; done
echo DONE
