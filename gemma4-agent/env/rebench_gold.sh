#!/usr/bin/env bash
# Gold check of all rebench tasks: gold patch must resolve, empty patch must not (swegemma Phase 2). 48 shards.
set -uo pipefail
cd "$(dirname "$0")/.."
N=${N:-48}
for s in $(seq 0 $((N - 1))); do
  (./rebench_run.sh agents/budget_v2_t1 rb_gold_s$s --gold --shard $s/$N > /dev/null 2>&1 &)
  sleep 1
done
sleep 30; while docker ps --format '{{.Names}}' | grep -q "g4-rb_gold_s"; do sleep 30; done
cat runs/rb_gold_s*/results.jsonl > ext/rebench_filter/gold_check.jsonl
python3 -c "
import json, collections
rs = [json.loads(l) for l in open('ext/rebench_filter/gold_check.jsonl')]
print(len(rs), 'checked; usable', sum(r['usable'] for r in rs), '; gold fails', sum(not r['gold'] for r in rs), '; empty passes', sum(r['empty'] for r in rs))"
echo DONE
