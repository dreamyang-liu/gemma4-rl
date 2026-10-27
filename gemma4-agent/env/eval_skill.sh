#!/usr/bin/env bash
# skill_v1 (skill-script tools with call log and escalating warnings, 30 calls) vs shell_v3 (native tools, 30 calls,
# prompt rules keyed on the harness budget_warning), both on the QAT W4A16 model, 106 dev tasks x 4. Log: runs/eval_skill.log
set -uo pipefail
cd "$(dirname "$0")/.."
AGENT=agents/skill_v1 env/eval106.sh skill_v1 models/gemma-4-31b-it-qat-w4a16-ct 0,1,2,3 &
AGENT=agents/shell_v3 env/eval106.sh shell_v3 models/gemma-4-31b-it-qat-w4a16-ct 4,5,6,7 &
wait
echo ALLDONE
