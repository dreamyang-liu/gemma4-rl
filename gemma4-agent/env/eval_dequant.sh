#!/usr/bin/env bash
# Dequantized bf16 QAT checkpoint vs the original QAT W4A16 model on the 106 dev tasks (same settings as the
# base_bf16 / SFT evals). Log: runs/eval_dequant.log
set -uo pipefail
cd "$(dirname "$0")/.."
env/eval106.sh qat_deq_bf16 models/gemma-4-31b-it-qat-dequant-bf16 0,1,2,3 &
env/eval106.sh qat_w4a16 models/gemma-4-31b-it-qat-w4a16-ct 4,5,6,7 &
wait
echo ALLDONE
