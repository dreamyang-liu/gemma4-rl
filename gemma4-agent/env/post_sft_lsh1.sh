#!/usr/bin/env bash
# After ckpt/sft_lsh1 training: evaluate base bf16 vs the final SFT model on the 106 dev tasks (4 GPUs each, same
# settings), then two intermediate checkpoints. Log: runs/post_sft_lsh1.log
set -uo pipefail
cd "$(dirname "$0")/.."
while docker ps --format '{{.Names}}' | grep -q "^g4-sft-lsh1$"; do sleep 60; done
echo "training container exited: $(docker inspect g4-sft-lsh1 --format '{{.State.ExitCode}}')"; docker logs --tail 3 g4-sft-lsh1 2>&1
docker run --rm -v "$PWD/ckpt/sft_lsh1:/o" alpine:3 chown -R "$(id -u):$(id -g)" /o
[ -f ckpt/sft_lsh1/config.json ] || { echo "no final checkpoint"; exit 1; }
env/eval106.sh base_bf16 models/gemma-4-31B-it-bf16 0,1,2,3 &
env/eval106.sh lsh1_final ckpt/sft_lsh1 4,5,6,7 &
wait
env/eval106.sh lsh1_s60 ckpt/sft_lsh1/step60 0,1,2,3 &
env/eval106.sh lsh1_s100 ckpt/sft_lsh1/step100 4,5,6,7 &
wait
echo ALLDONE
