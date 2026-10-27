#!/usr/bin/env bash
# One training step on replayed trajectories (no SGLang, no agents), to compare settings on identical inputs.
#   ./replay_experiment.sh <name> <gpus e.g. 0,1,2,3> <attention backend: unfused|auto|flash|fused> [extra miles args...]
# SGD with lr 1 and no clipping, so the saved LoRA B matrices (zero-initialized) are minus the gradient.
# Outputs runs/replay/<name>/: train.log, megatron/iter_0000001/adapter, train_data_*.pt (per-token log_probs).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
NAME=$1; GPUS=$2; ATTN=$3; shift 3
MODEL=${MODEL:-$(cd "$HERE/.." && pwd)/gemma4-agent/models/gemma-4-31b-it-qat-dequant-bf16}
REPLAY=${REPLAY:-/root/g4rl/data/replay_smoke1_s0.json}
OUT=$HERE/runs/replay/$NAME; mkdir -p "$OUT"
LORA_TARGETS=${LORA_TARGETS:-mlp,model.language_model.layers.*.self_attn.o_proj}
docker rm -f g4rl-replay-$NAME >/dev/null 2>&1 || true
docker run --rm --name g4rl-replay-$NAME --network none --gpus "\"device=$GPUS\"" --ipc host --shm-size 64g \
  --ulimit memlock=-1 --ulimit stack=67108864 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v "$MODEL":/root/models/base:ro -v "$HERE":/root/g4rl:ro -v "$OUT":/root/out \
  --entrypoint bash ${MILES_IMAGE:-gemma4-miles:g4rl} -c "
set -ex
# Miles/Megatron fixes (miles_patches.py): baked into gemma4-miles:g4rl (marker /root/.g4rl_patched = sha256 of the
# applied script); applied here only on an unpatched image. A marker that does not match the mounted script means the
# image is stale: rebuild with docker/build_miles_image.sh.
if [ -f /root/.g4rl_patched ]; then
  [ \"\$(cat /root/.g4rl_patched)\" = \"\$(sha256sum /root/g4rl/miles_patches.py | cut -d\" \" -f1)\" ] || { echo \"miles_patches.py changed since the image was built: run docker/build_miles_image.sh\"; exit 2; }
else
  python3 /root/g4rl/miles_patches.py
fi
export PYTHONUNBUFFERED=1 MASTER_ADDR=127.0.0.1 no_proxy=127.0.0.1
ray start --head --node-ip-address 127.0.0.1 --num-gpus 4 --disable-usage-stats
MODEL_ARGS=\$(cd /root/miles && python3 -c 'import importlib.util as u; s=u.spec_from_file_location(\"m\",\"scripts/models/gemma-4-31b-it.py\"); m=u.module_from_spec(s); s.loader.exec_module(m); print(m.model_args())')
ray job submit --address=http://127.0.0.1:8265 --runtime-env-json='{\"env_vars\":{\"PYTHONUNBUFFERED\":\"1\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\",\"MASTER_ADDR\":\"127.0.0.1\",\"no_proxy\":\"127.0.0.1\",\"HF_HUB_OFFLINE\":\"1\",\"PYTHONPATH\":\"/root/miles:/root/Megatron-LM:/root/g4rl\",\"G4RL_REPLAY\":\"$REPLAY\",\"G4RL_SLIDING_FLASH\":\"${SLIDING_FLASH:-1}\",\"G4RL_GLOBAL_D512\":\"${GLOBAL_D512:-0}\"}}' -- \
 python3 /root/miles/train.py \$MODEL_ARGS \
 --hf-checkpoint /root/models/base --megatron-to-hf-mode bridge \
 --save /root/out/megatron --save-interval 1 --save-debug-train-data '/root/out/train_data_{rollout_id}_{rank}.pt' \
 --lora-rank 32 --lora-alpha 64 --lora-train-only --target-modules $LORA_TARGETS \
 --debug-train-only --rollout-function-path replay_rollout.generate_rollout \
 --prompt-data /root/g4rl/data/rl_tasks_usable.jsonl --input-key prompt --metadata-key metadata \
 --num-rollout 1 --rollout-batch-size 2 --n-samples-per-prompt 4 --global-batch-size 8 --micro-batch-size 1 \
 --advantage-estimator grpo --entropy-coef 0.0 --eps-clip 0.2 --eps-clip-high 0.28 \
 --optimizer sgd --lr 1.0 --lr-decay-style constant --weight-decay 0.0 --clip-grad 0 \
 --tensor-model-parallel-size 4 --pipeline-model-parallel-size 1 --context-parallel-size 1 \
 ${RECOMPUTE_ARGS:---recompute-granularity full --recompute-method uniform --recompute-num-layers 1} \
 --actor-num-nodes 1 --actor-num-gpus-per-node 4 --num-gpus-per-node 4 \
 --attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 --no-gradient-accumulation-fusion \
 --attention-softmax-in-fp32 --attention-backend $ATTN --qkv-format bshd --no-bias-swiglu-fusion \
 $*
" 2>&1 | tee "$OUT/train.log"
docker run --rm -v "$OUT":/o alpine:3 chown -R "$(id -u):$(id -g)" /o 2>/dev/null || true
