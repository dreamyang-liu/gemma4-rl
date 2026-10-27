#!/usr/bin/env bash
# Full-weight SFT of Gemma 4 31B with Miles (Megatron + bridge), pre-tokenized whole-trajectory rows.
# usage: ./launch.sh <run_name> <num_rollout> [index.parquet] [extra miles args...]
#   one rollout == one optimizer step (rollout_batch_size == global_batch_size == 24 rows, padded per step)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
NAME=$1; NROLL=$2; INDEX=${3:-index100.parquet}; shift 3 || shift $#
OUT=$ROOT/ckpt/$NAME; mkdir -p $OUT
docker rm -f g4-miles >/dev/null 2>&1 || true
docker run --rm --name g4-miles --network none --gpus all --ipc host --shm-size 64g --ulimit memlock=-1 --ulimit stack=67108864 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v $ROOT/models/gemma-4-31B-it-bf16:/root/models/gemma-4-31B-it:ro \
  -v $ROOT/ext/coderforge-fixed/tokenized:/root/data:ro \
  -v $ROOT/miles_sft:/root/sft:ro \
  -v $OUT:/root/out \
  --entrypoint bash ${MILES_IMAGE:-gemma4-miles:patched} -c "
set -ex
export PYTHONUNBUFFERED=1 MASTER_ADDR=127.0.0.1 no_proxy=127.0.0.1
ray start --head --node-ip-address 127.0.0.1 --num-gpus 8 --disable-usage-stats
MODEL_ARGS=\$(cd /root/miles && python3 -c 'import importlib.util as u; s=u.spec_from_file_location(\"m\",\"scripts/models/gemma-4-31b-it.py\"); m=u.module_from_spec(s); s.loader.exec_module(m); print(m.model_args())')
ray job submit --address=http://127.0.0.1:8265 --runtime-env-json='{\"env_vars\":{\"PYTHONUNBUFFERED\":\"1\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\",\"NCCL_NVLS_ENABLE\":\"1\",\"MASTER_ADDR\":\"127.0.0.1\",\"no_proxy\":\"127.0.0.1\",\"HF_HUB_OFFLINE\":\"1\",\"PYTHONPATH\":\"/root/miles:/root/Megatron-LM:/root/sft\"}}' -- \
 python3 /root/miles/train.py \$MODEL_ARGS \
 --actor-num-nodes 1 --actor-num-gpus-per-node 8 --num-gpus-per-node 8 \
 --hf-checkpoint /root/models/gemma-4-31B-it --megatron-to-hf-mode bridge \
 --load /root/out/megatron --save /root/out/megatron --save-interval 1000 --no-save-optim \
 --save-hf '/root/out/hf/iter_{rollout_id}' \
 --debug-train-only --rollout-function-path sft_rollout.generate_rollout \
 --prompt-data /root/sft/$INDEX --input-key id --metadata-key metadata \
 --rollout-batch-size 24 --global-batch-size 24 --num-rollout $NROLL --balance-data \
 --loss-type sft_loss --calculate-per-token-loss --disable-compute-advantages-and-returns \
 --tensor-model-parallel-size 4 --pipeline-model-parallel-size 1 --context-parallel-size 1 \
 --qkv-format bshd --micro-batch-size 1 --attention-backend unfused --attention-softmax-in-fp32 \
 --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
 --attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 --no-gradient-accumulation-fusion \
 --optimizer adam --lr 1e-5 --min-lr 1e-6 --lr-decay-style cosine --lr-warmup-iters 10 \
 --clip-grad 1.0 --weight-decay 0.0 --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 \
 --optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer \
 $*
" 2>&1 | tee $OUT/train.log
docker run --rm -v $OUT:/o alpine:3 chown -R $(id -u):$(id -g) /o 2>/dev/null || true
