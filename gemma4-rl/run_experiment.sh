#!/usr/bin/env bash
# GRPO on swegemma shell_v2 trajectories: TITO proxy + agent server + Miles (LoRA on the dequantized QAT base).
#   ./run_experiment.sh <name> <num_rollout> <rollout_batch_size> <n_samples_per_prompt> [extra miles args...]
# GPUs: 0-3 train (Megatron TP4), 4-7 rollout (4 SGLang engines, TP1). Outputs under runs/<name>/:
#   proxy_logs/ (one log per trajectory, collection-log format + tito), agent_server.log, train.log, megatron/.
# Harness run dirs are gemma4-agent/runs/<name>_<sample>. Stop everything with ./stop_experiment.sh <name>.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
NAME=$1; NROLL=$2; RBS=$3; NSPP=$4; shift 4
G4A=${G4A:-$(cd "$HERE/.." && pwd)/gemma4-agent}   # the harness tree, a sibling of this directory (override with G4A=...)
MODEL=$G4A/models/gemma-4-31b-it-qat-dequant-bf16
OUT=$HERE/runs/$NAME; mkdir -p "$OUT/proxy_logs" "$OUT/ray"  # ray/: the Ray session logs survive the --rm container (dead-rank logs)
date -u "+%Y-%m-%d %H:%M:%S" > "$OUT/started_at"  # stop_experiment.sh removes sandboxes created after this
PROXY_PORT=${PROXY_PORT:-8300}; AGENT_PORT=${AGENT_PORT:-8400}
LORA_RANK=${LORA_RANK:-32}; LORA_ALPHA=${LORA_ALPHA:-64}; LR=${LR:-1e-5}
# Not q/k/v yet: Megatron-Bridge's LoRA merge splits the fused linear_qkv with one head layout and fails on Gemma 4's
# global layers (head_dim 512, K=V). o_proj and the MLP merge cleanly; vLLM serves LoRA on all of them.
LORA_TARGETS=${LORA_TARGETS:-mlp,model.language_model.layers.*.self_attn.o_proj}
# DAPO=1: dynamic sampling (drop groups whose rewards are all equal, refill in waves; gemma4_rollout.py) and token-level
# loss (--calculate-per-token-loss). Clip-higher (--eps-clip-high 0.28) is on either way.
DAPO=${DAPO:-0}; OVERSAMPLE=${OVERSAMPLE:-1.5}; MAX_WAVES=${MAX_WAVES:-4}
# NUM_EPOCH set: num_rollout = NUM_EPOCH * 523 tasks // RBS (Miles --num-epoch) and the <num_rollout> argument is ignored.
SAVE_INTERVAL=${SAVE_INTERVAL:-10}; MAX_CONCURRENT=${MAX_CONCURRENT:-$(( (RBS * 3 / 2 + 1) * NSPP ))}
# AGENT: harness agent dir under gemma4-agent (default agents/shell_v2: 28 calls / 8 min; agents/shell_v2_15m: same with a
# 15-minute session budget in eval_config.yaml and in the system prompt's HARD LIMITS line).
AGENT=${AGENT:-agents/shell_v2}
# SKIP_SOLVED=1: skip tasks whose last group was solved by every trajectory (no DAPO signal), retrying them with
# probability SOLVED_RETRY_P; per-task counts persist in runs/<name>/task_stats.json (gemma4_rollout.py).
SKIP_SOLVED=${SKIP_SOLVED:-0}; SOLVED_RETRY_P=${SOLVED_RETRY_P:-0.2}
# ADAPTIVE_OVERSAMPLE=1: size the first wave from the observed usable-group rate (EMA in task_stats.json), between
# OVERSAMPLE x and 3.5 x the deficit, so most steps finish in one wave (gemma4_rollout.py).
ADAPTIVE_OVERSAMPLE=${ADAPTIVE_OVERSAMPLE:-0}
# D512_MIN_SEQ: with GLOBAL_D512=1, global layers use the d512 kernel only for sequences of at least this many tokens
# (shorter ones keep the faster fp32 score matrix; 0 = always). 16384 keeps the [8, s, s] fp32 matrix <= 8 GB.
D512_MIN_SEQ=${D512_MIN_SEQ:-0}
# REPLAY_STEPS=1: steps with runs/<name>/replay_steps/step<k>.json (tools/export_replay.py) are trained from those recorded
# rollouts instead of sampling; later steps roll out live.
REPLAY_STEPS=${REPLAY_STEPS:-0}; REPLAY_ENV=""; [ "$REPLAY_STEPS" = 1 ] && REPLAY_ENV=',"G4RL_REPLAY_DIR":"/root/out/replay_steps"'  # expanded verbatim into the single-quoted JSON
# RESUME=1: continue runs/<name> from its latest LoRA checkpoint (megatron/iter_*/adapter with optimizer state):
# --lora-adapter-path restores adapter + optimizer and the saved iteration, Miles resumes at iteration+1 and reloads the
# rollout-side state from megatron/rollout/<iteration> (miles_patches.py patch 12). task_stats.json/replay_steps stay.
RESUME_ARGS=""
if [ "${RESUME:-0}" = 1 ]; then
  LATEST=$(ls -d "$OUT"/megatron/iter_*/adapter 2>/dev/null | sort | while read -r d; do [ -f "$d/training_state_rank0.pt" ] && echo "$d"; done | tail -1)
  [ -n "$LATEST" ] || { echo "RESUME=1 but no LoRA checkpoint with optimizer state under $OUT/megatron"; exit 2; }
  ITER_DIR=$(basename "$(dirname "$LATEST")")            # iter_NNNNNNN
  ITER=$((10#${ITER_DIR#iter_}))
  # Miles sets start_rollout_id=0 in bridge mode whenever --load holds no Megatron tracker (always true for LoRA
  # checkpoints) and that explicit 0 overrides the iteration restored from the adapter, so pass it ourselves.
  RESUME_ARGS="--lora-adapter-path /root/out/megatron/$ITER_DIR/adapter --start-rollout-id $((ITER + 1))"
  echo "resuming from $LATEST (rollout $((ITER + 1)))"
fi
LENGTH_ARGS="--num-rollout $NROLL"; [ -n "${NUM_EPOCH:-}" ] && LENGTH_ARGS="--num-epoch $NUM_EPOCH"
# Attention (see DESIGN.md "Attention: the real finding"): SLIDING_FLASH=1 (default) runs FlashAttention-3 on the 50
# sliding layers; GLOBAL_D512=1 (experimental, default off) runs kernels/attn_d512.py on the 10 global layers.
# Both are applied by miles_patches.py together with the fused-softmax window fix, which is always on.
# COLOCATE=1: training and rollout take turns on all 8 GPUs (Megatron TP4 x DP2, 8 SGLang engines of one GPU); each side
# is offloaded to host memory while the other runs. ENGINE_TP (colocate only, default 1): GPUs per SGLang engine; 4 gives
# two TP4 engines (Miles' Gemma 4 recipe layout): ~2x faster per-sequence decode and 4x the KV per engine.
# Default: 4 GPUs train, 4 GPUs serve, both resident.
if [ "${COLOCATE:-0}" = 1 ]; then
  GPU_ARGS="--colocate --actor-num-nodes 1 --actor-num-gpus-per-node 8 --num-gpus-per-node 8 --rollout-num-gpus-per-engine ${ENGINE_TP:-1} --sglang-mem-fraction-static ${SGLANG_MEM:-0.75}"
else
  GPU_ARGS="--actor-num-nodes 1 --actor-num-gpus-per-node 4 --num-gpus-per-node 8 --rollout-num-gpus 4 --rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static ${SGLANG_MEM:-0.85}"
fi
# WANDB=1: the key is read from ~/wandb_key into WANDB_API_KEY and handed to docker by name only (-e WANDB_API_KEY), so
# it never appears in a command line or in train.log; wandb.init reads it from the environment (no --wandb-key).
WANDB_ARGS=""; WANDB_ENV=()
if [ "${WANDB:-0}" = 1 ]; then
  export WANDB_API_KEY=$(tr -d '[:space:]' < "$HOME/wandb_key")
  WANDB_ENV=(-e WANDB_API_KEY); WANDB_ARGS="--use-wandb --wandb-project ${WANDB_PROJECT:-gemma4-rl} --wandb-group $NAME"
fi
DAPO_ARGS=""; [ "$DAPO" = 1 ] && DAPO_ARGS="--calculate-per-token-loss"
# Host python for the agent server (fastapi, uvicorn): G4RL_PY, else ./.venv (../setup.sh venv), else the SPROUT venv.
PY=${G4RL_PY:-$HERE/.venv/bin/python}; [ -x "$PY" ] || PY=$HOME/project/SPROUT/.venv/bin/python

# 1. TITO proxy: the Kaggle vLLM serving layer, generating on Miles' SGLang router.
docker rm -f g4rl-proxy-$NAME >/dev/null 2>&1 || true
docker run -d --name g4rl-proxy-$NAME --network host -e CUDA_VISIBLE_DEVICES= \
  -v "$MODEL":/model:ro -v "$HERE":/opt/tito:ro -v "$OUT/proxy_logs":/logs \
  --entrypoint python gemma4-kaggle:v170 /opt/tito/tito_proxy.py \
  --log-dir /logs --host 127.0.0.1 --port $PROXY_PORT -- \
  /model --served-model-name gemma-4-31b-it --max-model-len 32768 --dtype bfloat16 --tool-call-parser gemma4 \
  --reasoning-parser gemma4 --enable-auto-tool-choice \
  --reasoning-config '{"reasoning_start_str": "<|channel>", "reasoning_end_str": "<channel|>"}' >/dev/null
for i in $(seq 1 60); do curl -sf -m 2 127.0.0.1:$PROXY_PORT/health >/dev/null && break; sleep 2; done
curl -sf -m 2 127.0.0.1:$PROXY_PORT/health >/dev/null || { docker logs --tail 30 g4rl-proxy-$NAME; exit 1; }
echo "proxy up on :$PROXY_PORT (the rollout function sets the SGLang router URL)"

# 2. Agent server: one harness run per trajectory.
ps -eo pid,args | awk -v p="--port $AGENT_PORT" '$2 ~ /python/ && $3 ~ /agent_server/ && index($0, p) {print $1}' | xargs -r kill
(cd "$HERE" && nohup "$PY" agent_server.py --gemma4-agent "$G4A" --proxy-url http://127.0.0.1:$PROXY_PORT --proxy-log-dir "$OUT/proxy_logs" --agent "$AGENT" \
  --run-prefix "${NAME}_$(date +%m%d%H%M)" --port $AGENT_PORT --max-concurrent $MAX_CONCURRENT --sample-mode ${SAMPLE_MODE:-merged} > "$OUT/agent_server.log" 2>&1 &)
for i in $(seq 1 30); do curl -sf -m 2 127.0.0.1:$AGENT_PORT/health >/dev/null && break; sleep 1; done
echo "agent server up on :$AGENT_PORT"

# 3. Miles.
docker rm -f g4rl-miles-$NAME >/dev/null 2>&1 || true
docker run --rm --name g4rl-miles-$NAME --network host --gpus all --ipc host --shm-size 64g --ulimit memlock=-1 \
  --ulimit stack=67108864 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 "${WANDB_ENV[@]}" \
  -v "$MODEL":/root/models/base:ro -v "$HERE":/root/g4rl:ro -v "$OUT":/root/out -v "$OUT/ray":/tmp/ray \
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
ray start --head --node-ip-address 127.0.0.1 --num-gpus 8 --disable-usage-stats
MODEL_ARGS=\$(cd /root/miles && python3 -c 'import importlib.util as u; s=u.spec_from_file_location(\"m\",\"scripts/models/gemma-4-31b-it.py\"); m=u.module_from_spec(s); s.loader.exec_module(m); print(m.model_args())')
ray job submit --address=http://127.0.0.1:8265 --runtime-env-json='{\"env_vars\":{\"PYTHONUNBUFFERED\":\"1\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\",\"NCCL_NVLS_ENABLE\":\"1\",\"MASTER_ADDR\":\"127.0.0.1\",\"no_proxy\":\"127.0.0.1\",\"HF_HUB_OFFLINE\":\"1\",\"PYTHONPATH\":\"/root/miles:/root/Megatron-LM:/root/g4rl\",\"G4RL_AGENT_SERVER\":\"http://127.0.0.1:$AGENT_PORT\",\"G4RL_PROXY\":\"http://127.0.0.1:$PROXY_PORT\",\"G4RL_RUN_TAG\":\"s\",\"G4RL_DYNAMIC_SAMPLING\":\"$DAPO\",\"G4RL_OVERSAMPLE\":\"$OVERSAMPLE\",\"G4RL_MAX_WAVES\":\"$MAX_WAVES\",\"G4RL_SLIDING_FLASH\":\"${SLIDING_FLASH:-1}\",\"G4RL_GLOBAL_D512\":\"${GLOBAL_D512:-0}\",\"G4RL_D512_MIN_SEQ\":\"$D512_MIN_SEQ\",\"G4RL_DUMP_LIVE\":\"${DUMP_LIVE:-0}\",\"G4RL_SKIP_SOLVED\":\"$SKIP_SOLVED\",\"G4RL_SOLVED_RETRY_P\":\"$SOLVED_RETRY_P\",\"G4RL_ADAPTIVE_OVERSAMPLE\":\"$ADAPTIVE_OVERSAMPLE\",\"G4RL_TASK_STATS\":\"/root/out/task_stats.json\"$REPLAY_ENV}}' -- \
 python3 /root/miles/train.py \$MODEL_ARGS \
 --hf-checkpoint /root/models/base --megatron-to-hf-mode bridge \
 --load /root/out/megatron --save /root/out/megatron --save-interval $SAVE_INTERVAL \
 --lora-rank $LORA_RANK --lora-alpha $LORA_ALPHA --lora-train-only --target-modules $LORA_TARGETS $RESUME_ARGS \
 --rollout-function-path gemma4_rollout.generate_rollout \
 --prompt-data /root/g4rl/data/rl_tasks_usable.jsonl --input-key prompt --metadata-key metadata --rollout-shuffle \
 $LENGTH_ARGS --rollout-batch-size $RBS --n-samples-per-prompt $NSPP --global-batch-size $((RBS * NSPP)) \
 --micro-batch-size 1 \
 --advantage-estimator grpo --entropy-coef 0.0 --eps-clip 0.2 --eps-clip-high 0.28 $DAPO_ARGS \
 --optimizer adam --lr $LR --lr-decay-style constant --weight-decay 0.0 --adam-beta1 0.9 --adam-beta2 0.98 \
 --tensor-model-parallel-size 4 --pipeline-model-parallel-size 1 --context-parallel-size 1 \
 --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
 $GPU_ARGS \
 --sglang-attention-backend triton --sglang-disable-custom-all-reduce --sglang-context-length 32768 \
 --attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 --no-gradient-accumulation-fusion \
 --attention-softmax-in-fp32 --attention-backend unfused --qkv-format bshd \
 $WANDB_ARGS --no-bias-swiglu-fusion `# the LoRA bridge path copies this into bias_activation_fusion; Gemma 4 uses gelu-tanh` \
 $*
" 2>&1 | tee "$OUT/train.log"
docker run --rm -v "$OUT":/o alpine:3 chown -R "$(id -u):$(id -g)" /o 2>/dev/null || true
