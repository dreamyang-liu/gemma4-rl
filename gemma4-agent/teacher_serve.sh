#!/usr/bin/env bash
# Serve Qwen3.8-Flash-Next-FP8 offline with sglang, context capped at the competition's 32768.
# usage: ./teacher_serve.sh 0,1,2,3 [extra sglang args]   -> container g4-teacher, http://127.0.0.1:30000/v1 inside its netns
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
GPUS=${1:-0,1,2,3}; TP=$(awk -F, '{print NF}' <<<"$GPUS")
docker run -d --name g4-teacher --network none --gpus "\"device=$GPUS\"" --ipc host --shm-size 32g \
  -e HF_HUB_OFFLINE=1 -v "$ROOT/teacher/Qwen3.8-Flash-Next-FP8:/model:ro" lmsysorg/sglang:latest \
  python3 -m sglang.launch_server --model-path /model --served-model-name qwen-teacher \
  --host 127.0.0.1 --port 30000 --tp "$TP" --context-length 32768 \
  --reasoning-parser auto --tool-call-parser auto --trust-remote-code "${@:2}"
