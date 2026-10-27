#!/usr/bin/env bash
# Bootstrap a machine for the Gemma 4 RL stack (README.md). Every step is idempotent: it skips what is already there.
#   ./setup.sh venv          gemma4-rl/.venv for the agent server and the tools (fastapi, uvicorn, huggingface_hub)
#   ./setup.sh miles-image   gemma4-miles:patched (radixark/miles + Gemma 4 bridge fix) and gemma4-miles:g4rl (+ miles_patches.py);
#                            MILES_HUB_IMAGE=timemagic/gemma4-miles:g4rl pulls a pushed copy instead (accepted only if it matches miles_patches.py)
#   ./setup.sh kaggle-image  gemma4-kaggle:v170 and :v170-plus (Kaggle GPU image + competition wheelhouse; kaggle CLI or gemma4-agent/wheelhouse)
#   ./setup.sh model         google/gemma-4-31b-it-qat-w4a16-ct from Hugging Face (gated: `hf auth login` first) + bf16 dequantization
#   ./setup.sh tasks         the 523 task images (docker.io/swerebenchv2, ~1.3 TB, Docker Hub limit ~100 pulls/h) and their repo snapshots
#   ./setup.sh check         report what is present and what is missing
#   ./setup.sh all           venv miles-image kaggle-image model tasks check
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
RL=$ROOT/gemma4-rl
G4A=$ROOT/gemma4-agent
PYTHON=${PYTHON:-$(command -v python3.12 || command -v python3)}
MANIFEST=${TASKS_MANIFEST:-$G4A/ext/rebench_filter/rebench_usable_sources.jsonl}
SNAP_DIR=${SNAP_DIR:-$G4A/ext/rebench_snapshots}
PULL_LOG=${PULL_LOG:-$G4A/ext/rebench_filter/pull_log.jsonl}
QAT=$G4A/models/gemma-4-31b-it-qat-w4a16-ct
DEQ=$G4A/models/gemma-4-31b-it-qat-dequant-bf16

have_image() { docker image inspect "$1" >/dev/null 2>&1; }
marker() { docker run --rm --entrypoint cat "$1" /root/.g4rl_patched 2>/dev/null || true; }
patches_sha() { sha256sum "$RL/miles_patches.py" | cut -d' ' -f1; }

step_venv() {
  [ -x "$RL/.venv/bin/python" ] || "$PYTHON" -m venv "$RL/.venv"
  "$RL/.venv/bin/pip" install -q --upgrade pip
  "$RL/.venv/bin/pip" install -q -r "$RL/requirements-host.txt"
  "$RL/.venv/bin/python" -c "import fastapi, uvicorn, huggingface_hub as h; print('venv ok: fastapi', fastapi.__version__, 'uvicorn', uvicorn.__version__, 'huggingface_hub', h.__version__)"
}

step_miles_image() {
  if [ "$(marker gemma4-miles:g4rl)" = "$(patches_sha)" ]; then echo "gemma4-miles:g4rl is current (its marker matches miles_patches.py)"; return; fi
  if [ -n "${MILES_HUB_IMAGE:-}" ]; then
    docker pull "$MILES_HUB_IMAGE"
    if [ "$(marker "$MILES_HUB_IMAGE")" = "$(patches_sha)" ]; then
      docker tag "$MILES_HUB_IMAGE" gemma4-miles:g4rl && echo "gemma4-miles:g4rl <- $MILES_HUB_IMAGE"; return
    fi
    echo "$MILES_HUB_IMAGE was built from another miles_patches.py (marker $(marker "$MILES_HUB_IMAGE" | cut -c1-16) vs $(patches_sha | cut -c1-16)): building locally"
  fi
  # FROM radixark/miles:latest (pulled on demand, 50 GB): fixes the Megatron-Bridge Gemma 4 attention signature.
  have_image gemma4-miles:patched || docker build -t gemma4-miles:patched "$G4A/miles_sft/image"
  "$RL/docker/build_miles_image.sh"
}

step_kaggle_image() {
  if have_image gemma4-kaggle:v170 && have_image gemma4-kaggle:v170-plus; then echo "gemma4-kaggle:v170 and :v170-plus present"; return; fi
  if [ "$(ls "$G4A"/wheelhouse/*.whl 2>/dev/null | wc -l)" -lt 41 ]; then
    command -v kaggle >/dev/null || { echo "the competition wheelhouse is missing: install the kaggle CLI (pip install kaggle; token in ~/.kaggle/kaggle.json)" \
      "or put the 41 wheels of the Kaggle dataset metric/gemma-4-developer-agent-wheelhouse into $G4A/wheelhouse"; exit 2; }
    mkdir -p "$G4A/wheelhouse"
    kaggle datasets download -d metric/gemma-4-developer-agent-wheelhouse -p "$G4A/wheelhouse" --unzip
    find "$G4A/wheelhouse" -mindepth 2 -name '*.whl' -exec mv -t "$G4A/wheelhouse" {} +   # in case the archive has a top-level folder
  fi
  if ! have_image gemma4-kaggle:v170; then
    # FROM gcr.io/kaggle-gpu-images/python:v170 (49 GB). Context = Dockerfile + wheelhouse only (never the data trees).
    CTX=$(mktemp -d)
    cp "$G4A/env/Dockerfile" "$CTX"/ && cp -r "$G4A/wheelhouse" "$CTX/wheelhouse"
    docker build -t gemma4-kaggle:v170 "$CTX"; rm -rf "$CTX"
  fi
  have_image gemma4-kaggle:v170-plus || docker build -t gemma4-kaggle:v170-plus -f "$G4A/env/Dockerfile.plus" "$G4A/env"
}

step_model() {
  if [ ! -f "$QAT/config.json" ]; then
    [ -x "$RL/.venv/bin/hf" ] || { echo "run ./setup.sh venv first (the hf CLI)"; exit 2; }
    "$RL/.venv/bin/hf" download google/gemma-4-31b-it-qat-w4a16-ct --local-dir "$QAT"
  fi
  if [ ! -f "$DEQ/config.json" ]; then
    have_image gemma4-kaggle:v170 || { echo "run ./setup.sh kaggle-image first (dequant_qat.py runs in that image)"; exit 2; }
    # W = q * scale exactly, CPU only, ~59 GB of bf16 safetensors.
    docker run --rm -v "$G4A/models":/models -v "$G4A/env:/env:ro" gemma4-kaggle:v170 \
      python /env/dequant_qat.py --src "/models/$(basename "$QAT")" --out "/models/$(basename "$DEQ")"
    docker run --rm -v "$G4A/models":/o alpine:3 chown -R "$(id -u):$(id -g)" /o
  fi
  echo "model: $DEQ"
}

step_tasks() {
  [ -f "$MANIFEST" ] || { echo "missing $MANIFEST"; exit 2; }
  echo "pulling $(wc -l < "$MANIFEST") task images (pinned digests; Docker Hub anonymous limit ~100 pulls/h; log: $PULL_LOG)"
  "$PYTHON" -I "$G4A/env/rebench_pull.py" "$MANIFEST" "$PULL_LOG" --per-hour "${PULL_PER_HOUR:-90}" --workers "${PULL_WORKERS:-6}"
  echo "building repo snapshots into $SNAP_DIR"
  "$PYTHON" -I "$G4A/env/rebench_snapshots.py" "$MANIFEST" "$SNAP_DIR" --workers "${SNAP_WORKERS:-16}"
}

step_check() {
  row() { printf '  %-26s %s\n' "$1" "$2"; }
  echo "setup status for $ROOT"
  row venv "$([ -x "$RL/.venv/bin/python" ] && echo OK || echo "missing  -> ./setup.sh venv")"
  if [ "$(marker gemma4-miles:g4rl)" = "$(patches_sha)" ]; then row gemma4-miles:g4rl "OK (matches miles_patches.py)"
  elif have_image gemma4-miles:g4rl; then row gemma4-miles:g4rl "STALE  -> ./setup.sh miles-image"
  else row gemma4-miles:g4rl "missing  -> ./setup.sh miles-image"; fi
  for i in gemma4-kaggle:v170 gemma4-kaggle:v170-plus; do row "$i" "$(have_image "$i" && echo OK || echo "missing  -> ./setup.sh kaggle-image")"; done
  row "model (dequant bf16)" "$([ -f "$DEQ/config.json" ] && echo OK || echo "missing  -> ./setup.sh model")"
  if [ -f "$MANIFEST" ]; then
    "$PYTHON" -I - "$MANIFEST" "$SNAP_DIR" <<'PY'
import json, os, subprocess, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
have = set(subprocess.run(['docker', 'images', '--format', '{{.Repository}}:{{.Tag}}'], capture_output=True, text=True).stdout.split())
imgs = sum(f"rebench/{r['instance_id'].lower()}:latest" in have for r in rows)
snaps = sum(os.path.exists(os.path.join(sys.argv[2], r['instance_id'] + '.tgz')) for r in rows)
hint = '' if imgs == len(rows) and snaps == len(rows) else '  -> ./setup.sh tasks'
print(f"  {'task images':<26} {imgs}/{len(rows)}{hint}")
print(f"  {'repo snapshots':<26} {snaps}/{len(rows)}{hint}")
PY
  fi
  row "HF token" "$([ -f "$HOME/.cache/huggingface/token" ] && echo OK || echo "missing  -> hf auth login (gated weights, tools/hf_sync.py)")"
  row "~/wandb_key" "$([ -f "$HOME/wandb_key" ] && echo OK || echo "missing (needed only with WANDB=1)")"
  row docker "$(docker info --format 'v{{.ServerVersion}}, runtimes:{{range $k, $v := .Runtimes}} {{$k}}{{end}}' 2>/dev/null || echo "not reachable")"
  row GPUs "$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | sort | uniq -c | sed 's/^ *//' | paste -sd';' || echo "nvidia-smi not found")"
}

case "${1:-}" in
  venv) step_venv ;;
  miles-image) step_miles_image ;;
  kaggle-image) step_kaggle_image ;;
  model) step_model ;;
  tasks) step_tasks ;;
  check) step_check ;;
  all) step_venv; step_miles_image; step_kaggle_image; step_model; step_tasks; step_check ;;
  *) sed -n '2,10p' "$0"; exit 1 ;;
esac
