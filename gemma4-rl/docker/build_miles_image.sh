#!/usr/bin/env bash
# Build gemma4-miles:g4rl (Miles image + miles_patches.py baked in). Context is a temp dir with only what the
# Dockerfile needs, so the 50 GB base layer is reused and runs/ is never sent to the docker daemon.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
CTX=$(mktemp -d)
trap 'rm -rf "$CTX"' EXIT
cp "$HERE/miles_patches.py" "$HERE/docker/Dockerfile.miles" "$CTX"/
cp -r "$HERE/kernels" "$CTX"/kernels
rm -rf "$CTX"/kernels/__pycache__
TAG=${1:-gemma4-miles:g4rl}
docker build -f "$CTX/Dockerfile.miles" -t "$TAG" "$CTX"
docker tag "$TAG" "gemma4-miles:g4rl-$(date +%m%d)"
echo "built $TAG (also tagged gemma4-miles:g4rl-$(date +%m%d)); patches sha256: $(sha256sum "$HERE/miles_patches.py" | cut -c1-16)..."
