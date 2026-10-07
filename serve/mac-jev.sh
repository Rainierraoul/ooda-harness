#!/usr/bin/env bash
# Serve Open-Jev on an Apple Silicon Mac's GPU (PyTorch MPS), following
# Open-Jev's docs/apple-silicon.md. The first run installs it and downloads the
# 2B package plus its Qwen3.5-2B base, checking every file against the manifest.
set -euo pipefail

JEV_DIR="${JEV_DIR:-$HOME/Development/Open-Jev}"
export JEV_MODEL_ROOT="${JEV_MODEL_ROOT:-$HOME/models/open-jev}"

if [ ! -d "$JEV_DIR" ]; then
  git clone https://github.com/Zefan-Cai/Open-Jev.git "$JEV_DIR"
fi
cd "$JEV_DIR"
if [ ! -x .venv/bin/python ]; then
  uv venv --python 3.12 .venv
  uv pip install --python .venv/bin/python -e '.[train]'
fi
CKPT="$JEV_MODEL_ROOT/Open-Jev-2B/package/checkpoint"
if [ ! -d "$CKPT" ]; then
  JEV_PACKAGE_REPO=ZefanCai/Open-Jev-2B \
  JEV_PACKAGE_REVISION=0c7aa498b1627be8da4acf34c863ff0ee0a92785 \
  JEV_BASE_MODEL=Qwen/Qwen3.5-2B \
  JEV_BASE_REVISION=15852e8c16360a2fea060d615a32b45270f8a8fc \
    .venv/bin/python docker/fetch_models.py
fi

# Leave PYTORCH_ENABLE_MPS_FALLBACK unset: with it, unsupported operations
# silently run on the CPU instead of failing.
unset PYTORCH_ENABLE_MPS_FALLBACK
export HF_HUB_CACHE="$JEV_MODEL_ROOT/hub" HF_HUB_OFFLINE=1
exec .venv/bin/python -m jev.server --checkpoint "$CKPT" --device mps \
  --max-length "${JEV_CONTEXT:-4096}" --host 127.0.0.1 --port "${JEV_PORT:-8791}"
