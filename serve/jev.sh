#!/usr/bin/env bash
# Serve Open-Jev, the open re-implementation of TypeSafe's Jev decision model.
# Needs a CUDA GPU (about 7 GiB for the 2B package). The harness finds it through
# JEV_BASE_URL (default http://127.0.0.1:8791/v1).
#
#   serve/jev.sh            # first run clones, installs and downloads the 2B package
#
# The commands and the pinned revision come from Open-Jev's README.
set -euo pipefail

JEV_DIR="${JEV_DIR:-$HOME/Open-Jev}"
PACKAGE="${JEV_PACKAGE:-Open-Jev-2B}"
REVISION="${JEV_REVISION:-0c7aa498b1627be8da4acf34c863ff0ee0a92785}"

if [ ! -d "$JEV_DIR" ]; then
  git clone https://github.com/Zefan-Cai/Open-Jev.git "$JEV_DIR"
fi
cd "$JEV_DIR"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/python -m pip install -e '.[train]'
fi
if [ ! -d "models/$PACKAGE" ]; then
  .venv/bin/hf download "ZefanCai/$PACKAGE" --revision "$REVISION" --local-dir "models/$PACKAGE"
fi

exec .venv/bin/python -m jev.server \
  --checkpoint "models/$PACKAGE/package/checkpoint" \
  --device "${JEV_DEVICE:-cuda:0}" \
  --max-length "${JEV_CONTEXT:-4096}" \
  --batch-size 1 \
  --no-prefix-cache \
  --host "${JEV_HOST:-127.0.0.1}" \
  --port "${JEV_PORT:-8791}"
