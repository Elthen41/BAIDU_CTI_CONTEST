#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$ROOT_DIR"

if [ -f "$ROOT_DIR/libraries/bin/ninja" ]; then
  chmod +x "$ROOT_DIR/libraries/bin/ninja"
  export PATH="$ROOT_DIR/libraries/bin:$PATH"
fi

export PYTHONPATH="$ROOT_DIR/libraries:${PYTHONPATH:-}"
export TORCH_EXTENSIONS_DIR="$ROOT_DIR/.torch_extensions"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"

echo "[INFO] build_env.sh finished"
