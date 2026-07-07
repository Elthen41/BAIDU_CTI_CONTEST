#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$ROOT_DIR"

find_local_ninja() {
  for candidate in \
    "$ROOT_DIR/libraries/bin/ninja" \
    "$ROOT_DIR/libraries/ninja/data/bin/ninja" \
    "$ROOT_DIR/ninja"
  do
    if [ -f "$candidate" ]; then
      echo "$candidate"
      return 0
    fi
  done

  if [ -d "$ROOT_DIR/libraries" ]; then
    find "$ROOT_DIR/libraries" -type f -name ninja -print -quit 2>/dev/null || true
  fi
}

mkdir -p "$ROOT_DIR/libraries"

export PIP_USER=false

LOCAL_NINJA="$(find_local_ninja)"
if [ -z "$LOCAL_NINJA" ]; then
  python -m pip install \
    --target "$ROOT_DIR/libraries" \
    --upgrade \
    -r "$ROOT_DIR/requirements.txt"
  LOCAL_NINJA="$(find_local_ninja)"
else
  echo "[INFO] found packaged ninja: $LOCAL_NINJA"
fi

if [ -n "$LOCAL_NINJA" ]; then
  chmod +x "$LOCAL_NINJA"
  export PATH="$(dirname "$LOCAL_NINJA"):${PATH:-}"
fi

export PYTHONPATH="$ROOT_DIR/libraries:${PYTHONPATH:-}"
export PATH="$ROOT_DIR/libraries/bin:$ROOT_DIR/libraries/ninja/data/bin:${PATH:-}"
export TORCH_EXTENSIONS_DIR="$ROOT_DIR/.torch_extensions"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"

CUDA_NORM_SRC=""
if [ -f "$ROOT_DIR/CUDA/norm_kernels.cu" ]; then
  CUDA_NORM_SRC="$ROOT_DIR/CUDA/norm_kernels.cu"
elif [ -f "$ROOT_DIR/norm_kernels.cu" ]; then
  CUDA_NORM_SRC="$ROOT_DIR/norm_kernels.cu"
fi

if [ -n "$CUDA_NORM_SRC" ] && command -v nvcc >/dev/null 2>&1; then
  export CUDA_NORM_SRC
  python - <<'PY' || echo "[WARNING] optional CUDA extension prebuild failed; infer.py will JIT build at runtime"
import os
from torch.utils.cpp_extension import load

load(
    name="layernorm_512_ext",
    sources=[os.environ["CUDA_NORM_SRC"]],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    verbose=True,
)
print("[INFO] prebuilt layernorm_512_ext")
PY
else
  echo "[INFO] skip optional CUDA extension prebuild"
fi

echo "[INFO] build_env.sh finished"
