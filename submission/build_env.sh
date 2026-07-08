#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/libraries:${PYTHONPATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"

if [ -n "${PYTHON_BIN:-}" ]; then
  if ! "$PYTHON_BIN" -c 'import torch' >/dev/null 2>&1; then
    echo "[ERROR] PYTHON_BIN=$PYTHON_BIN cannot import torch" >&2
    exit 1
  fi
else
  PYTHON_BIN=""
  for candidate in \
    /home/aistudio/.conda/envs/pytorch-env/bin/python \
    /opt/conda/envs/python35-paddle120-env/bin/python \
    python \
    python3
  do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import torch' >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
  if [ -z "$PYTHON_BIN" ]; then
    echo "[ERROR] no Python executable with torch was found" >&2
    exit 1
  fi
fi

if command -v cmake >/dev/null 2>&1; then
  CMAKE_BIN=cmake
elif "$PYTHON_BIN" -c 'import cmake' >/dev/null 2>&1; then
  CMAKE_BIN="$PYTHON_BIN -m cmake"
else
  echo "[INFO] cmake not found, installing local Python cmake wheel into ./libraries"
  "$PYTHON_BIN" -m pip install \
    --target "$ROOT_DIR/libraries" \
    -i https://mirrors.aliyun.com/pypi/simple \
    --upgrade cmake
  CMAKE_BIN="$PYTHON_BIN -m cmake"
fi

BUILD_DIR="$ROOT_DIR/build/cmake_cuda_ext"
INSTALL_DIR="$ROOT_DIR/cmake_extensions"

echo "[INFO] python: $($PYTHON_BIN -c 'import sys; print(sys.executable)')"
echo "[INFO] python version: $($PYTHON_BIN -c 'import sys; print(sys.version.replace(chr(10), " "))')"
echo "[INFO] cmake: $($CMAKE_BIN --version | head -n 1)"

TORCH_CMAKE_PREFIX="$("$PYTHON_BIN" - <<'PY'
import torch
print(torch.utils.cmake_prefix_path)
PY
)"

PYTHON_EXT_SUFFIX="$("$PYTHON_BIN" - <<'PY'
import sysconfig
print(sysconfig.get_config_var("EXT_SUFFIX") or ".so")
PY
)"

PYTHON_INCLUDE_DIR="$("$PYTHON_BIN" - <<'PY'
import sysconfig
print(sysconfig.get_paths().get("include") or sysconfig.get_config_var("INCLUDEPY") or "")
PY
)"

TORCH_CXX11_ABI="$("$PYTHON_BIN" - <<'PY'
import torch
print(int(torch._C._GLIBCXX_USE_CXX11_ABI))
PY
)"

CUDA_ARCH="${TORCH_CUDA_ARCH_LIST%%[; ]*}"
CUDA_ARCH="${CUDA_ARCH%%+*}"
CUDA_ARCH="$(printf '%s' "$CUDA_ARCH" | tr -d '.')"

rm -rf "$INSTALL_DIR"
mkdir -p "$BUILD_DIR" "$INSTALL_DIR"

$CMAKE_BIN -S "$ROOT_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$TORCH_CMAKE_PREFIX" \
  -DPython3_EXECUTABLE="$PYTHON_BIN" \
  -DPYTHON_INCLUDE_DIR="$PYTHON_INCLUDE_DIR" \
  -DPYTHON_EXTENSION_SUFFIX="$PYTHON_EXT_SUFFIX" \
  -DTORCH_CXX11_ABI="$TORCH_CXX11_ABI" \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DCTI_INSTALL_DIR="$INSTALL_DIR" \
  -DCTI_SMOE_MAXRREGCOUNT="${SMOE_MAXRREGCOUNT:-80}" \
  -DCTI_ENABLE_CUTLASS_SMOE="${USE_CUTLASS_SMOE:-0}" \
  -DCTI_USE_FAST_ATTENTION_EXP="${USE_FAST_ATTENTION_EXP:-0}" \
  -DCTI_CUTLASS_ROOT="${CUTLASS_ROOT:-$ROOT_DIR/third_party/cutlass}"

$CMAKE_BIN --build "$BUILD_DIR" --parallel "$CMAKE_BUILD_PARALLEL_LEVEL"
$CMAKE_BIN --install "$BUILD_DIR"

echo "[INFO] CMake CUDA extensions installed to $INSTALL_DIR"
