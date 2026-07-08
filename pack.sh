#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$ROOT_DIR"

OUT="${1:-submission.zip}"
STAGE_DIR="$ROOT_DIR/.pack_stage"
case "$OUT" in
  /*) OUT_PATH="$OUT" ;;
  *) OUT_PATH="$ROOT_DIR/$OUT" ;;
esac

rm -rf "$STAGE_DIR" "$OUT_PATH"
mkdir -p "$STAGE_DIR"

if find build_env.sh infer.py requirements.txt CMakeLists.txt CUDA -type l | grep -q .; then
  echo "[ERROR] submission inputs contain symlinks; copy real source files before packing" >&2
  find build_env.sh infer.py requirements.txt CMakeLists.txt CUDA -type l >&2
  exit 1
fi

cp build_env.sh infer.py requirements.txt CMakeLists.txt "$STAGE_DIR"/
cp -R CUDA "$STAGE_DIR"/CUDA

find "$STAGE_DIR" \( \
  -name '__pycache__' -o \
  -name '*.pyc' -o \
  -name '.DS_Store' -o \
  -name '*.o' -o \
  -name '*.so' -o \
  -name '*.a' -o \
  -name '*.nsys-rep' -o \
  -name '*.zip' \
\) -prune -exec rm -rf {} +

if find "$STAGE_DIR" -type l | grep -q .; then
  echo "[ERROR] staged submission contains symlinks" >&2
  find "$STAGE_DIR" -type l >&2
  exit 1
fi

(
  cd "$STAGE_DIR"
  zip -qr "$OUT_PATH" .
)

rm -rf "$STAGE_DIR"

echo "[INFO] wrote $OUT_PATH"
unzip -l "$OUT_PATH"
