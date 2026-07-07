# W4A4 CUDA Kernels

Custom W4A4 (4-bit weight, 4-bit activation) GEMM kernels used for inference in
this package. Two GEMM variants plus a fused activation quantization/compression
kernel are provided:

| Python wrapper (`csrc/__init__.py`) | pybind symbol | source | scales |
|-------------------------------------|---------------|--------|--------|
| `w4a4_mul_G`        | `apex4_gemm_w4a4_group`   | `csrc/apex4_gemm_w4a4_group.cu`   | activation/weight per-group, half |
| `w4a4_mul_CC_half`  | `apex4_gemm_w4a4_channel` | `csrc/apex4_gemm_w4a4_channel.cu` | activation/weight per-channel, half |
| `quantize_compress` | `quantize_compress_A`           | `csrc/quantize_kernel.cu` | fused 4-bit quant + int32 pack |

## Build

Requires GPU compute capability ≥ 8.0 (Ampere or later) and CUDA ≥ 11.4.

```bash
python setup.py build_ext --inplace   # produces APEX4/_CUDA*.so
```

## Test (kernel correctness + timing)

```bash
python -m unittest test_w4a4.Test.test_groups        # run from this kernels/ directory
```

## Attribution

These kernels are derived from **QQQ** (HandH1998 et al., arXiv:2406.09904),
which is in turn derived from the **Marlin** kernel by Elias Frantar (IST-DASLab).
The W4A4 GEMM kernels and the fused activation quant/compress kernel are our
contribution. Original Marlin/QQQ copyright headers are retained in the source
files. Licensed under Apache-2.0; see ../NOTICE.
