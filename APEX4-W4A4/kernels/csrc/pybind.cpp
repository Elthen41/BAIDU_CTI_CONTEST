#include "apex4_gemm_w4a4_group.h"    // W4A4, per-group scale, half
#include "apex4_gemm_w4a4_channel.h"  // W4A4, per-channel scale, half
#include "quantize_kernel.h"          // fused activation 4-bit quantization + compression
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // W4A4 GEMM, activation per-group / weight per-group, half scales.
  m.def("apex4_gemm_w4a4_group", &apex4_gemm_w4a4_group,
        "W4A4 INT4xINT4 GEMM (per-group, half).");

  // W4A4 GEMM, activation per-channel / weight per-channel, half scales.
  m.def("apex4_gemm_w4a4_channel", &apex4_gemm_w4a4_channel,
        "W4A4 INT4xINT4 GEMM (per-channel, half).");

  // Fused per-row activation 4-bit quantization + int32 packing.
  m.def("quantize_compress_A", &launch_quantize_compress_kernel,
        "Matrix A 4-bit quantization and compression.");
}
