/*
 * Matrix A 4-bit Quantization and Compression CUDA Kernel
 * Optimized for different group sizes with 3 strategies
 *
 * Features:
 * - Direct compression: 4bit -> int32, no intermediate buffers
 * - 3 optimization strategies: v1_small, v2_medium, v3_large
 * - Auto selection based on group_size
 * - Hardware-aware optimizations
 */
#pragma once

#include <torch/extension.h>

void launch_quantize_compress_kernel(
    const torch::Tensor& x_input,
          torch::Tensor& x_compressed,
          torch::Tensor& quant_scales,
          int group_size
);