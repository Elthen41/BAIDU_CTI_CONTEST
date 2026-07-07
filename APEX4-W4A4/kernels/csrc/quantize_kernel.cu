#include "quantize_kernel.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>

// =============================================================================
// =============================================================================
// =============================================================================

#define WARP_SIZE 32

__device__ __forceinline__ int padded_idx(int k) {
    return k + (k >> 5);  // k + k/32
}

template<int GROUP_SIZE>
struct PaddedSize {
    static constexpr int VALUE = (GROUP_SIZE / 32) * 33;  // 32 data + 1 pad per row
};

// =============================================================================
// Warp Reduction (fp32)
// =============================================================================
__device__ __forceinline__ float warp_reduce_max_f32(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xFFFFFFFF, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

// =============================================================================
// =============================================================================
template<int BLOCK_SIZE, int GROUP_SIZE>
__global__ void quantize_compress_unified(
    const __half* __restrict__ x_input,
    int32_t*     __restrict__ x_compressed,
    __half*      __restrict__ quant_scales,
    const int M, const int K
) {
    const int group_idx = blockIdx.x;
    const int row       = blockIdx.y;
    const int tid       = threadIdx.x;

    if (row >= M) return;

    const int num_groups  = K / GROUP_SIZE;
    const int group_start = row * K + group_idx * GROUP_SIZE;

    // ---- Shared memory: padded layout ----
    constexpr int PADDED_SZ = PaddedSize<GROUP_SIZE>::VALUE;
    __shared__ __half shared_data[PADDED_SZ];
    __shared__ float  shared_inv_scale;
    __shared__ __half shared_scale_out;

    // ================================================================
    // Stage 1: Global → Padded Shared Memory
    // ================================================================
    #pragma unroll 4
    for (int i = tid; i < GROUP_SIZE; i += BLOCK_SIZE) {
        shared_data[padded_idx(i)] = x_input[group_start + i];
    }
    __syncthreads();

    // ================================================================
    // ================================================================
    float local_max = 0.0f;

    #pragma unroll 4
    for (int i = tid; i < GROUP_SIZE; i += BLOCK_SIZE) {
        float v = fabsf(__half2float(shared_data[padded_idx(i)]));
        local_max = fmaxf(local_max, v);
    }

    // Warp-level reduction
    local_max = warp_reduce_max_f32(local_max);

    // Block-level reduction
    constexpr int NUM_WARPS = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float warp_max[NUM_WARPS];

    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    if (lane_id == 0) {
        warp_max[warp_id] = local_max;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_max = (lane_id < NUM_WARPS) ? warp_max[lane_id] : 0.0f;
        block_max = warp_reduce_max_f32(block_max);

        if (lane_id == 0) {
            float scale_f32 = block_max * (2.0f / 15.0f);
            scale_f32 = fmaxf(scale_f32, 1e-6f);

            shared_inv_scale = 1.0f / scale_f32;
            shared_scale_out = __float2half(scale_f32);
        }
    }
    __syncthreads();

    const float inv_scale = shared_inv_scale;

    if (tid == 0) {
        quant_scales[row * num_groups + group_idx] = shared_scale_out;
    }

    // ================================================================
    // ================================================================
    constexpr int PACKS_PER_GROUP = GROUP_SIZE / 8;
    const int output_base = row * (K / 8) + group_idx * PACKS_PER_GROUP;

    for (int pack_idx = tid; pack_idx < PACKS_PER_GROUP; pack_idx += BLOCK_SIZE) {
        const int base = pack_idx * 8;
        uint32_t packed = 0;

        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float scaled = __half2float(shared_data[padded_idx(base + j)]) * inv_scale;
            int q = __float2int_rn(scaled);
            q = max(-8, min(7, q));
            packed |= ((uint32_t)q & 0xFu) << (j * 4);
        }

        x_compressed[output_base + pack_idx] = (int32_t)packed;
    }
}

// =============================================================================
// =============================================================================
void launch_quantize_compress_kernel(
    const torch::Tensor& x_input,
          torch::Tensor& x_compressed,
          torch::Tensor& quant_scales,
          int group_size
) {
    const int M = x_input.size(0);
    const int K = x_input.size(1);

    TORCH_CHECK(x_input.dtype() == torch::kHalf,      "Input must be half precision");
    TORCH_CHECK(x_input.is_cuda(),                     "Input must be on CUDA device");
    TORCH_CHECK(x_input.is_contiguous(),               "Input must be contiguous");
    TORCH_CHECK(x_input.dim() == 2,                    "Input must be 2D tensor");
    TORCH_CHECK(K % group_size == 0,                   "K must be divisible by group_size");
    TORCH_CHECK(group_size % 8 == 0,                   "group_size must be divisible by 8");

    const int num_groups = K / group_size;

    TORCH_CHECK(x_compressed.dtype() == torch::kInt32, "Compressed tensor must be int32");
    TORCH_CHECK(quant_scales.dtype() == torch::kHalf,  "Scales tensor must be half");
    TORCH_CHECK(x_compressed.size(0) == M && x_compressed.size(1) == K / 8,
                "Compressed tensor shape mismatch");
    TORCH_CHECK(quant_scales.size(0) == M && quant_scales.size(1) == num_groups,
                "Scales tensor shape mismatch");
    TORCH_CHECK(x_compressed.is_cuda() && quant_scales.is_cuda(),
                "Output tensors must be on CUDA device");
    TORCH_CHECK(x_compressed.is_contiguous() && quant_scales.is_contiguous(),
                "Output tensors must be contiguous");

    const __half* input_ptr      = (const __half*)x_input.data_ptr();
    int32_t*      compressed_ptr = (int32_t*)x_compressed.data_ptr();
    __half*       scales_ptr     = (__half*)quant_scales.data_ptr();

    switch (group_size) {
        case 32: {
            dim3 grid(num_groups, M);
            dim3 block(32);
            quantize_compress_unified<32, 32><<<grid, block>>>(
                input_ptr, compressed_ptr, scales_ptr, M, K);
            break;
        }
        case 64: {
            dim3 grid(num_groups, M);
            dim3 block(64);
            quantize_compress_unified<64, 64><<<grid, block>>>(
                input_ptr, compressed_ptr, scales_ptr, M, K);
            break;
        }
        case 128: {
            dim3 grid(num_groups, M);
            dim3 block(128);
            quantize_compress_unified<128, 128><<<grid, block>>>(
                input_ptr, compressed_ptr, scales_ptr, M, K);
            break;
        }
        case 256: {
            dim3 grid(num_groups, M);
            dim3 block(128);
            quantize_compress_unified<128, 256><<<grid, block>>>(
                input_ptr, compressed_ptr, scales_ptr, M, K);
            break;
        }
        case 512: {
            dim3 grid(num_groups, M);
            dim3 block(256);
            quantize_compress_unified<256, 512><<<grid, block>>>(
                input_ptr, compressed_ptr, scales_ptr, M, K);
            break;
        }
        case 1024: {
            dim3 grid(num_groups, M);
            dim3 block(256);
            quantize_compress_unified<256, 1024><<<grid, block>>>(
                input_ptr, compressed_ptr, scales_ptr, M, K);
            break;
        }
        default:
            TORCH_CHECK(false, "Unsupported group_size: " + std::to_string(group_size) +
                        ". Supported: 32, 64, 128, 256, 512, 1024");
            return;
    }

    C10_CUDA_CHECK(cudaGetLastError());
}