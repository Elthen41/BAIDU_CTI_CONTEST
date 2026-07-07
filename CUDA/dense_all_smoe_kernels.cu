// CUDA dense-all SMoE kernels.
//
// This file keeps the current Python dense-all SMoE inference path available as
// three explicit stages:
//   1. fc1 + bias + ReLU:
//        x [N,512] x w1[e] [1024,512]^T -> h [8,N,1024]
//   2. fc2 + bias:
//        h[e] [N,1024] x w2[e] [512,1024]^T -> y [8,N,512]
//   3. top-2 gather:
//        out[n,d] = score0 * y[idx0,n,d] + score1 * y[idx1,n,d]
//
// The default dense_all_smoe_forward path fuses stage 2 and 3:
//   fc1 -> h [8,N,1024] -> fused fc2+top2 gather -> out [N,512]
//
// Weight layout expected by these kernels is CUDA TN-friendly:
//   w1: [8,1024,512], b1: [8,1024]
//   w2: [8,512,1024], b2: [8,512]
//
// The GEMM microkernel follows the LeetCUDA HGEMM TN structure:
// 128x128 CTA tile, 8 warps, ldmatrix + mma.sync.m16n8k16, fp32 accumulate.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <pybind11/stl.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <vector>

namespace {

constexpr int kNumExperts = 8;
constexpr int kHiddenDim = 512;
constexpr int kFfDim = 1024;
constexpr int kTopK = 2;

constexpr int kWarpSize = 32;
constexpr int kThreads = 256;
constexpr int kBlockM = 128;
constexpr int kBlockN = 128;
constexpr int kMmaM = 16;
constexpr int kMmaN = 8;
constexpr int kMmaK = 16;
constexpr int kWarpTileM = 4;
constexpr int kWarpTileN = 4;
constexpr int kPipelineStages = 2;

__device__ __forceinline__ uint32_t shared_addr(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void cp_async_cg_16(uint32_t dst, const void* src) {
    asm volatile(
        "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
        :
        : "r"(dst), "l"(src)
    );
}

__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::);
}

__device__ __forceinline__ void ldmatrix_x4(
    uint32_t& r0,
    uint32_t& r1,
    uint32_t& r2,
    uint32_t& r3,
    uint32_t addr
) {
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr)
    );
}

__device__ __forceinline__ void ldmatrix_x2(
    uint32_t& r0,
    uint32_t& r1,
    uint32_t addr
) {
    asm volatile(
        "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(r0), "=r"(r1)
        : "r"(addr)
    );
}

__device__ __forceinline__ void hmma16816_f32(
    uint32_t& d0,
    uint32_t& d1,
    uint32_t& d2,
    uint32_t& d3,
    uint32_t a0,
    uint32_t a1,
    uint32_t a2,
    uint32_t a3,
    uint32_t b0,
    uint32_t b1,
    uint32_t c0,
    uint32_t c1,
    uint32_t c2,
    uint32_t c3
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "r"(c0), "r"(c1), "r"(c2), "r"(c3)
    );
}

__device__ __forceinline__ float reg_as_float(uint32_t reg) {
    return __uint_as_float(reg);
}

__device__ __forceinline__ uint32_t pack_half2_bits(float lo, float hi) {
    union {
        __half2 h2;
        uint32_t u32;
    } packed;
    packed.h2 = __halves2half2(__float2half_rn(lo), __float2half_rn(hi));
    return packed.u32;
}

__device__ __forceinline__ void atomic_add_half(c10::Half* addr, float value) {
    atomicAdd(reinterpret_cast<__half*>(addr), __float2half_rn(value));
}

template <int InDim, int OutDim, bool InputHasExpertDim>
__device__ __forceinline__ void load_dense_all_tile_scalar(
    const c10::Half* __restrict__ input,
    const c10::Half* __restrict__ weight,
    c10::Half* __restrict__ a_shared,
    c10::Half* __restrict__ b_shared,
    int64_t n_tokens,
    int expert,
    int m_tile_base,
    int n_tile_base,
    int k_base,
    int tid
) {
    for (int idx = tid; idx < kBlockM * kMmaK; idx += kThreads) {
        const int local_m = idx / kMmaK;
        const int local_k = idx - local_m * kMmaK;
        const int global_m = m_tile_base + local_m;
        const int global_k = k_base + local_k;

        c10::Half val = static_cast<c10::Half>(0.0f);
        if (global_m < n_tokens) {
            int64_t input_idx;
            if constexpr (InputHasExpertDim) {
                input_idx = (static_cast<int64_t>(expert) * n_tokens + global_m) * InDim + global_k;
            } else {
                input_idx = static_cast<int64_t>(global_m) * InDim + global_k;
            }
            val = input[input_idx];
        }
        a_shared[idx] = val;
    }

    for (int idx = tid; idx < kBlockN * kMmaK; idx += kThreads) {
        const int local_n = idx / kMmaK;
        const int local_k = idx - local_n * kMmaK;
        const int global_n = n_tile_base + local_n;
        const int global_k = k_base + local_k;

        c10::Half val = static_cast<c10::Half>(0.0f);
        if (global_n < OutDim) {
            const int64_t weight_idx =
                (static_cast<int64_t>(expert) * OutDim + global_n) * InDim + global_k;
            val = weight[weight_idx];
        }
        b_shared[idx] = val;
    }
}

template <int InDim, int OutDim, bool InputHasExpertDim>
__device__ __forceinline__ void load_dense_all_tile_cp_async(
    const c10::Half* __restrict__ input,
    const c10::Half* __restrict__ weight,
    c10::Half* __restrict__ a_shared,
    c10::Half* __restrict__ b_shared,
    int64_t n_tokens,
    int expert,
    int m_tile_base,
    int n_tile_base,
    int k_base,
    int tid
) {
    const int local_row = tid >> 1;
    const int local_k = (tid & 1) * 8;
    const int global_m = m_tile_base + local_row;
    const int global_n = n_tile_base + local_row;
    const int global_k = k_base + local_k;

    int64_t input_idx;
    if constexpr (InputHasExpertDim) {
        input_idx = (static_cast<int64_t>(expert) * n_tokens + global_m) * InDim + global_k;
    } else {
        input_idx = static_cast<int64_t>(global_m) * InDim + global_k;
    }
    const int64_t weight_idx =
        (static_cast<int64_t>(expert) * OutDim + global_n) * InDim + global_k;

    const uint32_t a_addr = shared_addr(a_shared + local_row * kMmaK + local_k);
    const uint32_t b_addr = shared_addr(b_shared + local_row * kMmaK + local_k);
    cp_async_cg_16(a_addr, input + input_idx);
    cp_async_cg_16(b_addr, weight + weight_idx);
}

__device__ __forceinline__ void compute_dense_all_mma_stage(
    const c10::Half* __restrict__ a_shared,
    const c10::Half* __restrict__ b_shared,
    int warp_m,
    int warp_n,
    int lane,
    uint32_t (&acc)[kWarpTileM][kWarpTileN][4]
) {
    uint32_t a_frag[kWarpTileM][4];
    uint32_t b_frag[kWarpTileN][2];

#pragma unroll
    for (int i = 0; i < kWarpTileM; ++i) {
        const int warp_smem_a_m = warp_m * (kMmaM * kWarpTileM) + i * kMmaM;
        const int lane_smem_a_m = warp_smem_a_m + (lane & 15);
        const int lane_smem_a_k = (lane >= 16) ? 8 : 0;
        const uint32_t a_addr =
            shared_addr(a_shared + lane_smem_a_m * kMmaK + lane_smem_a_k);
        ldmatrix_x4(
            a_frag[i][0],
            a_frag[i][1],
            a_frag[i][2],
            a_frag[i][3],
            a_addr
        );
    }

#pragma unroll
    for (int j = 0; j < kWarpTileN; ++j) {
        const int warp_smem_b_n = warp_n * (kMmaN * kWarpTileN) + j * kMmaN;
        const int lane_smem_b_n = warp_smem_b_n + (lane & 7);
        const int lane_smem_b_k = ((lane >> 3) & 1) * 8;
        const uint32_t b_addr =
            shared_addr(b_shared + lane_smem_b_n * kMmaK + lane_smem_b_k);
        ldmatrix_x2(b_frag[j][0], b_frag[j][1], b_addr);
    }

#pragma unroll
    for (int i = 0; i < kWarpTileM; ++i) {
#pragma unroll
        for (int j = 0; j < kWarpTileN; ++j) {
            hmma16816_f32(
                acc[i][j][0],
                acc[i][j][1],
                acc[i][j][2],
                acc[i][j][3],
                a_frag[i][0],
                a_frag[i][1],
                a_frag[i][2],
                a_frag[i][3],
                b_frag[j][0],
                b_frag[j][1],
                acc[i][j][0],
                acc[i][j][1],
                acc[i][j][2],
                acc[i][j][3]
            );
        }
    }
}

template <int InDim, int OutDim, bool InputHasExpertDim, bool ApplyRelu>
__global__ __launch_bounds__(kThreads) void dense_all_linear_mma_tn_kernel(
    const c10::Half* __restrict__ input,
    const c10::Half* __restrict__ weight,
    const c10::Half* __restrict__ bias,
    c10::Half* __restrict__ output,
    int64_t n_tokens
) {
    static_assert(InDim % kMmaK == 0, "InDim must be divisible by 16");
    static_assert(OutDim % kBlockN == 0, "OutDim must be divisible by 128");

    __shared__ __align__(16) c10::Half a_shared[kPipelineStages * kBlockM * kMmaK];
    __shared__ __align__(16) c10::Half b_shared[kPipelineStages * kBlockN * kMmaK];

    const int tid = threadIdx.x;
    const int warp_id = tid / kWarpSize;
    const int lane = tid & (kWarpSize - 1);
    const int expert = blockIdx.z;
    const int m_tile_base = blockIdx.y * kBlockM;
    const int n_tile_base = blockIdx.x * kBlockN;

    const int warp_m = warp_id & 1;      // 0..1
    const int warp_n = warp_id >> 1;     // 0..3

    uint32_t acc[kWarpTileM][kWarpTileN][4];
#pragma unroll
    for (int i = 0; i < kWarpTileM; ++i) {
#pragma unroll
        for (int j = 0; j < kWarpTileN; ++j) {
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                acc[i][j][r] = 0;
            }
        }
    }

    const bool full_m_tile = (static_cast<int64_t>(m_tile_base) + kBlockM) <= n_tokens;
    constexpr int kNumKTiles = InDim / kMmaK;
    constexpr int kStageStrideA = kBlockM * kMmaK;
    constexpr int kStageStrideB = kBlockN * kMmaK;

    if (full_m_tile) {
        load_dense_all_tile_cp_async<InDim, OutDim, InputHasExpertDim>(
            input,
            weight,
            a_shared,
            b_shared,
            n_tokens,
            expert,
            m_tile_base,
            n_tile_base,
            0,
            tid
        );
        cp_async_commit_group();
        cp_async_wait_all();
        __syncthreads();

        for (int k_tile = 1; k_tile < kNumKTiles; ++k_tile) {
            const int compute_stage = (k_tile + 1) & 1;
            const int load_stage = k_tile & 1;

            load_dense_all_tile_cp_async<InDim, OutDim, InputHasExpertDim>(
                input,
                weight,
                a_shared + load_stage * kStageStrideA,
                b_shared + load_stage * kStageStrideB,
                n_tokens,
                expert,
                m_tile_base,
                n_tile_base,
                k_tile * kMmaK,
                tid
            );
            cp_async_commit_group();

            compute_dense_all_mma_stage(
                a_shared + compute_stage * kStageStrideA,
                b_shared + compute_stage * kStageStrideB,
                warp_m,
                warp_n,
                lane,
                acc
            );

            cp_async_wait_all();
            __syncthreads();
        }

        constexpr int last_stage = (kNumKTiles - 1) & 1;
        compute_dense_all_mma_stage(
            a_shared + last_stage * kStageStrideA,
            b_shared + last_stage * kStageStrideB,
            warp_m,
            warp_n,
            lane,
            acc
        );
    } else {
        for (int k_base = 0; k_base < InDim; k_base += kMmaK) {
            load_dense_all_tile_scalar<InDim, OutDim, InputHasExpertDim>(
                input,
                weight,
                a_shared,
                b_shared,
                n_tokens,
                expert,
                m_tile_base,
                n_tile_base,
                k_base,
                tid
            );
            __syncthreads();

            compute_dense_all_mma_stage(
                a_shared,
                b_shared,
                warp_m,
                warp_n,
                lane,
                acc
            );
            __syncthreads();
        }
    }

    const int row_base = m_tile_base + warp_m * (kMmaM * kWarpTileM);
    const int col_base = n_tile_base + warp_n * (kMmaN * kWarpTileN);
    const int frag_row0 = lane / 4;
    const int frag_row1 = frag_row0 + 8;
    const int frag_col_pair = (lane & 3) * 2;
    const int lane_group_base = lane & ~3;

#pragma unroll
    for (int i = 0; i < kWarpTileM; ++i) {
#pragma unroll
        for (int j = 0; j < kWarpTileN; ++j) {
            const int rows[2] = {
                row_base + i * kMmaM + frag_row0,
                row_base + i * kMmaM + frag_row1,
            };
            const int col0 = col_base + j * kMmaN + frag_col_pair;

#pragma unroll
            for (int row_slot = 0; row_slot < 2; ++row_slot) {
                const int row = rows[row_slot];
                const int col = col0;
                float value0 = reg_as_float(acc[i][j][row_slot * 2 + 0])
                    + static_cast<float>(bias[expert * OutDim + col + 0]);
                float value1 = reg_as_float(acc[i][j][row_slot * 2 + 1])
                    + static_cast<float>(bias[expert * OutDim + col + 1]);
                if constexpr (ApplyRelu) {
                    value0 = fmaxf(value0, 0.0f);
                    value1 = fmaxf(value1, 0.0f);
                }

                const uint32_t packed = pack_half2_bits(value0, value1);
                uint4 vec;
                vec.x = __shfl_sync(0xffffffff, packed, lane_group_base + 0);
                vec.y = __shfl_sync(0xffffffff, packed, lane_group_base + 1);
                vec.z = __shfl_sync(0xffffffff, packed, lane_group_base + 2);
                vec.w = __shfl_sync(0xffffffff, packed, lane_group_base + 3);

                if ((lane & 3) == 0 && row < n_tokens) {
                    const int64_t output_idx =
                        (static_cast<int64_t>(expert) * n_tokens + row) * OutDim
                        + col_base + j * kMmaN;
                    *reinterpret_cast<uint4*>(output + output_idx) = vec;
                }
            }
        }
    }
}

template <typename IndexT, typename ScoreT>
__global__ __launch_bounds__(kThreads) void dense_all_fc2_gather_top2_mma_tn_kernel(
    const c10::Half* __restrict__ h,
    const c10::Half* __restrict__ w2,
    const c10::Half* __restrict__ b2,
    const IndexT* __restrict__ topk_idx,
    const ScoreT* __restrict__ topk_score,
    c10::Half* __restrict__ out,
    int64_t n_tokens
) {
    constexpr int InDim = kFfDim;
    constexpr int OutDim = kHiddenDim;

    __shared__ __align__(16) c10::Half a_shared[kPipelineStages * kBlockM * kMmaK];
    __shared__ __align__(16) c10::Half b_shared[kPipelineStages * kBlockN * kMmaK];

    const int tid = threadIdx.x;
    const int warp_id = tid / kWarpSize;
    const int lane = tid & (kWarpSize - 1);
    const int expert = blockIdx.z;
    const int m_tile_base = blockIdx.y * kBlockM;
    const int n_tile_base = blockIdx.x * kBlockN;

    const int warp_m = warp_id & 1;
    const int warp_n = warp_id >> 1;

    uint32_t acc[kWarpTileM][kWarpTileN][4];
#pragma unroll
    for (int i = 0; i < kWarpTileM; ++i) {
#pragma unroll
        for (int j = 0; j < kWarpTileN; ++j) {
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                acc[i][j][r] = 0;
            }
        }
    }

    const bool full_m_tile = (static_cast<int64_t>(m_tile_base) + kBlockM) <= n_tokens;
    constexpr int kNumKTiles = InDim / kMmaK;
    constexpr int kStageStrideA = kBlockM * kMmaK;
    constexpr int kStageStrideB = kBlockN * kMmaK;

    if (full_m_tile) {
        load_dense_all_tile_cp_async<InDim, OutDim, true>(
            h,
            w2,
            a_shared,
            b_shared,
            n_tokens,
            expert,
            m_tile_base,
            n_tile_base,
            0,
            tid
        );
        cp_async_commit_group();
        cp_async_wait_all();
        __syncthreads();

        for (int k_tile = 1; k_tile < kNumKTiles; ++k_tile) {
            const int compute_stage = (k_tile + 1) & 1;
            const int load_stage = k_tile & 1;

            load_dense_all_tile_cp_async<InDim, OutDim, true>(
                h,
                w2,
                a_shared + load_stage * kStageStrideA,
                b_shared + load_stage * kStageStrideB,
                n_tokens,
                expert,
                m_tile_base,
                n_tile_base,
                k_tile * kMmaK,
                tid
            );
            cp_async_commit_group();

            compute_dense_all_mma_stage(
                a_shared + compute_stage * kStageStrideA,
                b_shared + compute_stage * kStageStrideB,
                warp_m,
                warp_n,
                lane,
                acc
            );

            cp_async_wait_all();
            __syncthreads();
        }

        constexpr int last_stage = (kNumKTiles - 1) & 1;
        compute_dense_all_mma_stage(
            a_shared + last_stage * kStageStrideA,
            b_shared + last_stage * kStageStrideB,
            warp_m,
            warp_n,
            lane,
            acc
        );
    } else {
        for (int k_base = 0; k_base < InDim; k_base += kMmaK) {
            load_dense_all_tile_scalar<InDim, OutDim, true>(
                h,
                w2,
                a_shared,
                b_shared,
                n_tokens,
                expert,
                m_tile_base,
                n_tile_base,
                k_base,
                tid
            );
            __syncthreads();

            compute_dense_all_mma_stage(
                a_shared,
                b_shared,
                warp_m,
                warp_n,
                lane,
                acc
            );
            __syncthreads();
        }
    }

    const int row_base = m_tile_base + warp_m * (kMmaM * kWarpTileM);
    const int col_base = n_tile_base + warp_n * (kMmaN * kWarpTileN);
    const int frag_row0 = lane / 4;
    const int frag_row1 = frag_row0 + 8;
    const int frag_col_pair = (lane & 3) * 2;

#pragma unroll
    for (int i = 0; i < kWarpTileM; ++i) {
#pragma unroll
        for (int j = 0; j < kWarpTileN; ++j) {
            const int rows[2] = {
                row_base + i * kMmaM + frag_row0,
                row_base + i * kMmaM + frag_row1,
            };
            const int col0 = col_base + j * kMmaN + frag_col_pair;

#pragma unroll
            for (int row_slot = 0; row_slot < 2; ++row_slot) {
                const int row = rows[row_slot];
                if (row >= n_tokens) {
                    continue;
                }

                float route_score = 0.0f;
                const int idx0 = static_cast<int>(topk_idx[row * kTopK + 0]);
                const int idx1 = static_cast<int>(topk_idx[row * kTopK + 1]);
                if (idx0 == expert) {
                    route_score += static_cast<float>(topk_score[row * kTopK + 0]);
                }
                if (idx1 == expert) {
                    route_score += static_cast<float>(topk_score[row * kTopK + 1]);
                }
                if (route_score == 0.0f) {
                    continue;
                }

#pragma unroll
                for (int value_slot = 0; value_slot < 2; ++value_slot) {
                    const int col = col0 + value_slot;
                    float value = reg_as_float(acc[i][j][row_slot * 2 + value_slot])
                        + static_cast<float>(b2[expert * OutDim + col]);
                    value *= route_score;
                    atomic_add_half(out + static_cast<int64_t>(row) * OutDim + col, value);
                }
            }
        }
    }
}

template <typename IndexT, typename ScoreT>
__global__ void dense_all_gather_top2_kernel(
    const c10::Half* __restrict__ y,
    const IndexT* __restrict__ topk_idx,
    const ScoreT* __restrict__ topk_score,
    c10::Half* __restrict__ out,
    int64_t n_tokens
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = n_tokens * kHiddenDim;
    if (idx >= total) {
        return;
    }

    const int64_t token = idx / kHiddenDim;
    const int dim = static_cast<int>(idx - token * kHiddenDim);

    const int expert0 = static_cast<int>(topk_idx[token * kTopK + 0]);
    const int expert1 = static_cast<int>(topk_idx[token * kTopK + 1]);
    const float score0 = static_cast<float>(topk_score[token * kTopK + 0]);
    const float score1 = static_cast<float>(topk_score[token * kTopK + 1]);

    float value = 0.0f;
    if (expert0 >= 0 && expert0 < kNumExperts) {
        value += score0 * static_cast<float>(
            y[(static_cast<int64_t>(expert0) * n_tokens + token) * kHiddenDim + dim]
        );
    }
    if (expert1 >= 0 && expert1 < kNumExperts) {
        value += score1 * static_cast<float>(
            y[(static_cast<int64_t>(expert1) * n_tokens + token) * kHiddenDim + dim]
        );
    }
    out[idx] = static_cast<c10::Half>(value);
}

int64_t ceil_div_int64(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

void check_half_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kHalf, name, " must be float16");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_same_device(const torch::Tensor& a, const torch::Tensor& b, const char* a_name, const char* b_name) {
    TORCH_CHECK(a.device() == b.device(), a_name, " and ", b_name, " must be on the same CUDA device");
}

void check_launch_n_tokens(int64_t n_tokens) {
    TORCH_CHECK(n_tokens >= 0, "n_tokens must be non-negative");
    TORCH_CHECK(
        n_tokens <= static_cast<int64_t>(std::numeric_limits<int>::max()),
        "n_tokens is too large"
    );
    TORCH_CHECK(
        ceil_div_int64(n_tokens, kBlockM) <= 65535,
        "n_tokens creates too many CTA rows for this first implementation"
    );
}

template <int InDim, int OutDim, bool InputHasExpertDim, bool ApplyRelu>
void launch_dense_all_linear(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int64_t n_tokens
) {
    if (n_tokens == 0) {
        return;
    }

    const dim3 block(kThreads);
    const dim3 grid(
        static_cast<unsigned int>(OutDim / kBlockN),
        static_cast<unsigned int>(ceil_div_int64(n_tokens, kBlockM)),
        kNumExperts
    );

    dense_all_linear_mma_tn_kernel<InDim, OutDim, InputHasExpertDim, ApplyRelu>
        <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            input.data_ptr<c10::Half>(),
            weight.data_ptr<c10::Half>(),
            bias.data_ptr<c10::Half>(),
            output.data_ptr<c10::Half>(),
            n_tokens
        );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

torch::Tensor dense_all_fc1_relu(torch::Tensor x, torch::Tensor w1, torch::Tensor b1) {
    check_half_cuda_contiguous(x, "x");
    check_half_cuda_contiguous(w1, "w1");
    check_half_cuda_contiguous(b1, "b1");
    check_same_device(x, w1, "x", "w1");
    check_same_device(x, b1, "x", "b1");
    TORCH_CHECK(x.dim() == 2, "x must have shape [N,512]");
    TORCH_CHECK(x.size(1) == kHiddenDim, "x must have shape [N,512]");
    TORCH_CHECK(w1.dim() == 3, "w1 must have shape [8,1024,512]");
    TORCH_CHECK(w1.size(0) == kNumExperts, "w1 must have shape [8,1024,512]");
    TORCH_CHECK(w1.size(1) == kFfDim, "w1 must have shape [8,1024,512]");
    TORCH_CHECK(w1.size(2) == kHiddenDim, "w1 must have shape [8,1024,512]");
    TORCH_CHECK(b1.dim() == 2, "b1 must have shape [8,1024]");
    TORCH_CHECK(b1.size(0) == kNumExperts, "b1 must have shape [8,1024]");
    TORCH_CHECK(b1.size(1) == kFfDim, "b1 must have shape [8,1024]");

    const int64_t n_tokens = x.size(0);
    check_launch_n_tokens(n_tokens);

    auto h = torch::empty({kNumExperts, n_tokens, kFfDim}, x.options());
    launch_dense_all_linear<kHiddenDim, kFfDim, false, true>(x, w1, b1, h, n_tokens);
    return h;
}

torch::Tensor dense_all_fc2(torch::Tensor h, torch::Tensor w2, torch::Tensor b2) {
    check_half_cuda_contiguous(h, "h");
    check_half_cuda_contiguous(w2, "w2");
    check_half_cuda_contiguous(b2, "b2");
    check_same_device(h, w2, "h", "w2");
    check_same_device(h, b2, "h", "b2");
    TORCH_CHECK(h.dim() == 3, "h must have shape [8,N,1024]");
    TORCH_CHECK(h.size(0) == kNumExperts, "h must have shape [8,N,1024]");
    TORCH_CHECK(h.size(2) == kFfDim, "h must have shape [8,N,1024]");
    TORCH_CHECK(w2.dim() == 3, "w2 must have shape [8,512,1024]");
    TORCH_CHECK(w2.size(0) == kNumExperts, "w2 must have shape [8,512,1024]");
    TORCH_CHECK(w2.size(1) == kHiddenDim, "w2 must have shape [8,512,1024]");
    TORCH_CHECK(w2.size(2) == kFfDim, "w2 must have shape [8,512,1024]");
    TORCH_CHECK(b2.dim() == 2, "b2 must have shape [8,512]");
    TORCH_CHECK(b2.size(0) == kNumExperts, "b2 must have shape [8,512]");
    TORCH_CHECK(b2.size(1) == kHiddenDim, "b2 must have shape [8,512]");

    const int64_t n_tokens = h.size(1);
    check_launch_n_tokens(n_tokens);

    auto y = torch::empty({kNumExperts, n_tokens, kHiddenDim}, h.options());
    launch_dense_all_linear<kFfDim, kHiddenDim, true, false>(h, w2, b2, y, n_tokens);
    return y;
}

torch::Tensor dense_all_fc2_gather_top2(
    torch::Tensor h,
    torch::Tensor w2,
    torch::Tensor b2,
    torch::Tensor topk_idx,
    torch::Tensor topk_score
) {
    check_half_cuda_contiguous(h, "h");
    check_half_cuda_contiguous(w2, "w2");
    check_half_cuda_contiguous(b2, "b2");
    TORCH_CHECK(topk_idx.is_cuda(), "topk_idx must be a CUDA tensor");
    TORCH_CHECK(topk_score.is_cuda(), "topk_score must be a CUDA tensor");
    TORCH_CHECK(topk_idx.is_contiguous(), "topk_idx must be contiguous");
    TORCH_CHECK(topk_score.is_contiguous(), "topk_score must be contiguous");
    TORCH_CHECK(topk_idx.scalar_type() == at::kLong || topk_idx.scalar_type() == at::kInt,
        "topk_idx must be int64 or int32");
    TORCH_CHECK(topk_score.scalar_type() == at::kHalf || topk_score.scalar_type() == at::kFloat,
        "topk_score must be float16 or float32");
    check_same_device(h, w2, "h", "w2");
    check_same_device(h, b2, "h", "b2");
    check_same_device(h, topk_idx, "h", "topk_idx");
    check_same_device(h, topk_score, "h", "topk_score");
    TORCH_CHECK(h.dim() == 3, "h must have shape [8,N,1024]");
    TORCH_CHECK(h.size(0) == kNumExperts, "h must have shape [8,N,1024]");
    TORCH_CHECK(h.size(2) == kFfDim, "h must have shape [8,N,1024]");
    TORCH_CHECK(w2.dim() == 3, "w2 must have shape [8,512,1024]");
    TORCH_CHECK(w2.size(0) == kNumExperts, "w2 must have shape [8,512,1024]");
    TORCH_CHECK(w2.size(1) == kHiddenDim, "w2 must have shape [8,512,1024]");
    TORCH_CHECK(w2.size(2) == kFfDim, "w2 must have shape [8,512,1024]");
    TORCH_CHECK(b2.dim() == 2, "b2 must have shape [8,512]");
    TORCH_CHECK(b2.size(0) == kNumExperts, "b2 must have shape [8,512]");
    TORCH_CHECK(b2.size(1) == kHiddenDim, "b2 must have shape [8,512]");

    const int64_t n_tokens = h.size(1);
    check_launch_n_tokens(n_tokens);
    TORCH_CHECK(topk_idx.dim() == 2, "topk_idx must have shape [N,2]");
    TORCH_CHECK(topk_idx.size(0) == n_tokens, "topk_idx must have shape [N,2]");
    TORCH_CHECK(topk_idx.size(1) == kTopK, "topk_idx must have shape [N,2]");
    TORCH_CHECK(topk_score.dim() == 2, "topk_score must have shape [N,2]");
    TORCH_CHECK(topk_score.size(0) == n_tokens, "topk_score must have shape [N,2]");
    TORCH_CHECK(topk_score.size(1) == kTopK, "topk_score must have shape [N,2]");

    auto out = torch::empty({n_tokens, kHiddenDim}, h.options());
    if (n_tokens == 0) {
        return out;
    }

    auto stream = at::cuda::getCurrentCUDAStream();
    cudaStream_t cuda_stream = stream.stream();
    C10_CUDA_CHECK(cudaMemsetAsync(
        out.data_ptr<c10::Half>(),
        0,
        static_cast<size_t>(out.numel()) * sizeof(c10::Half),
        cuda_stream
    ));

    const dim3 block(kThreads);
    const dim3 grid(
        static_cast<unsigned int>(kHiddenDim / kBlockN),
        static_cast<unsigned int>(ceil_div_int64(n_tokens, kBlockM)),
        kNumExperts
    );

    if (topk_idx.scalar_type() == at::kLong && topk_score.scalar_type() == at::kHalf) {
        dense_all_fc2_gather_top2_mma_tn_kernel<int64_t, c10::Half>
            <<<grid, block, 0, cuda_stream>>>(
                h.data_ptr<c10::Half>(),
                w2.data_ptr<c10::Half>(),
                b2.data_ptr<c10::Half>(),
                topk_idx.data_ptr<int64_t>(),
                topk_score.data_ptr<c10::Half>(),
                out.data_ptr<c10::Half>(),
                n_tokens
            );
    } else if (topk_idx.scalar_type() == at::kLong && topk_score.scalar_type() == at::kFloat) {
        dense_all_fc2_gather_top2_mma_tn_kernel<int64_t, float>
            <<<grid, block, 0, cuda_stream>>>(
                h.data_ptr<c10::Half>(),
                w2.data_ptr<c10::Half>(),
                b2.data_ptr<c10::Half>(),
                topk_idx.data_ptr<int64_t>(),
                topk_score.data_ptr<float>(),
                out.data_ptr<c10::Half>(),
                n_tokens
            );
    } else if (topk_idx.scalar_type() == at::kInt && topk_score.scalar_type() == at::kHalf) {
        dense_all_fc2_gather_top2_mma_tn_kernel<int32_t, c10::Half>
            <<<grid, block, 0, cuda_stream>>>(
                h.data_ptr<c10::Half>(),
                w2.data_ptr<c10::Half>(),
                b2.data_ptr<c10::Half>(),
                topk_idx.data_ptr<int32_t>(),
                topk_score.data_ptr<c10::Half>(),
                out.data_ptr<c10::Half>(),
                n_tokens
            );
    } else {
        dense_all_fc2_gather_top2_mma_tn_kernel<int32_t, float>
            <<<grid, block, 0, cuda_stream>>>(
                h.data_ptr<c10::Half>(),
                w2.data_ptr<c10::Half>(),
                b2.data_ptr<c10::Half>(),
                topk_idx.data_ptr<int32_t>(),
                topk_score.data_ptr<float>(),
                out.data_ptr<c10::Half>(),
                n_tokens
            );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

torch::Tensor dense_all_gather_top2(torch::Tensor y, torch::Tensor topk_idx, torch::Tensor topk_score) {
    check_half_cuda_contiguous(y, "y");
    TORCH_CHECK(topk_idx.is_cuda(), "topk_idx must be a CUDA tensor");
    TORCH_CHECK(topk_score.is_cuda(), "topk_score must be a CUDA tensor");
    TORCH_CHECK(topk_idx.is_contiguous(), "topk_idx must be contiguous");
    TORCH_CHECK(topk_score.is_contiguous(), "topk_score must be contiguous");
    TORCH_CHECK(topk_idx.scalar_type() == at::kLong || topk_idx.scalar_type() == at::kInt,
        "topk_idx must be int64 or int32");
    TORCH_CHECK(topk_score.scalar_type() == at::kHalf || topk_score.scalar_type() == at::kFloat,
        "topk_score must be float16 or float32");
    check_same_device(y, topk_idx, "y", "topk_idx");
    check_same_device(y, topk_score, "y", "topk_score");
    TORCH_CHECK(y.dim() == 3, "y must have shape [8,N,512]");
    TORCH_CHECK(y.size(0) == kNumExperts, "y must have shape [8,N,512]");
    TORCH_CHECK(y.size(2) == kHiddenDim, "y must have shape [8,N,512]");

    const int64_t n_tokens = y.size(1);
    TORCH_CHECK(topk_idx.dim() == 2, "topk_idx must have shape [N,2]");
    TORCH_CHECK(topk_idx.size(0) == n_tokens, "topk_idx must have shape [N,2]");
    TORCH_CHECK(topk_idx.size(1) == kTopK, "topk_idx must have shape [N,2]");
    TORCH_CHECK(topk_score.dim() == 2, "topk_score must have shape [N,2]");
    TORCH_CHECK(topk_score.size(0) == n_tokens, "topk_score must have shape [N,2]");
    TORCH_CHECK(topk_score.size(1) == kTopK, "topk_score must have shape [N,2]");

    auto out = torch::empty({n_tokens, kHiddenDim}, y.options());
    if (n_tokens == 0) {
        return out;
    }

    constexpr int kGatherThreads = 256;
    const int64_t total = n_tokens * kHiddenDim;
    const int blocks = static_cast<int>(ceil_div_int64(total, kGatherThreads));

    if (topk_idx.scalar_type() == at::kLong && topk_score.scalar_type() == at::kHalf) {
        dense_all_gather_top2_kernel<int64_t, c10::Half>
            <<<blocks, kGatherThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
                y.data_ptr<c10::Half>(),
                topk_idx.data_ptr<int64_t>(),
                topk_score.data_ptr<c10::Half>(),
                out.data_ptr<c10::Half>(),
                n_tokens
            );
    } else if (topk_idx.scalar_type() == at::kLong && topk_score.scalar_type() == at::kFloat) {
        dense_all_gather_top2_kernel<int64_t, float>
            <<<blocks, kGatherThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
                y.data_ptr<c10::Half>(),
                topk_idx.data_ptr<int64_t>(),
                topk_score.data_ptr<float>(),
                out.data_ptr<c10::Half>(),
                n_tokens
            );
    } else if (topk_idx.scalar_type() == at::kInt && topk_score.scalar_type() == at::kHalf) {
        dense_all_gather_top2_kernel<int32_t, c10::Half>
            <<<blocks, kGatherThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
                y.data_ptr<c10::Half>(),
                topk_idx.data_ptr<int32_t>(),
                topk_score.data_ptr<c10::Half>(),
                out.data_ptr<c10::Half>(),
                n_tokens
            );
    } else {
        dense_all_gather_top2_kernel<int32_t, float>
            <<<blocks, kGatherThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
                y.data_ptr<c10::Half>(),
                topk_idx.data_ptr<int32_t>(),
                topk_score.data_ptr<float>(),
                out.data_ptr<c10::Half>(),
                n_tokens
            );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

torch::Tensor dense_all_smoe_forward(
    torch::Tensor x,
    torch::Tensor w1,
    torch::Tensor b1,
    torch::Tensor w2,
    torch::Tensor b2,
    torch::Tensor topk_idx,
    torch::Tensor topk_score
) {
    auto h = dense_all_fc1_relu(x, w1, b1);
    return dense_all_fc2_gather_top2(h, w2, b2, topk_idx, topk_score);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dense_all_fc1_relu", &dense_all_fc1_relu, "Dense-all SMoE fc1 + bias + ReLU");
    m.def("dense_all_fc2", &dense_all_fc2, "Dense-all SMoE fc2 + bias");
    m.def("dense_all_fc2_gather_top2", &dense_all_fc2_gather_top2, "Dense-all SMoE fused fc2 + top-2 gather");
    m.def("dense_all_gather_top2", &dense_all_gather_top2, "Dense-all SMoE top-2 gather");
    m.def("dense_all_smoe_forward", &dense_all_smoe_forward, "Dense-all SMoE forward");
}
