// CUDA routed sparse Mixture-of-Experts kernels.
//
// This file implements the first routed sparse SMoE path:
//   topk_idx/topk_score
//     -> route count + padded expert offsets
//     -> pack x[N,512] by expert into x_route[pool,512]
//     -> grouped fc1 + bias + ReLU
//     -> grouped fc2 + bias
//     -> explicit top-2 reduce back to out[N,512]
//
// Weight layout is the same TN-friendly layout used by dense_all_smoe:
//   w1: [8,1024,512], b1: [8,1024]
//   w2: [8,512,1024], b2: [8,512]
//
// The grouped GEMM microkernel follows the LeetCUDA SM80 HGEMM TN shape:
// 128x128 CTA tile, 8 warps, cp.async for full M tiles, ldmatrix, and
// mma.sync.m16n8k16 with fp32 accumulation.
//
// CTA scheduling is pooled over the padded route pool:
//   grid = (N tiles, pooled M tiles, 1)
// and each CTA maps its pooled M tile back to an expert via offsets[].

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <pybind11/stl.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#if BAIDU_CTI_ENABLE_CUTLASS_SMOE
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#endif

#include <cstdint>
#include <cstdlib>
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
constexpr int kBlockM64 = 64;
constexpr int kMmaM = 16;
constexpr int kMmaN = 8;
constexpr int kMmaK = 16;
constexpr int kWarpTileM = 4;
constexpr int kWarpTileN = 4;
constexpr int kWarpTileM64 = 2;
constexpr int kPipelineStages = 2;
constexpr int kSimtFc2BlockM = 16;
constexpr int kSimtFc2BlockN = 16;
constexpr int kSimpleW4A4CtaM = 16;
constexpr int kSimpleW4A4CtaN = 128;
constexpr int kSimpleW4A4WarpsPerCta = 8;
constexpr int kSimpleW4A4WarpNFragments = 4;
constexpr int kSimpleW4A4PackRowsPerCta = 2;

struct RouteMetadata {
    torch::Tensor x_route;
    torch::Tensor route_pos;
    torch::Tensor route_token;
    torch::Tensor route_slot;
    torch::Tensor route_score;
    torch::Tensor counts;
    torch::Tensor offsets;
    int64_t max_routes_per_expert;
};

int64_t ceil_div_int64(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

int64_t max_pool_routes_for_tokens(int64_t n_tokens, int64_t pad_multiple = kBlockM) {
    if (n_tokens == 0) {
        return 0;
    }
    TORCH_CHECK(pad_multiple > 0, "pad_multiple must be positive");
    return n_tokens * kTopK + kNumExperts * (pad_multiple - 1);
}

bool env_flag_enabled(const char* name, bool default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }
    return !(value[0] == '0' && value[1] == '\0');
}

void check_half_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kHalf, name, " must be float16");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_i32_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kInt, name, " must be int32");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_i16_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kShort, name, " must be int16");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_scale_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(
        tensor.scalar_type() == at::kHalf || tensor.scalar_type() == at::kFloat,
        name, " must be float16 or float32"
    );
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_topk_idx(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(
        tensor.scalar_type() == at::kLong || tensor.scalar_type() == at::kInt,
        name, " must be int64 or int32"
    );
}

void check_topk_score(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(
        tensor.scalar_type() == at::kHalf || tensor.scalar_type() == at::kFloat,
        name, " must be float16 or float32"
    );
}

void check_same_device(const torch::Tensor& a, const torch::Tensor& b, const char* a_name, const char* b_name) {
    TORCH_CHECK(a.device() == b.device(), a_name, " and ", b_name, " must be on the same CUDA device");
}

void check_n_tokens(int64_t n_tokens) {
    TORCH_CHECK(n_tokens >= 0, "n_tokens must be non-negative");
    TORCH_CHECK(n_tokens <= static_cast<int64_t>(std::numeric_limits<int>::max()) / kTopK,
        "n_tokens is too large");
    TORCH_CHECK(max_pool_routes_for_tokens(n_tokens) <= static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
        "n_tokens creates a route pool that is too large for int32 metadata");
    TORCH_CHECK(ceil_div_int64(n_tokens * kTopK, kBlockM) <= 65535,
        "n_tokens creates too many CTA rows for this first implementation");
}

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

__device__ __forceinline__ uint32_t half2_as_uint(__half2 value) {
    union {
        __half2 h2;
        uint32_t u32;
    } packed;
    packed.h2 = value;
    return packed.u32;
}

__device__ __forceinline__ uint32_t pack_s4_pair_bits_from_half_rounded(
    float lo,
    float hi,
    int nibble_base,
    float inv_scale
) {
    int q0 = __float2int_rn(__half2float(__float2half_rn(lo)) * inv_scale);
    int q1 = __float2int_rn(__half2float(__float2half_rn(hi)) * inv_scale);
    q0 = max(-8, min(7, q0));
    q1 = max(-8, min(7, q1));
    return ((static_cast<uint32_t>(q0) & 0xFu) << (4 * nibble_base))
        | ((static_cast<uint32_t>(q1) & 0xFu) << (4 * (nibble_base + 1)));
}

template <int lut>
__device__ __forceinline__ int lop3(int a, int b, int c) {
    int res;
    asm volatile(
        "lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(res)
        : "r"(a), "r"(b), "r"(c), "n"(lut)
    );
    return res;
}

__device__ __forceinline__ int find_expert_for_pool_m(
    const int32_t* __restrict__ offsets,
    int pool_m_tile_base
) {
#pragma unroll
    for (int expert = 0; expert < kNumExperts; ++expert) {
        if (pool_m_tile_base < offsets[expert + 1]) {
            return expert;
        }
    }
    return kNumExperts;
}

template <typename IndexT>
__global__ void smoe_route_count_kernel(
    const IndexT* __restrict__ topk_idx,
    int32_t* __restrict__ counts,
    int64_t n_tokens
) {
    const int64_t route_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total_routes = n_tokens * kTopK;
    if (route_id >= total_routes) {
        return;
    }

    const int expert = static_cast<int>(topk_idx[route_id]);
    if (expert >= 0 && expert < kNumExperts) {
        atomicAdd(counts + expert, 1);
    }
}

__global__ void smoe_route_prefix_kernel(
    const int32_t* __restrict__ counts,
    int32_t* __restrict__ offsets,
    int32_t* __restrict__ cursors,
    int32_t pad_multiple
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    int32_t running = 0;
#pragma unroll
    for (int expert = 0; expert < kNumExperts; ++expert) {
        offsets[expert] = running;
        cursors[expert] = running;
        const int32_t count = counts[expert];
        const int32_t padded = ((count + pad_multiple - 1) / pad_multiple) * pad_multiple;
        running += padded;
    }
    offsets[kNumExperts] = running;
}

template <typename IndexT, typename ScoreT, bool WriteDebugMetadata>
__global__ void smoe_route_pack_one_route_kernel(
    const c10::Half* __restrict__ x,
    const IndexT* __restrict__ topk_idx,
    const ScoreT* __restrict__ topk_score,
    int32_t* __restrict__ cursors,
    c10::Half* __restrict__ x_route,
    int32_t* __restrict__ route_pos,
    int32_t* __restrict__ route_token,
    int32_t* __restrict__ route_slot,
    ScoreT* __restrict__ route_score,
    int64_t n_tokens
) {
    const int64_t route_id = blockIdx.x;
    const int token = static_cast<int>(route_id >> 1);
    const int slot = static_cast<int>(route_id & 1);
    if (route_id >= n_tokens * kTopK) {
        return;
    }

    const int expert = static_cast<int>(topk_idx[route_id]);
    if (expert < 0 || expert >= kNumExperts) {
        return;
    }

    __shared__ int32_t shared_pos;
    if (threadIdx.x == 0) {
        shared_pos = atomicAdd(cursors + expert, 1);
        const int32_t pos = shared_pos;
        route_pos[token * kTopK + slot] = pos;
        if constexpr (WriteDebugMetadata) {
            route_token[pos] = token;
            route_slot[pos] = slot;
            route_score[pos] = topk_score[route_id];
        }
    }
    __syncthreads();
    const int32_t pos = shared_pos;

    const uint4* __restrict__ x_vec =
        reinterpret_cast<const uint4*>(x + static_cast<int64_t>(token) * kHiddenDim);
    uint4* __restrict__ route_vec =
        reinterpret_cast<uint4*>(x_route + static_cast<int64_t>(pos) * kHiddenDim);
    constexpr int kVecsPerRow = kHiddenDim / 8;
    for (int vec = threadIdx.x; vec < kVecsPerRow; vec += blockDim.x) {
        route_vec[vec] = x_vec[vec];
    }
}

template <typename IndexT, typename ScoreT, bool WriteDebugMetadata>
__global__ void smoe_route_pack_kernel(
    const c10::Half* __restrict__ x,
    const IndexT* __restrict__ topk_idx,
    const ScoreT* __restrict__ topk_score,
    int32_t* __restrict__ cursors,
    c10::Half* __restrict__ x_route,
    int32_t* __restrict__ route_pos,
    int32_t* __restrict__ route_token,
    int32_t* __restrict__ route_slot,
    ScoreT* __restrict__ route_score,
    int64_t n_tokens
) {
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int64_t route_id = static_cast<int64_t>(blockIdx.x) * 4 + warp_id;
    const int token = static_cast<int>(route_id >> 1);
    const int slot = static_cast<int>(route_id & 1);
    if (route_id >= n_tokens * kTopK) {
        return;
    }

    const int expert = static_cast<int>(topk_idx[route_id]);
    if (expert < 0 || expert >= kNumExperts) {
        return;
    }

    int32_t pos = 0;
    if (lane == 0) {
        pos = atomicAdd(cursors + expert, 1);
        route_pos[token * kTopK + slot] = pos;
        if constexpr (WriteDebugMetadata) {
            route_token[pos] = token;
            route_slot[pos] = slot;
            route_score[pos] = topk_score[route_id];
        }
    }
    pos = __shfl_sync(0xffffffff, pos, 0);

    const uint4* __restrict__ x_vec =
        reinterpret_cast<const uint4*>(x + static_cast<int64_t>(token) * kHiddenDim);
    uint4* __restrict__ route_vec =
        reinterpret_cast<uint4*>(x_route + static_cast<int64_t>(pos) * kHiddenDim);
    constexpr int kVecsPerRow = kHiddenDim / 8;
    static_assert(kVecsPerRow == 64, "route pack assumes 64 uint4 vectors per row");
    route_vec[lane] = x_vec[lane];
    route_vec[lane + 32] = x_vec[lane + 32];
}

template <int InDim, int OutDim>
__device__ __forceinline__ void load_grouped_tile_scalar(
    const c10::Half* __restrict__ input,
    const c10::Half* __restrict__ weight,
    const int32_t* __restrict__ counts,
    const int32_t* __restrict__ offsets,
    c10::Half* __restrict__ a_shared,
    c10::Half* __restrict__ b_shared,
    int expert,
    int local_m_tile_base,
    int n_tile_base,
    int k_base,
    int tid
) {
    const int expert_count = counts[expert];
    const int expert_offset = offsets[expert];

    for (int idx = tid; idx < kBlockM * kMmaK; idx += kThreads) {
        const int local_m = idx / kMmaK;
        const int local_k = idx - local_m * kMmaK;
        const int route_m = local_m_tile_base + local_m;
        const int global_k = k_base + local_k;

        c10::Half val = static_cast<c10::Half>(0.0f);
        if (route_m < expert_count) {
            const int64_t input_idx =
                (static_cast<int64_t>(expert_offset) + route_m) * InDim + global_k;
            val = input[input_idx];
        }
        a_shared[idx] = val;
    }

    for (int idx = tid; idx < kBlockN * kMmaK; idx += kThreads) {
        const int local_n = idx / kMmaK;
        const int local_k = idx - local_n * kMmaK;
        const int global_n = n_tile_base + local_n;
        const int global_k = k_base + local_k;

        const int64_t weight_idx =
            (static_cast<int64_t>(expert) * OutDim + global_n) * InDim + global_k;
        b_shared[idx] = weight[weight_idx];
    }
}

template <int InDim, int OutDim>
__device__ __forceinline__ void load_grouped_tile_cp_async(
    const c10::Half* __restrict__ input,
    const c10::Half* __restrict__ weight,
    const int32_t* __restrict__ offsets,
    c10::Half* __restrict__ a_shared,
    c10::Half* __restrict__ b_shared,
    int expert,
    int local_m_tile_base,
    int n_tile_base,
    int k_base,
    int tid
) {
    const int local_row = tid >> 1;
    const int local_k = (tid & 1) * 8;
    const int expert_offset = offsets[expert];
    const int route_m = local_m_tile_base + local_row;
    const int global_n = n_tile_base + local_row;
    const int global_k = k_base + local_k;

    const int64_t input_idx =
        (static_cast<int64_t>(expert_offset) + route_m) * InDim + global_k;
    const int64_t weight_idx =
        (static_cast<int64_t>(expert) * OutDim + global_n) * InDim + global_k;

    const uint32_t a_addr = shared_addr(a_shared + local_row * kMmaK + local_k);
    const uint32_t b_addr = shared_addr(b_shared + local_row * kMmaK + local_k);
    cp_async_cg_16(a_addr, input + input_idx);
    cp_async_cg_16(b_addr, weight + weight_idx);
}

__device__ __forceinline__ void compute_grouped_mma_stage(
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

template <int InDim, int OutDim>
__device__ __forceinline__ void load_grouped_tile_scalar_m64(
    const c10::Half* __restrict__ input,
    const c10::Half* __restrict__ weight,
    const int32_t* __restrict__ counts,
    const int32_t* __restrict__ offsets,
    c10::Half* __restrict__ a_shared,
    c10::Half* __restrict__ b_shared,
    int expert,
    int local_m_tile_base,
    int n_tile_base,
    int k_base,
    int tid
) {
    const int expert_count = counts[expert];
    const int expert_offset = offsets[expert];

    for (int idx = tid; idx < kBlockM64 * kMmaK; idx += kThreads) {
        const int local_m = idx / kMmaK;
        const int local_k = idx - local_m * kMmaK;
        const int route_m = local_m_tile_base + local_m;
        const int global_k = k_base + local_k;

        c10::Half val = static_cast<c10::Half>(0.0f);
        if (route_m < expert_count) {
            const int64_t input_idx =
                (static_cast<int64_t>(expert_offset) + route_m) * InDim + global_k;
            val = input[input_idx];
        }
        a_shared[idx] = val;
    }

    for (int idx = tid; idx < kBlockN * kMmaK; idx += kThreads) {
        const int local_n = idx / kMmaK;
        const int local_k = idx - local_n * kMmaK;
        const int global_n = n_tile_base + local_n;
        const int global_k = k_base + local_k;

        const int64_t weight_idx =
            (static_cast<int64_t>(expert) * OutDim + global_n) * InDim + global_k;
        b_shared[idx] = weight[weight_idx];
    }
}

template <int InDim, int OutDim>
__device__ __forceinline__ void load_grouped_tile_cp_async_m64(
    const c10::Half* __restrict__ input,
    const c10::Half* __restrict__ weight,
    const int32_t* __restrict__ offsets,
    c10::Half* __restrict__ a_shared,
    c10::Half* __restrict__ b_shared,
    int expert,
    int local_m_tile_base,
    int n_tile_base,
    int k_base,
    int tid
) {
    constexpr int kVecHalf = 8;
    constexpr int kVecsPerM64ATile = kBlockM64 * kMmaK / kVecHalf;
    constexpr int kVecsPerBTile = kBlockN * kMmaK / kVecHalf;
    const int expert_offset = offsets[expert];

    for (int vec = tid; vec < kVecsPerM64ATile; vec += kThreads) {
        const int local_m = vec >> 1;
        const int local_k = (vec & 1) * kVecHalf;
        const int route_m = local_m_tile_base + local_m;
        const int global_k = k_base + local_k;
        const int64_t input_idx =
            (static_cast<int64_t>(expert_offset) + route_m) * InDim + global_k;
        const uint32_t a_addr = shared_addr(a_shared + local_m * kMmaK + local_k);
        cp_async_cg_16(a_addr, input + input_idx);
    }

    for (int vec = tid; vec < kVecsPerBTile; vec += kThreads) {
        const int local_n = vec >> 1;
        const int local_k = (vec & 1) * kVecHalf;
        const int global_n = n_tile_base + local_n;
        const int global_k = k_base + local_k;
        const int64_t weight_idx =
            (static_cast<int64_t>(expert) * OutDim + global_n) * InDim + global_k;
        const uint32_t b_addr = shared_addr(b_shared + local_n * kMmaK + local_k);
        cp_async_cg_16(b_addr, weight + weight_idx);
    }
}

__device__ __forceinline__ void compute_grouped_mma_stage_m64(
    const c10::Half* __restrict__ a_shared,
    const c10::Half* __restrict__ b_shared,
    int warp_m,
    int warp_n,
    int lane,
    uint32_t (&acc)[kWarpTileM64][kWarpTileN][4]
) {
    uint32_t a_frag[kWarpTileM64][4];
    uint32_t b_frag[kWarpTileN][2];

#pragma unroll
    for (int i = 0; i < kWarpTileM64; ++i) {
        const int warp_smem_a_m = warp_m * (kMmaM * kWarpTileM64) + i * kMmaM;
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
    for (int i = 0; i < kWarpTileM64; ++i) {
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

template <int InDim, int OutDim, bool ApplyRelu>
__global__ __launch_bounds__(kThreads) void smoe_grouped_linear_mma_tn_m64_kernel(
    const c10::Half* __restrict__ input,
    const c10::Half* __restrict__ weight,
    const c10::Half* __restrict__ bias,
    const int32_t* __restrict__ counts,
    const int32_t* __restrict__ offsets,
    c10::Half* __restrict__ output
) {
    static_assert(InDim % kMmaK == 0, "InDim must be divisible by 16");
    static_assert(OutDim % kBlockN == 0, "OutDim must be divisible by kBlockN");

    __shared__ __align__(16) c10::Half a_shared[kPipelineStages * kBlockM64 * kMmaK];
    __shared__ __align__(16) c10::Half b_shared[kPipelineStages * kBlockN * kMmaK];

    const int tid = threadIdx.x;
    const int warp_id = tid / kWarpSize;
    const int lane = tid & (kWarpSize - 1);
    const int pool_m_tile_base = blockIdx.y * kBlockM64;
    const int n_tile_base = blockIdx.x * kBlockN;

    const int total_padded_routes = offsets[kNumExperts];
    if (pool_m_tile_base >= total_padded_routes) {
        return;
    }

    const int expert = find_expert_for_pool_m(offsets, pool_m_tile_base);
    if (expert >= kNumExperts) {
        return;
    }

    const int expert_offset = offsets[expert];
    const int local_m_tile_base = pool_m_tile_base - expert_offset;
    const int expert_count = counts[expert];
    if (local_m_tile_base >= expert_count) {
        return;
    }

    const int warp_m = warp_id & 1;
    const int warp_n = warp_id >> 1;

    uint32_t acc[kWarpTileM64][kWarpTileN][4];
#pragma unroll
    for (int i = 0; i < kWarpTileM64; ++i) {
#pragma unroll
        for (int j = 0; j < kWarpTileN; ++j) {
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                acc[i][j][r] = 0;
            }
        }
    }

    const bool full_m_tile = (local_m_tile_base + kBlockM64) <= expert_count;
    constexpr int kNumKTiles = InDim / kMmaK;
    constexpr int kStageStrideA = kBlockM64 * kMmaK;
    constexpr int kStageStrideB = kBlockN * kMmaK;

    if (full_m_tile) {
        load_grouped_tile_cp_async_m64<InDim, OutDim>(
            input,
            weight,
            offsets,
            a_shared,
            b_shared,
            expert,
            local_m_tile_base,
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

            load_grouped_tile_cp_async_m64<InDim, OutDim>(
                input,
                weight,
                offsets,
                a_shared + load_stage * kStageStrideA,
                b_shared + load_stage * kStageStrideB,
                expert,
                local_m_tile_base,
                n_tile_base,
                k_tile * kMmaK,
                tid
            );
            cp_async_commit_group();

            compute_grouped_mma_stage_m64(
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
        compute_grouped_mma_stage_m64(
            a_shared + last_stage * kStageStrideA,
            b_shared + last_stage * kStageStrideB,
            warp_m,
            warp_n,
            lane,
            acc
        );
    } else {
        for (int k_base = 0; k_base < InDim; k_base += kMmaK) {
            load_grouped_tile_scalar_m64<InDim, OutDim>(
                input,
                weight,
                counts,
                offsets,
                a_shared,
                b_shared,
                expert,
                local_m_tile_base,
                n_tile_base,
                k_base,
                tid
            );
            __syncthreads();

            compute_grouped_mma_stage_m64(
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

    const int row_base = local_m_tile_base + warp_m * (kMmaM * kWarpTileM64);
    const int col_base = n_tile_base + warp_n * (kMmaN * kWarpTileN);
    const int frag_row0 = lane / 4;
    const int frag_row1 = frag_row0 + 8;
    const int frag_col_pair = (lane & 3) * 2;
    const int lane_group_base = lane & ~3;

#pragma unroll
    for (int i = 0; i < kWarpTileM64; ++i) {
#pragma unroll
        for (int j = 0; j < kWarpTileN; ++j) {
            const int rows[2] = {
                row_base + i * kMmaM + frag_row0,
                row_base + i * kMmaM + frag_row1,
            };
            const int col = col_base + j * kMmaN + frag_col_pair;

#pragma unroll
            for (int row_slot = 0; row_slot < 2; ++row_slot) {
                const int route_m = rows[row_slot];
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

                if ((lane & 3) == 0 && route_m < expert_count) {
                    const int64_t output_idx =
                        (static_cast<int64_t>(expert_offset) + route_m) * OutDim
                        + col_base + j * kMmaN;
                    *reinterpret_cast<uint4*>(output + output_idx) = vec;
                }
            }
        }
    }
}

template <int InDim, int OutDim, typename ScaleT>
__device__ __forceinline__ uint32_t dequant_w4_pair_bits_from_global(
    const int16_t* __restrict__ weight_pack,
    const ScaleT* __restrict__ weight_scale,
    const ScaleT* __restrict__ weight_zero,
    int expert,
    int global_n,
    int global_k,
    int group_size
);

template <int InDim, int OutDim, typename ScaleT>
__device__ __forceinline__ void dequant_w4_two_pair_bits_from_global(
    const int16_t* __restrict__ weight_pack,
    const ScaleT* __restrict__ weight_scale,
    const ScaleT* __restrict__ weight_zero,
    int expert,
    int global_n,
    int global_k_pair,
    int group_size,
    uint32_t& out0,
    uint32_t& out1
);

template <int InDim, int OutDim, typename ScaleT>
__device__ __forceinline__ void compute_grouped_mma_stage_w4a16_frag(
    const c10::Half* __restrict__ a_shared,
    const int16_t* __restrict__ weight_pack,
    const ScaleT* __restrict__ weight_scale,
    const ScaleT* __restrict__ weight_zero,
    int expert,
    int n_tile_base,
    int k_base,
    int group_size,
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

    // ldmatrix output mapping: lane groups of 4 cover one N row, lanes within the group cover K pairs.
    const int lane_n = lane >> 2;
    const int lane_k_pair = (lane & 3) * 2;

#pragma unroll
    for (int j = 0; j < kWarpTileN; ++j) {
        const int warp_b_n = warp_n * (kMmaN * kWarpTileN) + j * kMmaN;
        const int global_n = n_tile_base + warp_b_n + lane_n;
        dequant_w4_two_pair_bits_from_global<InDim, OutDim, ScaleT>(
            weight_pack,
            weight_scale,
            weight_zero,
            expert,
            global_n,
            k_base + lane_k_pair,
            group_size,
            b_frag[j][0],
            b_frag[j][1]
        );
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

template <int InDim, int OutDim, bool ApplyRelu>
__global__ __launch_bounds__(kThreads) void smoe_grouped_linear_mma_tn_kernel(
    const c10::Half* __restrict__ input,
    const c10::Half* __restrict__ weight,
    const c10::Half* __restrict__ bias,
    const int32_t* __restrict__ counts,
    const int32_t* __restrict__ offsets,
    c10::Half* __restrict__ output
) {
    static_assert(InDim % kMmaK == 0, "InDim must be divisible by 16");
    static_assert(OutDim % kBlockN == 0, "OutDim must be divisible by kBlockN");

    __shared__ __align__(16) c10::Half a_shared[kPipelineStages * kBlockM * kMmaK];
    __shared__ __align__(16) c10::Half b_shared[kPipelineStages * kBlockN * kMmaK];

    const int tid = threadIdx.x;
    const int warp_id = tid / kWarpSize;
    const int lane = tid & (kWarpSize - 1);
    const int pool_m_tile_base = blockIdx.y * kBlockM;
    const int n_tile_base = blockIdx.x * kBlockN;

    const int total_padded_routes = offsets[kNumExperts];
    if (pool_m_tile_base >= total_padded_routes) {
        return;
    }

    const int expert = find_expert_for_pool_m(offsets, pool_m_tile_base);
    if (expert >= kNumExperts) {
        return;
    }

    const int expert_offset = offsets[expert];
    const int local_m_tile_base = pool_m_tile_base - expert_offset;
    const int expert_count = counts[expert];
    if (local_m_tile_base >= expert_count) {
        return;
    }

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

    const bool full_m_tile = (local_m_tile_base + kBlockM) <= expert_count;
    constexpr int kNumKTiles = InDim / kMmaK;
    constexpr int kStageStrideA = kBlockM * kMmaK;
    constexpr int kStageStrideB = kBlockN * kMmaK;

    if (full_m_tile) {
        load_grouped_tile_cp_async<InDim, OutDim>(
            input,
            weight,
            offsets,
            a_shared,
            b_shared,
            expert,
            local_m_tile_base,
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

            load_grouped_tile_cp_async<InDim, OutDim>(
                input,
                weight,
                offsets,
                a_shared + load_stage * kStageStrideA,
                b_shared + load_stage * kStageStrideB,
                expert,
                local_m_tile_base,
                n_tile_base,
                k_tile * kMmaK,
                tid
            );
            cp_async_commit_group();

            compute_grouped_mma_stage(
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
        compute_grouped_mma_stage(
            a_shared + last_stage * kStageStrideA,
            b_shared + last_stage * kStageStrideB,
            warp_m,
            warp_n,
            lane,
            acc
        );
    } else {
        for (int k_base = 0; k_base < InDim; k_base += kMmaK) {
            load_grouped_tile_scalar<InDim, OutDim>(
                input,
                weight,
                counts,
                offsets,
                a_shared,
                b_shared,
                expert,
                local_m_tile_base,
                n_tile_base,
                k_base,
                tid
            );
            __syncthreads();

            compute_grouped_mma_stage(
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

    const int row_base = local_m_tile_base + warp_m * (kMmaM * kWarpTileM);
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
            const int col = col_base + j * kMmaN + frag_col_pair;

#pragma unroll
            for (int row_slot = 0; row_slot < 2; ++row_slot) {
                const int route_m = rows[row_slot];
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

                if ((lane & 3) == 0 && route_m < expert_count) {
                    const int64_t output_idx =
                        (static_cast<int64_t>(expert_offset) + route_m) * OutDim
                        + col_base + j * kMmaN;
                    *reinterpret_cast<uint4*>(output + output_idx) = vec;
                }
            }
        }
    }
}

template <int InDim, int OutDim, bool ApplyRelu>
__global__ __launch_bounds__(kThreads) void smoe_grouped_linear_mma_tn_pack_w4a4_kernel(
    const c10::Half* __restrict__ input,
    const c10::Half* __restrict__ weight,
    const c10::Half* __restrict__ bias,
    const int32_t* __restrict__ counts,
    const int32_t* __restrict__ offsets,
    int32_t* __restrict__ output_pack,
    float inv_act_scale
) {
    static_assert(InDim % kMmaK == 0, "InDim must be divisible by 16");
    static_assert(OutDim % kBlockN == 0, "OutDim must be divisible by kBlockN");
    static_assert(OutDim % 8 == 0, "OutDim must be divisible by 8");

    __shared__ __align__(16) c10::Half a_shared[kPipelineStages * kBlockM * kMmaK];
    __shared__ __align__(16) c10::Half b_shared[kPipelineStages * kBlockN * kMmaK];

    const int tid = threadIdx.x;
    const int warp_id = tid / kWarpSize;
    const int lane = tid & (kWarpSize - 1);
    const int pool_m_tile_base = blockIdx.y * kBlockM;
    const int n_tile_base = blockIdx.x * kBlockN;

    const int total_padded_routes = offsets[kNumExperts];
    if (pool_m_tile_base >= total_padded_routes) {
        return;
    }

    const int expert = find_expert_for_pool_m(offsets, pool_m_tile_base);
    if (expert >= kNumExperts) {
        return;
    }

    const int expert_offset = offsets[expert];
    const int local_m_tile_base = pool_m_tile_base - expert_offset;
    const int expert_count = counts[expert];
    if (local_m_tile_base >= expert_count) {
        return;
    }

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

    const bool full_m_tile = (local_m_tile_base + kBlockM) <= expert_count;
    constexpr int kNumKTiles = InDim / kMmaK;
    constexpr int kStageStrideA = kBlockM * kMmaK;
    constexpr int kStageStrideB = kBlockN * kMmaK;

    if (full_m_tile) {
        load_grouped_tile_cp_async<InDim, OutDim>(
            input,
            weight,
            offsets,
            a_shared,
            b_shared,
            expert,
            local_m_tile_base,
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

            load_grouped_tile_cp_async<InDim, OutDim>(
                input,
                weight,
                offsets,
                a_shared + load_stage * kStageStrideA,
                b_shared + load_stage * kStageStrideB,
                expert,
                local_m_tile_base,
                n_tile_base,
                k_tile * kMmaK,
                tid
            );
            cp_async_commit_group();

            compute_grouped_mma_stage(
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
        compute_grouped_mma_stage(
            a_shared + last_stage * kStageStrideA,
            b_shared + last_stage * kStageStrideB,
            warp_m,
            warp_n,
            lane,
            acc
        );
    } else {
        for (int k_base = 0; k_base < InDim; k_base += kMmaK) {
            load_grouped_tile_scalar<InDim, OutDim>(
                input,
                weight,
                counts,
                offsets,
                a_shared,
                b_shared,
                expert,
                local_m_tile_base,
                n_tile_base,
                k_base,
                tid
            );
            __syncthreads();

            compute_grouped_mma_stage(
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

    const int row_base = local_m_tile_base + warp_m * (kMmaM * kWarpTileM);
    const int col_base = n_tile_base + warp_n * (kMmaN * kWarpTileN);
    const int frag_row0 = lane / 4;
    const int frag_row1 = frag_row0 + 8;
    const int frag_col_pair = (lane & 3) * 2;
    const int lane_group_base = lane & ~3;
    constexpr int kPacksPerRow = OutDim / 8;

#pragma unroll
    for (int i = 0; i < kWarpTileM; ++i) {
#pragma unroll
        for (int j = 0; j < kWarpTileN; ++j) {
            const int rows[2] = {
                row_base + i * kMmaM + frag_row0,
                row_base + i * kMmaM + frag_row1,
            };
            const int col = col_base + j * kMmaN + frag_col_pair;

#pragma unroll
            for (int row_slot = 0; row_slot < 2; ++row_slot) {
                const int route_m = rows[row_slot];
                float value0 = reg_as_float(acc[i][j][row_slot * 2 + 0])
                    + static_cast<float>(bias[expert * OutDim + col + 0]);
                float value1 = reg_as_float(acc[i][j][row_slot * 2 + 1])
                    + static_cast<float>(bias[expert * OutDim + col + 1]);
                if constexpr (ApplyRelu) {
                    value0 = fmaxf(value0, 0.0f);
                    value1 = fmaxf(value1, 0.0f);
                }

                const uint32_t pair_bits = pack_s4_pair_bits_from_half_rounded(
                    value0,
                    value1,
                    frag_col_pair,
                    inv_act_scale
                );
                const uint32_t word =
                    __shfl_sync(0xffffffff, pair_bits, lane_group_base + 0)
                    | __shfl_sync(0xffffffff, pair_bits, lane_group_base + 1)
                    | __shfl_sync(0xffffffff, pair_bits, lane_group_base + 2)
                    | __shfl_sync(0xffffffff, pair_bits, lane_group_base + 3);

                if ((lane & 3) == 0 && route_m < expert_count) {
                    const int64_t output_idx =
                        (static_cast<int64_t>(expert_offset) + route_m) * kPacksPerRow
                        + (col_base + j * kMmaN) / 8;
                    output_pack[output_idx] = static_cast<int32_t>(word);
                }
            }
        }
    }
}

template <int InDim, int OutDim, bool ApplyRelu>
__global__ __launch_bounds__(kThreads) void smoe_grouped_linear_mma_tn_pack_w4a4_m64_kernel(
    const c10::Half* __restrict__ input,
    const c10::Half* __restrict__ weight,
    const c10::Half* __restrict__ bias,
    const int32_t* __restrict__ counts,
    const int32_t* __restrict__ offsets,
    int32_t* __restrict__ output_pack,
    float inv_act_scale
) {
    static_assert(InDim % kMmaK == 0, "InDim must be divisible by 16");
    static_assert(OutDim % kBlockN == 0, "OutDim must be divisible by kBlockN");
    static_assert(OutDim % 8 == 0, "OutDim must be divisible by 8");

    __shared__ __align__(16) c10::Half a_shared[kPipelineStages * kBlockM64 * kMmaK];
    __shared__ __align__(16) c10::Half b_shared[kPipelineStages * kBlockN * kMmaK];

    const int tid = threadIdx.x;
    const int warp_id = tid / kWarpSize;
    const int lane = tid & (kWarpSize - 1);
    const int pool_m_tile_base = blockIdx.y * kBlockM64;
    const int n_tile_base = blockIdx.x * kBlockN;

    const int total_padded_routes = offsets[kNumExperts];
    if (pool_m_tile_base >= total_padded_routes) {
        return;
    }

    const int expert = find_expert_for_pool_m(offsets, pool_m_tile_base);
    if (expert >= kNumExperts) {
        return;
    }

    const int expert_offset = offsets[expert];
    const int local_m_tile_base = pool_m_tile_base - expert_offset;
    const int expert_count = counts[expert];
    if (local_m_tile_base >= expert_count) {
        return;
    }

    const int warp_m = warp_id & 1;
    const int warp_n = warp_id >> 1;

    uint32_t acc[kWarpTileM64][kWarpTileN][4];
#pragma unroll
    for (int i = 0; i < kWarpTileM64; ++i) {
#pragma unroll
        for (int j = 0; j < kWarpTileN; ++j) {
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                acc[i][j][r] = 0;
            }
        }
    }

    const bool full_m_tile = (local_m_tile_base + kBlockM64) <= expert_count;
    constexpr int kNumKTiles = InDim / kMmaK;
    constexpr int kStageStrideA = kBlockM64 * kMmaK;
    constexpr int kStageStrideB = kBlockN * kMmaK;

    if (full_m_tile) {
        load_grouped_tile_cp_async_m64<InDim, OutDim>(
            input, weight, offsets, a_shared, b_shared,
            expert, local_m_tile_base, n_tile_base, 0, tid);
        cp_async_commit_group();
        cp_async_wait_all();
        __syncthreads();

        for (int k_tile = 1; k_tile < kNumKTiles; ++k_tile) {
            const int compute_stage = (k_tile + 1) & 1;
            const int load_stage = k_tile & 1;

            load_grouped_tile_cp_async_m64<InDim, OutDim>(
                input, weight, offsets,
                a_shared + load_stage * kStageStrideA,
                b_shared + load_stage * kStageStrideB,
                expert, local_m_tile_base, n_tile_base, k_tile * kMmaK, tid);
            cp_async_commit_group();

            compute_grouped_mma_stage_m64(
                a_shared + compute_stage * kStageStrideA,
                b_shared + compute_stage * kStageStrideB,
                warp_m, warp_n, lane, acc);

            cp_async_wait_all();
            __syncthreads();
        }

        constexpr int last_stage = (kNumKTiles - 1) & 1;
        compute_grouped_mma_stage_m64(
            a_shared + last_stage * kStageStrideA,
            b_shared + last_stage * kStageStrideB,
            warp_m, warp_n, lane, acc);
    } else {
        for (int k_base = 0; k_base < InDim; k_base += kMmaK) {
            load_grouped_tile_scalar_m64<InDim, OutDim>(
                input, weight, counts, offsets, a_shared, b_shared,
                expert, local_m_tile_base, n_tile_base, k_base, tid);
            __syncthreads();

            compute_grouped_mma_stage_m64(
                a_shared, b_shared, warp_m, warp_n, lane, acc);
            __syncthreads();
        }
    }

    const int row_base = local_m_tile_base + warp_m * (kMmaM * kWarpTileM64);
    const int col_base = n_tile_base + warp_n * (kMmaN * kWarpTileN);
    const int frag_row0 = lane / 4;
    const int frag_row1 = frag_row0 + 8;
    const int frag_col_pair = (lane & 3) * 2;
    const int lane_group_base = lane & ~3;
    constexpr int kPacksPerRow = OutDim / 8;

#pragma unroll
    for (int i = 0; i < kWarpTileM64; ++i) {
#pragma unroll
        for (int j = 0; j < kWarpTileN; ++j) {
            const int rows[2] = {
                row_base + i * kMmaM + frag_row0,
                row_base + i * kMmaM + frag_row1,
            };
            const int col = col_base + j * kMmaN + frag_col_pair;

#pragma unroll
            for (int row_slot = 0; row_slot < 2; ++row_slot) {
                const int route_m = rows[row_slot];
                float value0 = reg_as_float(acc[i][j][row_slot * 2 + 0])
                    + static_cast<float>(bias[expert * OutDim + col + 0]);
                float value1 = reg_as_float(acc[i][j][row_slot * 2 + 1])
                    + static_cast<float>(bias[expert * OutDim + col + 1]);
                if constexpr (ApplyRelu) {
                    value0 = fmaxf(value0, 0.0f);
                    value1 = fmaxf(value1, 0.0f);
                }

                const uint32_t pair_bits = pack_s4_pair_bits_from_half_rounded(
                    value0, value1, frag_col_pair, inv_act_scale);
                const uint32_t word =
                    __shfl_sync(0xffffffff, pair_bits, lane_group_base + 0)
                    | __shfl_sync(0xffffffff, pair_bits, lane_group_base + 1)
                    | __shfl_sync(0xffffffff, pair_bits, lane_group_base + 2)
                    | __shfl_sync(0xffffffff, pair_bits, lane_group_base + 3);

                if ((lane & 3) == 0 && route_m < expert_count) {
                    const int64_t output_idx =
                        (static_cast<int64_t>(expert_offset) + route_m) * kPacksPerRow
                        + (col_base + j * kMmaN) / 8;
                    output_pack[output_idx] = static_cast<int32_t>(word);
                }
            }
        }
    }
}

template <typename ScaleT>
__device__ __forceinline__ float scalar_to_float(ScaleT value) {
    return static_cast<float>(value);
}

__device__ __forceinline__ uint32_t dequant_w4_pair_bits_from_packed(
    uint16_t packed,
    int shift,
    __half2 scale_h2,
    __half2 zero_h2
) {
    const int q0 = static_cast<int>((packed >> shift) & 0x000f);
    const int q1 = static_cast<int>((packed >> (shift + 4)) & 0x000f);
    const __half2 q_h2 = __halves2half2(
        __float2half_rn(static_cast<float>(q0)),
        __float2half_rn(static_cast<float>(q1))
    );
    return half2_as_uint(__hfma2(q_h2, scale_h2, zero_h2));
}

__device__ __forceinline__ uint32_t dequant_w4_contiguous_pair_bits_lop3(
    uint16_t packed,
    bool high_pair,
    __half2 scale_h2,
    __half2 zero_h2
) {
    const int pair_bits = high_pair
        ? (static_cast<int>((packed >> 8) & 0x000f)
            | static_cast<int>((packed & 0xf000) << 4))
        : (static_cast<int>(packed & 0x000f)
            | static_cast<int>((packed & 0x00f0) << 12));

    constexpr int kLoMask = 0x000f000f;
    constexpr int kExponent = 0x64006400;
    constexpr int kLop3Or = (0xf0 & 0xcc) | 0xaa;
    const int q_plus_base = lop3<kLop3Or>(pair_bits, kLoMask, kExponent);
    const int exponent = kExponent;
    const __half2 q_h2 = __hsub2(
        *reinterpret_cast<const __half2*>(&q_plus_base),
        *reinterpret_cast<const __half2*>(&exponent)
    );
    return half2_as_uint(__hfma2(q_h2, scale_h2, zero_h2));
}

template <int InDim>
__device__ __forceinline__ void load_grouped_a_tile_w4a16(
    const c10::Half* __restrict__ input,
    const int32_t* __restrict__ counts,
    const int32_t* __restrict__ offsets,
    c10::Half* __restrict__ a_shared,
    int expert,
    int local_m_tile_base,
    int k_base,
    int tid
) {
    const int expert_count = counts[expert];
    const int expert_offset = offsets[expert];
    constexpr int kAVecHalf = 8;
    constexpr int kAVecsPerRow = kMmaK / kAVecHalf;

    for (int idx = tid; idx < kBlockM * kAVecsPerRow; idx += kThreads) {
        const int local_m = idx / kAVecsPerRow;
        const int local_vec = idx - local_m * kAVecsPerRow;
        const int local_k = local_vec * kAVecHalf;
        const int route_m = local_m_tile_base + local_m;
        const int global_k = k_base + local_k;

        uint4 val = make_uint4(0, 0, 0, 0);
        if (route_m < expert_count) {
            const int64_t input_idx =
                (static_cast<int64_t>(expert_offset) + route_m) * InDim + global_k;
            val = *reinterpret_cast<const uint4*>(input + input_idx);
        }
        *reinterpret_cast<uint4*>(a_shared + local_m * kMmaK + local_k) = val;
    }
}

template <int InDim, int OutDim, typename ScaleT>
__device__ __forceinline__ uint32_t dequant_w4_pair_bits_from_global(
    const int16_t* __restrict__ weight_pack,
    const ScaleT* __restrict__ weight_scale,
    const ScaleT* __restrict__ weight_zero,
    int expert,
    int global_n,
    int global_k,
    int group_size
) {
    constexpr int kPackedInDim = InDim / 4;
    const int groups_per_row = InDim / group_size;
    const int packed_k = global_k >> 2;
    const int shift = (global_k & 3) << 2;
    const int group_idx = global_k / group_size;

    const int64_t row = static_cast<int64_t>(expert) * OutDim + global_n;
    const int64_t packed_idx = row * kPackedInDim + packed_k;
    const int64_t scale_idx = row * groups_per_row + group_idx;

    const uint16_t packed = static_cast<uint16_t>(weight_pack[packed_idx]);
    const __half2 scale_h2 = __float2half2_rn(scalar_to_float(weight_scale[scale_idx]));
    const __half2 zero_h2 = __float2half2_rn(scalar_to_float(weight_zero[scale_idx]));
    return dequant_w4_pair_bits_from_packed(packed, shift, scale_h2, zero_h2);
}

template <int InDim, int OutDim, typename ScaleT>
__device__ __forceinline__ void dequant_w4_two_pair_bits_from_global(
    const int16_t* __restrict__ weight_pack,
    const ScaleT* __restrict__ weight_scale,
    const ScaleT* __restrict__ weight_zero,
    int expert,
    int global_n,
    int global_k_pair,
    int group_size,
    uint32_t& out0,
    uint32_t& out1
) {
    constexpr int kPackedInDim = InDim / 4;
    const int groups_per_row = InDim / group_size;

    const int64_t row = static_cast<int64_t>(expert) * OutDim + global_n;
    const int group_idx = global_k_pair / group_size;
    const int64_t scale_idx = row * groups_per_row + group_idx;
    const __half2 scale_h2 = __float2half2_rn(scalar_to_float(weight_scale[scale_idx]));
    const __half2 zero_h2 = __float2half2_rn(scalar_to_float(weight_zero[scale_idx]));

    const int packed_k0 = global_k_pair >> 2;
    const int shift0 = (global_k_pair & 3) << 2;
    const int global_k1 = global_k_pair + 8;
    const int packed_k1 = global_k1 >> 2;
    const int shift1 = (global_k1 & 3) << 2;

    const uint16_t packed0 = static_cast<uint16_t>(weight_pack[row * kPackedInDim + packed_k0]);
    const uint16_t packed1 = static_cast<uint16_t>(weight_pack[row * kPackedInDim + packed_k1]);
    out0 = dequant_w4_pair_bits_from_packed(packed0, shift0, scale_h2, zero_h2);
    out1 = dequant_w4_pair_bits_from_packed(packed1, shift1, scale_h2, zero_h2);
}

template <int InDim, int OutDim, typename ScaleT, bool UseLop3Dequant>
__device__ __forceinline__ void load_grouped_tile_w4a16_scalar(
    const c10::Half* __restrict__ input,
    const int16_t* __restrict__ weight_pack,
    const ScaleT* __restrict__ weight_scale,
    const ScaleT* __restrict__ weight_zero,
    const int32_t* __restrict__ counts,
    const int32_t* __restrict__ offsets,
    c10::Half* __restrict__ a_shared,
    c10::Half* __restrict__ b_shared,
    int expert,
    int local_m_tile_base,
    int n_tile_base,
    int k_base,
    int group_size,
    int tid
) {
    const int expert_count = counts[expert];
    const int expert_offset = offsets[expert];
    constexpr int kPackedInDim = InDim / 4;
    constexpr int kAVecHalf = 8;
    constexpr int kAVecsPerRow = kMmaK / kAVecHalf;

    for (int idx = tid; idx < kBlockM * kAVecsPerRow; idx += kThreads) {
        const int local_m = idx / kAVecsPerRow;
        const int local_vec = idx - local_m * kAVecsPerRow;
        const int local_k = local_vec * kAVecHalf;
        const int route_m = local_m_tile_base + local_m;
        const int global_k = k_base + local_k;

        uint4 val = make_uint4(0, 0, 0, 0);
        if (route_m < expert_count) {
            const int64_t input_idx =
                (static_cast<int64_t>(expert_offset) + route_m) * InDim + global_k;
            val = *reinterpret_cast<const uint4*>(input + input_idx);
        }
        *reinterpret_cast<uint4*>(a_shared + local_m * kMmaK + local_k) = val;
    }

    constexpr int kPackedMmaK = kMmaK / 4;
    const int groups_per_row = InDim / group_size;

    for (int idx = tid; idx < kBlockN * kPackedMmaK; idx += kThreads) {
        const int local_n = idx / kPackedMmaK;
        const int local_pack_k = idx - local_n * kPackedMmaK;
        const int global_n = n_tile_base + local_n;
        const int global_k = k_base + local_pack_k * 4;
        const int packed_k = global_k >> 2;
        const int group_idx = global_k / group_size;

        const int64_t row = static_cast<int64_t>(expert) * OutDim + global_n;
        const int64_t packed_idx = row * kPackedInDim + packed_k;
        const int64_t scale_idx = row * groups_per_row + group_idx;

        const uint16_t packed = static_cast<uint16_t>(weight_pack[packed_idx]);
        const float scale = scalar_to_float(weight_scale[scale_idx]);
        const float zero = scalar_to_float(weight_zero[scale_idx]);
        const int shared_base = local_n * kMmaK + local_pack_k * 4;
        if constexpr (UseLop3Dequant) {
            const __half2 scale_h2 = __float2half2_rn(scale);
            const __half2 zero_h2 = __float2half2_rn(zero);
            *reinterpret_cast<uint32_t*>(b_shared + shared_base) =
                dequant_w4_contiguous_pair_bits_lop3(packed, false, scale_h2, zero_h2);
            *reinterpret_cast<uint32_t*>(b_shared + shared_base + 2) =
                dequant_w4_contiguous_pair_bits_lop3(packed, true, scale_h2, zero_h2);
        } else {
#pragma unroll
            for (int q_idx = 0; q_idx < 4; ++q_idx) {
                const int q = static_cast<int>((packed >> (q_idx * 4)) & 0x000f);
                b_shared[shared_base + q_idx] =
                    static_cast<c10::Half>(static_cast<float>(q) * scale + zero);
            }
        }
    }
}

template <int InDim, int OutDim, typename ScaleT>
__global__ __launch_bounds__(kThreads) void debug_w4a16_bfrag_layout_kernel(
    const int16_t* __restrict__ weight_pack,
    const ScaleT* __restrict__ weight_scale,
    const ScaleT* __restrict__ weight_zero,
    int32_t* __restrict__ dump,
    int group_size,
    int expert,
    int n_tile_base,
    int k_base
) {
    static_assert(InDim % kMmaK == 0, "InDim must be divisible by 16");
    static_assert(InDim % 4 == 0, "InDim must be divisible by 4");
    static_assert(OutDim % kBlockN == 0, "OutDim must be divisible by kBlockN");

    __shared__ __align__(16) c10::Half b_shared[kBlockN * kMmaK];

    const int tid = threadIdx.x;
    const int warp_id = tid / kWarpSize;
    const int lane = tid & (kWarpSize - 1);

    constexpr int kPackedInDim = InDim / 4;
    const int groups_per_row = InDim / group_size;

    for (int idx = tid; idx < kBlockN * kMmaK; idx += kThreads) {
        const int local_n = idx / kMmaK;
        const int local_k = idx - local_n * kMmaK;
        const int global_n = n_tile_base + local_n;
        const int global_k = k_base + local_k;
        const int packed_k = global_k >> 2;
        const int q_shift = (global_k & 3) << 2;
        const int group_idx = global_k / group_size;

        const int64_t row = static_cast<int64_t>(expert) * OutDim + global_n;
        const int64_t packed_idx = row * kPackedInDim + packed_k;
        const int64_t scale_idx = row * groups_per_row + group_idx;

        const uint16_t packed = static_cast<uint16_t>(weight_pack[packed_idx]);
        const int q = static_cast<int>((packed >> q_shift) & 0x000f);
        const float scale = scalar_to_float(weight_scale[scale_idx]);
        const float zero = scalar_to_float(weight_zero[scale_idx]);
        b_shared[idx] = static_cast<c10::Half>(static_cast<float>(q) * scale + zero);
    }
    __syncthreads();

    const int warp_n = warp_id >> 1;
    // Match the ldmatrix output mapping used by mma, not the address-provider lane mapping.
    const int lane_n = lane >> 2;
    const int lane_k_pair = (lane & 3) * 2;

#pragma unroll
    for (int j = 0; j < kWarpTileN; ++j) {
        const int warp_smem_b_n = warp_n * (kMmaN * kWarpTileN) + j * kMmaN;
        const int lane_smem_b_n = warp_smem_b_n + (lane & 7);
        const int lane_smem_b_k = ((lane >> 3) & 1) * 8;
        const uint32_t b_addr =
            shared_addr(b_shared + lane_smem_b_n * kMmaK + lane_smem_b_k);

        uint32_t ref0;
        uint32_t ref1;
        ldmatrix_x2(ref0, ref1, b_addr);

        const int global_n = n_tile_base + warp_smem_b_n + lane_n;
        uint32_t direct0;
        uint32_t direct1;
        dequant_w4_two_pair_bits_from_global<InDim, OutDim, ScaleT>(
            weight_pack,
            weight_scale,
            weight_zero,
            expert,
            global_n,
            k_base + lane_k_pair,
            group_size,
            direct0,
            direct1
        );

        const int64_t base =
            (((static_cast<int64_t>(warp_id) * kWarpSize + lane) * kWarpTileN + j) * 4);
        uint32_t* __restrict__ dump_u32 = reinterpret_cast<uint32_t*>(dump);
        dump_u32[base + 0] = ref0;
        dump_u32[base + 1] = ref1;
        dump_u32[base + 2] = direct0;
        dump_u32[base + 3] = direct1;
    }
}

template <int InDim, int OutDim, bool ApplyRelu, typename ScaleT, bool UseLop3Dequant>
__global__ __launch_bounds__(kThreads) void smoe_grouped_linear_w4a16_mma_tn_kernel(
    const c10::Half* __restrict__ input,
    const int16_t* __restrict__ weight_pack,
    const ScaleT* __restrict__ weight_scale,
    const ScaleT* __restrict__ weight_zero,
    const c10::Half* __restrict__ bias,
    const int32_t* __restrict__ counts,
    const int32_t* __restrict__ offsets,
    c10::Half* __restrict__ output,
    int group_size
) {
    static_assert(InDim % kMmaK == 0, "InDim must be divisible by 16");
    static_assert(InDim % 4 == 0, "InDim must be divisible by 4");
    static_assert(OutDim % kBlockN == 0, "OutDim must be divisible by kBlockN");

    __shared__ __align__(16) c10::Half a_shared[kBlockM * kMmaK];
    __shared__ __align__(16) c10::Half b_shared[kBlockN * kMmaK];

    const int tid = threadIdx.x;
    const int warp_id = tid / kWarpSize;
    const int lane = tid & (kWarpSize - 1);
    const int pool_m_tile_base = blockIdx.y * kBlockM;
    const int n_tile_base = blockIdx.x * kBlockN;

    const int total_padded_routes = offsets[kNumExperts];
    if (pool_m_tile_base >= total_padded_routes) {
        return;
    }

    const int expert = find_expert_for_pool_m(offsets, pool_m_tile_base);
    if (expert >= kNumExperts) {
        return;
    }

    const int expert_offset = offsets[expert];
    const int local_m_tile_base = pool_m_tile_base - expert_offset;
    const int expert_count = counts[expert];
    if (local_m_tile_base >= expert_count) {
        return;
    }

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

    for (int k_base = 0; k_base < InDim; k_base += kMmaK) {
        load_grouped_tile_w4a16_scalar<InDim, OutDim, ScaleT, UseLop3Dequant>(
            input,
            weight_pack,
            weight_scale,
            weight_zero,
            counts,
            offsets,
            a_shared,
            b_shared,
            expert,
            local_m_tile_base,
            n_tile_base,
            k_base,
            group_size,
            tid
        );
        __syncthreads();

        compute_grouped_mma_stage(
            a_shared,
            b_shared,
            warp_m,
            warp_n,
            lane,
            acc
        );
        __syncthreads();
    }

    const int row_base = local_m_tile_base + warp_m * (kMmaM * kWarpTileM);
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
            const int col = col_base + j * kMmaN + frag_col_pair;

#pragma unroll
            for (int row_slot = 0; row_slot < 2; ++row_slot) {
                const int route_m = rows[row_slot];
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

                if ((lane & 3) == 0 && route_m < expert_count) {
                    const int64_t output_idx =
                        (static_cast<int64_t>(expert_offset) + route_m) * OutDim
                        + col_base + j * kMmaN;
                    *reinterpret_cast<uint4*>(output + output_idx) = vec;
                }
            }
        }
    }
}

template <int InDim, int OutDim, bool ApplyRelu, typename ScaleT>
__global__ __launch_bounds__(kThreads) void smoe_grouped_linear_w4a16_frag_mma_tn_kernel(
    const c10::Half* __restrict__ input,
    const int16_t* __restrict__ weight_pack,
    const ScaleT* __restrict__ weight_scale,
    const ScaleT* __restrict__ weight_zero,
    const c10::Half* __restrict__ bias,
    const int32_t* __restrict__ counts,
    const int32_t* __restrict__ offsets,
    c10::Half* __restrict__ output,
    int group_size
) {
    static_assert(InDim % kMmaK == 0, "InDim must be divisible by 16");
    static_assert(InDim % 4 == 0, "InDim must be divisible by 4");
    static_assert(OutDim % kBlockN == 0, "OutDim must be divisible by kBlockN");

    __shared__ __align__(16) c10::Half a_shared[kBlockM * kMmaK];

    const int tid = threadIdx.x;
    const int warp_id = tid / kWarpSize;
    const int lane = tid & (kWarpSize - 1);
    const int pool_m_tile_base = blockIdx.y * kBlockM;
    const int n_tile_base = blockIdx.x * kBlockN;

    const int total_padded_routes = offsets[kNumExperts];
    if (pool_m_tile_base >= total_padded_routes) {
        return;
    }

    const int expert = find_expert_for_pool_m(offsets, pool_m_tile_base);
    if (expert >= kNumExperts) {
        return;
    }

    const int expert_offset = offsets[expert];
    const int local_m_tile_base = pool_m_tile_base - expert_offset;
    const int expert_count = counts[expert];
    if (local_m_tile_base >= expert_count) {
        return;
    }

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

    for (int k_base = 0; k_base < InDim; k_base += kMmaK) {
        load_grouped_a_tile_w4a16<InDim>(
            input,
            counts,
            offsets,
            a_shared,
            expert,
            local_m_tile_base,
            k_base,
            tid
        );
        __syncthreads();

        compute_grouped_mma_stage_w4a16_frag<InDim, OutDim, ScaleT>(
            a_shared,
            weight_pack,
            weight_scale,
            weight_zero,
            expert,
            n_tile_base,
            k_base,
            group_size,
            warp_m,
            warp_n,
            lane,
            acc
        );
        __syncthreads();
    }

    const int row_base = local_m_tile_base + warp_m * (kMmaM * kWarpTileM);
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
            const int col = col_base + j * kMmaN + frag_col_pair;

#pragma unroll
            for (int row_slot = 0; row_slot < 2; ++row_slot) {
                const int route_m = rows[row_slot];
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

                if ((lane & 3) == 0 && route_m < expert_count) {
                    const int64_t output_idx =
                        (static_cast<int64_t>(expert_offset) + route_m) * OutDim
                        + col_base + j * kMmaN;
                    *reinterpret_cast<uint4*>(output + output_idx) = vec;
                }
            }
        }
    }
}

template <typename ScoreT>
__global__ void smoe_route_reduce_kernel(
    const c10::Half* __restrict__ y_route,
    const int32_t* __restrict__ route_pos,
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

    float value = 0.0f;
    const int32_t pos0 = route_pos[token * kTopK + 0];
    const int32_t pos1 = route_pos[token * kTopK + 1];
    if (pos0 >= 0) {
        value += static_cast<float>(topk_score[token * kTopK + 0])
            * static_cast<float>(y_route[static_cast<int64_t>(pos0) * kHiddenDim + dim]);
    }
    if (pos1 >= 0) {
        value += static_cast<float>(topk_score[token * kTopK + 1])
            * static_cast<float>(y_route[static_cast<int64_t>(pos1) * kHiddenDim + dim]);
    }
    out[idx] = static_cast<c10::Half>(value);
}

template <typename ScoreT>
__global__ void smoe_route_reduce_residual_kernel(
    const c10::Half* __restrict__ y_route,
    const int32_t* __restrict__ route_pos,
    const ScoreT* __restrict__ topk_score,
    const c10::Half* __restrict__ residual,
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

    float value = static_cast<float>(residual[idx]);
    const int32_t pos0 = route_pos[token * kTopK + 0];
    const int32_t pos1 = route_pos[token * kTopK + 1];
    if (pos0 >= 0) {
        value += static_cast<float>(topk_score[token * kTopK + 0])
            * static_cast<float>(y_route[static_cast<int64_t>(pos0) * kHiddenDim + dim]);
    }
    if (pos1 >= 0) {
        value += static_cast<float>(topk_score[token * kTopK + 1])
            * static_cast<float>(y_route[static_cast<int64_t>(pos1) * kHiddenDim + dim]);
    }
    out[idx] = static_cast<c10::Half>(value);
}

union Half8Vector {
    uint4 u4;
    __half h[8];
};

template <typename ScoreT>
__global__ void smoe_route_reduce_vec8_kernel(
    const c10::Half* __restrict__ y_route,
    const int32_t* __restrict__ route_pos,
    const ScoreT* __restrict__ topk_score,
    c10::Half* __restrict__ out,
    int64_t n_tokens
) {
    constexpr int kVecElems = 8;
    constexpr int kVecsPerRow = kHiddenDim / kVecElems;
    const int64_t vec_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = n_tokens * kVecsPerRow;
    if (vec_idx >= total) {
        return;
    }

    const int64_t token = vec_idx / kVecsPerRow;
    const int vec = static_cast<int>(vec_idx - token * kVecsPerRow);
    const int dim_base = vec * kVecElems;
    const int32_t pos0 = route_pos[token * kTopK + 0];
    const int32_t pos1 = route_pos[token * kTopK + 1];
    const float score0 = pos0 >= 0 ? static_cast<float>(topk_score[token * kTopK + 0]) : 0.0f;
    const float score1 = pos1 >= 0 ? static_cast<float>(topk_score[token * kTopK + 1]) : 0.0f;

    Half8Vector v0;
    Half8Vector v1;
    v0.u4 = make_uint4(0, 0, 0, 0);
    v1.u4 = make_uint4(0, 0, 0, 0);
    if (pos0 >= 0) {
        v0.u4 = *reinterpret_cast<const uint4*>(
            y_route + static_cast<int64_t>(pos0) * kHiddenDim + dim_base);
    }
    if (pos1 >= 0) {
        v1.u4 = *reinterpret_cast<const uint4*>(
            y_route + static_cast<int64_t>(pos1) * kHiddenDim + dim_base);
    }

    Half8Vector result;
#pragma unroll
    for (int i = 0; i < kVecElems; ++i) {
        const float value = score0 * __half2float(v0.h[i]) + score1 * __half2float(v1.h[i]);
        result.h[i] = __float2half_rn(value);
    }
    *reinterpret_cast<uint4*>(out + token * kHiddenDim + dim_base) = result.u4;
}

template <typename ScoreT>
__global__ void smoe_route_reduce_residual_vec8_kernel(
    const c10::Half* __restrict__ y_route,
    const int32_t* __restrict__ route_pos,
    const ScoreT* __restrict__ topk_score,
    const c10::Half* __restrict__ residual,
    c10::Half* __restrict__ out,
    int64_t n_tokens
) {
    constexpr int kVecElems = 8;
    constexpr int kVecsPerRow = kHiddenDim / kVecElems;
    const int64_t vec_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = n_tokens * kVecsPerRow;
    if (vec_idx >= total) {
        return;
    }

    const int64_t token = vec_idx / kVecsPerRow;
    const int vec = static_cast<int>(vec_idx - token * kVecsPerRow);
    const int dim_base = vec * kVecElems;
    const int32_t pos0 = route_pos[token * kTopK + 0];
    const int32_t pos1 = route_pos[token * kTopK + 1];
    const float score0 = pos0 >= 0 ? static_cast<float>(topk_score[token * kTopK + 0]) : 0.0f;
    const float score1 = pos1 >= 0 ? static_cast<float>(topk_score[token * kTopK + 1]) : 0.0f;

    Half8Vector v0;
    Half8Vector v1;
    Half8Vector residual_vec;
    v0.u4 = make_uint4(0, 0, 0, 0);
    v1.u4 = make_uint4(0, 0, 0, 0);
    residual_vec.u4 = *reinterpret_cast<const uint4*>(residual + token * kHiddenDim + dim_base);
    if (pos0 >= 0) {
        v0.u4 = *reinterpret_cast<const uint4*>(
            y_route + static_cast<int64_t>(pos0) * kHiddenDim + dim_base);
    }
    if (pos1 >= 0) {
        v1.u4 = *reinterpret_cast<const uint4*>(
            y_route + static_cast<int64_t>(pos1) * kHiddenDim + dim_base);
    }

    Half8Vector result;
#pragma unroll
    for (int i = 0; i < kVecElems; ++i) {
        const float value = __half2float(residual_vec.h[i])
            + score0 * __half2float(v0.h[i])
            + score1 * __half2float(v1.h[i]);
        result.h[i] = __float2half_rn(value);
    }
    *reinterpret_cast<uint4*>(out + token * kHiddenDim + dim_base) = result.u4;
}

template <typename IndexT>
void launch_route_count(
    const torch::Tensor& topk_idx,
    torch::Tensor& counts,
    int64_t n_tokens
) {
    const int64_t total_routes = n_tokens * kTopK;
    if (total_routes == 0) {
        return;
    }

    constexpr int kCountThreads = 256;
    const int blocks = static_cast<int>(ceil_div_int64(total_routes, kCountThreads));
    smoe_route_count_kernel<IndexT>
        <<<blocks, kCountThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
            topk_idx.data_ptr<IndexT>(),
            counts.data_ptr<int32_t>(),
            n_tokens
        );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename IndexT, typename ScoreT, bool WriteDebugMetadata>
void launch_route_pack(
    const torch::Tensor& x,
    const torch::Tensor& topk_idx,
    const torch::Tensor& topk_score,
    torch::Tensor& cursors,
    torch::Tensor& x_route,
    torch::Tensor& route_pos,
    torch::Tensor& route_token,
    torch::Tensor& route_slot,
    torch::Tensor& route_score,
    int64_t n_tokens
) {
    const int64_t total_routes = n_tokens * kTopK;
    if (total_routes == 0) {
        return;
    }

    constexpr int kRoutePackThreads = 4 * kWarpSize;
    constexpr int kRoutesPerCta = 4;
    int32_t* route_token_ptr = nullptr;
    int32_t* route_slot_ptr = nullptr;
    ScoreT* route_score_ptr = nullptr;
    if constexpr (WriteDebugMetadata) {
        route_token_ptr = route_token.data_ptr<int32_t>();
        route_slot_ptr = route_slot.data_ptr<int32_t>();
        route_score_ptr = route_score.data_ptr<ScoreT>();
    }
    if (env_flag_enabled("USE_SMOE_ROUTE_PACK_4ROUTES", true)) {
        smoe_route_pack_kernel<IndexT, ScoreT, WriteDebugMetadata>
            <<<static_cast<unsigned int>(ceil_div_int64(total_routes, kRoutesPerCta)),
               kRoutePackThreads,
               0,
               at::cuda::getCurrentCUDAStream()>>>(
                x.data_ptr<c10::Half>(),
                topk_idx.data_ptr<IndexT>(),
                topk_score.data_ptr<ScoreT>(),
                cursors.data_ptr<int32_t>(),
                x_route.data_ptr<c10::Half>(),
                route_pos.data_ptr<int32_t>(),
                route_token_ptr,
                route_slot_ptr,
                route_score_ptr,
                n_tokens
            );
    } else {
        smoe_route_pack_one_route_kernel<IndexT, ScoreT, WriteDebugMetadata>
            <<<static_cast<unsigned int>(total_routes),
               kThreads,
               0,
               at::cuda::getCurrentCUDAStream()>>>(
                x.data_ptr<c10::Half>(),
                topk_idx.data_ptr<IndexT>(),
                topk_score.data_ptr<ScoreT>(),
                cursors.data_ptr<int32_t>(),
                x_route.data_ptr<c10::Half>(),
                route_pos.data_ptr<int32_t>(),
                route_token_ptr,
                route_slot_ptr,
                route_score_ptr,
                n_tokens
            );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int InDim, int OutDim, bool ApplyRelu>
void launch_grouped_linear(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& counts,
    const torch::Tensor& offsets,
    torch::Tensor& output,
    int64_t max_routes_per_expert
) {
    if (max_routes_per_expert == 0 || input.size(0) == 0) {
        return;
    }
    TORCH_CHECK(ceil_div_int64(input.size(0), kBlockM) <= 65535,
        "route pool creates too many CTA rows");

    const dim3 block(kThreads);
    const dim3 grid(
        static_cast<unsigned int>(OutDim / kBlockN),
        static_cast<unsigned int>(ceil_div_int64(input.size(0), kBlockM)),
        1
    );

    smoe_grouped_linear_mma_tn_kernel<InDim, OutDim, ApplyRelu>
        <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            input.data_ptr<c10::Half>(),
            weight.data_ptr<c10::Half>(),
            bias.data_ptr<c10::Half>(),
            counts.data_ptr<int32_t>(),
            offsets.data_ptr<int32_t>(),
            output.data_ptr<c10::Half>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int InDim, int OutDim, bool ApplyRelu>
void launch_grouped_linear_pack_simple_w4a4(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& counts,
    const torch::Tensor& offsets,
    torch::Tensor& output_pack,
    int64_t max_routes_per_expert,
    double act_scale
) {
    if (max_routes_per_expert == 0 || input.size(0) == 0) {
        return;
    }
    check_half_cuda_contiguous(input, "input");
    check_half_cuda_contiguous(weight, "weight");
    check_half_cuda_contiguous(bias, "bias");
    check_i32_cuda_contiguous(counts, "counts");
    check_i32_cuda_contiguous(offsets, "offsets");
    check_i32_cuda_contiguous(output_pack, "output_pack");
    check_same_device(input, weight, "input", "weight");
    check_same_device(input, bias, "input", "bias");
    check_same_device(input, counts, "input", "counts");
    check_same_device(input, offsets, "input", "offsets");
    check_same_device(input, output_pack, "input", "output_pack");
    TORCH_CHECK(input.dim() == 2 && input.size(1) == InDim, "input has wrong shape");
    TORCH_CHECK(
        weight.dim() == 3
        && weight.size(0) == kNumExperts
        && weight.size(1) == OutDim
        && weight.size(2) == InDim,
        "weight has wrong shape"
    );
    TORCH_CHECK(bias.dim() == 2 && bias.size(0) == kNumExperts && bias.size(1) == OutDim,
        "bias has wrong shape");
    TORCH_CHECK(output_pack.dim() == 2 && output_pack.size(0) == input.size(0)
        && output_pack.size(1) == OutDim / 8, "output_pack has wrong shape");
    TORCH_CHECK(act_scale > 0.0, "simple W4A4 act_scale must be positive");
    TORCH_CHECK(ceil_div_int64(input.size(0), kBlockM) <= 65535,
        "route pool creates too many CTA rows");

    const dim3 block(kThreads);
    const dim3 grid(
        static_cast<unsigned int>(OutDim / kBlockN),
        static_cast<unsigned int>(ceil_div_int64(input.size(0), kBlockM)),
        1
    );
    smoe_grouped_linear_mma_tn_pack_w4a4_kernel<InDim, OutDim, ApplyRelu>
        <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            input.data_ptr<c10::Half>(),
            weight.data_ptr<c10::Half>(),
            bias.data_ptr<c10::Half>(),
            counts.data_ptr<int32_t>(),
            offsets.data_ptr<int32_t>(),
            output_pack.data_ptr<int32_t>(),
            1.0f / static_cast<float>(act_scale)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int InDim, int OutDim, bool ApplyRelu>
void launch_grouped_linear_pack_simple_w4a4_m64(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& counts,
    const torch::Tensor& offsets,
    torch::Tensor& output_pack,
    int64_t max_routes_per_expert,
    double act_scale
) {
    if (max_routes_per_expert == 0 || input.size(0) == 0) {
        return;
    }
    check_half_cuda_contiguous(input, "input");
    check_half_cuda_contiguous(weight, "weight");
    check_half_cuda_contiguous(bias, "bias");
    check_i32_cuda_contiguous(counts, "counts");
    check_i32_cuda_contiguous(offsets, "offsets");
    check_i32_cuda_contiguous(output_pack, "output_pack");
    check_same_device(input, weight, "input", "weight");
    check_same_device(input, bias, "input", "bias");
    check_same_device(input, counts, "input", "counts");
    check_same_device(input, offsets, "input", "offsets");
    check_same_device(input, output_pack, "input", "output_pack");
    TORCH_CHECK(input.dim() == 2 && input.size(1) == InDim, "input has wrong shape");
    TORCH_CHECK(
        weight.dim() == 3
        && weight.size(0) == kNumExperts
        && weight.size(1) == OutDim
        && weight.size(2) == InDim,
        "weight has wrong shape"
    );
    TORCH_CHECK(bias.dim() == 2 && bias.size(0) == kNumExperts && bias.size(1) == OutDim,
        "bias has wrong shape");
    TORCH_CHECK(output_pack.dim() == 2 && output_pack.size(0) == input.size(0)
        && output_pack.size(1) == OutDim / 8, "output_pack has wrong shape");
    TORCH_CHECK(act_scale > 0.0, "simple W4A4 act_scale must be positive");
    TORCH_CHECK(ceil_div_int64(input.size(0), kBlockM64) <= 65535,
        "route pool creates too many CTA rows for M64 simple W4A4 pack");

    const dim3 block(kThreads);
    const dim3 grid(
        static_cast<unsigned int>(OutDim / kBlockN),
        static_cast<unsigned int>(ceil_div_int64(input.size(0), kBlockM64)),
        1
    );
    smoe_grouped_linear_mma_tn_pack_w4a4_m64_kernel<InDim, OutDim, ApplyRelu>
        <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            input.data_ptr<c10::Half>(),
            weight.data_ptr<c10::Half>(),
            bias.data_ptr<c10::Half>(),
            counts.data_ptr<int32_t>(),
            offsets.data_ptr<int32_t>(),
            output_pack.data_ptr<int32_t>(),
            1.0f / static_cast<float>(act_scale)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int InDim, int OutDim, bool ApplyRelu>
void launch_grouped_linear_m64(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& counts,
    const torch::Tensor& offsets,
    torch::Tensor& output,
    int64_t max_routes_per_expert
) {
    if (max_routes_per_expert == 0 || input.size(0) == 0) {
        return;
    }
    TORCH_CHECK(ceil_div_int64(input.size(0), kBlockM64) <= 65535,
        "route pool creates too many CTA rows for M64 SMoE");

    const dim3 block(kThreads);
    const dim3 grid(
        static_cast<unsigned int>(OutDim / kBlockN),
        static_cast<unsigned int>(ceil_div_int64(input.size(0), kBlockM64)),
        1
    );

    smoe_grouped_linear_mma_tn_m64_kernel<InDim, OutDim, ApplyRelu>
        <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            input.data_ptr<c10::Half>(),
            weight.data_ptr<c10::Half>(),
            bias.data_ptr<c10::Half>(),
            counts.data_ptr<int32_t>(),
            offsets.data_ptr<int32_t>(),
            output.data_ptr<c10::Half>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#if BAIDU_CTI_ENABLE_CUTLASS_SMOE
using CutlassFc2Element = cutlass::half_t;
using CutlassFc2LayoutA = cutlass::layout::RowMajor;
using CutlassFc2LayoutB = cutlass::layout::ColumnMajor;
using CutlassFc2LayoutC = cutlass::layout::RowMajor;
using CutlassFc2Accumulator = float;
using CutlassFc2Epilogue = cutlass::epilogue::thread::LinearCombination<
    CutlassFc2Element,
    128 / cutlass::sizeof_bits<CutlassFc2Element>::value,
    CutlassFc2Accumulator,
    CutlassFc2Accumulator>;
using CutlassFc2GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    CutlassFc2Element,
    CutlassFc2LayoutA,
    cutlass::ComplexTransform::kNone,
    8,
    CutlassFc2Element,
    CutlassFc2LayoutB,
    cutlass::ComplexTransform::kNone,
    8,
    CutlassFc2Element,
    CutlassFc2LayoutC,
    CutlassFc2Accumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    CutlassFc2Epilogue,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    4,
    cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly>::GemmKernel;
using CutlassFc2GroupedGemm = cutlass::gemm::device::GemmGrouped<CutlassFc2GemmKernel>;

static_assert(sizeof(cutlass::gemm::GemmCoord) == 3 * sizeof(int),
    "CUTLASS GemmCoord layout must stay int[3]-compatible");

__global__ void prepare_cutlass_fc2_grouped_args_kernel(
    const c10::Half* __restrict__ input,
    const c10::Half* __restrict__ weight,
    const c10::Half* __restrict__ bias,
    const int32_t* __restrict__ counts,
    const int32_t* __restrict__ offsets,
    c10::Half* __restrict__ output,
    cutlass::gemm::GemmCoord* __restrict__ problem_sizes,
    CutlassFc2Element** __restrict__ ptr_A,
    CutlassFc2Element** __restrict__ ptr_B,
    CutlassFc2Element** __restrict__ ptr_C,
    CutlassFc2Element** __restrict__ ptr_D,
    int64_t* __restrict__ lda,
    int64_t* __restrict__ ldb,
    int64_t* __restrict__ ldc,
    int64_t* __restrict__ ldd
) {
    const int expert = threadIdx.x;
    if (expert >= kNumExperts) {
        return;
    }

    const int32_t expert_offset = offsets[expert];
    const int32_t expert_count = counts[expert];
    problem_sizes[expert] = cutlass::gemm::GemmCoord(expert_count, kHiddenDim, kFfDim);

    ptr_A[expert] = reinterpret_cast<CutlassFc2Element*>(
        const_cast<c10::Half*>(input + static_cast<int64_t>(expert_offset) * kFfDim));
    ptr_B[expert] = reinterpret_cast<CutlassFc2Element*>(
        const_cast<c10::Half*>(weight + static_cast<int64_t>(expert) * kHiddenDim * kFfDim));
    ptr_C[expert] = reinterpret_cast<CutlassFc2Element*>(
        const_cast<c10::Half*>(bias + static_cast<int64_t>(expert) * kHiddenDim));
    ptr_D[expert] = reinterpret_cast<CutlassFc2Element*>(
        output + static_cast<int64_t>(expert_offset) * kHiddenDim);

    lda[expert] = kFfDim;
    ldb[expert] = kFfDim;
    ldc[expert] = 0;
    ldd[expert] = kHiddenDim;
}

void launch_grouped_linear_cutlass_fc2(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& counts,
    const torch::Tensor& offsets,
    torch::Tensor& output,
    int64_t max_routes_per_expert
) {
    if (max_routes_per_expert == 0 || input.size(0) == 0) {
        return;
    }
    TORCH_CHECK(input.size(1) == kFfDim, "CUTLASS fc2 input must have shape [pool,1024]");
    TORCH_CHECK(weight.dim() == 3 && weight.size(0) == kNumExperts
        && weight.size(1) == kHiddenDim && weight.size(2) == kFfDim,
        "CUTLASS fc2 weight must have shape [8,512,1024]");
    TORCH_CHECK(bias.dim() == 2 && bias.size(0) == kNumExperts && bias.size(1) == kHiddenDim,
        "CUTLASS fc2 bias must have shape [8,512]");
    TORCH_CHECK(output.size(0) == input.size(0) && output.size(1) == kHiddenDim,
        "CUTLASS fc2 output must have shape [pool,512]");

    auto int_options = torch::TensorOptions().device(input.device()).dtype(torch::kInt32);
    auto ptr_options = torch::TensorOptions().device(input.device()).dtype(torch::kInt64);
    auto byte_options = torch::TensorOptions().device(input.device()).dtype(torch::kUInt8);

    auto problem_storage = torch::empty({kNumExperts, 3}, int_options);
    auto ptr_A_storage = torch::empty({kNumExperts}, ptr_options);
    auto ptr_B_storage = torch::empty({kNumExperts}, ptr_options);
    auto ptr_C_storage = torch::empty({kNumExperts}, ptr_options);
    auto ptr_D_storage = torch::empty({kNumExperts}, ptr_options);
    auto lda_storage = torch::empty({kNumExperts}, ptr_options);
    auto ldb_storage = torch::empty({kNumExperts}, ptr_options);
    auto ldc_storage = torch::empty({kNumExperts}, ptr_options);
    auto ldd_storage = torch::empty({kNumExperts}, ptr_options);

    auto stream = at::cuda::getCurrentCUDAStream();
    prepare_cutlass_fc2_grouped_args_kernel<<<1, kNumExperts, 0, stream>>>(
        input.data_ptr<c10::Half>(),
        weight.data_ptr<c10::Half>(),
        bias.data_ptr<c10::Half>(),
        counts.data_ptr<int32_t>(),
        offsets.data_ptr<int32_t>(),
        output.data_ptr<c10::Half>(),
        reinterpret_cast<cutlass::gemm::GemmCoord*>(problem_storage.data_ptr<int32_t>()),
        reinterpret_cast<CutlassFc2Element**>(ptr_A_storage.data_ptr<int64_t>()),
        reinterpret_cast<CutlassFc2Element**>(ptr_B_storage.data_ptr<int64_t>()),
        reinterpret_cast<CutlassFc2Element**>(ptr_C_storage.data_ptr<int64_t>()),
        reinterpret_cast<CutlassFc2Element**>(ptr_D_storage.data_ptr<int64_t>()),
        lda_storage.data_ptr<int64_t>(),
        ldb_storage.data_ptr<int64_t>(),
        ldc_storage.data_ptr<int64_t>(),
        ldd_storage.data_ptr<int64_t>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    CutlassFc2GroupedGemm gemm;
    const int threadblock_count = CutlassFc2GroupedGemm::sufficient();
    TORCH_CHECK(threadblock_count > 0, "CUTLASS fc2 grouped GEMM has no available threadblocks");
    typename CutlassFc2GroupedGemm::EpilogueOutputOp::Params epilogue_op(1.0f, 1.0f);
    typename CutlassFc2GroupedGemm::Arguments args(
        reinterpret_cast<cutlass::gemm::GemmCoord*>(problem_storage.data_ptr<int32_t>()),
        kNumExperts,
        threadblock_count,
        epilogue_op,
        reinterpret_cast<CutlassFc2Element**>(ptr_A_storage.data_ptr<int64_t>()),
        reinterpret_cast<CutlassFc2Element**>(ptr_B_storage.data_ptr<int64_t>()),
        reinterpret_cast<CutlassFc2Element**>(ptr_C_storage.data_ptr<int64_t>()),
        reinterpret_cast<CutlassFc2Element**>(ptr_D_storage.data_ptr<int64_t>()),
        lda_storage.data_ptr<int64_t>(),
        ldb_storage.data_ptr<int64_t>(),
        ldc_storage.data_ptr<int64_t>(),
        ldd_storage.data_ptr<int64_t>(),
        nullptr
    );

    cutlass::Status status = gemm.can_implement(args);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
        "CUTLASS fc2 grouped GEMM cannot implement problem: ", cutlass::cutlassGetStatusString(status));

    size_t workspace_size = gemm.get_workspace_size(args);
    torch::Tensor workspace;
    void* workspace_ptr = nullptr;
    if (workspace_size > 0) {
        workspace = torch::empty({static_cast<int64_t>(workspace_size)}, byte_options);
        workspace_ptr = workspace.data_ptr<uint8_t>();
    }

    status = gemm.initialize(args, workspace_ptr, stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
        "CUTLASS fc2 grouped GEMM initialize failed: ", cutlass::cutlassGetStatusString(status));
    status = gemm.run(stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
        "CUTLASS fc2 grouped GEMM run failed: ", cutlass::cutlassGetStatusString(status));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
#endif

template <int InDim, int OutDim, bool ApplyRelu, typename ScaleT, bool UseLop3Dequant>
void launch_grouped_linear_w4a16_typed(
    const torch::Tensor& input,
    const torch::Tensor& weight_pack,
    const torch::Tensor& weight_scale,
    const torch::Tensor& weight_zero,
    const torch::Tensor& bias,
    const torch::Tensor& counts,
    const torch::Tensor& offsets,
    torch::Tensor& output,
    int64_t max_routes_per_expert,
    int64_t group_size
) {
    if (max_routes_per_expert == 0 || input.size(0) == 0) {
        return;
    }
    TORCH_CHECK(ceil_div_int64(input.size(0), kBlockM) <= 65535,
        "route pool creates too many CTA rows");

    const dim3 block(kThreads);
    const dim3 grid(
        static_cast<unsigned int>(OutDim / kBlockN),
        static_cast<unsigned int>(ceil_div_int64(input.size(0), kBlockM)),
        1
    );

    smoe_grouped_linear_w4a16_mma_tn_kernel<
        InDim,
        OutDim,
        ApplyRelu,
        ScaleT,
        UseLop3Dequant>
        <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            input.data_ptr<c10::Half>(),
            weight_pack.data_ptr<int16_t>(),
            weight_scale.data_ptr<ScaleT>(),
            weight_zero.data_ptr<ScaleT>(),
            bias.data_ptr<c10::Half>(),
            counts.data_ptr<int32_t>(),
            offsets.data_ptr<int32_t>(),
            output.data_ptr<c10::Half>(),
            static_cast<int>(group_size)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int InDim, int OutDim, bool ApplyRelu>
void launch_grouped_linear_w4a16(
    const torch::Tensor& input,
    const torch::Tensor& weight_pack,
    const torch::Tensor& weight_scale,
    const torch::Tensor& weight_zero,
    const torch::Tensor& bias,
    const torch::Tensor& counts,
    const torch::Tensor& offsets,
    torch::Tensor& output,
    int64_t max_routes_per_expert,
    int64_t group_size
) {
    TORCH_CHECK(
        weight_scale.scalar_type() == weight_zero.scalar_type(),
        "weight_scale and weight_zero must have the same dtype"
    );
    if (weight_scale.scalar_type() == at::kHalf) {
        launch_grouped_linear_w4a16_typed<InDim, OutDim, ApplyRelu, c10::Half, false>(
            input,
            weight_pack,
            weight_scale,
            weight_zero,
            bias,
            counts,
            offsets,
            output,
            max_routes_per_expert,
            group_size
        );
    } else {
        launch_grouped_linear_w4a16_typed<InDim, OutDim, ApplyRelu, float, false>(
            input,
            weight_pack,
            weight_scale,
            weight_zero,
            bias,
            counts,
            offsets,
            output,
            max_routes_per_expert,
            group_size
        );
    }
}

template <int InDim, int OutDim, bool ApplyRelu>
void launch_grouped_linear_w4a16_lop3(
    const torch::Tensor& input,
    const torch::Tensor& weight_pack,
    const torch::Tensor& weight_scale,
    const torch::Tensor& weight_zero,
    const torch::Tensor& bias,
    const torch::Tensor& counts,
    const torch::Tensor& offsets,
    torch::Tensor& output,
    int64_t max_routes_per_expert,
    int64_t group_size
) {
    TORCH_CHECK(
        weight_scale.scalar_type() == weight_zero.scalar_type(),
        "weight_scale and weight_zero must have the same dtype"
    );
    if (weight_scale.scalar_type() == at::kHalf) {
        launch_grouped_linear_w4a16_typed<InDim, OutDim, ApplyRelu, c10::Half, true>(
            input,
            weight_pack,
            weight_scale,
            weight_zero,
            bias,
            counts,
            offsets,
            output,
            max_routes_per_expert,
            group_size
        );
    } else {
        launch_grouped_linear_w4a16_typed<InDim, OutDim, ApplyRelu, float, true>(
            input,
            weight_pack,
            weight_scale,
            weight_zero,
            bias,
            counts,
            offsets,
            output,
            max_routes_per_expert,
            group_size
        );
    }
}

template <typename ScaleT>
__global__ __launch_bounds__(kSimtFc2BlockM * kSimtFc2BlockN) void smoe_grouped_linear_w4a16_simt_fc2_kernel(
    const c10::Half* __restrict__ input,
    const int16_t* __restrict__ weight_pack,
    const ScaleT* __restrict__ weight_scale,
    const ScaleT* __restrict__ weight_zero,
    const c10::Half* __restrict__ bias,
    const int32_t* __restrict__ counts,
    const int32_t* __restrict__ offsets,
    c10::Half* __restrict__ output,
    int group_size
) {
    static_assert(kFfDim % 4 == 0, "fc2 input dimension must be divisible by 4");
    const int local_n = threadIdx.x;
    const int local_m = threadIdx.y;
    const int col = blockIdx.x * kSimtFc2BlockN + local_n;
    const int pool_m_tile_base = blockIdx.y * kSimtFc2BlockM;
    if (col >= kHiddenDim) {
        return;
    }

    const int total_padded_routes = offsets[kNumExperts];
    if (pool_m_tile_base >= total_padded_routes) {
        return;
    }

    const int expert = find_expert_for_pool_m(offsets, pool_m_tile_base);
    if (expert >= kNumExperts) {
        return;
    }

    const int expert_offset = offsets[expert];
    const int route_m = pool_m_tile_base - expert_offset + local_m;
    if (route_m >= counts[expert]) {
        return;
    }

    constexpr int kPackedInDim = kFfDim / 4;
    const int groups_per_row = kFfDim / group_size;
    const int packs_per_group = group_size / 4;
    const int64_t route_base =
        (static_cast<int64_t>(expert_offset) + route_m) * kFfDim;
    const int64_t weight_row =
        (static_cast<int64_t>(expert) * kHiddenDim + col);

    float acc = 0.0f;
#pragma unroll
    for (int group = 0; group < groups_per_row; ++group) {
        const int64_t scale_idx = weight_row * groups_per_row + group;
        const float scale = scalar_to_float(weight_scale[scale_idx]);
        const float zero = scalar_to_float(weight_zero[scale_idx]);
        const int pack_base = group * packs_per_group;

        for (int pack_offset = 0; pack_offset < packs_per_group; ++pack_offset) {
            const int pack_k = pack_base + pack_offset;
            const uint16_t packed =
                static_cast<uint16_t>(weight_pack[weight_row * kPackedInDim + pack_k]);
            const int k_base = pack_k * 4;
#pragma unroll
            for (int q_idx = 0; q_idx < 4; ++q_idx) {
                const int q = static_cast<int>((packed >> (q_idx * 4)) & 0x000f);
                const float w = __half2float(__float2half_rn(static_cast<float>(q) * scale + zero));
                const float x_val = static_cast<float>(input[route_base + k_base + q_idx]);
                acc = fmaf(x_val, w, acc);
            }
        }
    }

    acc += static_cast<float>(bias[static_cast<int64_t>(expert) * kHiddenDim + col]);
    output[(static_cast<int64_t>(expert_offset) + route_m) * kHiddenDim + col] =
        static_cast<c10::Half>(acc);
}

template <typename ScaleT>
void launch_grouped_linear_w4a16_simt_fc2_typed(
    const torch::Tensor& input,
    const torch::Tensor& weight_pack,
    const torch::Tensor& weight_scale,
    const torch::Tensor& weight_zero,
    const torch::Tensor& bias,
    const torch::Tensor& counts,
    const torch::Tensor& offsets,
    torch::Tensor& output,
    int64_t max_routes_per_expert,
    int64_t group_size
) {
    if (max_routes_per_expert == 0 || input.size(0) == 0) {
        return;
    }
    TORCH_CHECK(group_size % 4 == 0, "SIMT W4A16 fc2 requires group_size divisible by 4");
    TORCH_CHECK(ceil_div_int64(input.size(0), kSimtFc2BlockM) <= 65535,
        "route pool creates too many CTA rows for SIMT W4A16 fc2");

    const dim3 block(kSimtFc2BlockN, kSimtFc2BlockM);
    const dim3 grid(
        static_cast<unsigned int>(ceil_div_int64(kHiddenDim, kSimtFc2BlockN)),
        static_cast<unsigned int>(ceil_div_int64(input.size(0), kSimtFc2BlockM)),
        1
    );

    smoe_grouped_linear_w4a16_simt_fc2_kernel<ScaleT>
        <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            input.data_ptr<c10::Half>(),
            weight_pack.data_ptr<int16_t>(),
            weight_scale.data_ptr<ScaleT>(),
            weight_zero.data_ptr<ScaleT>(),
            bias.data_ptr<c10::Half>(),
            counts.data_ptr<int32_t>(),
            offsets.data_ptr<int32_t>(),
            output.data_ptr<c10::Half>(),
            static_cast<int>(group_size)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_grouped_linear_w4a16_simt_fc2(
    const torch::Tensor& input,
    const torch::Tensor& weight_pack,
    const torch::Tensor& weight_scale,
    const torch::Tensor& weight_zero,
    const torch::Tensor& bias,
    const torch::Tensor& counts,
    const torch::Tensor& offsets,
    torch::Tensor& output,
    int64_t max_routes_per_expert,
    int64_t group_size
) {
    TORCH_CHECK(
        weight_scale.scalar_type() == weight_zero.scalar_type(),
        "weight_scale and weight_zero must have the same dtype"
    );
    if (weight_scale.scalar_type() == at::kHalf) {
        launch_grouped_linear_w4a16_simt_fc2_typed<c10::Half>(
            input,
            weight_pack,
            weight_scale,
            weight_zero,
            bias,
            counts,
            offsets,
            output,
            max_routes_per_expert,
            group_size
        );
    } else {
        launch_grouped_linear_w4a16_simt_fc2_typed<float>(
            input,
            weight_pack,
            weight_scale,
            weight_zero,
            bias,
            counts,
            offsets,
            output,
            max_routes_per_expert,
            group_size
        );
    }
}

__device__ __forceinline__ int unpack_s4(uint32_t word, int idx) {
    int q = static_cast<int>((word >> (4 * idx)) & 0xFu);
    return q >= 8 ? q - 16 : q;
}

template <int InDim>
__global__ void smoe_simple_w4a4_pack_activation_kernel(
    const __half* __restrict__ input,
    int32_t* __restrict__ packed,
    const int32_t* __restrict__ counts,
    const int32_t* __restrict__ offsets,
    int total_rows,
    float inv_scale
) {
    constexpr int kPacksPerRow = InDim / 8;
    const int row = blockIdx.x * kSimpleW4A4PackRowsPerCta + threadIdx.x / kPacksPerRow;
    const int tid = threadIdx.x % kPacksPerRow;
    if (row >= total_rows || tid >= kPacksPerRow) {
        return;
    }

    const int expert = find_expert_for_pool_m(offsets, row);
    uint32_t word = 0;
    if (expert < kNumExperts) {
        const int local = row - offsets[expert];
        if (local < counts[expert]) {
            const int base = row * InDim + tid * 8;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float v = __half2float(input[base + i]) * inv_scale;
                int q = __float2int_rn(v);
                q = max(-8, min(7, q));
                word |= (static_cast<uint32_t>(q) & 0xFu) << (4 * i);
            }
        }
    }
    packed[row * kPacksPerRow + tid] = static_cast<int32_t>(word);
}

template <int InDim, int OutDim, bool ApplyRelu>
__global__ void smoe_grouped_linear_simple_w4a4_kernel(
    const int32_t* __restrict__ a_pack,
    const int32_t* __restrict__ w_pack,
    const __half* __restrict__ bias,
    const int32_t* __restrict__ counts,
    const int32_t* __restrict__ offsets,
    __half* __restrict__ output,
    int total_rows,
    float out_scale,
    float fc2_output_scale
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)
    using namespace nvcuda;
    constexpr int kPacksPerRow = InDim / 8;
    using FragA = wmma::fragment<
        wmma::matrix_a,
        8,
        8,
        32,
        wmma::experimental::precision::s4,
        wmma::row_major>;
    using FragB = wmma::fragment<
        wmma::matrix_b,
        8,
        8,
        32,
        wmma::experimental::precision::s4,
        wmma::col_major>;
    using FragC = wmma::fragment<wmma::accumulator, 8, 8, 32, int>;

    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int warp_row_tile = warp_id >> 2;
    const int warp_col_pair = warp_id & 3;
    const int row_base = blockIdx.y * kSimpleW4A4CtaM + warp_row_tile * 8;
    const int col_base = blockIdx.x * kSimpleW4A4CtaN
        + warp_col_pair * (8 * kSimpleW4A4WarpNFragments);
    if (row_base >= total_rows || col_base >= OutDim) {
        return;
    }

    const int expert = find_expert_for_pool_m(offsets, row_base);
    if (expert >= kNumExperts) {
        return;
    }
    const int local_base = row_base - offsets[expert];
    if (local_base >= counts[expert]) {
        return;
    }

    FragC acc0;
    FragC acc1;
    FragC acc2;
    FragC acc3;
    wmma::fill_fragment(acc0, 0);
    wmma::fill_fragment(acc1, 0);
    wmma::fill_fragment(acc2, 0);
    wmma::fill_fragment(acc3, 0);
#pragma unroll
    for (int k = 0; k < InDim; k += 32) {
        FragA a_frag;
        FragB b_frag0;
        FragB b_frag1;
        FragB b_frag2;
        FragB b_frag3;
        const void* a_ptr = static_cast<const void*>(
            a_pack + row_base * kPacksPerRow + (k / 8));
        const void* b_ptr0 = static_cast<const void*>(
            w_pack + (expert * OutDim + col_base) * kPacksPerRow + (k / 8));
        const void* b_ptr1 = static_cast<const void*>(
            w_pack + (expert * OutDim + col_base + 8) * kPacksPerRow + (k / 8));
        const void* b_ptr2 = static_cast<const void*>(
            w_pack + (expert * OutDim + col_base + 16) * kPacksPerRow + (k / 8));
        const void* b_ptr3 = static_cast<const void*>(
            w_pack + (expert * OutDim + col_base + 24) * kPacksPerRow + (k / 8));
        wmma::load_matrix_sync(a_frag, a_ptr, InDim);
        wmma::load_matrix_sync(b_frag0, b_ptr0, InDim);
        wmma::load_matrix_sync(b_frag1, b_ptr1, InDim);
        wmma::load_matrix_sync(b_frag2, b_ptr2, InDim);
        wmma::load_matrix_sync(b_frag3, b_ptr3, InDim);
        wmma::mma_sync(acc0, a_frag, b_frag0, acc0, false);
        wmma::mma_sync(acc1, a_frag, b_frag1, acc1, false);
        wmma::mma_sync(acc2, a_frag, b_frag2, acc2, false);
        wmma::mma_sync(acc3, a_frag, b_frag3, acc3, false);
    }

    __shared__ int acc_tile[kSimpleW4A4WarpsPerCta][kSimpleW4A4WarpNFragments][8 * 8];
    wmma::store_matrix_sync(acc_tile[warp_id][0], acc0, 8, wmma::mem_row_major);
    wmma::store_matrix_sync(acc_tile[warp_id][1], acc1, 8, wmma::mem_row_major);
    wmma::store_matrix_sync(acc_tile[warp_id][2], acc2, 8, wmma::mem_row_major);
    wmma::store_matrix_sync(acc_tile[warp_id][3], acc3, 8, wmma::mem_row_major);
    __syncwarp();

#pragma unroll
    for (int frag_n = 0; frag_n < kSimpleW4A4WarpNFragments; ++frag_n) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int idx = lane + i * 32;
            const int local_row = idx / 8;
            const int local_col = idx - local_row * 8;
            const int row = row_base + local_row;
            const int col = col_base + frag_n * 8 + local_col;
            const int expert_local_row = local_base + local_row;
            if (row < total_rows && col < OutDim && expert_local_row < counts[expert]) {
                float value = static_cast<float>(acc_tile[warp_id][frag_n][idx]) * out_scale;
                value += __half2float(bias[expert * OutDim + col]);
                if constexpr (ApplyRelu) {
                    value = value > 0.0f ? value : 0.0f;
                }
                value *= fc2_output_scale;
                output[row * OutDim + col] = __float2half_rn(value);
            }
        }
    }
#endif
}

template <int InDim, int OutDim>
void validate_simple_w4a4_linear_inputs(
    const torch::Tensor& input,
    const torch::Tensor& weight_pack,
    const torch::Tensor& bias,
    const torch::Tensor& counts,
    const torch::Tensor& offsets
) {
    check_half_cuda_contiguous(input, "input");
    check_i32_cuda_contiguous(weight_pack, "weight_pack");
    check_half_cuda_contiguous(bias, "bias");
    check_i32_cuda_contiguous(counts, "counts");
    check_i32_cuda_contiguous(offsets, "offsets");
    check_same_device(input, weight_pack, "input", "weight_pack");
    check_same_device(input, bias, "input", "bias");
    check_same_device(input, counts, "input", "counts");
    check_same_device(input, offsets, "input", "offsets");
    TORCH_CHECK(input.dim() == 2 && input.size(1) == InDim,
        "simple W4A4 input has wrong shape");
    TORCH_CHECK(
        weight_pack.dim() == 3
        && weight_pack.size(0) == kNumExperts
        && weight_pack.size(1) == OutDim
        && weight_pack.size(2) == InDim / 8,
        "simple W4A4 weight_pack has wrong shape"
    );
    TORCH_CHECK(bias.dim() == 2 && bias.size(0) == kNumExperts && bias.size(1) == OutDim,
        "simple W4A4 bias has wrong shape");
    TORCH_CHECK(counts.numel() == kNumExperts, "counts must have shape [8]");
    TORCH_CHECK(offsets.numel() == kNumExperts + 1, "offsets must have shape [9]");
}

template <int InDim, int OutDim>
void validate_simple_w4a4_packed_linear_inputs(
    const torch::Tensor& input_pack,
    const torch::Tensor& weight_pack,
    const torch::Tensor& bias,
    const torch::Tensor& counts,
    const torch::Tensor& offsets,
    const torch::Tensor& output
) {
    check_i32_cuda_contiguous(input_pack, "input_pack");
    check_i32_cuda_contiguous(weight_pack, "weight_pack");
    check_half_cuda_contiguous(bias, "bias");
    check_i32_cuda_contiguous(counts, "counts");
    check_i32_cuda_contiguous(offsets, "offsets");
    check_half_cuda_contiguous(output, "output");
    check_same_device(input_pack, weight_pack, "input_pack", "weight_pack");
    check_same_device(input_pack, bias, "input_pack", "bias");
    check_same_device(input_pack, counts, "input_pack", "counts");
    check_same_device(input_pack, offsets, "input_pack", "offsets");
    check_same_device(input_pack, output, "input_pack", "output");
    TORCH_CHECK(input_pack.dim() == 2 && input_pack.size(1) == InDim / 8,
        "simple W4A4 input_pack has wrong shape");
    TORCH_CHECK(
        weight_pack.dim() == 3
        && weight_pack.size(0) == kNumExperts
        && weight_pack.size(1) == OutDim
        && weight_pack.size(2) == InDim / 8,
        "simple W4A4 weight_pack has wrong shape"
    );
    TORCH_CHECK(bias.dim() == 2 && bias.size(0) == kNumExperts && bias.size(1) == OutDim,
        "simple W4A4 bias has wrong shape");
    TORCH_CHECK(output.dim() == 2 && output.size(0) == input_pack.size(0)
        && output.size(1) == OutDim, "simple W4A4 output has wrong shape");
    TORCH_CHECK(counts.numel() == kNumExperts, "counts must have shape [8]");
    TORCH_CHECK(offsets.numel() == kNumExperts + 1, "offsets must have shape [9]");
}

template <int InDim, int OutDim, bool ApplyRelu>
void launch_grouped_linear_simple_w4a4_packed_input(
    const torch::Tensor& input_pack,
    const torch::Tensor& weight_pack,
    const torch::Tensor& bias,
    const torch::Tensor& counts,
    const torch::Tensor& offsets,
    torch::Tensor& output,
    int64_t max_routes_per_expert,
    double act_scale,
    double weight_scale,
    double output_scale
) {
    if (max_routes_per_expert == 0 || input_pack.size(0) == 0) {
        return;
    }
    validate_simple_w4a4_packed_linear_inputs<InDim, OutDim>(
        input_pack, weight_pack, bias, counts, offsets, output);
    TORCH_CHECK(act_scale > 0.0, "simple W4A4 act_scale must be positive");
    TORCH_CHECK(weight_scale > 0.0, "simple W4A4 weight_scale must be positive");
    TORCH_CHECK(output_scale > 0.0, "simple W4A4 output_scale must be positive");
    TORCH_CHECK(ceil_div_int64(input_pack.size(0), kSimpleW4A4CtaM) <= 65535,
        "route pool creates too many CTA rows for simple W4A4");

    const dim3 block(kSimpleW4A4WarpsPerCta * kWarpSize);
    const dim3 grid(
        static_cast<unsigned int>(ceil_div_int64(OutDim, kSimpleW4A4CtaN)),
        static_cast<unsigned int>(ceil_div_int64(input_pack.size(0), kSimpleW4A4CtaM)),
        1
    );
    smoe_grouped_linear_simple_w4a4_kernel<InDim, OutDim, ApplyRelu>
        <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            input_pack.data_ptr<int32_t>(),
            weight_pack.data_ptr<int32_t>(),
            reinterpret_cast<const __half*>(bias.data_ptr<c10::Half>()),
            counts.data_ptr<int32_t>(),
            offsets.data_ptr<int32_t>(),
            reinterpret_cast<__half*>(output.data_ptr<c10::Half>()),
            static_cast<int>(input_pack.size(0)),
            static_cast<float>(act_scale * weight_scale),
            static_cast<float>(output_scale)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int InDim, int OutDim, bool ApplyRelu>
void launch_grouped_linear_simple_w4a4(
    const torch::Tensor& input,
    const torch::Tensor& weight_pack,
    const torch::Tensor& bias,
    const torch::Tensor& counts,
    const torch::Tensor& offsets,
    torch::Tensor& output,
    int64_t max_routes_per_expert,
    double act_scale,
    double weight_scale,
    double output_scale
) {
    if (max_routes_per_expert == 0 || input.size(0) == 0) {
        return;
    }
    validate_simple_w4a4_linear_inputs<InDim, OutDim>(input, weight_pack, bias, counts, offsets);
    TORCH_CHECK(act_scale > 0.0, "simple W4A4 act_scale must be positive");
    TORCH_CHECK(weight_scale > 0.0, "simple W4A4 weight_scale must be positive");
    TORCH_CHECK(output_scale > 0.0, "simple W4A4 output_scale must be positive");
    TORCH_CHECK(ceil_div_int64(input.size(0), kSimpleW4A4CtaM) <= 65535,
        "route pool creates too many CTA rows for simple W4A4");

    constexpr int kPacksPerRow = InDim / 8;
    auto pack_options = input.options().dtype(torch::kInt32);
    auto a_pack = torch::empty({input.size(0), kPacksPerRow}, pack_options);
    const float inv_act_scale = 1.0f / static_cast<float>(act_scale);
    smoe_simple_w4a4_pack_activation_kernel<InDim>
        <<<
            static_cast<unsigned int>(ceil_div_int64(input.size(0), kSimpleW4A4PackRowsPerCta)),
            kSimpleW4A4PackRowsPerCta * kPacksPerRow,
            0,
            at::cuda::getCurrentCUDAStream()
        >>>(
            reinterpret_cast<const __half*>(input.data_ptr<c10::Half>()),
            a_pack.data_ptr<int32_t>(),
            counts.data_ptr<int32_t>(),
            offsets.data_ptr<int32_t>(),
            static_cast<int>(input.size(0)),
            inv_act_scale
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    launch_grouped_linear_simple_w4a4_packed_input<InDim, OutDim, ApplyRelu>(
        a_pack,
        weight_pack,
        bias,
        counts,
        offsets,
        output,
        max_routes_per_expert,
        act_scale,
        weight_scale,
        output_scale
    );
}

template <int InDim, int OutDim, bool ApplyRelu, typename ScaleT>
void launch_grouped_linear_w4a16_frag_typed(
    const torch::Tensor& input,
    const torch::Tensor& weight_pack,
    const torch::Tensor& weight_scale,
    const torch::Tensor& weight_zero,
    const torch::Tensor& bias,
    const torch::Tensor& counts,
    const torch::Tensor& offsets,
    torch::Tensor& output,
    int64_t max_routes_per_expert,
    int64_t group_size
) {
    if (max_routes_per_expert == 0 || input.size(0) == 0) {
        return;
    }
    TORCH_CHECK(ceil_div_int64(input.size(0), kBlockM) <= 65535,
        "route pool creates too many CTA rows");
    TORCH_CHECK(group_size % kMmaK == 0,
        "frag-dequant W4A16 path requires group_size divisible by 16");

    const dim3 block(kThreads);
    const dim3 grid(
        static_cast<unsigned int>(OutDim / kBlockN),
        static_cast<unsigned int>(ceil_div_int64(input.size(0), kBlockM)),
        1
    );

    smoe_grouped_linear_w4a16_frag_mma_tn_kernel<InDim, OutDim, ApplyRelu, ScaleT>
        <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            input.data_ptr<c10::Half>(),
            weight_pack.data_ptr<int16_t>(),
            weight_scale.data_ptr<ScaleT>(),
            weight_zero.data_ptr<ScaleT>(),
            bias.data_ptr<c10::Half>(),
            counts.data_ptr<int32_t>(),
            offsets.data_ptr<int32_t>(),
            output.data_ptr<c10::Half>(),
            static_cast<int>(group_size)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int InDim, int OutDim, bool ApplyRelu>
void launch_grouped_linear_w4a16_frag(
    const torch::Tensor& input,
    const torch::Tensor& weight_pack,
    const torch::Tensor& weight_scale,
    const torch::Tensor& weight_zero,
    const torch::Tensor& bias,
    const torch::Tensor& counts,
    const torch::Tensor& offsets,
    torch::Tensor& output,
    int64_t max_routes_per_expert,
    int64_t group_size
) {
    TORCH_CHECK(
        weight_scale.scalar_type() == weight_zero.scalar_type(),
        "weight_scale and weight_zero must have the same dtype"
    );

    if (weight_scale.scalar_type() == at::kHalf) {
        launch_grouped_linear_w4a16_frag_typed<InDim, OutDim, ApplyRelu, c10::Half>(
            input,
            weight_pack,
            weight_scale,
            weight_zero,
            bias,
            counts,
            offsets,
            output,
            max_routes_per_expert,
            group_size
        );
    } else {
        launch_grouped_linear_w4a16_frag_typed<InDim, OutDim, ApplyRelu, float>(
            input,
            weight_pack,
            weight_scale,
            weight_zero,
            bias,
            counts,
            offsets,
            output,
            max_routes_per_expert,
            group_size
        );
    }
}

template <int InDim, int OutDim, typename ScaleT>
void launch_debug_w4a16_bfrag_layout_typed(
    const torch::Tensor& weight_pack,
    const torch::Tensor& weight_scale,
    const torch::Tensor& weight_zero,
    torch::Tensor& dump,
    int64_t group_size,
    int64_t expert,
    int64_t n_tile_base,
    int64_t k_base
) {
    const dim3 block(kThreads);
    debug_w4a16_bfrag_layout_kernel<InDim, OutDim, ScaleT>
        <<<1, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            weight_pack.data_ptr<int16_t>(),
            weight_scale.data_ptr<ScaleT>(),
            weight_zero.data_ptr<ScaleT>(),
            dump.data_ptr<int32_t>(),
            static_cast<int>(group_size),
            static_cast<int>(expert),
            static_cast<int>(n_tile_base),
            static_cast<int>(k_base)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int InDim, int OutDim>
void launch_debug_w4a16_bfrag_layout(
    const torch::Tensor& weight_pack,
    const torch::Tensor& weight_scale,
    const torch::Tensor& weight_zero,
    torch::Tensor& dump,
    int64_t group_size,
    int64_t expert,
    int64_t n_tile_base,
    int64_t k_base
) {
    TORCH_CHECK(
        weight_scale.scalar_type() == weight_zero.scalar_type(),
        "weight_scale and weight_zero must have the same dtype"
    );
    if (weight_scale.scalar_type() == at::kHalf) {
        launch_debug_w4a16_bfrag_layout_typed<InDim, OutDim, c10::Half>(
            weight_pack,
            weight_scale,
            weight_zero,
            dump,
            group_size,
            expert,
            n_tile_base,
            k_base
        );
    } else {
        launch_debug_w4a16_bfrag_layout_typed<InDim, OutDim, float>(
            weight_pack,
            weight_scale,
            weight_zero,
            dump,
            group_size,
            expert,
            n_tile_base,
            k_base
        );
    }
}

template <typename ScoreT>
void launch_route_reduce(
    const torch::Tensor& y_route,
    const torch::Tensor& route_pos,
    const torch::Tensor& topk_score,
    torch::Tensor& out,
    int64_t n_tokens
) {
    if (n_tokens == 0) {
        return;
    }

    constexpr int kVecElems = 8;
    static_assert(kHiddenDim % kVecElems == 0, "hidden dim must be divisible by vector width");
    constexpr int kReduceThreads = 256;
    const int64_t total = n_tokens * (kHiddenDim / kVecElems);
    const int blocks = static_cast<int>(ceil_div_int64(total, kReduceThreads));
    smoe_route_reduce_vec8_kernel<ScoreT>
        <<<blocks, kReduceThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
            y_route.data_ptr<c10::Half>(),
            route_pos.data_ptr<int32_t>(),
            topk_score.data_ptr<ScoreT>(),
            out.data_ptr<c10::Half>(),
            n_tokens
        );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename ScoreT>
void launch_route_reduce_with_residual(
    const torch::Tensor& y_route,
    const torch::Tensor& route_pos,
    const torch::Tensor& topk_score,
    const torch::Tensor& residual,
    torch::Tensor& out,
    int64_t n_tokens
) {
    if (n_tokens == 0) {
        return;
    }

    constexpr int kVecElems = 8;
    static_assert(kHiddenDim % kVecElems == 0, "hidden dim must be divisible by vector width");
    constexpr int kReduceThreads = 256;
    const int64_t total = n_tokens * (kHiddenDim / kVecElems);
    const int blocks = static_cast<int>(ceil_div_int64(total, kReduceThreads));
    smoe_route_reduce_residual_vec8_kernel<ScoreT>
        <<<blocks, kReduceThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
            y_route.data_ptr<c10::Half>(),
            route_pos.data_ptr<int32_t>(),
            topk_score.data_ptr<ScoreT>(),
            residual.data_ptr<c10::Half>(),
            out.data_ptr<c10::Half>(),
            n_tokens
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void validate_route_inputs(
    const torch::Tensor& x,
    const torch::Tensor& topk_idx,
    const torch::Tensor& topk_score
) {
    check_half_cuda_contiguous(x, "x");
    check_topk_idx(topk_idx, "topk_idx");
    check_topk_score(topk_score, "topk_score");
    check_same_device(x, topk_idx, "x", "topk_idx");
    check_same_device(x, topk_score, "x", "topk_score");
    TORCH_CHECK(x.dim() == 2 && x.size(1) == kHiddenDim, "x must have shape [N,512]");
    TORCH_CHECK(topk_idx.dim() == 2, "topk_idx must have shape [N,2]");
    TORCH_CHECK(topk_idx.size(0) == x.size(0), "topk_idx must have shape [N,2]");
    TORCH_CHECK(topk_idx.size(1) == kTopK, "topk_idx must have shape [N,2]");
    TORCH_CHECK(topk_score.dim() == 2, "topk_score must have shape [N,2]");
    TORCH_CHECK(topk_score.size(0) == x.size(0), "topk_score must have shape [N,2]");
    TORCH_CHECK(topk_score.size(1) == kTopK, "topk_score must have shape [N,2]");
    check_n_tokens(x.size(0));
}

void validate_fc1_inputs(
    const torch::Tensor& x_route,
    const torch::Tensor& w1,
    const torch::Tensor& b1,
    const torch::Tensor& counts,
    const torch::Tensor& offsets
) {
    check_half_cuda_contiguous(x_route, "x_route");
    check_half_cuda_contiguous(w1, "w1");
    check_half_cuda_contiguous(b1, "b1");
    check_i32_cuda_contiguous(counts, "counts");
    check_i32_cuda_contiguous(offsets, "offsets");
    check_same_device(x_route, w1, "x_route", "w1");
    check_same_device(x_route, b1, "x_route", "b1");
    check_same_device(x_route, counts, "x_route", "counts");
    check_same_device(x_route, offsets, "x_route", "offsets");
    TORCH_CHECK(x_route.dim() == 2 && x_route.size(1) == kHiddenDim,
        "x_route must have shape [pool,512]");
    TORCH_CHECK(w1.dim() == 3 && w1.size(0) == kNumExperts && w1.size(1) == kFfDim && w1.size(2) == kHiddenDim,
        "w1 must have shape [8,1024,512]");
    TORCH_CHECK(b1.dim() == 2 && b1.size(0) == kNumExperts && b1.size(1) == kFfDim,
        "b1 must have shape [8,1024]");
    TORCH_CHECK(counts.numel() == kNumExperts, "counts must have shape [8]");
    TORCH_CHECK(offsets.numel() == kNumExperts + 1, "offsets must have shape [9]");
}

void validate_fc2_inputs(
    const torch::Tensor& h_route,
    const torch::Tensor& w2,
    const torch::Tensor& b2,
    const torch::Tensor& counts,
    const torch::Tensor& offsets
) {
    check_half_cuda_contiguous(h_route, "h_route");
    check_half_cuda_contiguous(w2, "w2");
    check_half_cuda_contiguous(b2, "b2");
    check_i32_cuda_contiguous(counts, "counts");
    check_i32_cuda_contiguous(offsets, "offsets");
    check_same_device(h_route, w2, "h_route", "w2");
    check_same_device(h_route, b2, "h_route", "b2");
    check_same_device(h_route, counts, "h_route", "counts");
    check_same_device(h_route, offsets, "h_route", "offsets");
    TORCH_CHECK(h_route.dim() == 2 && h_route.size(1) == kFfDim,
        "h_route must have shape [pool,1024]");
    TORCH_CHECK(w2.dim() == 3 && w2.size(0) == kNumExperts && w2.size(1) == kHiddenDim && w2.size(2) == kFfDim,
        "w2 must have shape [8,512,1024]");
    TORCH_CHECK(b2.dim() == 2 && b2.size(0) == kNumExperts && b2.size(1) == kHiddenDim,
        "b2 must have shape [8,512]");
    TORCH_CHECK(counts.numel() == kNumExperts, "counts must have shape [8]");
    TORCH_CHECK(offsets.numel() == kNumExperts + 1, "offsets must have shape [9]");
}

void validate_w4a16_inputs(
    const torch::Tensor& x,
    const torch::Tensor& w1_pack,
    const torch::Tensor& w1_scale,
    const torch::Tensor& w1_zero,
    const torch::Tensor& b1,
    const torch::Tensor& w2_pack,
    const torch::Tensor& w2_scale,
    const torch::Tensor& w2_zero,
    const torch::Tensor& b2,
    const torch::Tensor& topk_idx,
    const torch::Tensor& topk_score,
    int64_t group_size
) {
    validate_route_inputs(x, topk_idx, topk_score);
    check_i16_cuda_contiguous(w1_pack, "w1_pack");
    check_scale_cuda_contiguous(w1_scale, "w1_scale");
    check_scale_cuda_contiguous(w1_zero, "w1_zero");
    check_half_cuda_contiguous(b1, "b1");
    check_i16_cuda_contiguous(w2_pack, "w2_pack");
    check_scale_cuda_contiguous(w2_scale, "w2_scale");
    check_scale_cuda_contiguous(w2_zero, "w2_zero");
    check_half_cuda_contiguous(b2, "b2");

    check_same_device(x, w1_pack, "x", "w1_pack");
    check_same_device(x, w1_scale, "x", "w1_scale");
    check_same_device(x, w1_zero, "x", "w1_zero");
    check_same_device(x, b1, "x", "b1");
    check_same_device(x, w2_pack, "x", "w2_pack");
    check_same_device(x, w2_scale, "x", "w2_scale");
    check_same_device(x, w2_zero, "x", "w2_zero");
    check_same_device(x, b2, "x", "b2");

    TORCH_CHECK(group_size > 0, "group_size must be positive");
    TORCH_CHECK(group_size % 4 == 0, "group_size must be divisible by 4");
    TORCH_CHECK(kHiddenDim % group_size == 0, "group_size must divide 512");
    TORCH_CHECK(kFfDim % group_size == 0, "group_size must divide 1024");

    TORCH_CHECK(
        w1_pack.dim() == 3
        && w1_pack.size(0) == kNumExperts
        && w1_pack.size(1) == kFfDim
        && w1_pack.size(2) == kHiddenDim / 4,
        "w1_pack must have shape [8,1024,128]"
    );
    TORCH_CHECK(
        w1_scale.dim() == 3
        && w1_scale.size(0) == kNumExperts
        && w1_scale.size(1) == kFfDim
        && w1_scale.size(2) == kHiddenDim / group_size,
        "w1_scale must have shape [8,1024,512/group_size]"
    );
    TORCH_CHECK(
        w1_zero.dim() == w1_scale.dim()
        && w1_zero.size(0) == w1_scale.size(0)
        && w1_zero.size(1) == w1_scale.size(1)
        && w1_zero.size(2) == w1_scale.size(2),
        "w1_zero must match w1_scale shape"
    );
    TORCH_CHECK(
        b1.dim() == 2 && b1.size(0) == kNumExperts && b1.size(1) == kFfDim,
        "b1 must have shape [8,1024]"
    );

    TORCH_CHECK(
        w2_pack.dim() == 3
        && w2_pack.size(0) == kNumExperts
        && w2_pack.size(1) == kHiddenDim
        && w2_pack.size(2) == kFfDim / 4,
        "w2_pack must have shape [8,512,256]"
    );
    TORCH_CHECK(
        w2_scale.dim() == 3
        && w2_scale.size(0) == kNumExperts
        && w2_scale.size(1) == kHiddenDim
        && w2_scale.size(2) == kFfDim / group_size,
        "w2_scale must have shape [8,512,1024/group_size]"
    );
    TORCH_CHECK(
        w2_zero.dim() == w2_scale.dim()
        && w2_zero.size(0) == w2_scale.size(0)
        && w2_zero.size(1) == w2_scale.size(1)
        && w2_zero.size(2) == w2_scale.size(2),
        "w2_zero must match w2_scale shape"
    );
    TORCH_CHECK(
        b2.dim() == 2 && b2.size(0) == kNumExperts && b2.size(1) == kHiddenDim,
        "b2 must have shape [8,512]"
    );
}

void validate_w4a16_residual(
    const torch::Tensor& x,
    const torch::Tensor& residual
) {
    check_half_cuda_contiguous(residual, "residual");
    check_same_device(x, residual, "x", "residual");
    TORCH_CHECK(
        residual.dim() == 2 && residual.size(0) == x.size(0) && residual.size(1) == kHiddenDim,
        "residual must have shape [N,512]"
    );
}

template <int InDim, int OutDim>
void validate_w4a16_debug_weight(
    const torch::Tensor& weight_pack,
    const torch::Tensor& weight_scale,
    const torch::Tensor& weight_zero,
    int64_t group_size,
    int64_t expert,
    int64_t n_tile_base,
    int64_t k_base
) {
    check_i16_cuda_contiguous(weight_pack, "weight_pack");
    check_scale_cuda_contiguous(weight_scale, "weight_scale");
    check_scale_cuda_contiguous(weight_zero, "weight_zero");
    check_same_device(weight_pack, weight_scale, "weight_pack", "weight_scale");
    check_same_device(weight_pack, weight_zero, "weight_pack", "weight_zero");

    TORCH_CHECK(group_size > 0, "group_size must be positive");
    TORCH_CHECK(group_size % kMmaK == 0,
        "debug W4A16 B-fragment path requires group_size divisible by 16");
    TORCH_CHECK(InDim % group_size == 0, "group_size must divide InDim");
    TORCH_CHECK(expert >= 0 && expert < kNumExperts, "expert must be in [0, 8)");
    TORCH_CHECK(n_tile_base >= 0 && n_tile_base + kBlockN <= OutDim,
        "n_tile_base must select a full kBlockN-wide output tile");
    TORCH_CHECK(n_tile_base % kBlockN == 0, "n_tile_base must be divisible by kBlockN");
    TORCH_CHECK(k_base >= 0 && k_base + kMmaK <= InDim,
        "k_base must select a full 16-wide K tile");
    TORCH_CHECK(k_base % kMmaK == 0, "k_base must be divisible by 16");

    TORCH_CHECK(
        weight_pack.dim() == 3
        && weight_pack.size(0) == kNumExperts
        && weight_pack.size(1) == OutDim
        && weight_pack.size(2) == InDim / 4,
        "weight_pack has unexpected shape"
    );
    TORCH_CHECK(
        weight_scale.dim() == 3
        && weight_scale.size(0) == kNumExperts
        && weight_scale.size(1) == OutDim
        && weight_scale.size(2) == InDim / group_size,
        "weight_scale has unexpected shape"
    );
    TORCH_CHECK(
        weight_zero.dim() == weight_scale.dim()
        && weight_zero.size(0) == weight_scale.size(0)
        && weight_zero.size(1) == weight_scale.size(1)
        && weight_zero.size(2) == weight_scale.size(2),
        "weight_zero must match weight_scale shape"
    );
}

RouteMetadata build_route_metadata_impl(
    torch::Tensor x,
    torch::Tensor topk_idx,
    torch::Tensor topk_score,
    int64_t pad_multiple,
    bool include_debug_metadata
) {
    validate_route_inputs(x, topk_idx, topk_score);
    TORCH_CHECK(pad_multiple > 0, "pad_multiple must be positive");
    TORCH_CHECK(pad_multiple <= static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
        "pad_multiple is too large");

    const int64_t n_tokens = x.size(0);
    const int64_t max_pool_routes = max_pool_routes_for_tokens(n_tokens, pad_multiple);
    const int64_t max_routes_per_expert = n_tokens * kTopK;

    auto int_opts = x.options().dtype(torch::kInt32);
    auto counts = torch::empty({kNumExperts}, int_opts);
    auto cursors = torch::empty({kNumExperts}, int_opts);
    auto offsets = torch::empty({kNumExperts + 1}, int_opts);
    auto route_pos = torch::empty({n_tokens, kTopK}, int_opts);
    torch::Tensor route_token;
    torch::Tensor route_slot;
    auto x_route = torch::empty({max_pool_routes, kHiddenDim}, x.options());
    torch::Tensor route_score;
    if (include_debug_metadata) {
        route_token = torch::empty({max_pool_routes}, int_opts);
        route_slot = torch::empty({max_pool_routes}, int_opts);
        route_score = torch::empty({max_pool_routes}, topk_score.options());
    }

    auto stream = at::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaMemsetAsync(counts.data_ptr<int32_t>(), 0, counts.numel() * sizeof(int32_t), stream.stream()));
    C10_CUDA_CHECK(cudaMemsetAsync(route_pos.data_ptr<int32_t>(), 0xff, route_pos.numel() * sizeof(int32_t), stream.stream()));

    if (n_tokens == 0) {
        C10_CUDA_CHECK(cudaMemsetAsync(offsets.data_ptr<int32_t>(), 0, offsets.numel() * sizeof(int32_t), stream.stream()));
        return {x_route, route_pos, route_token, route_slot, route_score, counts, offsets, max_routes_per_expert};
    }

    if (topk_idx.scalar_type() == at::kLong) {
        launch_route_count<int64_t>(topk_idx, counts, n_tokens);
    } else {
        launch_route_count<int32_t>(topk_idx, counts, n_tokens);
    }

    smoe_route_prefix_kernel<<<1, 1, 0, stream>>>(
        counts.data_ptr<int32_t>(),
        offsets.data_ptr<int32_t>(),
        cursors.data_ptr<int32_t>(),
        static_cast<int32_t>(pad_multiple)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if (topk_idx.scalar_type() == at::kLong && topk_score.scalar_type() == at::kHalf) {
        if (include_debug_metadata) {
            launch_route_pack<int64_t, c10::Half, true>(
                x, topk_idx, topk_score, cursors, x_route, route_pos,
                route_token, route_slot, route_score, n_tokens);
        } else {
            launch_route_pack<int64_t, c10::Half, false>(
                x, topk_idx, topk_score, cursors, x_route, route_pos,
                route_token, route_slot, route_score, n_tokens);
        }
    } else if (topk_idx.scalar_type() == at::kLong && topk_score.scalar_type() == at::kFloat) {
        if (include_debug_metadata) {
            launch_route_pack<int64_t, float, true>(
                x, topk_idx, topk_score, cursors, x_route, route_pos,
                route_token, route_slot, route_score, n_tokens);
        } else {
            launch_route_pack<int64_t, float, false>(
                x, topk_idx, topk_score, cursors, x_route, route_pos,
                route_token, route_slot, route_score, n_tokens);
        }
    } else if (topk_idx.scalar_type() == at::kInt && topk_score.scalar_type() == at::kHalf) {
        if (include_debug_metadata) {
            launch_route_pack<int32_t, c10::Half, true>(
                x, topk_idx, topk_score, cursors, x_route, route_pos,
                route_token, route_slot, route_score, n_tokens);
        } else {
            launch_route_pack<int32_t, c10::Half, false>(
                x, topk_idx, topk_score, cursors, x_route, route_pos,
                route_token, route_slot, route_score, n_tokens);
        }
    } else {
        if (include_debug_metadata) {
            launch_route_pack<int32_t, float, true>(
                x, topk_idx, topk_score, cursors, x_route, route_pos,
                route_token, route_slot, route_score, n_tokens);
        } else {
            launch_route_pack<int32_t, float, false>(
                x, topk_idx, topk_score, cursors, x_route, route_pos,
                route_token, route_slot, route_score, n_tokens);
        }
    }

    return {x_route, route_pos, route_token, route_slot, route_score, counts, offsets, max_routes_per_expert};
}

RouteMetadata build_route_metadata_forward_only(
    torch::Tensor x,
    torch::Tensor topk_idx,
    torch::Tensor topk_score,
    int64_t pad_multiple = kBlockM
) {
    return build_route_metadata_impl(x, topk_idx, topk_score, pad_multiple, false);
}

RouteMetadata build_route_metadata_debug(
    torch::Tensor x,
    torch::Tensor topk_idx,
    torch::Tensor topk_score,
    int64_t pad_multiple = kBlockM
) {
    return build_route_metadata_impl(x, topk_idx, topk_score, pad_multiple, true);
}

RouteMetadata build_route_metadata(
    torch::Tensor x,
    torch::Tensor topk_idx,
    torch::Tensor topk_score,
    int64_t pad_multiple = kBlockM
) {
    if (env_flag_enabled("USE_SMOE_FORWARD_ONLY_ROUTE_METADATA", true)) {
        return build_route_metadata_forward_only(x, topk_idx, topk_score, pad_multiple);
    }
    return build_route_metadata_debug(x, topk_idx, topk_score, pad_multiple);
}

}  // namespace

torch::Tensor smoe_route_count(torch::Tensor topk_idx) {
    check_topk_idx(topk_idx, "topk_idx");
    TORCH_CHECK(topk_idx.dim() == 2 && topk_idx.size(1) == kTopK, "topk_idx must have shape [N,2]");
    const int64_t n_tokens = topk_idx.size(0);
    check_n_tokens(n_tokens);

    auto counts = torch::empty({kNumExperts}, topk_idx.options().dtype(torch::kInt32));
    C10_CUDA_CHECK(cudaMemsetAsync(
        counts.data_ptr<int32_t>(),
        0,
        counts.numel() * sizeof(int32_t),
        at::cuda::getCurrentCUDAStream().stream()
    ));

    if (topk_idx.scalar_type() == at::kLong) {
        launch_route_count<int64_t>(topk_idx, counts, n_tokens);
    } else {
        launch_route_count<int32_t>(topk_idx, counts, n_tokens);
    }
    return counts;
}

std::vector<torch::Tensor> smoe_route_pack(
    torch::Tensor x,
    torch::Tensor topk_idx,
    torch::Tensor topk_score
) {
    auto meta = build_route_metadata_debug(x, topk_idx, topk_score);
    return {
        meta.x_route,
        meta.route_pos,
        meta.route_token,
        meta.route_slot,
        meta.route_score,
        meta.counts,
        meta.offsets,
    };
}

torch::Tensor smoe_grouped_fc1_relu(
    torch::Tensor x_route,
    torch::Tensor w1,
    torch::Tensor b1,
    torch::Tensor counts,
    torch::Tensor offsets,
    int64_t max_routes_per_expert
) {
    validate_fc1_inputs(x_route, w1, b1, counts, offsets);
    TORCH_CHECK(max_routes_per_expert >= 0, "max_routes_per_expert must be non-negative");
    TORCH_CHECK(ceil_div_int64(max_routes_per_expert, kBlockM) <= 65535,
        "max_routes_per_expert creates too many CTA rows");

    auto h_route = torch::empty({x_route.size(0), kFfDim}, x_route.options());
    launch_grouped_linear<kHiddenDim, kFfDim, true>(
        x_route,
        w1,
        b1,
        counts,
        offsets,
        h_route,
        max_routes_per_expert
    );
    return h_route;
}

torch::Tensor smoe_grouped_fc2(
    torch::Tensor h_route,
    torch::Tensor w2,
    torch::Tensor b2,
    torch::Tensor counts,
    torch::Tensor offsets,
    int64_t max_routes_per_expert
) {
    validate_fc2_inputs(h_route, w2, b2, counts, offsets);
    TORCH_CHECK(max_routes_per_expert >= 0, "max_routes_per_expert must be non-negative");
    TORCH_CHECK(ceil_div_int64(max_routes_per_expert, kBlockM) <= 65535,
        "max_routes_per_expert creates too many CTA rows");

    auto y_route = torch::empty({h_route.size(0), kHiddenDim}, h_route.options());
    launch_grouped_linear<kFfDim, kHiddenDim, false>(
        h_route,
        w2,
        b2,
        counts,
        offsets,
        y_route,
        max_routes_per_expert
    );
    return y_route;
}

torch::Tensor smoe_route_reduce(
    torch::Tensor y_route,
    torch::Tensor route_pos,
    torch::Tensor topk_score
) {
    check_half_cuda_contiguous(y_route, "y_route");
    check_i32_cuda_contiguous(route_pos, "route_pos");
    check_topk_score(topk_score, "topk_score");
    check_same_device(y_route, route_pos, "y_route", "route_pos");
    check_same_device(y_route, topk_score, "y_route", "topk_score");
    TORCH_CHECK(y_route.dim() == 2 && y_route.size(1) == kHiddenDim,
        "y_route must have shape [pool,512]");
    TORCH_CHECK(route_pos.dim() == 2 && route_pos.size(1) == kTopK,
        "route_pos must have shape [N,2]");
    TORCH_CHECK(topk_score.dim() == 2 && topk_score.size(0) == route_pos.size(0) && topk_score.size(1) == kTopK,
        "topk_score must have shape [N,2]");

    const int64_t n_tokens = route_pos.size(0);
    check_n_tokens(n_tokens);
    auto out = torch::empty({n_tokens, kHiddenDim}, y_route.options());
    if (n_tokens == 0) {
        return out;
    }

    if (topk_score.scalar_type() == at::kHalf) {
        launch_route_reduce<c10::Half>(y_route, route_pos, topk_score, out, n_tokens);
    } else {
        launch_route_reduce<float>(y_route, route_pos, topk_score, out, n_tokens);
    }
    return out;
}

torch::Tensor smoe_route_reduce_with_residual(
    torch::Tensor y_route,
    torch::Tensor route_pos,
    torch::Tensor topk_score,
    torch::Tensor residual
) {
    check_half_cuda_contiguous(y_route, "y_route");
    check_i32_cuda_contiguous(route_pos, "route_pos");
    check_topk_score(topk_score, "topk_score");
    check_half_cuda_contiguous(residual, "residual");
    check_same_device(y_route, route_pos, "y_route", "route_pos");
    check_same_device(y_route, topk_score, "y_route", "topk_score");
    check_same_device(y_route, residual, "y_route", "residual");
    TORCH_CHECK(y_route.dim() == 2 && y_route.size(1) == kHiddenDim,
        "y_route must have shape [pool,512]");
    TORCH_CHECK(route_pos.dim() == 2 && route_pos.size(1) == kTopK,
        "route_pos must have shape [N,2]");
    TORCH_CHECK(topk_score.dim() == 2 && topk_score.size(0) == route_pos.size(0) && topk_score.size(1) == kTopK,
        "topk_score must have shape [N,2]");

    const int64_t n_tokens = route_pos.size(0);
    check_n_tokens(n_tokens);
    TORCH_CHECK(residual.dim() == 2 && residual.size(0) == n_tokens && residual.size(1) == kHiddenDim,
        "residual must have shape [N,512]");

    auto out = torch::empty({n_tokens, kHiddenDim}, y_route.options());
    if (n_tokens == 0) {
        return out;
    }

    if (topk_score.scalar_type() == at::kHalf) {
        launch_route_reduce_with_residual<c10::Half>(y_route, route_pos, topk_score, residual, out, n_tokens);
    } else {
        launch_route_reduce_with_residual<float>(y_route, route_pos, topk_score, residual, out, n_tokens);
    }
    return out;
}

torch::Tensor smoe_forward(
    torch::Tensor x,
    torch::Tensor w1,
    torch::Tensor b1,
    torch::Tensor w2,
    torch::Tensor b2,
    torch::Tensor topk_idx,
    torch::Tensor topk_score
) {
    auto meta = build_route_metadata(x, topk_idx, topk_score);

    validate_fc1_inputs(meta.x_route, w1, b1, meta.counts, meta.offsets);

    auto h_route = torch::empty({meta.x_route.size(0), kFfDim}, x.options());
    validate_fc2_inputs(h_route, w2, b2, meta.counts, meta.offsets);

    auto y_route = torch::empty({meta.x_route.size(0), kHiddenDim}, x.options());
    auto out = torch::empty({x.size(0), kHiddenDim}, x.options());

    launch_grouped_linear<kHiddenDim, kFfDim, true>(
        meta.x_route,
        w1,
        b1,
        meta.counts,
        meta.offsets,
        h_route,
        meta.max_routes_per_expert
    );
    launch_grouped_linear<kFfDim, kHiddenDim, false>(
        h_route,
        w2,
        b2,
        meta.counts,
        meta.offsets,
        y_route,
        meta.max_routes_per_expert
    );

    if (x.size(0) == 0) {
        return out;
    }
    if (topk_score.scalar_type() == at::kHalf) {
        launch_route_reduce<c10::Half>(y_route, meta.route_pos, topk_score, out, x.size(0));
    } else {
        launch_route_reduce<float>(y_route, meta.route_pos, topk_score, out, x.size(0));
    }
    return out;
}

torch::Tensor smoe_forward_with_residual(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor w1,
    torch::Tensor b1,
    torch::Tensor w2,
    torch::Tensor b2,
    torch::Tensor topk_idx,
    torch::Tensor topk_score
) {
    check_half_cuda_contiguous(residual, "residual");
    check_same_device(x, residual, "x", "residual");
    TORCH_CHECK(residual.dim() == 2 && residual.size(0) == x.size(0) && residual.size(1) == kHiddenDim,
        "residual must have shape [N,512]");

    auto meta = build_route_metadata(x, topk_idx, topk_score);

    validate_fc1_inputs(meta.x_route, w1, b1, meta.counts, meta.offsets);

    auto h_route = torch::empty({meta.x_route.size(0), kFfDim}, x.options());
    validate_fc2_inputs(h_route, w2, b2, meta.counts, meta.offsets);

    auto y_route = torch::empty({meta.x_route.size(0), kHiddenDim}, x.options());
    auto out = torch::empty({x.size(0), kHiddenDim}, x.options());

    launch_grouped_linear<kHiddenDim, kFfDim, true>(
        meta.x_route,
        w1,
        b1,
        meta.counts,
        meta.offsets,
        h_route,
        meta.max_routes_per_expert
    );
    launch_grouped_linear<kFfDim, kHiddenDim, false>(
        h_route,
        w2,
        b2,
        meta.counts,
        meta.offsets,
        y_route,
        meta.max_routes_per_expert
    );

    if (x.size(0) == 0) {
        return out;
    }
    if (topk_score.scalar_type() == at::kHalf) {
        launch_route_reduce_with_residual<c10::Half>(
            y_route, meta.route_pos, topk_score, residual, out, x.size(0));
    } else {
        launch_route_reduce_with_residual<float>(
            y_route, meta.route_pos, topk_score, residual, out, x.size(0));
    }
    return out;
}

#if BAIDU_CTI_ENABLE_CUTLASS_SMOE
torch::Tensor smoe_forward_cutlass_fc2(
    torch::Tensor x,
    torch::Tensor w1,
    torch::Tensor b1,
    torch::Tensor w2,
    torch::Tensor b2,
    torch::Tensor topk_idx,
    torch::Tensor topk_score
) {
    auto meta = build_route_metadata(x, topk_idx, topk_score);

    validate_fc1_inputs(meta.x_route, w1, b1, meta.counts, meta.offsets);

    auto h_route = torch::empty({meta.x_route.size(0), kFfDim}, x.options());
    validate_fc2_inputs(h_route, w2, b2, meta.counts, meta.offsets);

    auto y_route = torch::empty({meta.x_route.size(0), kHiddenDim}, x.options());
    auto out = torch::empty({x.size(0), kHiddenDim}, x.options());

    launch_grouped_linear<kHiddenDim, kFfDim, true>(
        meta.x_route,
        w1,
        b1,
        meta.counts,
        meta.offsets,
        h_route,
        meta.max_routes_per_expert
    );
    launch_grouped_linear_cutlass_fc2(
        h_route,
        w2,
        b2,
        meta.counts,
        meta.offsets,
        y_route,
        meta.max_routes_per_expert
    );

    if (x.size(0) == 0) {
        return out;
    }
    if (topk_score.scalar_type() == at::kHalf) {
        launch_route_reduce<c10::Half>(y_route, meta.route_pos, topk_score, out, x.size(0));
    } else {
        launch_route_reduce<float>(y_route, meta.route_pos, topk_score, out, x.size(0));
    }
    return out;
}

torch::Tensor smoe_forward_cutlass_fc2_with_residual(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor w1,
    torch::Tensor b1,
    torch::Tensor w2,
    torch::Tensor b2,
    torch::Tensor topk_idx,
    torch::Tensor topk_score
) {
    check_half_cuda_contiguous(residual, "residual");
    check_same_device(x, residual, "x", "residual");
    TORCH_CHECK(residual.dim() == 2 && residual.size(0) == x.size(0) && residual.size(1) == kHiddenDim,
        "residual must have shape [N,512]");

    auto meta = build_route_metadata(x, topk_idx, topk_score);

    validate_fc1_inputs(meta.x_route, w1, b1, meta.counts, meta.offsets);

    auto h_route = torch::empty({meta.x_route.size(0), kFfDim}, x.options());
    validate_fc2_inputs(h_route, w2, b2, meta.counts, meta.offsets);

    auto y_route = torch::empty({meta.x_route.size(0), kHiddenDim}, x.options());
    auto out = torch::empty({x.size(0), kHiddenDim}, x.options());

    launch_grouped_linear<kHiddenDim, kFfDim, true>(
        meta.x_route,
        w1,
        b1,
        meta.counts,
        meta.offsets,
        h_route,
        meta.max_routes_per_expert
    );
    launch_grouped_linear_cutlass_fc2(
        h_route,
        w2,
        b2,
        meta.counts,
        meta.offsets,
        y_route,
        meta.max_routes_per_expert
    );

    if (x.size(0) == 0) {
        return out;
    }
    if (topk_score.scalar_type() == at::kHalf) {
        launch_route_reduce_with_residual<c10::Half>(
            y_route, meta.route_pos, topk_score, residual, out, x.size(0));
    } else {
        launch_route_reduce_with_residual<float>(
            y_route, meta.route_pos, topk_score, residual, out, x.size(0));
    }
    return out;
}
#endif

torch::Tensor smoe_forward_m64(
    torch::Tensor x,
    torch::Tensor w1,
    torch::Tensor b1,
    torch::Tensor w2,
    torch::Tensor b2,
    torch::Tensor topk_idx,
    torch::Tensor topk_score
) {
    auto meta = build_route_metadata(x, topk_idx, topk_score, kBlockM64);

    validate_fc1_inputs(meta.x_route, w1, b1, meta.counts, meta.offsets);

    auto h_route = torch::empty({meta.x_route.size(0), kFfDim}, x.options());
    validate_fc2_inputs(h_route, w2, b2, meta.counts, meta.offsets);

    auto y_route = torch::empty({meta.x_route.size(0), kHiddenDim}, x.options());
    auto out = torch::empty({x.size(0), kHiddenDim}, x.options());

    launch_grouped_linear_m64<kHiddenDim, kFfDim, true>(
        meta.x_route,
        w1,
        b1,
        meta.counts,
        meta.offsets,
        h_route,
        meta.max_routes_per_expert
    );
    launch_grouped_linear_m64<kFfDim, kHiddenDim, false>(
        h_route,
        w2,
        b2,
        meta.counts,
        meta.offsets,
        y_route,
        meta.max_routes_per_expert
    );

    if (x.size(0) == 0) {
        return out;
    }
    if (topk_score.scalar_type() == at::kHalf) {
        launch_route_reduce<c10::Half>(y_route, meta.route_pos, topk_score, out, x.size(0));
    } else {
        launch_route_reduce<float>(y_route, meta.route_pos, topk_score, out, x.size(0));
    }
    return out;
}

torch::Tensor smoe_forward_m64_with_residual(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor w1,
    torch::Tensor b1,
    torch::Tensor w2,
    torch::Tensor b2,
    torch::Tensor topk_idx,
    torch::Tensor topk_score
) {
    check_half_cuda_contiguous(residual, "residual");
    check_same_device(x, residual, "x", "residual");
    TORCH_CHECK(residual.dim() == 2 && residual.size(0) == x.size(0) && residual.size(1) == kHiddenDim,
        "residual must have shape [N,512]");

    auto meta = build_route_metadata(x, topk_idx, topk_score, kBlockM64);

    validate_fc1_inputs(meta.x_route, w1, b1, meta.counts, meta.offsets);

    auto h_route = torch::empty({meta.x_route.size(0), kFfDim}, x.options());
    validate_fc2_inputs(h_route, w2, b2, meta.counts, meta.offsets);

    auto y_route = torch::empty({meta.x_route.size(0), kHiddenDim}, x.options());
    auto out = torch::empty({x.size(0), kHiddenDim}, x.options());

    launch_grouped_linear_m64<kHiddenDim, kFfDim, true>(
        meta.x_route,
        w1,
        b1,
        meta.counts,
        meta.offsets,
        h_route,
        meta.max_routes_per_expert
    );
    launch_grouped_linear_m64<kFfDim, kHiddenDim, false>(
        h_route,
        w2,
        b2,
        meta.counts,
        meta.offsets,
        y_route,
        meta.max_routes_per_expert
    );

    if (x.size(0) == 0) {
        return out;
    }
    if (topk_score.scalar_type() == at::kHalf) {
        launch_route_reduce_with_residual<c10::Half>(
            y_route, meta.route_pos, topk_score, residual, out, x.size(0));
    } else {
        launch_route_reduce_with_residual<float>(
            y_route, meta.route_pos, topk_score, residual, out, x.size(0));
    }
    return out;
}

torch::Tensor smoe_forward_simple_w4a4_fc2(
    torch::Tensor x,
    torch::Tensor w1,
    torch::Tensor b1,
    torch::Tensor w2_pack,
    torch::Tensor b2,
    torch::Tensor topk_idx,
    torch::Tensor topk_score,
    double act_scale,
    double weight_scale,
    double fc2_output_scale
) {
    auto meta = build_route_metadata(x, topk_idx, topk_score);

    validate_fc1_inputs(meta.x_route, w1, b1, meta.counts, meta.offsets);

    auto h_pack = torch::empty({meta.x_route.size(0), kFfDim / 8}, x.options().dtype(torch::kInt32));
    auto y_route = torch::empty({meta.x_route.size(0), kHiddenDim}, x.options());
    auto out = torch::empty({x.size(0), kHiddenDim}, x.options());

    launch_grouped_linear_pack_simple_w4a4<kHiddenDim, kFfDim, true>(
        meta.x_route,
        w1,
        b1,
        meta.counts,
        meta.offsets,
        h_pack,
        meta.max_routes_per_expert,
        act_scale
    );
    launch_grouped_linear_simple_w4a4_packed_input<kFfDim, kHiddenDim, false>(
        h_pack,
        w2_pack,
        b2,
        meta.counts,
        meta.offsets,
        y_route,
        meta.max_routes_per_expert,
        act_scale,
        weight_scale,
        fc2_output_scale
    );

    if (x.size(0) == 0) {
        return out;
    }
    if (topk_score.scalar_type() == at::kHalf) {
        launch_route_reduce<c10::Half>(y_route, meta.route_pos, topk_score, out, x.size(0));
    } else {
        launch_route_reduce<float>(y_route, meta.route_pos, topk_score, out, x.size(0));
    }
    return out;
}

torch::Tensor smoe_forward_simple_w4a4_fc2_with_residual(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor w1,
    torch::Tensor b1,
    torch::Tensor w2_pack,
    torch::Tensor b2,
    torch::Tensor topk_idx,
    torch::Tensor topk_score,
    double act_scale,
    double weight_scale,
    double fc2_output_scale
) {
    check_half_cuda_contiguous(residual, "residual");
    check_same_device(x, residual, "x", "residual");
    TORCH_CHECK(residual.dim() == 2 && residual.size(0) == x.size(0) && residual.size(1) == kHiddenDim,
        "residual must have shape [N,512]");

    auto meta = build_route_metadata(x, topk_idx, topk_score);

    validate_fc1_inputs(meta.x_route, w1, b1, meta.counts, meta.offsets);

    auto h_pack = torch::empty({meta.x_route.size(0), kFfDim / 8}, x.options().dtype(torch::kInt32));
    auto y_route = torch::empty({meta.x_route.size(0), kHiddenDim}, x.options());
    auto out = torch::empty({x.size(0), kHiddenDim}, x.options());

    launch_grouped_linear_pack_simple_w4a4<kHiddenDim, kFfDim, true>(
        meta.x_route,
        w1,
        b1,
        meta.counts,
        meta.offsets,
        h_pack,
        meta.max_routes_per_expert,
        act_scale
    );
    launch_grouped_linear_simple_w4a4_packed_input<kFfDim, kHiddenDim, false>(
        h_pack,
        w2_pack,
        b2,
        meta.counts,
        meta.offsets,
        y_route,
        meta.max_routes_per_expert,
        act_scale,
        weight_scale,
        fc2_output_scale
    );

    if (x.size(0) == 0) {
        return out;
    }
    if (topk_score.scalar_type() == at::kHalf) {
        launch_route_reduce_with_residual<c10::Half>(
            y_route, meta.route_pos, topk_score, residual, out, x.size(0));
    } else {
        launch_route_reduce_with_residual<float>(
            y_route, meta.route_pos, topk_score, residual, out, x.size(0));
    }
    return out;
}

torch::Tensor smoe_forward_simple_w4a4_fc2_m64(
    torch::Tensor x,
    torch::Tensor w1,
    torch::Tensor b1,
    torch::Tensor w2_pack,
    torch::Tensor b2,
    torch::Tensor topk_idx,
    torch::Tensor topk_score,
    double act_scale,
    double weight_scale,
    double fc2_output_scale
) {
    auto meta = build_route_metadata(x, topk_idx, topk_score, kBlockM64);

    validate_fc1_inputs(meta.x_route, w1, b1, meta.counts, meta.offsets);

    auto h_pack = torch::empty({meta.x_route.size(0), kFfDim / 8}, x.options().dtype(torch::kInt32));
    auto y_route = torch::empty({meta.x_route.size(0), kHiddenDim}, x.options());
    auto out = torch::empty({x.size(0), kHiddenDim}, x.options());

    launch_grouped_linear_pack_simple_w4a4_m64<kHiddenDim, kFfDim, true>(
        meta.x_route,
        w1,
        b1,
        meta.counts,
        meta.offsets,
        h_pack,
        meta.max_routes_per_expert,
        act_scale
    );
    launch_grouped_linear_simple_w4a4_packed_input<kFfDim, kHiddenDim, false>(
        h_pack,
        w2_pack,
        b2,
        meta.counts,
        meta.offsets,
        y_route,
        meta.max_routes_per_expert,
        act_scale,
        weight_scale,
        fc2_output_scale
    );

    if (x.size(0) == 0) {
        return out;
    }
    if (topk_score.scalar_type() == at::kHalf) {
        launch_route_reduce<c10::Half>(y_route, meta.route_pos, topk_score, out, x.size(0));
    } else {
        launch_route_reduce<float>(y_route, meta.route_pos, topk_score, out, x.size(0));
    }
    return out;
}

torch::Tensor smoe_forward_simple_w4a4_fc2_m64_with_residual(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor w1,
    torch::Tensor b1,
    torch::Tensor w2_pack,
    torch::Tensor b2,
    torch::Tensor topk_idx,
    torch::Tensor topk_score,
    double act_scale,
    double weight_scale,
    double fc2_output_scale
) {
    check_half_cuda_contiguous(residual, "residual");
    check_same_device(x, residual, "x", "residual");
    TORCH_CHECK(residual.dim() == 2 && residual.size(0) == x.size(0) && residual.size(1) == kHiddenDim,
        "residual must have shape [N,512]");

    auto meta = build_route_metadata(x, topk_idx, topk_score, kBlockM64);

    validate_fc1_inputs(meta.x_route, w1, b1, meta.counts, meta.offsets);

    auto h_pack = torch::empty({meta.x_route.size(0), kFfDim / 8}, x.options().dtype(torch::kInt32));
    auto y_route = torch::empty({meta.x_route.size(0), kHiddenDim}, x.options());
    auto out = torch::empty({x.size(0), kHiddenDim}, x.options());

    launch_grouped_linear_pack_simple_w4a4_m64<kHiddenDim, kFfDim, true>(
        meta.x_route,
        w1,
        b1,
        meta.counts,
        meta.offsets,
        h_pack,
        meta.max_routes_per_expert,
        act_scale
    );
    launch_grouped_linear_simple_w4a4_packed_input<kFfDim, kHiddenDim, false>(
        h_pack,
        w2_pack,
        b2,
        meta.counts,
        meta.offsets,
        y_route,
        meta.max_routes_per_expert,
        act_scale,
        weight_scale,
        fc2_output_scale
    );

    if (x.size(0) == 0) {
        return out;
    }
    if (topk_score.scalar_type() == at::kHalf) {
        launch_route_reduce_with_residual<c10::Half>(
            y_route, meta.route_pos, topk_score, residual, out, x.size(0));
    } else {
        launch_route_reduce_with_residual<float>(
            y_route, meta.route_pos, topk_score, residual, out, x.size(0));
    }
    return out;
}

torch::Tensor smoe_forward_simple_w4a4(
    torch::Tensor x,
    torch::Tensor w1_pack,
    torch::Tensor b1,
    torch::Tensor w2_pack,
    torch::Tensor b2,
    torch::Tensor topk_idx,
    torch::Tensor topk_score,
    double fc1_act_scale,
    double fc1_weight_scale,
    double fc1_output_scale,
    double fc2_act_scale,
    double fc2_weight_scale,
    double fc2_output_scale
) {
    auto meta = build_route_metadata(x, topk_idx, topk_score);

    auto h_route = torch::empty({meta.x_route.size(0), kFfDim}, x.options());
    auto y_route = torch::empty({meta.x_route.size(0), kHiddenDim}, x.options());
    auto out = torch::empty({x.size(0), kHiddenDim}, x.options());

    launch_grouped_linear_simple_w4a4<kHiddenDim, kFfDim, true>(
        meta.x_route,
        w1_pack,
        b1,
        meta.counts,
        meta.offsets,
        h_route,
        meta.max_routes_per_expert,
        fc1_act_scale,
        fc1_weight_scale,
        fc1_output_scale
    );
    launch_grouped_linear_simple_w4a4<kFfDim, kHiddenDim, false>(
        h_route,
        w2_pack,
        b2,
        meta.counts,
        meta.offsets,
        y_route,
        meta.max_routes_per_expert,
        fc2_act_scale,
        fc2_weight_scale,
        fc2_output_scale
    );

    if (x.size(0) == 0) {
        return out;
    }
    if (topk_score.scalar_type() == at::kHalf) {
        launch_route_reduce<c10::Half>(y_route, meta.route_pos, topk_score, out, x.size(0));
    } else {
        launch_route_reduce<float>(y_route, meta.route_pos, topk_score, out, x.size(0));
    }
    return out;
}

torch::Tensor smoe_forward_simple_w4a4_with_residual(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor w1_pack,
    torch::Tensor b1,
    torch::Tensor w2_pack,
    torch::Tensor b2,
    torch::Tensor topk_idx,
    torch::Tensor topk_score,
    double fc1_act_scale,
    double fc1_weight_scale,
    double fc1_output_scale,
    double fc2_act_scale,
    double fc2_weight_scale,
    double fc2_output_scale
) {
    check_half_cuda_contiguous(residual, "residual");
    check_same_device(x, residual, "x", "residual");
    TORCH_CHECK(residual.dim() == 2 && residual.size(0) == x.size(0) && residual.size(1) == kHiddenDim,
        "residual must have shape [N,512]");

    auto meta = build_route_metadata(x, topk_idx, topk_score);

    auto h_route = torch::empty({meta.x_route.size(0), kFfDim}, x.options());
    auto y_route = torch::empty({meta.x_route.size(0), kHiddenDim}, x.options());
    auto out = torch::empty({x.size(0), kHiddenDim}, x.options());

    launch_grouped_linear_simple_w4a4<kHiddenDim, kFfDim, true>(
        meta.x_route,
        w1_pack,
        b1,
        meta.counts,
        meta.offsets,
        h_route,
        meta.max_routes_per_expert,
        fc1_act_scale,
        fc1_weight_scale,
        fc1_output_scale
    );
    launch_grouped_linear_simple_w4a4<kFfDim, kHiddenDim, false>(
        h_route,
        w2_pack,
        b2,
        meta.counts,
        meta.offsets,
        y_route,
        meta.max_routes_per_expert,
        fc2_act_scale,
        fc2_weight_scale,
        fc2_output_scale
    );

    if (x.size(0) == 0) {
        return out;
    }
    if (topk_score.scalar_type() == at::kHalf) {
        launch_route_reduce_with_residual<c10::Half>(
            y_route, meta.route_pos, topk_score, residual, out, x.size(0));
    } else {
        launch_route_reduce_with_residual<float>(
            y_route, meta.route_pos, topk_score, residual, out, x.size(0));
    }
    return out;
}

torch::Tensor smoe_forward_w4a16(
    torch::Tensor x,
    torch::Tensor w1_pack,
    torch::Tensor w1_scale,
    torch::Tensor w1_zero,
    torch::Tensor b1,
    torch::Tensor w2_pack,
    torch::Tensor w2_scale,
    torch::Tensor w2_zero,
    torch::Tensor b2,
    torch::Tensor topk_idx,
    torch::Tensor topk_score,
    int64_t group_size
) {
    validate_w4a16_inputs(
        x,
        w1_pack,
        w1_scale,
        w1_zero,
        b1,
        w2_pack,
        w2_scale,
        w2_zero,
        b2,
        topk_idx,
        topk_score,
        group_size
    );

    auto meta = build_route_metadata(x, topk_idx, topk_score);
    auto h_route = torch::empty({meta.x_route.size(0), kFfDim}, x.options());
    auto y_route = torch::empty({meta.x_route.size(0), kHiddenDim}, x.options());
    auto out = torch::empty({x.size(0), kHiddenDim}, x.options());

    launch_grouped_linear_w4a16<kHiddenDim, kFfDim, true>(
        meta.x_route,
        w1_pack,
        w1_scale,
        w1_zero,
        b1,
        meta.counts,
        meta.offsets,
        h_route,
        meta.max_routes_per_expert,
        group_size
    );
    launch_grouped_linear_w4a16<kFfDim, kHiddenDim, false>(
        h_route,
        w2_pack,
        w2_scale,
        w2_zero,
        b2,
        meta.counts,
        meta.offsets,
        y_route,
        meta.max_routes_per_expert,
        group_size
    );

    if (x.size(0) == 0) {
        return out;
    }
    if (topk_score.scalar_type() == at::kHalf) {
        launch_route_reduce<c10::Half>(y_route, meta.route_pos, topk_score, out, x.size(0));
    } else {
        launch_route_reduce<float>(y_route, meta.route_pos, topk_score, out, x.size(0));
    }
    return out;
}

torch::Tensor smoe_forward_w4a16_with_residual(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor w1_pack,
    torch::Tensor w1_scale,
    torch::Tensor w1_zero,
    torch::Tensor b1,
    torch::Tensor w2_pack,
    torch::Tensor w2_scale,
    torch::Tensor w2_zero,
    torch::Tensor b2,
    torch::Tensor topk_idx,
    torch::Tensor topk_score,
    int64_t group_size
) {
    validate_w4a16_inputs(
        x,
        w1_pack,
        w1_scale,
        w1_zero,
        b1,
        w2_pack,
        w2_scale,
        w2_zero,
        b2,
        topk_idx,
        topk_score,
        group_size
    );
    validate_w4a16_residual(x, residual);

    auto meta = build_route_metadata(x, topk_idx, topk_score);
    auto h_route = torch::empty({meta.x_route.size(0), kFfDim}, x.options());
    auto y_route = torch::empty({meta.x_route.size(0), kHiddenDim}, x.options());
    auto out = torch::empty({x.size(0), kHiddenDim}, x.options());

    launch_grouped_linear_w4a16<kHiddenDim, kFfDim, true>(
        meta.x_route,
        w1_pack,
        w1_scale,
        w1_zero,
        b1,
        meta.counts,
        meta.offsets,
        h_route,
        meta.max_routes_per_expert,
        group_size
    );
    launch_grouped_linear_w4a16<kFfDim, kHiddenDim, false>(
        h_route,
        w2_pack,
        w2_scale,
        w2_zero,
        b2,
        meta.counts,
        meta.offsets,
        y_route,
        meta.max_routes_per_expert,
        group_size
    );

    if (x.size(0) == 0) {
        return out;
    }
    if (topk_score.scalar_type() == at::kHalf) {
        launch_route_reduce_with_residual<c10::Half>(
            y_route, meta.route_pos, topk_score, residual, out, x.size(0));
    } else {
        launch_route_reduce_with_residual<float>(
            y_route, meta.route_pos, topk_score, residual, out, x.size(0));
    }
    return out;
}

torch::Tensor smoe_forward_w4a16_lop3(
    torch::Tensor x,
    torch::Tensor w1_pack,
    torch::Tensor w1_scale,
    torch::Tensor w1_zero,
    torch::Tensor b1,
    torch::Tensor w2_pack,
    torch::Tensor w2_scale,
    torch::Tensor w2_zero,
    torch::Tensor b2,
    torch::Tensor topk_idx,
    torch::Tensor topk_score,
    int64_t group_size
) {
    validate_w4a16_inputs(
        x,
        w1_pack,
        w1_scale,
        w1_zero,
        b1,
        w2_pack,
        w2_scale,
        w2_zero,
        b2,
        topk_idx,
        topk_score,
        group_size
    );

    auto meta = build_route_metadata(x, topk_idx, topk_score);
    auto h_route = torch::empty({meta.x_route.size(0), kFfDim}, x.options());
    auto y_route = torch::empty({meta.x_route.size(0), kHiddenDim}, x.options());
    auto out = torch::empty({x.size(0), kHiddenDim}, x.options());

    launch_grouped_linear_w4a16_lop3<kHiddenDim, kFfDim, true>(
        meta.x_route,
        w1_pack,
        w1_scale,
        w1_zero,
        b1,
        meta.counts,
        meta.offsets,
        h_route,
        meta.max_routes_per_expert,
        group_size
    );
    launch_grouped_linear_w4a16_lop3<kFfDim, kHiddenDim, false>(
        h_route,
        w2_pack,
        w2_scale,
        w2_zero,
        b2,
        meta.counts,
        meta.offsets,
        y_route,
        meta.max_routes_per_expert,
        group_size
    );

    if (x.size(0) == 0) {
        return out;
    }
    if (topk_score.scalar_type() == at::kHalf) {
        launch_route_reduce<c10::Half>(y_route, meta.route_pos, topk_score, out, x.size(0));
    } else {
        launch_route_reduce<float>(y_route, meta.route_pos, topk_score, out, x.size(0));
    }
    return out;
}

torch::Tensor smoe_forward_w4a16_lop3_with_residual(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor w1_pack,
    torch::Tensor w1_scale,
    torch::Tensor w1_zero,
    torch::Tensor b1,
    torch::Tensor w2_pack,
    torch::Tensor w2_scale,
    torch::Tensor w2_zero,
    torch::Tensor b2,
    torch::Tensor topk_idx,
    torch::Tensor topk_score,
    int64_t group_size
) {
    validate_w4a16_inputs(
        x,
        w1_pack,
        w1_scale,
        w1_zero,
        b1,
        w2_pack,
        w2_scale,
        w2_zero,
        b2,
        topk_idx,
        topk_score,
        group_size
    );
    validate_w4a16_residual(x, residual);

    auto meta = build_route_metadata(x, topk_idx, topk_score);
    auto h_route = torch::empty({meta.x_route.size(0), kFfDim}, x.options());
    auto y_route = torch::empty({meta.x_route.size(0), kHiddenDim}, x.options());
    auto out = torch::empty({x.size(0), kHiddenDim}, x.options());

    launch_grouped_linear_w4a16_lop3<kHiddenDim, kFfDim, true>(
        meta.x_route,
        w1_pack,
        w1_scale,
        w1_zero,
        b1,
        meta.counts,
        meta.offsets,
        h_route,
        meta.max_routes_per_expert,
        group_size
    );
    launch_grouped_linear_w4a16_lop3<kFfDim, kHiddenDim, false>(
        h_route,
        w2_pack,
        w2_scale,
        w2_zero,
        b2,
        meta.counts,
        meta.offsets,
        y_route,
        meta.max_routes_per_expert,
        group_size
    );

    if (x.size(0) == 0) {
        return out;
    }
    if (topk_score.scalar_type() == at::kHalf) {
        launch_route_reduce_with_residual<c10::Half>(
            y_route, meta.route_pos, topk_score, residual, out, x.size(0));
    } else {
        launch_route_reduce_with_residual<float>(
            y_route, meta.route_pos, topk_score, residual, out, x.size(0));
    }
    return out;
}

torch::Tensor smoe_forward_w4a16_simt_fc2(
    torch::Tensor x,
    torch::Tensor w1_pack,
    torch::Tensor w1_scale,
    torch::Tensor w1_zero,
    torch::Tensor b1,
    torch::Tensor w2_pack,
    torch::Tensor w2_scale,
    torch::Tensor w2_zero,
    torch::Tensor b2,
    torch::Tensor topk_idx,
    torch::Tensor topk_score,
    int64_t group_size
) {
    validate_w4a16_inputs(
        x,
        w1_pack,
        w1_scale,
        w1_zero,
        b1,
        w2_pack,
        w2_scale,
        w2_zero,
        b2,
        topk_idx,
        topk_score,
        group_size
    );

    auto meta = build_route_metadata(x, topk_idx, topk_score);
    auto h_route = torch::empty({meta.x_route.size(0), kFfDim}, x.options());
    auto y_route = torch::empty({meta.x_route.size(0), kHiddenDim}, x.options());
    auto out = torch::empty({x.size(0), kHiddenDim}, x.options());

    launch_grouped_linear_w4a16<kHiddenDim, kFfDim, true>(
        meta.x_route,
        w1_pack,
        w1_scale,
        w1_zero,
        b1,
        meta.counts,
        meta.offsets,
        h_route,
        meta.max_routes_per_expert,
        group_size
    );
    launch_grouped_linear_w4a16_simt_fc2(
        h_route,
        w2_pack,
        w2_scale,
        w2_zero,
        b2,
        meta.counts,
        meta.offsets,
        y_route,
        meta.max_routes_per_expert,
        group_size
    );

    if (x.size(0) == 0) {
        return out;
    }
    if (topk_score.scalar_type() == at::kHalf) {
        launch_route_reduce<c10::Half>(y_route, meta.route_pos, topk_score, out, x.size(0));
    } else {
        launch_route_reduce<float>(y_route, meta.route_pos, topk_score, out, x.size(0));
    }
    return out;
}

torch::Tensor smoe_forward_w4a16_simt_fc2_with_residual(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor w1_pack,
    torch::Tensor w1_scale,
    torch::Tensor w1_zero,
    torch::Tensor b1,
    torch::Tensor w2_pack,
    torch::Tensor w2_scale,
    torch::Tensor w2_zero,
    torch::Tensor b2,
    torch::Tensor topk_idx,
    torch::Tensor topk_score,
    int64_t group_size
) {
    validate_w4a16_inputs(
        x,
        w1_pack,
        w1_scale,
        w1_zero,
        b1,
        w2_pack,
        w2_scale,
        w2_zero,
        b2,
        topk_idx,
        topk_score,
        group_size
    );
    validate_w4a16_residual(x, residual);

    auto meta = build_route_metadata(x, topk_idx, topk_score);
    auto h_route = torch::empty({meta.x_route.size(0), kFfDim}, x.options());
    auto y_route = torch::empty({meta.x_route.size(0), kHiddenDim}, x.options());
    auto out = torch::empty({x.size(0), kHiddenDim}, x.options());

    launch_grouped_linear_w4a16<kHiddenDim, kFfDim, true>(
        meta.x_route,
        w1_pack,
        w1_scale,
        w1_zero,
        b1,
        meta.counts,
        meta.offsets,
        h_route,
        meta.max_routes_per_expert,
        group_size
    );
    launch_grouped_linear_w4a16_simt_fc2(
        h_route,
        w2_pack,
        w2_scale,
        w2_zero,
        b2,
        meta.counts,
        meta.offsets,
        y_route,
        meta.max_routes_per_expert,
        group_size
    );

    if (x.size(0) == 0) {
        return out;
    }
    if (topk_score.scalar_type() == at::kHalf) {
        launch_route_reduce_with_residual<c10::Half>(
            y_route, meta.route_pos, topk_score, residual, out, x.size(0));
    } else {
        launch_route_reduce_with_residual<float>(
            y_route, meta.route_pos, topk_score, residual, out, x.size(0));
    }
    return out;
}

torch::Tensor smoe_forward_w4a16_frag(
    torch::Tensor x,
    torch::Tensor w1_pack,
    torch::Tensor w1_scale,
    torch::Tensor w1_zero,
    torch::Tensor b1,
    torch::Tensor w2_pack,
    torch::Tensor w2_scale,
    torch::Tensor w2_zero,
    torch::Tensor b2,
    torch::Tensor topk_idx,
    torch::Tensor topk_score,
    int64_t group_size
) {
    validate_w4a16_inputs(
        x,
        w1_pack,
        w1_scale,
        w1_zero,
        b1,
        w2_pack,
        w2_scale,
        w2_zero,
        b2,
        topk_idx,
        topk_score,
        group_size
    );

    auto meta = build_route_metadata(x, topk_idx, topk_score);
    auto h_route = torch::empty({meta.x_route.size(0), kFfDim}, x.options());
    auto y_route = torch::empty({meta.x_route.size(0), kHiddenDim}, x.options());
    auto out = torch::empty({x.size(0), kHiddenDim}, x.options());

    launch_grouped_linear_w4a16_frag<kHiddenDim, kFfDim, true>(
        meta.x_route,
        w1_pack,
        w1_scale,
        w1_zero,
        b1,
        meta.counts,
        meta.offsets,
        h_route,
        meta.max_routes_per_expert,
        group_size
    );
    launch_grouped_linear_w4a16_frag<kFfDim, kHiddenDim, false>(
        h_route,
        w2_pack,
        w2_scale,
        w2_zero,
        b2,
        meta.counts,
        meta.offsets,
        y_route,
        meta.max_routes_per_expert,
        group_size
    );

    if (x.size(0) == 0) {
        return out;
    }
    if (topk_score.scalar_type() == at::kHalf) {
        launch_route_reduce<c10::Half>(y_route, meta.route_pos, topk_score, out, x.size(0));
    } else {
        launch_route_reduce<float>(y_route, meta.route_pos, topk_score, out, x.size(0));
    }
    return out;
}

torch::Tensor smoe_forward_w4a16_frag_with_residual(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor w1_pack,
    torch::Tensor w1_scale,
    torch::Tensor w1_zero,
    torch::Tensor b1,
    torch::Tensor w2_pack,
    torch::Tensor w2_scale,
    torch::Tensor w2_zero,
    torch::Tensor b2,
    torch::Tensor topk_idx,
    torch::Tensor topk_score,
    int64_t group_size
) {
    validate_w4a16_inputs(
        x,
        w1_pack,
        w1_scale,
        w1_zero,
        b1,
        w2_pack,
        w2_scale,
        w2_zero,
        b2,
        topk_idx,
        topk_score,
        group_size
    );
    validate_w4a16_residual(x, residual);

    auto meta = build_route_metadata(x, topk_idx, topk_score);
    auto h_route = torch::empty({meta.x_route.size(0), kFfDim}, x.options());
    auto y_route = torch::empty({meta.x_route.size(0), kHiddenDim}, x.options());
    auto out = torch::empty({x.size(0), kHiddenDim}, x.options());

    launch_grouped_linear_w4a16_frag<kHiddenDim, kFfDim, true>(
        meta.x_route,
        w1_pack,
        w1_scale,
        w1_zero,
        b1,
        meta.counts,
        meta.offsets,
        h_route,
        meta.max_routes_per_expert,
        group_size
    );
    launch_grouped_linear_w4a16_frag<kFfDim, kHiddenDim, false>(
        h_route,
        w2_pack,
        w2_scale,
        w2_zero,
        b2,
        meta.counts,
        meta.offsets,
        y_route,
        meta.max_routes_per_expert,
        group_size
    );

    if (x.size(0) == 0) {
        return out;
    }
    if (topk_score.scalar_type() == at::kHalf) {
        launch_route_reduce_with_residual<c10::Half>(
            y_route, meta.route_pos, topk_score, residual, out, x.size(0));
    } else {
        launch_route_reduce_with_residual<float>(
            y_route, meta.route_pos, topk_score, residual, out, x.size(0));
    }
    return out;
}

torch::Tensor debug_w4a16_bfrag_fc1(
    torch::Tensor w1_pack,
    torch::Tensor w1_scale,
    torch::Tensor w1_zero,
    int64_t group_size,
    int64_t expert,
    int64_t n_tile_base,
    int64_t k_base
) {
    validate_w4a16_debug_weight<kHiddenDim, kFfDim>(
        w1_pack,
        w1_scale,
        w1_zero,
        group_size,
        expert,
        n_tile_base,
        k_base
    );

    auto dump = torch::empty(
        {kThreads / kWarpSize, kWarpSize, kWarpTileN, 4},
        w1_pack.options().dtype(torch::kInt32)
    );
    launch_debug_w4a16_bfrag_layout<kHiddenDim, kFfDim>(
        w1_pack,
        w1_scale,
        w1_zero,
        dump,
        group_size,
        expert,
        n_tile_base,
        k_base
    );
    return dump;
}

torch::Tensor debug_w4a16_bfrag_fc2(
    torch::Tensor w2_pack,
    torch::Tensor w2_scale,
    torch::Tensor w2_zero,
    int64_t group_size,
    int64_t expert,
    int64_t n_tile_base,
    int64_t k_base
) {
    validate_w4a16_debug_weight<kFfDim, kHiddenDim>(
        w2_pack,
        w2_scale,
        w2_zero,
        group_size,
        expert,
        n_tile_base,
        k_base
    );

    auto dump = torch::empty(
        {kThreads / kWarpSize, kWarpSize, kWarpTileN, 4},
        w2_pack.options().dtype(torch::kInt32)
    );
    launch_debug_w4a16_bfrag_layout<kFfDim, kHiddenDim>(
        w2_pack,
        w2_scale,
        w2_zero,
        dump,
        group_size,
        expert,
        n_tile_base,
        k_base
    );
    return dump;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("smoe_route_count", &smoe_route_count, "Routed SMoE route count");
    m.def("smoe_route_pack", &smoe_route_pack, "Routed SMoE route pack");
    m.def("smoe_grouped_fc1_relu", &smoe_grouped_fc1_relu, "Routed SMoE grouped fc1 + bias + ReLU");
    m.def("smoe_grouped_fc2", &smoe_grouped_fc2, "Routed SMoE grouped fc2 + bias");
    m.def("smoe_route_reduce", &smoe_route_reduce, "Routed SMoE explicit top-2 reduce");
    m.def("smoe_route_reduce_with_residual", &smoe_route_reduce_with_residual,
        "Routed SMoE explicit top-2 reduce fused with residual add");
    m.def("smoe_forward", &smoe_forward, "Routed sparse SMoE forward");
    m.def("smoe_forward_with_residual", &smoe_forward_with_residual,
        "Routed sparse SMoE forward fused with residual add");
#if BAIDU_CTI_ENABLE_CUTLASS_SMOE
    m.def("smoe_forward_cutlass_fc2", &smoe_forward_cutlass_fc2,
        "Routed sparse SMoE forward with CUTLASS grouped GEMM fc2");
    m.def("smoe_forward_cutlass_fc2_with_residual", &smoe_forward_cutlass_fc2_with_residual,
        "Routed sparse SMoE forward with CUTLASS grouped GEMM fc2 fused with residual add");
#endif
    m.def("smoe_forward_m64", &smoe_forward_m64,
        "Routed sparse SMoE forward with M64 grouped GEMM tiles");
    m.def("smoe_forward_m64_with_residual", &smoe_forward_m64_with_residual,
        "Routed sparse SMoE forward with M64 grouped GEMM tiles fused with residual add");
    m.def("smoe_forward_simple_w4a4_fc2", &smoe_forward_simple_w4a4_fc2,
        "Routed sparse SMoE forward with simple uniform-scale W4A4 fc2");
    m.def("smoe_forward_simple_w4a4_fc2_with_residual", &smoe_forward_simple_w4a4_fc2_with_residual,
        "Routed sparse SMoE forward with simple uniform-scale W4A4 fc2 fused with residual add");
    m.def("smoe_forward_simple_w4a4_fc2_m64", &smoe_forward_simple_w4a4_fc2_m64,
        "Routed sparse SMoE M64 forward with simple uniform-scale W4A4 fc2");
    m.def("smoe_forward_simple_w4a4_fc2_m64_with_residual", &smoe_forward_simple_w4a4_fc2_m64_with_residual,
        "Routed sparse SMoE M64 forward with simple uniform-scale W4A4 fc2 fused with residual add");
    m.def("smoe_forward_simple_w4a4", &smoe_forward_simple_w4a4,
        "Routed sparse SMoE forward with simple uniform-scale W4A4 fc1/fc2");
    m.def("smoe_forward_simple_w4a4_with_residual", &smoe_forward_simple_w4a4_with_residual,
        "Routed sparse SMoE forward with simple uniform-scale W4A4 fc1/fc2 fused with residual add");
    m.def("smoe_forward_w4a16", &smoe_forward_w4a16,
        "Routed sparse SMoE W4A16 forward");
    m.def("smoe_forward_w4a16_with_residual", &smoe_forward_w4a16_with_residual,
        "Routed sparse SMoE W4A16 forward fused with residual add");
    m.def("smoe_forward_w4a16_lop3", &smoe_forward_w4a16_lop3,
        "Routed sparse SMoE W4A16 forward with BitDecoding-style LOP3 dequant");
    m.def("smoe_forward_w4a16_lop3_with_residual", &smoe_forward_w4a16_lop3_with_residual,
        "Routed sparse SMoE W4A16 forward with BitDecoding-style LOP3 dequant fused with residual add");
    m.def("smoe_forward_w4a16_simt_fc2", &smoe_forward_w4a16_simt_fc2,
        "Routed sparse SMoE W4A16 forward with CUDA-core SIMT fc2");
    m.def("smoe_forward_w4a16_simt_fc2_with_residual", &smoe_forward_w4a16_simt_fc2_with_residual,
        "Routed sparse SMoE W4A16 forward with CUDA-core SIMT fc2 fused with residual add");
    m.def("smoe_forward_w4a16_frag", &smoe_forward_w4a16_frag,
        "Routed sparse SMoE W4A16 forward with direct B-fragment dequant");
    m.def("smoe_forward_w4a16_frag_with_residual", &smoe_forward_w4a16_frag_with_residual,
        "Routed sparse SMoE W4A16 forward with direct B-fragment dequant fused with residual add");
    m.def("debug_w4a16_bfrag_fc1", &debug_w4a16_bfrag_fc1,
        "Dump W4A16 fc1 B-fragment registers: [warp, lane, j, ref0/ref1/direct0/direct1]");
    m.def("debug_w4a16_bfrag_fc2", &debug_w4a16_bfrag_fc2,
        "Dump W4A16 fc2 B-fragment registers: [warp, lane, j, ref0/ref1/direct0/direct1]");
}
