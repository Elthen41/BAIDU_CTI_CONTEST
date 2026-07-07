// CUDA fp16 GEMM kernels for fixed-shape Linear layers.
//
// First target:
//   qkv_proj_fused_layout:
//     x      [1,S,512] or [S,512]
//     weight [1536,512]  PyTorch Linear weight, row-major [out,in]
//     bias   [1536]
//     -> q,k,v [1,8,S,64]
//
// The kernel follows the LeetCUDA SM80 HGEMM TN structure:
// 128x128 CTA tile, 8 warps, 2-stage cp.async pipeline,
// ldmatrix + mma.sync.m16n8k16, fp32 accumulate.

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

constexpr int kHiddenDim = 512;
constexpr int kQkvDim = 1536;
constexpr int kHeads = 8;
constexpr int kHeadDim = 64;
constexpr int kQkvPerHead = 3 * kHeadDim;

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

__device__ __forceinline__ c10::Half* qkv_output_ptr(
    c10::Half* __restrict__ q,
    c10::Half* __restrict__ k,
    c10::Half* __restrict__ v,
    int qkv_part
) {
    if (qkv_part == 0) {
        return q;
    }
    if (qkv_part == 1) {
        return k;
    }
    return v;
}

__device__ __forceinline__ void load_qkv_tile_scalar(
    const c10::Half* __restrict__ x,
    const c10::Half* __restrict__ weight,
    c10::Half* __restrict__ a_shared,
    c10::Half* __restrict__ b_shared,
    int64_t n_tokens,
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
            val = x[static_cast<int64_t>(global_m) * kHiddenDim + global_k];
        }
        a_shared[idx] = val;
    }

    for (int idx = tid; idx < kBlockN * kMmaK; idx += kThreads) {
        const int local_n = idx / kMmaK;
        const int local_k = idx - local_n * kMmaK;
        const int global_n = n_tile_base + local_n;
        const int global_k = k_base + local_k;
        b_shared[idx] = weight[static_cast<int64_t>(global_n) * kHiddenDim + global_k];
    }
}

__device__ __forceinline__ void load_qkv_tile_cp_async(
    const c10::Half* __restrict__ x,
    const c10::Half* __restrict__ weight,
    c10::Half* __restrict__ a_shared,
    c10::Half* __restrict__ b_shared,
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

    const uint32_t a_addr = shared_addr(a_shared + local_row * kMmaK + local_k);
    const uint32_t b_addr = shared_addr(b_shared + local_row * kMmaK + local_k);
    cp_async_cg_16(a_addr, x + static_cast<int64_t>(global_m) * kHiddenDim + global_k);
    cp_async_cg_16(b_addr, weight + static_cast<int64_t>(global_n) * kHiddenDim + global_k);
}

__device__ __forceinline__ void compute_qkv_mma_stage(
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

__global__ __launch_bounds__(kThreads) void qkv_proj_fused_layout_kernel(
    const c10::Half* __restrict__ x,
    const c10::Half* __restrict__ weight,
    const c10::Half* __restrict__ bias,
    c10::Half* __restrict__ q,
    c10::Half* __restrict__ k,
    c10::Half* __restrict__ v,
    int64_t n_tokens
) {
    __shared__ __align__(16) c10::Half a_shared[kPipelineStages * kBlockM * kMmaK];
    __shared__ __align__(16) c10::Half b_shared[kPipelineStages * kBlockN * kMmaK];

    const int tid = threadIdx.x;
    const int warp_id = tid / kWarpSize;
    const int lane = tid & (kWarpSize - 1);
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
    constexpr int kNumKTiles = kHiddenDim / kMmaK;
    constexpr int kStageStrideA = kBlockM * kMmaK;
    constexpr int kStageStrideB = kBlockN * kMmaK;

    if (full_m_tile) {
        load_qkv_tile_cp_async(
            x,
            weight,
            a_shared,
            b_shared,
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

            load_qkv_tile_cp_async(
                x,
                weight,
                a_shared + load_stage * kStageStrideA,
                b_shared + load_stage * kStageStrideB,
                m_tile_base,
                n_tile_base,
                k_tile * kMmaK,
                tid
            );
            cp_async_commit_group();

            compute_qkv_mma_stage(
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
        compute_qkv_mma_stage(
            a_shared + last_stage * kStageStrideA,
            b_shared + last_stage * kStageStrideB,
            warp_m,
            warp_n,
            lane,
            acc
        );
    } else {
        for (int k_base = 0; k_base < kHiddenDim; k_base += kMmaK) {
            load_qkv_tile_scalar(
                x,
                weight,
                a_shared,
                b_shared,
                n_tokens,
                m_tile_base,
                n_tile_base,
                k_base,
                tid
            );
            __syncthreads();

            compute_qkv_mma_stage(
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
            const int col_group = col_base + j * kMmaN;

#pragma unroll
            for (int row_slot = 0; row_slot < 2; ++row_slot) {
                const int row = rows[row_slot];
                const int col = col0;
                float value0 = reg_as_float(acc[i][j][row_slot * 2 + 0])
                    + static_cast<float>(bias[col + 0]);
                float value1 = reg_as_float(acc[i][j][row_slot * 2 + 1])
                    + static_cast<float>(bias[col + 1]);

                const uint32_t packed = pack_half2_bits(value0, value1);
                uint4 vec;
                vec.x = __shfl_sync(0xffffffff, packed, lane_group_base + 0);
                vec.y = __shfl_sync(0xffffffff, packed, lane_group_base + 1);
                vec.z = __shfl_sync(0xffffffff, packed, lane_group_base + 2);
                vec.w = __shfl_sync(0xffffffff, packed, lane_group_base + 3);

                if ((lane & 3) == 0 && row < n_tokens) {
                    const int head = col_group / kQkvPerHead;
                    const int rem = col_group - head * kQkvPerHead;
                    const int qkv_part = rem / kHeadDim;
                    const int head_dim_base = rem - qkv_part * kHeadDim;
                    c10::Half* out = qkv_output_ptr(q, k, v, qkv_part);
                    const int64_t out_idx =
                        (static_cast<int64_t>(head) * n_tokens + row) * kHeadDim
                        + head_dim_base;
                    *reinterpret_cast<uint4*>(out + out_idx) = vec;
                }
            }
        }
    }
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

int64_t check_qkv_x_shape(const torch::Tensor& x) {
    TORCH_CHECK(x.dim() == 2 || x.dim() == 3, "x must have shape [S,512] or [1,S,512]");
    if (x.dim() == 2) {
        TORCH_CHECK(x.size(1) == kHiddenDim, "x must have shape [S,512]");
        return x.size(0);
    }

    TORCH_CHECK(x.size(0) == 1, "x must have shape [1,S,512]");
    TORCH_CHECK(x.size(2) == kHiddenDim, "x must have shape [1,S,512]");
    return x.size(1);
}

}  // namespace

std::vector<torch::Tensor> qkv_proj_fused_layout(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias
) {
    check_half_cuda_contiguous(x, "x");
    check_half_cuda_contiguous(weight, "weight");
    check_half_cuda_contiguous(bias, "bias");
    check_same_device(x, weight, "x", "weight");
    check_same_device(x, bias, "x", "bias");

    const int64_t n_tokens = check_qkv_x_shape(x);
    TORCH_CHECK(weight.dim() == 2, "weight must have shape [1536,512]");
    TORCH_CHECK(weight.size(0) == kQkvDim, "weight must have shape [1536,512]");
    TORCH_CHECK(weight.size(1) == kHiddenDim, "weight must have shape [1536,512]");
    TORCH_CHECK(bias.dim() == 1, "bias must have shape [1536]");
    TORCH_CHECK(bias.size(0) == kQkvDim, "bias must have shape [1536]");
    TORCH_CHECK(n_tokens >= 0, "n_tokens must be non-negative");
    TORCH_CHECK(
        n_tokens <= static_cast<int64_t>(std::numeric_limits<int>::max()),
        "n_tokens is too large"
    );
    TORCH_CHECK(
        ceil_div_int64(n_tokens, kBlockM) <= 65535,
        "n_tokens creates too many CTA rows"
    );

    auto q = torch::empty({1, kHeads, n_tokens, kHeadDim}, x.options());
    auto k = torch::empty({1, kHeads, n_tokens, kHeadDim}, x.options());
    auto v = torch::empty({1, kHeads, n_tokens, kHeadDim}, x.options());

    if (n_tokens == 0) {
        return {q, k, v};
    }

    const dim3 block(kThreads);
    const dim3 grid(
        static_cast<unsigned int>(kQkvDim / kBlockN),
        static_cast<unsigned int>(ceil_div_int64(n_tokens, kBlockM))
    );

    qkv_proj_fused_layout_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<c10::Half>(),
        weight.data_ptr<c10::Half>(),
        bias.data_ptr<c10::Half>(),
        q.data_ptr<c10::Half>(),
        k.data_ptr<c10::Half>(),
        v.data_ptr<c10::Half>(),
        n_tokens
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {q, k, v};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "qkv_proj_fused_layout",
        &qkv_proj_fused_layout,
        "fp16 QKV projection fused with [1,8,S,64] layout write"
    );
}
