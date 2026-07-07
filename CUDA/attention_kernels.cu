// CUDA variable-length causal attention kernels.
//
// Correctness-first implementation for the current Transformer path:
// q/k/v: [B=1, heads=8, S, head_dim=64], contiguous float16.
// user_offsets: [U + 1], int64 CUDA tensor.
// The exported varlen_causal_attention entry point keeps the original
// one-query-per-block implementation as a stable fallback. The tiled entry
// point below is the first FlashAttention-style step: one CTA handles several
// query tokens and reuses K/V tiles through shared memory, with fp32 online
// softmax state.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cmath>

namespace {

constexpr int kHeadDim = 64;
constexpr int kQkvParts = 3;
constexpr int kWarpSize = 32;
constexpr int kHalfPer128Bits = 8;
constexpr int kHeadDimVec128 = kHeadDim / kHalfPer128Bits;
constexpr float kScale = 0.125f;  // 1 / sqrt(64)
constexpr float kNegInf = -3.4028234663852886e38f;

#ifndef USE_FAST_ATTENTION_EXP
#define USE_FAST_ATTENTION_EXP 0
#endif

static_assert(kHeadDim % kHalfPer128Bits == 0, "head_dim must be divisible by 128-bit half vector width");

__device__ __forceinline__ float attention_exp(float x) {
#if USE_FAST_ATTENTION_EXP
    return __expf(x);
#else
    return expf(x);
#endif
}

__device__ __forceinline__ void copy_half8_128b(
    c10::Half* __restrict__ dst,
    const c10::Half* __restrict__ src
) {
    *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
}

__device__ __forceinline__ void zero_half8_128b(c10::Half* __restrict__ dst) {
    *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
}

__device__ __forceinline__ uint32_t shared_addr(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void cp_async_cg_16(void* smem_dst, const void* gmem_src) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile(
        "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
        :
        : "r"(shared_addr(smem_dst)), "l"(gmem_src)
    );
#else
    copy_half8_128b(
        reinterpret_cast<c10::Half*>(smem_dst),
        reinterpret_cast<const c10::Half*>(gmem_src)
    );
#endif
}

__device__ __forceinline__ void cp_async_commit_group() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template <int PendingGroups>
__device__ __forceinline__ void cp_async_wait_group() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("cp.async.wait_group %0;\n" :: "n"(PendingGroups));
#endif
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

__device__ __forceinline__ void ldmatrix_x2_trans(
    uint32_t& r0,
    uint32_t& r1,
    uint32_t addr
) {
    asm volatile(
        "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"
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

__device__ __forceinline__ int64_t interleaved_qkv_offset(
    int head,
    int heads,
    int token,
    int qkv_part,
    int dim
) {
    return (
        (static_cast<int64_t>(token) * heads + head) * kQkvParts * kHeadDim
        + static_cast<int64_t>(qkv_part) * kHeadDim
        + dim
    );
}

template <int Bc, int NumThreads, int KSmemHeadStride, int VSmemHeadStride>
__device__ __forceinline__ void load_kv_tile_async(
    const c10::Half* __restrict__ k,
    const c10::Half* __restrict__ v,
    c10::Half* __restrict__ k_shared,
    c10::Half* __restrict__ v_shared,
    int64_t head_base,
    int k_tile_start,
    int k_tile_len,
    int tid
) {
    for (int vec = tid; vec < Bc * kHeadDimVec128; vec += NumThreads) {
        const int key_row = vec / kHeadDimVec128;
        const int vec_col = vec - key_row * kHeadDimVec128;
        const int dim = vec_col * kHalfPer128Bits;
        c10::Half* k_dst = k_shared + key_row * KSmemHeadStride + dim;
        c10::Half* v_dst = v_shared + key_row * VSmemHeadStride + dim;
        if (key_row < k_tile_len) {
            const int key = k_tile_start + key_row;
            const int64_t src = head_base + static_cast<int64_t>(key) * kHeadDim + dim;
            cp_async_cg_16(k_dst, k + src);
            cp_async_cg_16(v_dst, v + src);
        } else {
            zero_half8_128b(k_dst);
            zero_half8_128b(v_dst);
        }
    }
    cp_async_commit_group();
}

template <int Bc, int NumThreads, int SmemHeadStride, int QkvPart>
__device__ __forceinline__ void load_interleaved_qkv_tile_async(
    const c10::Half* __restrict__ qkv,
    c10::Half* __restrict__ dst_shared,
    int head,
    int heads,
    int tile_start,
    int tile_len,
    int tid
) {
    static_assert(QkvPart >= 0 && QkvPart < kQkvParts, "invalid qkv part");
    for (int vec = tid; vec < Bc * kHeadDimVec128; vec += NumThreads) {
        const int row = vec / kHeadDimVec128;
        const int vec_col = vec - row * kHeadDimVec128;
        const int dim = vec_col * kHalfPer128Bits;
        c10::Half* dst = dst_shared + row * SmemHeadStride + dim;
        if (row < tile_len) {
            const int token = tile_start + row;
            const int64_t src = interleaved_qkv_offset(head, heads, token, QkvPart, dim);
            cp_async_cg_16(dst, qkv + src);
        } else {
            zero_half8_128b(dst);
        }
    }
    cp_async_commit_group();
}

template <int Bc, int NumThreads, int SmemHeadStride>
__device__ __forceinline__ void load_tensor_tile_async(
    const c10::Half* __restrict__ src_tensor,
    c10::Half* __restrict__ dst_shared,
    int64_t head_base,
    int tile_start,
    int tile_len,
    int tid
) {
    for (int vec = tid; vec < Bc * kHeadDimVec128; vec += NumThreads) {
        const int row = vec / kHeadDimVec128;
        const int vec_col = vec - row * kHeadDimVec128;
        const int dim = vec_col * kHalfPer128Bits;
        c10::Half* dst = dst_shared + row * SmemHeadStride + dim;
        if (row < tile_len) {
            const int token = tile_start + row;
            const int64_t src = head_base + static_cast<int64_t>(token) * kHeadDim + dim;
            cp_async_cg_16(dst, src_tensor + src);
        } else {
            zero_half8_128b(dst);
        }
    }
    cp_async_commit_group();
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int mask = kWarpSize >> 1; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template <int Width>
__device__ __forceinline__ float warp_reduce_sum_width(float val) {
    static_assert(Width > 0 && Width <= kWarpSize, "invalid warp reduce width");
#pragma unroll
    for (int mask = Width >> 1; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask, Width);
    }
    return val;
}

template <int Width>
__device__ __forceinline__ float warp_reduce_max_width(float val) {
    static_assert(Width > 0 && Width <= kWarpSize, "invalid warp reduce width");
#pragma unroll
    for (int mask = Width >> 1; mask > 0; mask >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask, Width));
    }
    return val;
}

template <int NumThreads>
__device__ __forceinline__ float block_reduce_sum(float val) {
    constexpr int kNumWarps = (NumThreads + kWarpSize - 1) / kWarpSize;
    static __shared__ float warp_sums[kNumWarps];

    const int tid = threadIdx.x;
    const int lane = tid & (kWarpSize - 1);
    const int warp = tid / kWarpSize;

    val = warp_reduce_sum(val);
    if (lane == 0) {
        warp_sums[warp] = val;
    }
    __syncthreads();

    val = (warp == 0 && lane < kNumWarps) ? warp_sums[lane] : 0.0f;
    if (warp == 0) {
        val = warp_reduce_sum(val);
    }
    return val;
}

__device__ __forceinline__ int find_user_for_token(
    int token,
    const int64_t* __restrict__ user_offsets,
    int num_users
) {
    int lo = 0;
    int hi = num_users;
    while (lo + 1 < hi) {
        const int mid = (lo + hi) >> 1;
        if (user_offsets[mid] <= token) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return lo;
}

template <
    int Br,
    int Bc,
    int NumThreads,
    bool Compact,
    bool UseMmaQk,
    bool UseMmaPv,
    bool UsePaddedSmem = false,
    bool UseSharedKv = false,
    bool UseTileMeta = false,
    bool InterleavedQkv = false,
    bool OutputTokenMajor = false>
__global__ void varlen_causal_attention_tiled_kernel(
    const c10::Half* __restrict__ q,
    const c10::Half* __restrict__ k,
    const c10::Half* __restrict__ v,
    const int64_t* __restrict__ user_offsets,
    const int64_t* __restrict__ tile_starts,
    const int32_t* __restrict__ tile_meta,
    c10::Half* __restrict__ out,
    int heads,
    int seq_len,
    int num_users,
    int num_tiles
) {
    const int head = blockIdx.y;
    const int tid = threadIdx.x;

    if (head >= heads) {
        return;
    }

    int q_tile_start = 0;
    int user_start = 0;
    int user_end = 0;
    int q_tile_len = 0;
    int q_tile_end = 0;

    if constexpr (Compact && UseTileMeta) {
        const int tile_idx = static_cast<int>(blockIdx.x);
        if (tile_idx >= num_tiles || tile_meta == nullptr) {
            return;
        }

        const int32_t* meta = tile_meta + tile_idx * 4;
        user_start = static_cast<int>(meta[0]);
        user_end = static_cast<int>(meta[1]);
        q_tile_start = static_cast<int>(meta[2]);
        q_tile_len = static_cast<int>(meta[3]);

        if (
            user_start < 0
            || user_end > seq_len
            || user_start >= user_end
            || q_tile_start < user_start
            || q_tile_start >= user_end
            || q_tile_len <= 0
            || q_tile_len > Br
            || q_tile_start + q_tile_len > user_end
        ) {
            return;
        }
        q_tile_end = q_tile_start + q_tile_len - 1;
    } else {
        if (Compact) {
            const int tile_idx = static_cast<int>(blockIdx.x);
            if (tile_idx >= num_tiles) {
                return;
            }

            const int64_t q_tile_start_i64 = tile_starts[tile_idx];
            if (q_tile_start_i64 < 0 || q_tile_start_i64 >= seq_len) {
                return;
            }
            q_tile_start = static_cast<int>(q_tile_start_i64);
        } else {
            // Reverse candidate scheduling keeps later, heavier causal tiles near
            // the front of the launch without needing a per-user tile-offset table.
            const int candidate_query = seq_len - 1 - static_cast<int>(blockIdx.x);
            if (candidate_query < 0) {
                return;
            }
            q_tile_start = candidate_query;
        }

        const int user = find_user_for_token(q_tile_start, user_offsets, num_users);
        user_start = static_cast<int>(user_offsets[user]);
        user_end = static_cast<int>(user_offsets[user + 1]);
        const int user_len = user_end - user_start;
        if (user_len <= 0) {
            return;
        }

        const int local_query = q_tile_start - user_start;
        if (local_query < 0 || local_query >= user_len || (local_query % Br) != 0) {
            return;
        }

        const int q_tile_remaining = user_end - q_tile_start;
        q_tile_len = q_tile_remaining < Br ? q_tile_remaining : Br;
        q_tile_end = q_tile_start + q_tile_len - 1;
    }

    const int64_t head_base = static_cast<int64_t>(head) * seq_len * kHeadDim;
    constexpr int kRowReduceWidth = NumThreads / Br;
    static_assert(NumThreads % Br == 0, "NumThreads must be divisible by Br");
    static_assert((kRowReduceWidth & (kRowReduceWidth - 1)) == 0, "row reduce width must be a power of two");
    static_assert(kRowReduceWidth <= kWarpSize, "row reduce width must fit in one warp");
    if constexpr (UseMmaQk) {
        static_assert(Br == 16, "MMA QK kernel currently requires Br=16");
        static_assert(Bc == 64, "MMA QK kernel currently requires Bc=64");
        static_assert(NumThreads == 128, "MMA QK kernel currently requires 128 threads");
    }
    if constexpr (UseMmaPv) {
        static_assert(UseMmaQk, "MMA PV kernel requires MMA QK score generation");
        static_assert(Br == 16, "MMA PV kernel currently requires Br=16");
        static_assert(Bc == 64, "MMA PV kernel currently requires Bc=64");
        static_assert(NumThreads == 128, "MMA PV kernel currently requires 128 threads");
    }
    if constexpr (UseSharedKv) {
        static_assert(UsePaddedSmem, "Shared-KV experiment currently requires padded dynamic SMEM");
        static_assert(UseMmaQk && UseMmaPv, "Shared-KV experiment requires MMA QK and PV");
    }
    if constexpr (UseTileMeta) {
        static_assert(Compact, "Tile meta path requires compact tile scheduling");
    }
    if constexpr (OutputTokenMajor) {
        static_assert(InterleavedQkv, "Token-major output is only exposed for the interleaved-qkv path");
    }
    const int row_reduce_row = tid / kRowReduceWidth;
    const int row_reduce_lane = tid - row_reduce_row * kRowReduceWidth;
    constexpr int kKvStages = UseMmaQk ? 2 : 1;
    constexpr int kSmemPad = UsePaddedSmem ? 8 : 0;
    constexpr int kQSmemHeadStride = kHeadDim + kSmemPad;
    constexpr int kKSmemHeadStride = kHeadDim + (UsePaddedSmem ? 8 : 0);
    constexpr int kVSmemHeadStride = kHeadDim + kSmemPad;
    constexpr int kPSmemStride = Bc + kSmemPad;
    static_assert(kQSmemHeadStride % kHalfPer128Bits == 0, "Q SMEM stride must be 16-byte aligned");
    static_assert(kKSmemHeadStride % kHalfPer128Bits == 0, "K SMEM stride must be 16-byte aligned");
    static_assert(kVSmemHeadStride % kHalfPer128Bits == 0, "V SMEM stride must be 16-byte aligned");
    static_assert(kPSmemStride % kHalfPer128Bits == 0, "P SMEM stride must be 16-byte aligned");
    __shared__ __align__(16) c10::Half q_static[UsePaddedSmem ? 1 : Br * kQSmemHeadStride];
    __shared__ __align__(16) c10::Half k_static[UsePaddedSmem ? 1 : kKvStages * Bc * kKSmemHeadStride];
    __shared__ __align__(16) c10::Half v_static[UsePaddedSmem ? 1 : kKvStages * Bc * kVSmemHeadStride];
    __shared__ __align__(16) c10::Half p_static[UsePaddedSmem ? 1 : (UseMmaPv ? Br * kPSmemStride : 1)];
    __shared__ float scores_shared[Br * Bc];
    __shared__ float out_shared[Br * kHeadDim];
    __shared__ float m_shared[Br];
    __shared__ float l_shared[Br];
    __shared__ float tile_m_shared[Br];
    __shared__ float tile_l_shared[Br];
    __shared__ float new_m_shared[Br];
    __shared__ float old_scale_shared[Br];

    extern __shared__ uint4 dynamic_smem_u4[];
    c10::Half* q_shared = q_static;
    c10::Half* k_shared = k_static;
    c10::Half* v_shared = v_static;
    c10::Half* p_shared = p_static;
    if constexpr (UsePaddedSmem) {
        c10::Half* dynamic_half_smem = reinterpret_cast<c10::Half*>(dynamic_smem_u4);
        int smem_half_offset = 0;
        q_shared = dynamic_half_smem + smem_half_offset;
        smem_half_offset += Br * kQSmemHeadStride;
        k_shared = dynamic_half_smem + smem_half_offset;
        smem_half_offset += kKvStages * Bc * kKSmemHeadStride;
        if constexpr (UseSharedKv) {
            v_shared = k_shared;
        } else {
            v_shared = dynamic_half_smem + smem_half_offset;
            smem_half_offset += kKvStages * Bc * kVSmemHeadStride;
        }
        p_shared = dynamic_half_smem + smem_half_offset;
    }

    for (int vec = tid; vec < Br * kHeadDimVec128; vec += NumThreads) {
        const int row = vec / kHeadDimVec128;
        const int vec_col = vec - row * kHeadDimVec128;
        const int dim = vec_col * kHalfPer128Bits;
        const int query = q_tile_start + row;
        c10::Half* dst = q_shared + row * kQSmemHeadStride + dim;
        if (row < q_tile_len) {
            const c10::Half* src = nullptr;
            if constexpr (InterleavedQkv) {
                src = q + interleaved_qkv_offset(head, heads, query, 0, dim);
            } else {
                src = q + head_base + static_cast<int64_t>(query) * kHeadDim + dim;
            }
            copy_half8_128b(dst, src);
        } else {
            zero_half8_128b(dst);
        }
    }

    for (int idx = tid; idx < Br * kHeadDim; idx += NumThreads) {
        out_shared[idx] = 0.0f;
    }

    if (tid < Br) {
        m_shared[tid] = kNegInf;
        l_shared[tid] = 0.0f;
    }
    __syncthreads();

    constexpr int kQFragDimBlocks = kHeadDim / 16;
    uint32_t q_frag_prefetch[UseMmaQk ? kQFragDimBlocks : 1][4];
    if constexpr (UseMmaQk) {
        const int lane = tid & (kWarpSize - 1);
#pragma unroll
        for (int dim_block = 0; dim_block < kHeadDim; dim_block += 16) {
            const int dim_tile = dim_block / 16;
            const int q_row = lane & 15;
            const int q_dim = dim_block + (lane >= 16 ? 8 : 0);
            const uint32_t q_addr = shared_addr(q_shared + q_row * kQSmemHeadStride + q_dim);
            ldmatrix_x4(
                q_frag_prefetch[dim_tile][0],
                q_frag_prefetch[dim_tile][1],
                q_frag_prefetch[dim_tile][2],
                q_frag_prefetch[dim_tile][3],
                q_addr
            );
        }
    }

    if constexpr (UseMmaQk) {
        const int first_k_tile_remaining = q_tile_end + 1 - user_start;
        const int first_k_tile_len = first_k_tile_remaining < Bc ? first_k_tile_remaining : Bc;
        if constexpr (UseSharedKv) {
            if constexpr (InterleavedQkv) {
                load_interleaved_qkv_tile_async<Bc, NumThreads, kKSmemHeadStride, 1>(
                    q,
                    k_shared,
                    head,
                    heads,
                    user_start,
                    first_k_tile_len,
                    tid
                );
            } else {
                load_tensor_tile_async<Bc, NumThreads, kKSmemHeadStride>(
                    k,
                    k_shared,
                    head_base,
                    user_start,
                    first_k_tile_len,
                    tid
                );
            }
        } else {
            load_kv_tile_async<Bc, NumThreads, kKSmemHeadStride, kVSmemHeadStride>(
                k,
                v,
                k_shared,
                v_shared,
                head_base,
                user_start,
                first_k_tile_len,
                tid
            );
        }
    }

    for (int k_tile_start = user_start; k_tile_start <= q_tile_end; k_tile_start += Bc) {
        const int k_tile_remaining = q_tile_end + 1 - k_tile_start;
        const int k_tile_len = k_tile_remaining < Bc ? k_tile_remaining : Bc;

        c10::Half* k_tile_shared = k_shared;
        c10::Half* v_tile_shared = v_shared;
        if constexpr (UseMmaQk) {
            const int tile_idx = (k_tile_start - user_start) / Bc;
            const int stage = tile_idx & 1;
            k_tile_shared = k_shared + stage * Bc * kKSmemHeadStride;
            v_tile_shared = v_shared + stage * Bc * kVSmemHeadStride;

            cp_async_wait_group<0>();
            __syncthreads();

            const int next_k_tile_start = k_tile_start + Bc;
            if (next_k_tile_start <= q_tile_end) {
                const int next_k_tile_remaining = q_tile_end + 1 - next_k_tile_start;
                const int next_k_tile_len = next_k_tile_remaining < Bc ? next_k_tile_remaining : Bc;
                const int next_stage = stage ^ 1;
                if constexpr (UseSharedKv) {
                    if constexpr (InterleavedQkv) {
                        load_interleaved_qkv_tile_async<Bc, NumThreads, kKSmemHeadStride, 1>(
                            q,
                            k_shared + next_stage * Bc * kKSmemHeadStride,
                            head,
                            heads,
                            next_k_tile_start,
                            next_k_tile_len,
                            tid
                        );
                    } else {
                        load_tensor_tile_async<Bc, NumThreads, kKSmemHeadStride>(
                            k,
                            k_shared + next_stage * Bc * kKSmemHeadStride,
                            head_base,
                            next_k_tile_start,
                            next_k_tile_len,
                            tid
                        );
                    }
                } else {
                    load_kv_tile_async<Bc, NumThreads, kKSmemHeadStride, kVSmemHeadStride>(
                        k,
                        v,
                        k_shared + next_stage * Bc * kKSmemHeadStride,
                        v_shared + next_stage * Bc * kVSmemHeadStride,
                        head_base,
                        next_k_tile_start,
                        next_k_tile_len,
                        tid
                    );
                }
            }
        } else {
            for (int vec = tid; vec < k_tile_len * kHeadDimVec128; vec += NumThreads) {
                const int key_row = vec / kHeadDimVec128;
                const int vec_col = vec - key_row * kHeadDimVec128;
                const int dim = vec_col * kHalfPer128Bits;
                const int key = k_tile_start + key_row;
                c10::Half* k_dst = k_shared + key_row * kKSmemHeadStride + dim;
                c10::Half* v_dst = v_shared + key_row * kVSmemHeadStride + dim;
                const int64_t src = head_base + static_cast<int64_t>(key) * kHeadDim + dim;
                copy_half8_128b(k_dst, k + src);
                copy_half8_128b(v_dst, v + src);
            }
            __syncthreads();
        }

        if constexpr (UseMmaQk) {
            const int warp = tid / kWarpSize;
            const int lane = tid & (kWarpSize - 1);
            const int col_block_base = warp * 16;
            const int row0 = lane / 4;
            const int row1 = row0 + 8;
            const int col_pair = (lane & 3) * 2;

            uint32_t score_frag[2][4] = {
                {0, 0, 0, 0},
                {0, 0, 0, 0},
            };


                uint32_t k_frag[2][2];

#pragma unroll
                for (int dim_block = 0; dim_block < kHeadDim; dim_block += 16) {
                    const int dim_tile = dim_block / 16;
                    const uint32_t* q_frag = q_frag_prefetch[dim_tile];

#pragma unroll
                    for (int n_block = 0; n_block < 2; ++n_block) {
                        const int key_row = col_block_base + n_block * 8 + (lane & 7);
                        const int k_dim = dim_block + ((lane >> 3) & 1) * 8;
                        const uint32_t k_addr = shared_addr(k_tile_shared + key_row * kKSmemHeadStride + k_dim);
                        ldmatrix_x2(k_frag[n_block][0], k_frag[n_block][1], k_addr);

                        hmma16816_f32(
                            score_frag[n_block][0],
                            score_frag[n_block][1],
                            score_frag[n_block][2],
                            score_frag[n_block][3],
                            q_frag[0],
                            q_frag[1],
                            q_frag[2],
                            q_frag[3],
                            k_frag[n_block][0],
                            k_frag[n_block][1],
                            score_frag[n_block][0],
                            score_frag[n_block][1],
                            score_frag[n_block][2],
                            score_frag[n_block][3]
                        );
                    }
                }


#pragma unroll
            for (int n_block = 0; n_block < 2; ++n_block) {
                const int key_col = col_block_base + n_block * 8 + col_pair;
                const uint32_t* regs = score_frag[n_block];
                const int rows[2] = {row0, row1};
#pragma unroll
                for (int row_slot = 0; row_slot < 2; ++row_slot) {
                    const int row = rows[row_slot];
                    const int query = q_tile_start + row;
#pragma unroll
                    for (int value_slot = 0; value_slot < 2; ++value_slot) {
                        const int key_row = key_col + value_slot;
                        const int key = k_tile_start + key_row;
                        if (row < q_tile_len && key_row < k_tile_len) {
                            float score = kNegInf;
                            if (key <= query) {
                                score = reg_as_float(regs[row_slot * 2 + value_slot]) * kScale;
                            }
                            scores_shared[row * k_tile_len + key_row] = score;
                        }
                    }
                }
            }
        } else {
            for (int cell = tid; cell < Br * k_tile_len; cell += NumThreads) {
                const int row = cell / k_tile_len;
                const int key_row = cell - row * k_tile_len;
                const int query = q_tile_start + row;
                const int key = k_tile_start + key_row;

                float score = kNegInf;
                if (row < q_tile_len && key <= query) {
                    float dot = 0.0f;
#pragma unroll
                    for (int dim = 0; dim < kHeadDim; ++dim) {
                        dot += static_cast<float>(q_shared[row * kQSmemHeadStride + dim])
                            * static_cast<float>(k_tile_shared[key_row * kKSmemHeadStride + dim]);
                    }
                    score = dot * kScale;
                }
                scores_shared[cell] = score;
            }
        }
        __syncthreads();

        if constexpr (UseSharedKv) {
            if constexpr (InterleavedQkv) {
                load_interleaved_qkv_tile_async<Bc, NumThreads, kVSmemHeadStride, 2>(
                    q,
                    v_tile_shared,
                    head,
                    heads,
                    k_tile_start,
                    k_tile_len,
                    tid
                );
            } else {
                load_tensor_tile_async<Bc, NumThreads, kVSmemHeadStride>(
                    v,
                    v_tile_shared,
                    head_base,
                    k_tile_start,
                    k_tile_len,
                    tid
                );
            }
        }

        float row_max = kNegInf;
        if (row_reduce_row < q_tile_len) {
            for (int key_row = row_reduce_lane; key_row < k_tile_len; key_row += kRowReduceWidth) {
                row_max = fmaxf(row_max, scores_shared[row_reduce_row * k_tile_len + key_row]);
            }
        }
        row_max = warp_reduce_max_width<kRowReduceWidth>(row_max);

        if (row_reduce_lane == 0) {
            const int row = row_reduce_row;
            const float old_m = m_shared[row];
            float new_m = old_m;
            if (row_max != kNegInf) {
                new_m = fmaxf(old_m, row_max);
            }

            tile_m_shared[row] = row_max;
            tile_l_shared[row] = 0.0f;
            new_m_shared[row] = new_m;
            old_scale_shared[row] = (old_m == kNegInf) ? 0.0f : attention_exp(old_m - new_m);
        }
        __syncthreads();

        if constexpr (UseMmaPv) {
            for (int idx = tid; idx < Br * kPSmemStride; idx += NumThreads) {
                p_shared[idx] = static_cast<c10::Half>(0.0f);
            }
            __syncthreads();
        }

        float row_l = 0.0f;
        if (row_reduce_row < q_tile_len && tile_m_shared[row_reduce_row] != kNegInf) {
            for (int key_row = row_reduce_lane; key_row < k_tile_len; key_row += kRowReduceWidth) {
                const int cell = row_reduce_row * k_tile_len + key_row;
                float prob = 0.0f;
                const float score = scores_shared[cell];
                if (score != kNegInf) {
                    prob = attention_exp(score - new_m_shared[row_reduce_row]);
                }
                scores_shared[cell] = prob;
                if constexpr (UseMmaPv) {
                    p_shared[row_reduce_row * kPSmemStride + key_row] = static_cast<c10::Half>(prob);
                }
                row_l += prob;
            }
        }
        row_l = warp_reduce_sum_width<kRowReduceWidth>(row_l);
        if (row_reduce_lane == 0) {
            tile_l_shared[row_reduce_row] = row_l;
        }
        __syncthreads();

        if constexpr (UseMmaPv) {
            if constexpr (UseSharedKv) {
                cp_async_wait_group<0>();
                __syncthreads();
            }

            const int warp = tid / kWarpSize;
            const int lane = tid & (kWarpSize - 1);
            const int out_col_base = warp * 16;
            const int row0 = lane / 4;
            const int row1 = row0 + 8;
            const int col_pair = (lane & 3) * 2;

            uint32_t p_frag[4];
            uint32_t out_frag[2][4] = {
                {0, 0, 0, 0},
                {0, 0, 0, 0},
            };

#pragma unroll
            for (int key_block = 0; key_block < Bc; key_block += 16) {
                const int p_row = lane & 15;
                const int p_col = key_block + (lane >= 16 ? 8 : 0);
                const uint32_t p_addr = shared_addr(p_shared + p_row * kPSmemStride + p_col);
                ldmatrix_x4(p_frag[0], p_frag[1], p_frag[2], p_frag[3], p_addr);


                    uint32_t v_frag[2][2];

#pragma unroll
                    for (int n_block = 0; n_block < 2; ++n_block) {
                        const int v_row = key_block + (lane & 15);
                        const int v_col = out_col_base + n_block * 8;
                        const uint32_t v_addr = shared_addr(v_tile_shared + v_row * kVSmemHeadStride + v_col);
                        ldmatrix_x2_trans(v_frag[n_block][0], v_frag[n_block][1], v_addr);

                        hmma16816_f32(
                            out_frag[n_block][0],
                            out_frag[n_block][1],
                            out_frag[n_block][2],
                            out_frag[n_block][3],
                            p_frag[0],
                            p_frag[1],
                            p_frag[2],
                            p_frag[3],
                            v_frag[n_block][0],
                            v_frag[n_block][1],
                            out_frag[n_block][0],
                            out_frag[n_block][1],
                            out_frag[n_block][2],
                            out_frag[n_block][3]
                        );
                    }

            }

#pragma unroll
            for (int n_block = 0; n_block < 2; ++n_block) {
                const int dim_col = out_col_base + n_block * 8 + col_pair;
                const uint32_t* regs = out_frag[n_block];
                const int rows[2] = {row0, row1};
#pragma unroll
                for (int row_slot = 0; row_slot < 2; ++row_slot) {
                    const int row = rows[row_slot];
                    if (row < q_tile_len && tile_m_shared[row] != kNegInf) {
#pragma unroll
                        for (int value_slot = 0; value_slot < 2; ++value_slot) {
                            const int dim = dim_col + value_slot;
                            const int idx = row * kHeadDim + dim;
                            const float tile_acc = reg_as_float(regs[row_slot * 2 + value_slot]);
                            out_shared[idx] = out_shared[idx] * old_scale_shared[row] + tile_acc;
                        }
                    }
                }
            }
        } else {
            for (int idx = tid; idx < Br * kHeadDim; idx += NumThreads) {
                const int row = idx / kHeadDim;
                const int dim = idx - row * kHeadDim;
                if (row >= q_tile_len || tile_m_shared[row] == kNegInf) {
                    continue;
                }

                float tile_acc = 0.0f;
                for (int key_row = 0; key_row < k_tile_len; ++key_row) {
                    const float prob = scores_shared[row * k_tile_len + key_row];
                    if (prob != 0.0f) {
                        tile_acc += prob * static_cast<float>(v_tile_shared[key_row * kVSmemHeadStride + dim]);
                    }
                }
                out_shared[idx] = out_shared[idx] * old_scale_shared[row] + tile_acc;
            }
        }
        __syncthreads();

        if (tid < Br) {
            const int row = tid;
            const float row_max = tile_m_shared[row];
            if (row < q_tile_len && row_max != kNegInf) {
                l_shared[row] = l_shared[row] * old_scale_shared[row] + tile_l_shared[row];
                m_shared[row] = new_m_shared[row];
            }
        }
        __syncthreads();
    }

    for (int idx = tid; idx < Br * kHeadDim; idx += NumThreads) {
        const int row = idx / kHeadDim;
        const int dim = idx - row * kHeadDim;
        if (row < q_tile_len) {
            const int query = q_tile_start + row;
            int64_t dst = head_base + static_cast<int64_t>(query) * kHeadDim + dim;
            if constexpr (OutputTokenMajor) {
                dst = (static_cast<int64_t>(query) * heads + head) * kHeadDim + dim;
            }
            out[dst] =
                static_cast<c10::Half>(out_shared[idx] / l_shared[row]);
        }
    }
}

void check_varlen_attention_inputs(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& user_offsets
) {
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    TORCH_CHECK(k.is_cuda(), "k must be a CUDA tensor");
    TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");
    TORCH_CHECK(user_offsets.is_cuda(), "user_offsets must be a CUDA tensor");
    TORCH_CHECK(q.scalar_type() == at::kHalf, "q must be float16");
    TORCH_CHECK(q.scalar_type() == k.scalar_type(), "q and k must have the same dtype");
    TORCH_CHECK(q.scalar_type() == v.scalar_type(), "q and v must have the same dtype");
    TORCH_CHECK(user_offsets.scalar_type() == at::kLong, "user_offsets must be int64");
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "v must be contiguous");
    TORCH_CHECK(user_offsets.is_contiguous(), "user_offsets must be contiguous");
    TORCH_CHECK(q.sizes() == k.sizes(), "q and k must have the same shape");
    TORCH_CHECK(q.sizes() == v.sizes(), "q and v must have the same shape");
    TORCH_CHECK(q.dim() == 4, "q must have shape [1, heads, seq_len, 64]");
    TORCH_CHECK(q.size(0) == 1, "only batch size 1 is supported");
    TORCH_CHECK(q.size(3) == kHeadDim, "head_dim must be 64");
    TORCH_CHECK(user_offsets.dim() == 1, "user_offsets must be 1D");
    TORCH_CHECK(user_offsets.numel() >= 2, "user_offsets must contain at least 2 elements");
    TORCH_CHECK(q.device() == k.device(), "q and k must be on the same CUDA device");
    TORCH_CHECK(q.device() == v.device(), "q and v must be on the same CUDA device");
    TORCH_CHECK(q.device() == user_offsets.device(), "q and user_offsets must be on the same CUDA device");
}

void check_varlen_attention_tile_meta(
    const torch::Tensor& q,
    const torch::Tensor& tile_meta
) {
    TORCH_CHECK(tile_meta.is_cuda(), "tile_meta must be a CUDA tensor");
    TORCH_CHECK(tile_meta.scalar_type() == at::kInt, "tile_meta must be int32");
    TORCH_CHECK(tile_meta.is_contiguous(), "tile_meta must be contiguous");
    TORCH_CHECK(tile_meta.dim() == 2, "tile_meta must have shape [num_tiles,4]");
    TORCH_CHECK(tile_meta.size(1) == 4, "tile_meta must have shape [num_tiles,4]");
    TORCH_CHECK(q.device() == tile_meta.device(), "q and tile_meta must be on the same CUDA device");
}

void check_varlen_attention_interleaved_qkv_inputs(
    const torch::Tensor& qkv,
    const torch::Tensor& user_offsets
) {
    TORCH_CHECK(qkv.is_cuda(), "qkv must be a CUDA tensor");
    TORCH_CHECK(user_offsets.is_cuda(), "user_offsets must be a CUDA tensor");
    TORCH_CHECK(qkv.scalar_type() == at::kHalf, "qkv must be float16");
    TORCH_CHECK(user_offsets.scalar_type() == at::kLong, "user_offsets must be int64");
    TORCH_CHECK(qkv.is_contiguous(), "qkv must be contiguous");
    TORCH_CHECK(user_offsets.is_contiguous(), "user_offsets must be contiguous");
    TORCH_CHECK(qkv.dim() == 4, "qkv must have shape [1, seq_len, heads, 192]");
    TORCH_CHECK(qkv.size(0) == 1, "only batch size 1 is supported");
    TORCH_CHECK(qkv.size(2) > 0, "heads must be positive");
    TORCH_CHECK(qkv.size(3) == kQkvParts * kHeadDim, "interleaved qkv last dim must be 192");
    TORCH_CHECK(user_offsets.dim() == 1, "user_offsets must be 1D");
    TORCH_CHECK(user_offsets.numel() >= 2, "user_offsets must contain at least 2 elements");
    TORCH_CHECK(qkv.device() == user_offsets.device(), "qkv and user_offsets must be on the same CUDA device");
}

}  // namespace

torch::Tensor varlen_causal_attention_mma_qk_pv_padded_shared_kv_meta(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor user_offsets,
    torch::Tensor tile_meta
) {
    check_varlen_attention_inputs(q, k, v, user_offsets);
    check_varlen_attention_tile_meta(q, tile_meta);

    constexpr int kMmaBr = 16;
    constexpr int kMmaBc = 64;
    constexpr int kMmaThreads = 128;
    constexpr int kSmemPad = 8;
    constexpr int kKvStages = 2;
    constexpr int kQSmemHeadStride = kHeadDim + kSmemPad;
    constexpr int kKvSmemHeadStride = kHeadDim + kSmemPad;
    constexpr int kPSmemStride = kMmaBc + kSmemPad;
    constexpr int kDynamicSmemHalfCount =
        kMmaBr * kQSmemHeadStride
        + kKvStages * kMmaBc * kKvSmemHeadStride
        + kMmaBr * kPSmemStride;
    constexpr int kDynamicSmemBytes = kDynamicSmemHalfCount * static_cast<int>(sizeof(c10::Half));

    const int heads = static_cast<int>(q.size(1));
    const int seq_len = static_cast<int>(q.size(2));
    const int num_users = static_cast<int>(user_offsets.numel() - 1);
    const int num_tiles = static_cast<int>(tile_meta.size(0));

    auto out = torch::empty_like(q);
    if (seq_len == 0) {
        return out;
    }
    TORCH_CHECK(num_tiles > 0, "tile_meta must contain at least one tile when seq_len > 0");

    const dim3 grid(num_tiles, heads);
    auto kernel = varlen_causal_attention_tiled_kernel<kMmaBr, kMmaBc, kMmaThreads, true, true, true, true, true, true>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kDynamicSmemBytes
    ));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared
    ));
    varlen_causal_attention_tiled_kernel<kMmaBr, kMmaBc, kMmaThreads, true, true, true, true, true, true>
        <<<grid, kMmaThreads, kDynamicSmemBytes, at::cuda::getCurrentCUDAStream()>>>(
            q.data_ptr<c10::Half>(),
            k.data_ptr<c10::Half>(),
            v.data_ptr<c10::Half>(),
            user_offsets.data_ptr<int64_t>(),
            nullptr,
            tile_meta.data_ptr<int32_t>(),
            out.data_ptr<c10::Half>(),
            heads,
            seq_len,
            num_users,
            num_tiles
        );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

torch::Tensor varlen_causal_attention_qkv_interleaved_mma_qk_pv_padded_shared_kv_meta(
    torch::Tensor qkv,
    torch::Tensor user_offsets,
    torch::Tensor tile_meta,
    bool token_major_out
) {
    check_varlen_attention_interleaved_qkv_inputs(qkv, user_offsets);
    check_varlen_attention_tile_meta(qkv, tile_meta);

    constexpr int kMmaBr = 16;
    constexpr int kMmaBc = 64;
    constexpr int kMmaThreads = 128;
    constexpr int kSmemPad = 8;
    constexpr int kKvStages = 2;
    constexpr int kQSmemHeadStride = kHeadDim + kSmemPad;
    constexpr int kKvSmemHeadStride = kHeadDim + kSmemPad;
    constexpr int kPSmemStride = kMmaBc + kSmemPad;
    constexpr int kDynamicSmemHalfCount =
        kMmaBr * kQSmemHeadStride
        + kKvStages * kMmaBc * kKvSmemHeadStride
        + kMmaBr * kPSmemStride;
    constexpr int kDynamicSmemBytes = kDynamicSmemHalfCount * static_cast<int>(sizeof(c10::Half));

    const int seq_len = static_cast<int>(qkv.size(1));
    const int heads = static_cast<int>(qkv.size(2));
    const int num_users = static_cast<int>(user_offsets.numel() - 1);
    const int num_tiles = static_cast<int>(tile_meta.size(0));

    torch::Tensor out;
    if (token_major_out) {
        out = torch::empty({1, seq_len, heads, kHeadDim}, qkv.options());
    } else {
        out = torch::empty({1, heads, seq_len, kHeadDim}, qkv.options());
    }
    if (seq_len == 0) {
        return out;
    }
    TORCH_CHECK(num_tiles > 0, "tile_meta must contain at least one tile when seq_len > 0");

    const dim3 grid(num_tiles, heads);
    if (token_major_out) {
        auto kernel = varlen_causal_attention_tiled_kernel<kMmaBr, kMmaBc, kMmaThreads, true, true, true, true, true, true, true, true>;
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kDynamicSmemBytes
        ));
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxShared
        ));
        varlen_causal_attention_tiled_kernel<kMmaBr, kMmaBc, kMmaThreads, true, true, true, true, true, true, true, true>
            <<<grid, kMmaThreads, kDynamicSmemBytes, at::cuda::getCurrentCUDAStream()>>>(
                qkv.data_ptr<c10::Half>(),
                nullptr,
                nullptr,
                user_offsets.data_ptr<int64_t>(),
                nullptr,
                tile_meta.data_ptr<int32_t>(),
                out.data_ptr<c10::Half>(),
                heads,
                seq_len,
                num_users,
                num_tiles
            );
    } else {
        auto kernel = varlen_causal_attention_tiled_kernel<kMmaBr, kMmaBc, kMmaThreads, true, true, true, true, true, true, true, false>;
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kDynamicSmemBytes
        ));
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxShared
        ));
        varlen_causal_attention_tiled_kernel<kMmaBr, kMmaBc, kMmaThreads, true, true, true, true, true, true, true, false>
            <<<grid, kMmaThreads, kDynamicSmemBytes, at::cuda::getCurrentCUDAStream()>>>(
                qkv.data_ptr<c10::Half>(),
                nullptr,
                nullptr,
                user_offsets.data_ptr<int64_t>(),
                nullptr,
                tile_meta.data_ptr<int32_t>(),
                out.data_ptr<c10::Half>(),
                heads,
                seq_len,
                num_users,
                num_tiles
            );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "varlen_causal_attention_mma_qk_pv_padded_shared_kv_meta",
        &varlen_causal_attention_mma_qk_pv_padded_shared_kv_meta,
        "Tile-meta padded-SMEM shared-KV variable-length causal attention with Tensor Core QK and PV MMA (CUDA, float16)"
    );
    m.def(
        "varlen_causal_attention_qkv_interleaved_mma_qk_pv_padded_shared_kv_meta",
        &varlen_causal_attention_qkv_interleaved_mma_qk_pv_padded_shared_kv_meta,
        "Tile-meta padded-SMEM shared-KV variable-length causal attention reading contiguous [1,S,H,192] interleaved QKV (CUDA, float16)"
    );
}
