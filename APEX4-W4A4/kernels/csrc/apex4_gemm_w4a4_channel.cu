/*
 * Adapted from https://github.com/IST-DASLab/marlin/blob/master/marlin/marlin_cuda_kernel.cu
 * https://github.com/IST-DASLab/marlin/blob/master/marlin/marlin_cuda.cpp
 * Modified by HandH1998
 * Copyright (C) 2024 HandH1998
 * Modified for W4A4 by the APEX4 authors.
 * Copyright (C) Marlin.2024 Elias Frantar (elias.frantar@ist.ac.at)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// W4A4 GEMM kernel; weights are pre-processed offline into int4-vector layout.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>




__host__ __device__ constexpr int ceildiv(int a, int b) {//
  return (a + b - 1) / b;
}

// Instances of `Vec` are used to organize groups of >>registers<<, as needed for instance as inputs to tensor core
// operations. Consequently, all corresponding index accesses must be compile-time constants, which is why we
// extensively use `#pragma unroll` throughout the kernel code to guarantee this.
template <typename T, int n>
struct Vec {
  T elems[n];
  __device__ T& operator[](int i) {
    return elems[i];
  }
};

using I4 = Vec<int, 4>;

// Matrix fragments for tensor core instructions; their precise layout is documented here:
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-integer-type
using FragA = Vec<uint32_t, 2>;
using FragB = Vec<uint32_t, 1>;
using FragC = Vec<int, 4>;
using FragC_f = Vec<int, 4>;

using FragS_CHANNEL = Vec<half, 2>; // weight per-channel quantization scales or activaton per-token quantization scales

// Predicated asynchronous global->shared copy; used for inputs A where we apply predication to handle batchsizes that
// are not multiples of 16.
__device__ inline void cp_async4_pred(void* smem_ptr, const void* glob_ptr, bool pred = true) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
    "{\n"
    "   .reg .pred p;\n"
    "   setp.ne.b32 p, %0, 0;\n"
    "   @p cp.async.cg.shared.global [%1], [%2], %3;\n"
    "}\n" :: "r"((int) pred), "r"(smem), "l"(glob_ptr), "n"(BYTES)
  );
}

// Asynchronous global->shared copy
__device__ inline void cp_async4(void* smem_ptr, const void* glob_ptr) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "{\n"
      "   cp.async.cg.shared.global [%0], [%1], %2;\n"
      "}\n" ::"r"(smem),
      "l"(glob_ptr), "n"(BYTES));
}


__device__ inline void cp_async1(void* smem_ptr, const void* glob_ptr) {
  const int BYTES = 4;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "{\n"
      "   cp.async.ca.shared.global [%0], [%1], %2;\n"
      "}\n" ::"r"(smem),
      "l"(glob_ptr), "n"(BYTES));
}
__device__ inline void cp_async4_stream(void* smem_ptr, const void* glob_ptr) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
    "{\n"
    "   .reg .b64 p;\n"
    "   createpolicy.fractional.L2::evict_first.b64 p, 1.0;"
    "   cp.async.cg.shared.global.L2::cache_hint [%0], [%1], %2, p;\n"
    "}\n" :: "r"(smem), "l"(glob_ptr), "n"(BYTES)
  );
}

__device__ inline void cp_async1_stream(void* smem_ptr, const void* glob_ptr) {
  const int BYTES = 4;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
    "{\n"
    "   .reg .b64 p;\n"
    "   createpolicy.fractional.L2::evict_first.b64 p, 1.0;"
    "   cp.async.ca.shared.global.L2::cache_hint [%0], [%1], %2, p;\n"
    "}\n" :: "r"(smem), "l"(glob_ptr), "n"(BYTES)
  );
}

// Async copy fence.
__device__ inline void cp_async_fence() {
  asm volatile("cp.async.commit_group;\n" ::);
}

// Wait until at most `n` async copy stages are still pending.
template <int n>
__device__ inline void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(n));
}

// m16n8k16 tensor core mma instruction with int8 inputs and int32 output/accumulation.
__device__ inline void mma(const FragA& a_frag, const FragB& frag_b, FragC& frag_c) {
  const uint32_t* a = reinterpret_cast<const uint32_t*>(&a_frag);
  const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
  int* c = reinterpret_cast<int*>(&frag_c);
  asm volatile(
    "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s4.s4.s32 "
    "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
    : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
    :  "r"(a[0]),  "r"(a[1]),  "r"(b[0]),
       "r"(c[0]),  "r"(c[1]),  "r"(c[2]),  "r"(c[3])
  );
}

// Instruction for loading a full 16x16 matrix fragment of operand A from shared memory, directly in tensor core layout.
__device__ inline void ldsm4(FragA& frag_a, const void* smem_ptr) {
  uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
    "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
    : "=r"(a[0]), "=r"(a[1]) : "r"(smem)
  );
}
// Instruction for loading a full 16x16 matrix fragment of operand A from shared memory, directly in tensor core layout.
__device__ inline void ldsm4_b(FragB& frag_b, const void* smem_ptr) {
  uint32_t* b = reinterpret_cast<uint32_t*>(&frag_b);
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
    "ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
    : "=r"(b[0]) : "r"(smem)
  );
}

inline __device__ half2 float2_to_half2(float2 f) {
  uint32_t res;
  uint16_t h0, h1;
  asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(h0) : "f"(f.x));
  asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(h1) : "f"(f.y));
  asm volatile("mov.b32 %0, {%1, %2};\n" : "=r"(res) : "h"(h0), "h"(h1));
  return reinterpret_cast<half2&>(res);
}

inline __device__ float int32_to_float(int h) {
  float res;
  asm volatile("cvt.rn.f32.s32 %0, %1;\n" : "=f"(res) : "r"(h));
  return res;
}

// Efficiently dequantize an int32 value into a full B-fragment of 4 int8 values for weight per channel dequant.
__device__ inline FragB dequant_per_channel(int q) {
  static constexpr int MASK = 0xf0f0f0f0;
  FragB frag_b;
  frag_b[0] = (q & MASK);
  return frag_b;
}

// Lookup-table based 3-input logical operation; explicitly used for dequantization as the compiler does not seem to
// automatically recognize it in all cases.
template <int lut>
__device__ inline uint32_t lop3(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t res;
  asm volatile(
    "lop3.b32 %0, %1, %2, %3, %4;\n"
    : "=r"(res) : "r"(a), "r"(b), "r"(c), "n"(lut)
  );
  return res;
}

// Wait until barrier reaches `count`, then lock for current threadblock.
__device__ inline void barrier_acquire(int* lock, int count) {
  if (threadIdx.x == 0) {
    int state = -1;
    do
      // Guarantee that subsequent writes by this threadblock will be visible globally.
      asm volatile ("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(state) : "l"(lock));
    while (state != count);
  }
  __syncthreads();
}

// Release barrier and increment visitation count.
__device__ inline void barrier_release(int* lock, bool reset = false) {
  __syncthreads();
  if (threadIdx.x == 0) {
    if (reset) {
      lock[0] = 0;
      return;
    }
    int val = 1;
    // Make sure that all writes since acquiring this barrier are visible globally, while releasing the barrier.
    asm volatile ("fence.acq_rel.gpu;\n");
    asm volatile ("red.relaxed.gpu.global.add.s32 [%0], %1;\n" : : "l"(lock), "r"(val));
  }
}


template <
  const int threads, // number of threads in a threadblock
  const int thread_m_blocks, // number of 16x16 blocks in the m dimension (batchsize) of the threadblock
  const int thread_n_blocks, // same for n dimension (output)
  const int thread_k_blocks, // same for k dimension (reduction)
  const int stages, // number of stages for the async global->shared fetch pipeline
  const int group_blocks = -1 // number of consecutive 16x16 blocks with a separate quantization scale
>
__global__ void Marlin_w4a4_7(
  const int4* __restrict__ A, //  input matrix of shape mxk
  const int4* __restrict__ B, // 4bit quantized weight matrix of shape kxn
        int4* __restrict__ C, //  global_reduce buffer of shape (max_par*16*4)xn , as int8 tensor core's output is int32 dtype
        int4* __restrict__ D, //  output buffer of shape mxn
  const half* __restrict__ s11,
  const int4* __restrict__ s22,
  int  prob_m, // batch dimension m
  int  prob_n, // output dimension n
  int  prob_k, // reduction dimension k
  int* locks // extra global storage for barrier synchronization
) {
  // Each threadblock processes one "stripe" of the B matrix with (roughly) the same size, which might involve multiple
  // column "slices" (of width 16 * `thread_n_blocks`). Stripes are defined as shown in the 3x3 matrix 5 SM example:
  //   0 1 3
  //   0 2 3
  //   1 2 4
  // While this kind of partitioning makes things somewhat more complicated, it ensures good utilization of all SMs
  // for many kinds of shape and GPU configurations, while requiring as few slow global cross-threadblock reductions as
  // possible.

  int parallel = 1;
  if (prob_m > 16 * thread_m_blocks) {
    parallel = prob_m / (16 * thread_m_blocks);
    prob_m = 16 * thread_m_blocks;
  }

  int k_tiles = prob_k / 32 / thread_k_blocks;
  int n_tiles = prob_n / 16 / thread_n_blocks;
  int iters = ceildiv(k_tiles * n_tiles * parallel, gridDim.x);

  // Ensure that the number of tiles in each stripe is a multiple of the groupsize; this avoids an annoying special case
  // where a stripe starts in the middle of group.
  if constexpr (group_blocks  >= thread_k_blocks)
    iters = (group_blocks / thread_k_blocks) * ceildiv(iters, (group_blocks / thread_k_blocks));

  int slice_row = (iters * blockIdx.x) % k_tiles;
  int slice_col_par = (iters * blockIdx.x) / k_tiles;

  int slice_col = slice_col_par;
  int slice_iters; // number of threadblock tiles in the current slice
  int slice_count = 0; // total number of active threadblocks in the current slice
  int slice_idx; // index of threadblock in current slice; numbered bottom to top

  // We can easily implement parallel problem execution by just remapping indices and advancing global pointers
  if (slice_col_par >= n_tiles) {
    A += (slice_col_par / n_tiles) * 16 * thread_m_blocks * prob_k / 32;
    C += (slice_col_par / n_tiles) * 16 * thread_m_blocks * prob_n / 4;
    D += (slice_col_par / n_tiles) * 16 * thread_m_blocks * prob_n / 8;
    s11 += (slice_col_par / n_tiles) * 16 * thread_m_blocks;
    locks += (slice_col_par / n_tiles) * n_tiles;
    slice_col = slice_col_par % n_tiles;
  }
  // Compute all information about the current slice which is required for synchronization.
  auto init_slice = [&] () {
    slice_iters = iters * (blockIdx.x + 1) - (k_tiles * slice_col_par + slice_row);
    if (slice_iters < 0 || slice_col_par >= n_tiles * parallel)
      slice_iters = 0;
    if (slice_iters == 0)
      return;
    if (slice_row + slice_iters > k_tiles)
      slice_iters = k_tiles - slice_row;
    slice_count = 1;
    slice_idx = 0;
    int col_first = iters * ceildiv(k_tiles * slice_col_par, iters);
    if (col_first <= k_tiles * (slice_col_par + 1)) {
      int col_off = col_first - k_tiles * slice_col_par;
      slice_count = ceildiv(k_tiles - col_off, iters);
      if (col_off > 0)
        slice_count++;
      int delta_first = iters * blockIdx.x - col_first;
      if (delta_first < 0 || (col_off == 0 && delta_first == 0))
        slice_idx = slice_count - 1;
      else {
        slice_idx = slice_count - 1 - delta_first / iters;
        if (col_off > 0)
          slice_idx--;
      }
    }
    if (slice_col == n_tiles) {
      A += 16 * thread_m_blocks * prob_k / 32;
      C += 16 * thread_m_blocks * prob_n / 4;
      D += 16 * thread_m_blocks * prob_n / 8;
      s11 += 16 * thread_m_blocks;
      locks += n_tiles;
      slice_col = 0;
    }
  };
  init_slice();

  int a_gl_stride = prob_k / 32; // stride of the A matrix in global memory
  // We typically use `constexpr` to indicate that this value is a compile-time constant
  constexpr int a_sh_stride = 32 * thread_k_blocks / 32; // stride of an A matrix tile in shared memory
  constexpr int a_gl_rd_delta_o = 32 * thread_k_blocks / 32; // delta between subsequent A tiles in global memory
  int a_gl_rd_delta_i = a_gl_stride * (threads / a_gl_rd_delta_o); // between subsequent accesses within a tile
  constexpr int a_sh_wr_delta = a_sh_stride * (threads / a_gl_rd_delta_o); // between shared memory writes
  constexpr int a_sh_rd_delta_o = 1 * ((threads / 32) / (thread_n_blocks / 4)); // between shared memory tile reads
  constexpr int a_sh_rd_delta_i = a_sh_stride * 16; // within a shared memory tile
  constexpr int a_sh_stage = a_sh_stride * (16 * thread_m_blocks); // overall size of a tile
  constexpr int a_sh_wr_iters = ceildiv(a_sh_stage, a_sh_wr_delta); // number of shared write iterations for a tile

  int b_gl_stride = prob_n;
  constexpr int b_sh_stride =  16 * thread_n_blocks ;
  constexpr int b_gl_rd_delta_o = 32 * thread_k_blocks / 32;
  constexpr int b_sh_wr_delta = 16 * thread_n_blocks ;
  constexpr int b_sh_rd_delta_o = 2 * b_sh_stride;
  constexpr int b_sh_rd_delta_i =  16;
  constexpr int b_sh_stage = b_gl_rd_delta_o * thread_n_blocks * 16;
  constexpr int b_sh_wr_iters = b_sh_stage / b_sh_stride;

  constexpr int s11_sh_stride = 16 * thread_m_blocks;
  constexpr int s22_sh_stride = 16 * thread_n_blocks / 8;

  int s11_gl_rd = threadIdx.x;
  int s11_sh_wr = threadIdx.x / 16 * 16 + (threadIdx.x % 16 % 8) * 2 + (threadIdx.x % 16 / 8);

  int s11_sh_rd = (threadIdx.x % 32) / 4;
  bool s11_sh_wr_pred = threadIdx.x < prob_m;

  int s22_gl_rd = s22_sh_stride * slice_col + threadIdx.x;
  int s22_sh_wr = threadIdx.x;
  int s22_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) +  ((threadIdx.x % 32) % 4);
  bool s22_sh_wr_pred = threadIdx.x < s22_sh_stride;

  // Global A read index of current thread.
  int a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);
  a_gl_rd += a_gl_rd_delta_o * slice_row;
  // Shared write index of current thread.
  int a_sh_wr = a_sh_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);
  // Shared read index.
  int a_sh_rd = a_sh_stride * ((threadIdx.x % 32) % 16);
  a_sh_rd += 1 * ((threadIdx.x / 32) / (thread_n_blocks / 4));
  // Global B read index of current thread.
  int b_gl_rd = threadIdx.x;
  b_gl_rd += b_gl_stride * b_gl_rd_delta_o * slice_row;
  b_gl_rd += b_sh_stride * slice_col;
  // Shared write index of current thread.
  int b_sh_wr = threadIdx.x;
  // Shared read index.
  int b_sh_rd = b_sh_stride * ((threadIdx.x % 32) % 8);

  b_sh_rd += 1 * ((threadIdx.x / 32) / (thread_n_blocks / 4)) + (threadIdx.x /32 %4) * 256 ;


  // Precompute which thread should not read memory in which iterations; this is needed if there are more threads than
  // required for a certain tilesize or when the batchsize is not a multiple of 16.
  bool a_sh_wr_pred[a_sh_wr_iters];
  #pragma unroll
  for (int i = 0; i < a_sh_wr_iters; i++)
    a_sh_wr_pred[i] = a_sh_wr_delta * i + a_sh_wr < a_sh_stride * prob_m;


  // To ensure that writing and reading A tiles to/from shared memory, the latter in fragment format, is fully bank
  // conflict free, we need to use a rather fancy XOR-based layout. The key here is that neither reads nor writes of
  // the 16-byte `int4` blocks of 8 consecutive threads involve the same shared memory banks. Further, it seems (based
  // on NSight-Compute) that each warp must also write a consecutive memory segment?
  auto transform_a = [&] (int i) {
    int row = i / a_gl_rd_delta_o;
    return a_gl_rd_delta_o * row + (i % a_gl_rd_delta_o) ^ row;
  };
  // Since the computation of this remapping is non-trivial and, due to our main loop unrolls, all shared memory
  // accesses are static, we simply precompute both transformed reads and writes.
  int a_sh_wr_trans[a_sh_wr_iters];
  #pragma unroll
  for (int i = 0; i < a_sh_wr_iters; i++)
    a_sh_wr_trans[i] = transform_a(a_sh_wr_delta * i + a_sh_wr);

  int a_sh_rd_trans[b_sh_wr_iters/2][thread_m_blocks];

  #pragma unroll
  for (int i = 0; i < b_sh_wr_iters/2; i++) {
    #pragma unroll
    for (int j = 0; j < thread_m_blocks; j++) {
        a_sh_rd_trans[i][j] = transform_a(a_sh_rd_delta_o * i + a_sh_rd_delta_i * j + a_sh_rd);

    }
  }

  // Since B-accesses have non-constant stride they have to be computed at runtime; we break dependicies between
  // subsequent accesses with a tile by maintining multiple pointers (we have enough registers), a tiny optimization.
  const int4* B_ptr[b_sh_wr_iters];
  #pragma unroll
  for (int i = 0; i < b_sh_wr_iters; i++)
    B_ptr[i] = B + b_gl_stride * i + b_gl_rd;

  int b_sh_wr_trans[b_sh_wr_iters];
  #pragma unroll
  for (int i = 0; i < b_sh_wr_iters; i++)
    b_sh_wr_trans[i] = b_sh_wr_delta * i + b_sh_wr;

  int b_sh_rd_trans[b_sh_wr_iters/2][4][2];
  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;
  if (lane_id < 8) {
    int base_offset = (lane_id % 8)  +  warp_id / (thread_n_blocks / 4) * b_sh_stride + (warp_id%4)* 64;
    #pragma unroll
    for (int i = 0; i < b_sh_wr_iters/2; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int offset = b_sh_rd_delta_o * i + b_sh_rd_delta_i * j + base_offset;
            b_sh_rd_trans[i][j][0] = offset;
            b_sh_rd_trans[i][j][1] = offset+8;
        }
    }
  }

  extern __shared__ int4 sh[];
  int4* sh_a = sh;
  int4* sh_b = sh_a + (stages * a_sh_stage);
  int4* sh_s11 = sh_b+ stages * b_sh_stage;
  int4* sh_s22 = sh_s11 + s11_sh_stride/8; //
  // Register storage for double buffer of shared memory reads.
  FragA frag_a[2][thread_m_blocks];
  FragB frag_b[2][4][2];
  FragC_f frag_c_f[thread_m_blocks][4][2];
  FragS_CHANNEL frag_s11[thread_m_blocks];
  FragS_CHANNEL frag_s22[2][4];

  // Zero accumulators.
  auto zero_accums = [&] () {
    #pragma unroll
    for (int i = 0; i < thread_m_blocks * 4 * 2 * 4; i++)
      reinterpret_cast<int*>(frag_c_f)[i] = 0;
  };

  // Asynchronously fetch the next A, B and s tile from global to the next shared memory pipeline location.
  auto fetch_to_shared = [&] (int pipe, int a_off, bool pred = true) {
    if (pred) {
      int4* sh_a_stage = sh_a + a_sh_stage * pipe;
      #pragma unroll
      for (int i = 0; i < a_sh_wr_iters; i++) {
        cp_async4_pred(
          &sh_a_stage[a_sh_wr_trans[i]],
          &A[a_gl_rd_delta_i * i + a_gl_rd + a_gl_rd_delta_o * a_off],
          a_sh_wr_pred[i]
        );
      }

      int4* sh_b_stage = sh_b + b_sh_stage * pipe;
      #pragma unroll
      for (int i = 0; i < b_sh_wr_iters; i++) {
        cp_async4(&sh_b_stage[b_sh_wr_trans[i]], B_ptr[i]);
        B_ptr[i] += b_gl_stride * b_gl_rd_delta_o;
      }
    }
    // Insert a fence even when we are winding down the pipeline to ensure that waiting is also correct at this point.
    cp_async_fence();
  };

  // Wait until the next thread tile has been loaded to shared memory.
  auto wait_for_stage = [&] () {
    // We only have `stages - 2` active fetches since we are double buffering and can only issue the next fetch when
    // it is guaranteed that the previous shared memory load is fully complete (as it may otherwise be overwritten).
    cp_async_wait<stages - 2>();
    __syncthreads();
  };

  // Load the next sub-tile from the current location in the shared memory pipe into the current register buffer.
  auto fetch_to_registers = [&] (int k, int pipe) {
    // It may seem inefficient that we reload the groups for every sub-tile; however, this does not seem to be a
    // significant bottleneck, while some theoretically better attempts have lead to bad instruction ordering by the
    // compiler and correspondingly a noticable drop in performance.
    int4* sh_a_stage = sh_a + a_sh_stage * pipe;
    #pragma unroll
    for (int i = 0; i < thread_m_blocks; i++)
       ldsm4(frag_a[k % 2][i], &sh_a_stage[a_sh_rd_trans[k % (b_sh_wr_iters/2)][i]]);

    int4* sh_b_stage = sh_b + b_sh_stage * pipe;
    #pragma unroll
    for (int j = 0; j < 4; j++){
       ldsm4_b(frag_b[k % 2][j][0], &sh_b_stage[b_sh_rd_trans[k % (b_sh_wr_iters/2)][j][0]]);
       ldsm4_b(frag_b[k % 2][j][1], &sh_b_stage[b_sh_rd_trans[k % (b_sh_wr_iters/2)][j][1]]);
    }
  };


  // Execute the actual tensor core matmul of a sub-tile.
  auto matmul = [&] (int k) {
    // We have the m dimension as the inner loop in order to encourage overlapping dequantization and matmul operations.
    #pragma unroll
    for (int j = 0; j < 4; j++) {
      #pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
        mma(frag_a[k % 2][i], frag_b[k%2][j][0], frag_c_f[i][j][0]);
        mma(frag_a[k % 2][i], frag_b[k%2][j][1], frag_c_f[i][j][1]);
      }
    }

  };

  // Since we slice across the k dimension of a tile in order to increase the number of warps while keeping the n
  // dimension of a tile reasonable, we have multiple warps that accumulate their partial sums of the same output
  // location; which we have to reduce over in the end. We do in shared memory.
  auto thread_block_reduce = [&] () {
    constexpr int b_sh_stride_1=128;
    constexpr int red_off = threads / b_sh_stride_1 / 2;
    if (red_off >= 1) {
      int red_idx = threadIdx.x / b_sh_stride_1;
      constexpr int red_sh_stride = b_sh_stride_1 * 4 * 2;
      constexpr int red_sh_delta = b_sh_stride_1;
      int red_sh_rd = red_sh_stride * (threadIdx.x / b_sh_stride_1) + (threadIdx.x % b_sh_stride_1);
      #pragma unroll
      for (int m_block = 0; m_block < thread_m_blocks; m_block++) {
        #pragma unroll
        for (int i = red_off; i > 0; i /= 2) {
          if (i <= red_idx && red_idx < 2 * i) {
            #pragma unroll
            for (int j = 0; j < 4 * 2; j++) {
              int red_sh_wr = red_sh_delta * j + (red_sh_rd - red_sh_stride * i);
              if (i < red_off) {
                int* c_rd = reinterpret_cast<int*>(&sh[red_sh_delta * j + red_sh_rd]);
                int* c_wr = reinterpret_cast<int*>(&sh[red_sh_wr]);
                #pragma unroll
                for (int k = 0; k < 4; k++)
                  reinterpret_cast<FragC_f*>(frag_c_f)[4 * 2 * m_block + j][k] += c_rd[k] + c_wr[k];
              }
              sh[red_sh_wr] = reinterpret_cast<int4*>(&frag_c_f)[4 * 2 * m_block + j];
            }
          }
          __syncthreads();
        }
        if (red_idx == 0) {
          #pragma unroll
          for (int i = 0; i < 4 * 2; i++) {
            int* c_rd = reinterpret_cast<int*>(&sh[red_sh_delta * i + red_sh_rd]);
            #pragma unroll
            for (int j = 0; j < 4; j++)
              reinterpret_cast<FragC_f*>(frag_c_f)[4 * 2 * m_block + i][j] += c_rd[j];
          }
        }
        __syncthreads();
      }
    }
  };


  auto global_reduce = [&] (bool first = false, bool last = false) {
    constexpr int active_threads = 32 * thread_n_blocks / 4;
    if (threadIdx.x < active_threads) {
      int c_gl_stride = prob_n / 4;
      int c_gl_wr_delta_o = 8 * c_gl_stride;
      int c_gl_wr_delta_i = 8 * (active_threads / 32);
      int c_gl_wr = c_gl_stride * ((threadIdx.x % 32) / 4) + 8 * (threadIdx.x / 32) + (threadIdx.x % 4) * 2;
      c_gl_wr += (4 * thread_n_blocks) * slice_col;
      constexpr int c_sh_wr_delta = active_threads * 2;
      int c_sh_wr = 2 * threadIdx.x;
      int row = (threadIdx.x % 32) / 4;

      if (!first) {
        #pragma unroll
        for (int i = 0; i < thread_m_blocks * 4; i++) {
          cp_async4_pred(
            &sh[c_sh_wr + c_sh_wr_delta * i],
            &C[c_gl_wr + c_gl_wr_delta_o * (i / 2) + c_gl_wr_delta_i * (i % 2)],
            i < (thread_m_blocks - 1) * 4 || 8 * (i / 2) + row < prob_m
          );

          cp_async4_pred(
            &sh[c_sh_wr + c_sh_wr_delta * i + 1],
            &C[c_gl_wr + c_gl_wr_delta_o * (i / 2) + c_gl_wr_delta_i * (i % 2) + 1],
            i < (thread_m_blocks - 1) * 4 || 8 * (i / 2) + row < prob_m
          );
        }
        cp_async_fence();
        cp_async_wait<0>();
      }

      #pragma unroll
      for (int i = 0; i < thread_m_blocks * 4; i++) {
        if (i < (thread_m_blocks - 1) * 4 || 8 * (i / 2) + row < prob_m) {
          if (!first) {
            int4 d_red1 = sh[c_sh_wr + i * c_sh_wr_delta];
            int4 d_red2 = sh[c_sh_wr + i * c_sh_wr_delta + 1];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
              reinterpret_cast<int*>(&frag_c_f)[4 * 2 * 4 * (i / 4) + 4 * j + (i % 4)] +=
                reinterpret_cast<int*>(&d_red1)[j];
            }
            #pragma unroll
            for (int j = 0; j < 4; j++) {
              reinterpret_cast<int*>(&frag_c_f)[4 * 2 * 4 * (i / 4) + 4 * (j + 4) + (i % 4)] +=
                reinterpret_cast<int*>(&d_red2)[j];
            }
          }
          if (!last) {
            int4 d1, d2;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
              reinterpret_cast<int*>(&d1)[j] =
                reinterpret_cast<int*>(&frag_c_f)[4 * 2 * 4 * (i / 4) + 4 * j + (i % 4)];
            }
            #pragma unroll
            for (int j = 0; j < 4; j++) {
              reinterpret_cast<int*>(&d2)[j] =
                reinterpret_cast<int*>(&frag_c_f)[4 * 2 * 4 * (i / 4) + 4 * (j + 4) + (i % 4)];
            }
            C[c_gl_wr + c_gl_wr_delta_o * (i / 2) + c_gl_wr_delta_i * (i % 2)] = d1;
            C[c_gl_wr + c_gl_wr_delta_o * (i / 2) + c_gl_wr_delta_i * (i % 2) + 1] = d2;
          }
        }
      }
    }
  };

  // Write out the reduce final result in the correct layout. We only actually reshuffle matrix fragments in this step,
  // the reduction above is performed in fragment layout.
  auto write_result = [&] () {
    int d_gl_stride = prob_n / 8;
    constexpr int d_sh_stride = 2 * thread_n_blocks + 1;
    int d_gl_wr_delta = d_gl_stride * (threads / (2 * thread_n_blocks));
    constexpr int d_sh_rd_delta = d_sh_stride * (threads / (2 * thread_n_blocks));

    int d_gl_wr = d_gl_stride * (threadIdx.x / (2 * thread_n_blocks)) + (threadIdx.x % (2 * thread_n_blocks));
    d_gl_wr += (2 * thread_n_blocks) * slice_col;
    int d_sh_wr = (4 * d_sh_stride) * ((threadIdx.x % 32) / 4) + (threadIdx.x % 32) % 4;
    d_sh_wr += 32 * (threadIdx.x / 32);
    int d_sh_rd = d_sh_stride * (threadIdx.x / (2 * thread_n_blocks)) + (threadIdx.x % (2 * thread_n_blocks));

    int d_gl_wr_end = d_gl_stride * prob_m;

    // We first reorder in shared memory to guarantee the most efficient final global write patterns
    auto write = [&] (int idx, int c0, int c1, half a_s, FragS_CHANNEL& w_s) {
      float2 deq_res;
      deq_res.x = int32_to_float(c0) * __half2float(w_s[0]) * __half2float(a_s);
      deq_res.y = int32_to_float(c1) * __half2float(w_s[1]) * __half2float(a_s);
      ((half2*) sh)[idx] = float2_to_half2(deq_res);
    };

    if (threadIdx.x / 32 < thread_n_blocks / 4) {
      #pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
          int wr = d_sh_wr + 8 * j;
          write(wr + (4 * d_sh_stride) * 0 + 0, frag_c_f[i][j][0][0], frag_c_f[i][j][0][1], frag_s11[i][0], frag_s22[j / 2][2 * (j % 2) + 0]);
          write(wr + (4 * d_sh_stride) * 8 + 0, frag_c_f[i][j][0][2], frag_c_f[i][j][0][3], frag_s11[i][1], frag_s22[j / 2][2 * (j % 2) + 0]);
          write(wr + (4 * d_sh_stride) * 0 + 4, frag_c_f[i][j][1][0], frag_c_f[i][j][1][1], frag_s11[i][0], frag_s22[j / 2][2 * (j % 2) + 1]);
          write(wr + (4 * d_sh_stride) * 8 + 4, frag_c_f[i][j][1][2], frag_c_f[i][j][1][3], frag_s11[i][1], frag_s22[j / 2][2 * (j % 2) + 1]);
        }
        d_sh_wr += 16 * (4 * d_sh_stride);
      }
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < ceildiv(16 * thread_m_blocks, threads / (2 * thread_n_blocks)); i++) {
      if (d_gl_wr < d_gl_wr_end) {
        D[d_gl_wr] = sh[d_sh_rd];
        d_gl_wr += d_gl_wr_delta;
        d_sh_rd += d_sh_rd_delta;
      }
    }
  };

  // Start global fetch and register load pipelines.
  auto start_pipes = [&] () {
    #pragma unroll
    for (int i = 0; i < stages - 1; i++){
      fetch_to_shared(i, i, i < slice_iters);
    }

    zero_accums();
    wait_for_stage();
    fetch_to_registers(0, 0);
    a_gl_rd += a_gl_rd_delta_o * (stages - 1);
  };
  start_pipes();

  // Main loop.
  while (slice_iters) {
    // We unroll over both the global fetch and the register load pipeline to ensure all shared memory accesses are
    // static. Note that both pipelines have even length meaning that the next iteration will always start at index 0.
    #pragma unroll
    for (int pipe = 0; pipe < stages;) {
      #pragma unroll
      for (int k = 0; k < b_sh_wr_iters/2; k++) {
        fetch_to_registers(k + 1, pipe % stages);
        if (k == b_sh_wr_iters/2 - 2) {
          fetch_to_shared((pipe + stages - 1) % stages, pipe, slice_iters >= stages);
          pipe++;
          wait_for_stage();
        }
        matmul(k);
      }
      slice_iters--;
      if (slice_iters == 0)
        break;
    }
    a_gl_rd += a_gl_rd_delta_o * stages;
    // Process results and, if necessary, proceed to the next column slice. While this pattern may not be the most
    // readable, other ways of writing the loop seemed to noticeably worse performance after compliation.
    if (slice_iters == 0) {
      cp_async_wait<0>();
      bool last = slice_idx == slice_count - 1;
      // For per-column scales, we only fetch them here in the final step before write-out
      if (last) {
        if (s11_sh_wr_pred) {
            reinterpret_cast<half*>(sh_s11)[s11_sh_wr] = s11[s11_gl_rd];
        }
        if (s22_sh_wr_pred) {
          cp_async4(&sh_s22[s22_sh_wr], &s22[s22_gl_rd]);
        }
        cp_async_fence();
      }
      thread_block_reduce();
      if (last) {
        cp_async_wait<0>();
        __syncthreads();
        if (threadIdx.x / 32 < thread_n_blocks / 4) {
          #pragma unroll
          for (int i = 0; i < thread_m_blocks; i++) {
            frag_s11[i][0] = reinterpret_cast<const half*>(sh_s11)[16 * i + 2 * s11_sh_rd];
            frag_s11[i][1] = reinterpret_cast<const half*>(sh_s11)[16 * i + 2 * s11_sh_rd + 1];
          }
          reinterpret_cast<int4*>(&frag_s22)[0] = sh_s22[s22_sh_rd + 0];
          reinterpret_cast<int4*>(&frag_s22)[1] = sh_s22[s22_sh_rd + 4];
        }
      }

      if (slice_count > 1) { // only globally reduce if there is more than one block in a slice
        barrier_acquire(&locks[slice_col], slice_idx);
        global_reduce(slice_idx == 0, last);
        barrier_release(&locks[slice_col], last);
      }
      if (last) // only the last block in a slice actually writes the result
        write_result();
      slice_row = 0;
      slice_col_par++;
      slice_col++;

      init_slice();
      if (slice_iters) {
        a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);
        #pragma unroll
        for (int i = 0; i < b_sh_wr_iters; i++)
          B_ptr[i] +=  b_sh_stride - b_gl_stride * b_gl_rd_delta_o * k_tiles;
        if (slice_col == 0) {
          #pragma unroll
          for (int i = 0; i < b_sh_wr_iters; i++)
            B_ptr[i] -= prob_n;
        }
        s22_gl_rd = s22_sh_stride * slice_col + threadIdx.x;
        start_pipes();
      }
    }
  }
}



// 8 warps are a good choice since every SM has 4 schedulers and having more than 1 warp per schedule allows some more
// latency hiding. At the same time, we want relatively few warps to have many registers per warp and small tiles.
const int THREADS = 256;
const int STAGES = 4; // 4 pipeline stages fit into shared memory

#define CALL_IF(THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, GROUP_BLOCKS) \
  else if ( \
    thread_m_blocks == THREAD_M_BLOCKS && thread_n_blocks == THREAD_N_BLOCKS && thread_k_blocks == THREAD_K_BLOCKS && \
    group_blocks == GROUP_BLOCKS \
  ) { \
    cudaFuncSetAttribute( \
      Marlin_w4a4_7<THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, STAGES, GROUP_BLOCKS>, \
      cudaFuncAttributeMaxDynamicSharedMemorySize, \
      max_shared_mem \
    ); \
    Marlin_w4a4_7< \
      THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, STAGES, GROUP_BLOCKS \
    ><<<blocks, THREADS, max_shared_mem, stream>>>( \
      A_ptr, B_ptr, C_ptr, D_ptr, s11_ptr, s22_ptr, \
      prob_m, prob_n, prob_k, \
      locks \
    ); \
  }

const int ERR_PROB_SHAPE = 1;
const int ERR_KERN_SHAPE = 2;

int apex4_cuda_w4a4_channel(
  const void* A,
  const void* B,
        void* C,
        void* D,
        void* s11,
        void* s22,
  int prob_m,
  int prob_n,
  int prob_k,
  void* workspace,
  int groupsize = -1,
  int dev = 0,
  cudaStream_t stream = 0,
  int thread_k = -1,
  int thread_n = -1,
  int sms = -1,
  int max_par = 16
) {
  int tot_m = prob_m;
  int tot_m_blocks = ceildiv(tot_m, 16);
  int pad = 16 * tot_m_blocks - tot_m;
  if (sms == -1)
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  int max_shared_mem = 0;
  cudaDeviceGetAttribute(&max_shared_mem,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  if (thread_k == -1 || thread_n == -1) {
    if (prob_m <= 16) {
      // For small batchizes, better partioning is slightly more important than better compute utilization
      thread_k = 128;
      thread_n = 128;
    } else {
      thread_k = 128;
      thread_n = 256;
    }
  }

  int thread_k_blocks = thread_k / 32;
  int thread_n_blocks = thread_n / 16;
  int group_blocks = (groupsize == -1) ? -1 : groupsize / 32;
  int blocks = sms;

  if (prob_n % thread_n != 0 || prob_k % thread_k != 0 || (group_blocks != -1 && prob_k % group_blocks != 0))
    return ERR_PROB_SHAPE;
  if (prob_m == 0 || prob_n == 0 || prob_k == 0)
    return 0;

  const int4* A_ptr = (const int4*) A;
  const int4* B_ptr = (const int4*) B;
  int4* C_ptr = (int4*) C;
  int4* D_ptr = (int4*) D;
  const half* s11_ptr = (const half*) s11;
  const int4* s22_ptr = (const int4*) s22;

  int* locks = (int*) workspace;

  int ret = 0;
  for (int i = 0; i < tot_m_blocks; i += 4) {
    int thread_m_blocks = tot_m_blocks - i;
    prob_m = tot_m - 16 * i;
    int par = 1;
    if (thread_m_blocks > 4) {
      // Note that parallel > 1 currently only works for inputs without any padding
      par = (16 * thread_m_blocks - pad) / 64;
      if (par > max_par)
        par = max_par;
      prob_m = 64 * par;
      i += 4 * (par - 1);
      thread_m_blocks = 4;
    }

    if (false) {}
    //thread_m_blocks, thread_n_blocks,thread_k_blocks, group_blocks = groupsize / 16
    CALL_IF(1, 16,  4,  -1)
    CALL_IF(2, 16,  4,  -1)
    CALL_IF(3, 16,  4,  -1)
    CALL_IF(4, 16,  4,  -1)

    CALL_IF(1,  8,  4,  1)
    CALL_IF(1,  8,  4,  2)
    CALL_IF(1,  8,  4,  4)
    CALL_IF(1,  8,  4,  8)
    CALL_IF(1,  8,  4,  16)
    CALL_IF(1,  8,  4,  32)

    CALL_IF(1, 16,  4,  1)
    CALL_IF(1, 16,  4,  2)
    CALL_IF(1, 16,  4,  4)
    CALL_IF(1, 16,  4,  8)
    CALL_IF(1, 16,  4,  16)
    CALL_IF(1, 16,  4,  32)

    CALL_IF(2, 16,  4,  1)
    CALL_IF(2, 16,  4,  2)
    CALL_IF(2, 16,  4,  4)
    CALL_IF(2, 16,  4,  8)
    CALL_IF(2, 16,  4,  16)
    CALL_IF(2, 16,  4,  32)

    CALL_IF(3, 16,  4,  1)
    CALL_IF(3, 16,  4,  2)
    CALL_IF(3, 16,  4,  4)
    CALL_IF(3, 16,  4,  8)
    CALL_IF(3, 16,  4,  16)
    CALL_IF(3, 16,  4,  32)

    CALL_IF(4, 16,  4,  1)
    CALL_IF(4, 16,  4,  2)
    CALL_IF(4, 16,  4,  4)
    CALL_IF(4, 16,  4,  8)
    CALL_IF(4, 16,  4,  16)
    CALL_IF(4, 16,  4,  32)
    else
      ret = ERR_KERN_SHAPE;

    A_ptr += 16 * thread_m_blocks * (prob_k / 32) * par;
    D_ptr += 16 * thread_m_blocks * (prob_n / 8) * par;
    s11_ptr += 16 * thread_m_blocks * par;


  }

  return ret;
}

void apex4_gemm_w4a4_channel(
  const torch::Tensor& A,
  const torch::Tensor& B,
        torch::Tensor& C,
        torch::Tensor& D,
  const torch::Tensor& s11,
  const torch::Tensor& s22,
        torch::Tensor& workspace,
  int thread_k = -1,
  int thread_n = -1,
  int sms = -1,
  int max_par = 8
) {
  int prob_m = A.size(0);
  int prob_n = C.size(1);
  int prob_k = A.size(1)*8;//Here A is int, prob_k is the size of 4bit_A

  int groupsize = -1;

  if (workspace.numel() < prob_n / 128 * max_par)
    AT_ERROR("workspace must be of size at least ", prob_n / 128 * max_par, ".");
  if (s11.dtype() != torch::kFloat16)
     AT_ERROR("s1 dtype must be kFloat16, but got ", s11.dtype(), ".");
  if (s22.dtype() != torch::kFloat16)
     AT_ERROR("s2 dtype must be kFloat16, but got ", s22.dtype(), ".");

  int dev = A.get_device();
  int err = apex4_cuda_w4a4_channel(
    A.data_ptr(),
    B.data_ptr(),
    C.data_ptr(),
    D.data_ptr(),
    s11.data_ptr(),
    s22.data_ptr(),
    prob_m, prob_n, prob_k,
    workspace.data_ptr(),
    groupsize,
    dev,
    at::cuda::getCurrentCUDAStream(dev),
    thread_k,
    thread_n,
    sms,
    max_par
  );

  if (err == ERR_PROB_SHAPE) {
    AT_ERROR(
      "Problem (m=", prob_m, ", n=", prob_n, ", k=", prob_k, ")",
      " not compatible with thread_k=", thread_k, ", thread_n=", thread_n, "."
    );
  } else if (err == ERR_KERN_SHAPE) {
    AT_ERROR(
      "No kernel implementation for thread_k=", thread_k, ", thread_n=", thread_n, ", groupsize=", groupsize, "."
    );
  }
}
