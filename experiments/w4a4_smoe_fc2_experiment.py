#!/usr/bin/env python3
"""Isolated APEX4-style W4A4 experiment for the SMoE fc2 shape.

This script does not change infer.py execution. It benchmarks one routed fc2
layer shape:

    h_route[pool, 1024] x w2[expert, 512, 1024]^T -> y_route[pool, 512]

Each expert is executed with the dense APEX4 W4A4 GEMM independently. That keeps
the prototype small and tells us whether the activation-quantized INT4 GEMM is
worth turning into a real grouped route-pool kernel later.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
PROJECT_LIBRARIES = (ROOT / "libraries").resolve()
TORCH_EXTENSIONS_DIR = (ROOT / ".torch_extensions").resolve()

if PROJECT_LIBRARIES.exists():
    sys.path.insert(0, str(PROJECT_LIBRARIES))

os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(TORCH_EXTENSIONS_DIR))

LOCAL_NINJA = PROJECT_LIBRARIES / "bin" / "ninja"
if LOCAL_NINJA.exists():
    os.environ["PATH"] = str(LOCAL_NINJA.parent) + os.pathsep + os.environ.get("PATH", "")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline


APEX4_KERNELS = ROOT / "APEX4-W4A4" / "kernels"
NUM_EXPERTS = 8
IN_DIM = 1024
OUT_DIM = 512


@dataclass
class ExpertW4A4:
    layer: nn.Module
    qdq_weight: torch.Tensor


_UNIFORM_PACK_EXT = None
_GROUPED_INT4_EXT = None


def _require_experiment_enabled() -> None:
    if os.environ.get("USE_W4A4_SMOE_EXPERIMENT", "0") == "0":
        raise SystemExit(
            "Refusing to run unless USE_W4A4_SMOE_EXPERIMENT=1 is set. "
            "This keeps the W4A4 prototype isolated from normal inference."
        )


def _load_apex4(build: bool):
    sys.path.insert(0, str(APEX4_KERNELS))
    try:
        import csrc  # type: ignore

        return csrc
    except Exception as exc:
        if not build:
            raise RuntimeError(
                "Could not import APEX4 kernels. Re-run with --build-apex4 "
                "or build them manually with: cd APEX4-W4A4/kernels && "
                "python setup.py build_ext --inplace"
            ) from exc

    build_env = os.environ.copy()
    build_env["PYTHONPATH"] = (
        str(PROJECT_LIBRARIES)
        + os.pathsep
        + build_env.get("PYTHONPATH", "")
    )
    subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=APEX4_KERNELS,
        env=build_env,
        check=True,
    )
    import csrc  # type: ignore

    return csrc


def _load_uniform_pack_ext():
    global _UNIFORM_PACK_EXT
    if _UNIFORM_PACK_EXT is not None:
        return _UNIFORM_PACK_EXT

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
    cpp_source = """
void uniform_pack_w4a4(torch::Tensor x, torch::Tensor packed, torch::Tensor scales, double scale);
"""
    cuda_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

constexpr int kInDim = 1024;
constexpr int kPacksPerRow = kInDim / 8;

__global__ void uniform_pack_w4a4_kernel(
    const __half* __restrict__ x,
    int32_t* __restrict__ packed,
    __half* __restrict__ scales,
    int rows,
    int groups,
    float inv_scale,
    __half scale
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows) {
        return;
    }

    if (tid < kPacksPerRow) {
        uint32_t word = 0;
        const int base = row * kInDim + tid * 8;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float v = __half2float(x[base + i]) * inv_scale;
            int q = __float2int_rn(v);
            q = max(-8, min(7, q));
            word |= (static_cast<uint32_t>(q) & 0xFu) << (4 * i);
        }
        packed[row * kPacksPerRow + tid] = static_cast<int32_t>(word);
    }

    if (tid < groups) {
        scales[row * groups + tid] = scale;
    }
}

void uniform_pack_w4a4(torch::Tensor x, torch::Tensor packed, torch::Tensor scales, double scale) {
    TORCH_CHECK(x.is_cuda() && packed.is_cuda() && scales.is_cuda(), "uniform_pack_w4a4 expects CUDA tensors");
    TORCH_CHECK(x.dtype() == torch::kFloat16, "x must be fp16");
    TORCH_CHECK(packed.dtype() == torch::kInt32, "packed must be int32");
    TORCH_CHECK(scales.dtype() == torch::kFloat16, "scales must be fp16");
    TORCH_CHECK(x.is_contiguous() && packed.is_contiguous() && scales.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(x.dim() == 2 && x.size(1) == kInDim, "x must have shape [M,1024]");
    TORCH_CHECK(packed.dim() == 2 && packed.size(0) == x.size(0) && packed.size(1) == kPacksPerRow,
        "packed must have shape [M,128]");
    TORCH_CHECK(scales.dim() == 2 && scales.size(0) == x.size(0), "scales must have shape [M,groups]");
    TORCH_CHECK(scale > 0.0, "scale must be positive");

    const int rows = static_cast<int>(x.size(0));
    const int groups = static_cast<int>(scales.size(1));
    const float scale_f = static_cast<float>(scale);
    const float inv_scale = 1.0f / scale_f;
    const dim3 grid(rows);
    const dim3 block(128);
    uniform_pack_w4a4_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(x.data_ptr<c10::Half>()),
        packed.data_ptr<int32_t>(),
        reinterpret_cast<__half*>(scales.data_ptr<c10::Half>()),
        rows,
        groups,
        inv_scale,
        __float2half(scale_f)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""
    _UNIFORM_PACK_EXT = load_inline(
        name="w4a4_uniform_pack_ext",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["uniform_pack_w4a4"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )
    return _UNIFORM_PACK_EXT


def _load_grouped_int4_ext():
    global _GROUPED_INT4_EXT
    if _GROUPED_INT4_EXT is not None:
        return _GROUPED_INT4_EXT

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
    cpp_source = """
void pack_routes_uniform_int4(torch::Tensor x, torch::Tensor packed, torch::Tensor counts, torch::Tensor offsets, double scale);
void grouped_fc2_int4_simt(torch::Tensor a_pack, torch::Tensor w_pack, torch::Tensor bias, torch::Tensor counts, torch::Tensor offsets, torch::Tensor out, double out_scale);
void grouped_fc2_int4_wmma(torch::Tensor a_pack, torch::Tensor w_pack, torch::Tensor bias, torch::Tensor counts, torch::Tensor offsets, torch::Tensor out, double out_scale);
"""
    cuda_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

constexpr int kNumExperts = 8;
constexpr int kInDim = 1024;
constexpr int kOutDim = 512;
constexpr int kPacksPerRow = kInDim / 8;
constexpr int kTileM = 16;
constexpr int kTileN = 16;
constexpr int kWmmaM = 8;
constexpr int kWmmaN = 8;
constexpr int kWmmaK = 32;
constexpr int kCtaM = 16;
constexpr int kCtaN = 128;
constexpr int kWarpsPerCta = 8;
constexpr int kWarpNFragments = 4;
constexpr int kPackRowsPerCta = 2;

__device__ __forceinline__ int find_expert(const int32_t* offsets, int pooled_row) {
#pragma unroll
    for (int expert = 0; expert < kNumExperts; ++expert) {
        if (pooled_row >= offsets[expert] && pooled_row < offsets[expert + 1]) {
            return expert;
        }
    }
    return kNumExperts;
}

__device__ __forceinline__ int unpack_s4(uint32_t word, int idx) {
    int q = static_cast<int>((word >> (4 * idx)) & 0xFu);
    return q >= 8 ? q - 16 : q;
}

__global__ void pack_routes_uniform_int4_kernel(
    const __half* __restrict__ x,
    int32_t* __restrict__ packed,
    const int32_t* __restrict__ counts,
    const int32_t* __restrict__ offsets,
    int total_rows,
    float inv_scale
) {
    const int row = blockIdx.x * kPackRowsPerCta + threadIdx.x / kPacksPerRow;
    const int tid = threadIdx.x % kPacksPerRow;
    if (row >= total_rows || tid >= kPacksPerRow) {
        return;
    }

    const int expert = find_expert(offsets, row);
    uint32_t word = 0;
    if (expert < kNumExperts) {
        const int local = row - offsets[expert];
        if (local < counts[expert]) {
            const int base = row * kInDim + tid * 8;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float v = __half2float(x[base + i]) * inv_scale;
                int q = __float2int_rn(v);
                q = max(-8, min(7, q));
                word |= (static_cast<uint32_t>(q) & 0xFu) << (4 * i);
            }
        }
    }
    packed[row * kPacksPerRow + tid] = static_cast<int32_t>(word);
}

__global__ void grouped_fc2_int4_simt_kernel(
    const int32_t* __restrict__ a_pack,
    const int32_t* __restrict__ w_pack,
    const __half* __restrict__ bias,
    const int32_t* __restrict__ counts,
    const int32_t* __restrict__ offsets,
    __half* __restrict__ out,
    int total_rows,
    float out_scale
) {
    const int pooled_row = blockIdx.y * kTileM + threadIdx.y;
    const int col = blockIdx.x * kTileN + threadIdx.x;
    if (pooled_row >= total_rows || col >= kOutDim) {
        return;
    }

    const int expert = find_expert(offsets, pooled_row);
    if (expert >= kNumExperts) {
        return;
    }
    const int local = pooled_row - offsets[expert];
    if (local >= counts[expert]) {
        return;
    }

    int acc = 0;
#pragma unroll 4
    for (int pack = 0; pack < kPacksPerRow; ++pack) {
        const uint32_t a_word = static_cast<uint32_t>(a_pack[pooled_row * kPacksPerRow + pack]);
        const uint32_t w_word = static_cast<uint32_t>(
            w_pack[(expert * kOutDim + col) * kPacksPerRow + pack]);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc += unpack_s4(a_word, i) * unpack_s4(w_word, i);
        }
    }

    float value = static_cast<float>(acc) * out_scale;
    if (bias != nullptr) {
        value += __half2float(bias[expert * kOutDim + col]);
    }
    out[pooled_row * kOutDim + col] = __float2half_rn(value);
}

__global__ void grouped_fc2_int4_wmma_kernel(
    const int32_t* __restrict__ a_pack,
    const int32_t* __restrict__ w_pack,
    const __half* __restrict__ bias,
    const int32_t* __restrict__ counts,
    const int32_t* __restrict__ offsets,
    __half* __restrict__ out,
    int total_rows,
    float out_scale
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)
    using namespace nvcuda;
    using FragA = wmma::fragment<
        wmma::matrix_a,
        kWmmaM,
        kWmmaN,
        kWmmaK,
        wmma::experimental::precision::s4,
        wmma::row_major>;
    using FragB = wmma::fragment<
        wmma::matrix_b,
        kWmmaM,
        kWmmaN,
        kWmmaK,
        wmma::experimental::precision::s4,
        wmma::col_major>;
    using FragC = wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, int>;

    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int warp_row_tile = warp_id >> 2;
    const int warp_col_pair = warp_id & 3;
    const int row_base = blockIdx.y * kCtaM + warp_row_tile * kWmmaM;
    const int col_base = blockIdx.x * kCtaN + warp_col_pair * (kWmmaN * kWarpNFragments);
    if (row_base >= total_rows || col_base >= kOutDim) {
        return;
    }

    const int expert = find_expert(offsets, row_base);
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
    for (int k = 0; k < kInDim; k += kWmmaK) {
        FragA a_frag;
        FragB b_frag0;
        FragB b_frag1;
        FragB b_frag2;
        FragB b_frag3;
        const void* a_ptr = static_cast<const void*>(
            a_pack + row_base * kPacksPerRow + (k / 8));
        const void* b_ptr0 = static_cast<const void*>(
            w_pack + (expert * kOutDim + col_base) * kPacksPerRow + (k / 8));
        const void* b_ptr1 = static_cast<const void*>(
            w_pack + (expert * kOutDim + col_base + kWmmaN) * kPacksPerRow + (k / 8));
        const void* b_ptr2 = static_cast<const void*>(
            w_pack + (expert * kOutDim + col_base + 2 * kWmmaN) * kPacksPerRow + (k / 8));
        const void* b_ptr3 = static_cast<const void*>(
            w_pack + (expert * kOutDim + col_base + 3 * kWmmaN) * kPacksPerRow + (k / 8));
        wmma::load_matrix_sync(a_frag, a_ptr, kInDim);
        wmma::load_matrix_sync(b_frag0, b_ptr0, kInDim);
        wmma::load_matrix_sync(b_frag1, b_ptr1, kInDim);
        wmma::load_matrix_sync(b_frag2, b_ptr2, kInDim);
        wmma::load_matrix_sync(b_frag3, b_ptr3, kInDim);
        wmma::mma_sync(acc0, a_frag, b_frag0, acc0, false);
        wmma::mma_sync(acc1, a_frag, b_frag1, acc1, false);
        wmma::mma_sync(acc2, a_frag, b_frag2, acc2, false);
        wmma::mma_sync(acc3, a_frag, b_frag3, acc3, false);
    }

    __shared__ int acc_tile[kWarpsPerCta][kWarpNFragments][kWmmaM * kWmmaN];
    wmma::store_matrix_sync(acc_tile[warp_id][0], acc0, kWmmaN, wmma::mem_row_major);
    wmma::store_matrix_sync(acc_tile[warp_id][1], acc1, kWmmaN, wmma::mem_row_major);
    wmma::store_matrix_sync(acc_tile[warp_id][2], acc2, kWmmaN, wmma::mem_row_major);
    wmma::store_matrix_sync(acc_tile[warp_id][3], acc3, kWmmaN, wmma::mem_row_major);
    __syncwarp();

#pragma unroll
    for (int frag_n = 0; frag_n < kWarpNFragments; ++frag_n) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int idx = lane + i * 32;
            const int local_row = idx / kWmmaN;
            const int local_col = idx - local_row * kWmmaN;
            const int row = row_base + local_row;
            const int col = col_base + frag_n * kWmmaN + local_col;
            const int expert_local_row = local_base + local_row;
            if (row < total_rows && col < kOutDim && expert_local_row < counts[expert]) {
                float value = static_cast<float>(acc_tile[warp_id][frag_n][idx]) * out_scale;
                if (bias != nullptr) {
                    value += __half2float(bias[expert * kOutDim + col]);
                }
                out[row * kOutDim + col] = __float2half_rn(value);
            }
        }
    }
#endif
}

void pack_routes_uniform_int4(torch::Tensor x, torch::Tensor packed, torch::Tensor counts, torch::Tensor offsets, double scale) {
    TORCH_CHECK(x.is_cuda() && packed.is_cuda() && counts.is_cuda() && offsets.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat16, "x must be fp16");
    TORCH_CHECK(packed.dtype() == torch::kInt32, "packed must be int32");
    TORCH_CHECK(counts.dtype() == torch::kInt32 && offsets.dtype() == torch::kInt32, "counts/offsets must be int32");
    TORCH_CHECK(x.is_contiguous() && packed.is_contiguous() && counts.is_contiguous() && offsets.is_contiguous(), "all tensors must be contiguous");
    TORCH_CHECK(x.dim() == 2 && x.size(1) == kInDim, "x must have shape [pool,1024]");
    TORCH_CHECK(packed.dim() == 2 && packed.size(0) == x.size(0) && packed.size(1) == kPacksPerRow, "packed must have shape [pool,128]");
    TORCH_CHECK(counts.numel() == kNumExperts && offsets.numel() == kNumExperts + 1, "counts [8], offsets [9] required");
    TORCH_CHECK(scale > 0.0, "scale must be positive");

    const int rows = static_cast<int>(x.size(0));
    pack_routes_uniform_int4_kernel<<<(rows + kPackRowsPerCta - 1) / kPackRowsPerCta, kPackRowsPerCta * kPacksPerRow, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(x.data_ptr<c10::Half>()),
        packed.data_ptr<int32_t>(),
        counts.data_ptr<int32_t>(),
        offsets.data_ptr<int32_t>(),
        rows,
        1.0f / static_cast<float>(scale));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void check_grouped_fc2_int4_args(torch::Tensor a_pack, torch::Tensor w_pack, torch::Tensor bias, torch::Tensor counts, torch::Tensor offsets, torch::Tensor out) {
    TORCH_CHECK(a_pack.is_cuda() && w_pack.is_cuda() && counts.is_cuda() && offsets.is_cuda() && out.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(a_pack.dtype() == torch::kInt32 && w_pack.dtype() == torch::kInt32, "packed tensors must be int32");
    TORCH_CHECK(out.dtype() == torch::kFloat16, "out must be fp16");
    TORCH_CHECK(counts.dtype() == torch::kInt32 && offsets.dtype() == torch::kInt32, "counts/offsets must be int32");
    TORCH_CHECK(a_pack.is_contiguous() && w_pack.is_contiguous() && out.is_contiguous(), "packed/out tensors must be contiguous");
    TORCH_CHECK(a_pack.dim() == 2 && a_pack.size(1) == kPacksPerRow, "a_pack must have shape [pool,128]");
    TORCH_CHECK(w_pack.dim() == 3 && w_pack.size(0) == kNumExperts && w_pack.size(1) == kOutDim && w_pack.size(2) == kPacksPerRow, "w_pack must have shape [8,512,128]");
    TORCH_CHECK(out.dim() == 2 && out.size(0) == a_pack.size(0) && out.size(1) == kOutDim, "out must have shape [pool,512]");
    TORCH_CHECK(counts.numel() == kNumExperts && offsets.numel() == kNumExperts + 1, "counts [8], offsets [9] required");
    TORCH_CHECK(!bias.defined() || (bias.is_cuda() && bias.dtype() == torch::kFloat16 && bias.is_contiguous() && bias.numel() == kNumExperts * kOutDim), "bias must be contiguous fp16 [8,512]");
}

void grouped_fc2_int4_simt(torch::Tensor a_pack, torch::Tensor w_pack, torch::Tensor bias, torch::Tensor counts, torch::Tensor offsets, torch::Tensor out, double out_scale) {
    check_grouped_fc2_int4_args(a_pack, w_pack, bias, counts, offsets, out);

    const int rows = static_cast<int>(a_pack.size(0));
    const dim3 grid((kOutDim + kTileN - 1) / kTileN, (rows + kTileM - 1) / kTileM);
    const dim3 block(kTileN, kTileM);
    const __half* bias_ptr = bias.defined() ? reinterpret_cast<const __half*>(bias.data_ptr<c10::Half>()) : nullptr;
    grouped_fc2_int4_simt_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        a_pack.data_ptr<int32_t>(),
        w_pack.data_ptr<int32_t>(),
        bias_ptr,
        counts.data_ptr<int32_t>(),
        offsets.data_ptr<int32_t>(),
        reinterpret_cast<__half*>(out.data_ptr<c10::Half>()),
        rows,
        static_cast<float>(out_scale));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void grouped_fc2_int4_wmma(torch::Tensor a_pack, torch::Tensor w_pack, torch::Tensor bias, torch::Tensor counts, torch::Tensor offsets, torch::Tensor out, double out_scale) {
    check_grouped_fc2_int4_args(a_pack, w_pack, bias, counts, offsets, out);

    const int rows = static_cast<int>(a_pack.size(0));
    const dim3 grid((kOutDim + kCtaN - 1) / kCtaN, (rows + kCtaM - 1) / kCtaM);
    const dim3 block(kWarpsPerCta * 32);
    const __half* bias_ptr = bias.defined() ? reinterpret_cast<const __half*>(bias.data_ptr<c10::Half>()) : nullptr;
    grouped_fc2_int4_wmma_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        a_pack.data_ptr<int32_t>(),
        w_pack.data_ptr<int32_t>(),
        bias_ptr,
        counts.data_ptr<int32_t>(),
        offsets.data_ptr<int32_t>(),
        reinterpret_cast<__half*>(out.data_ptr<c10::Half>()),
        rows,
        static_cast<float>(out_scale));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""
    _GROUPED_INT4_EXT = load_inline(
        name="w4a4_grouped_int4_route_pool_ext_v5",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["pack_routes_uniform_int4", "grouped_fc2_int4_simt", "grouped_fc2_int4_wmma"],
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
        verbose=False,
    )
    return _GROUPED_INT4_EXT


def _make_route_counts(total_routes: int, seed: int, device: torch.device):
    if total_routes < NUM_EXPERTS:
        raise ValueError(f"--routes must be at least {NUM_EXPERTS}")

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    probs = torch.rand(NUM_EXPERTS, generator=gen)
    probs = probs / probs.sum()
    counts = torch.multinomial(probs, total_routes, replacement=True, generator=gen)
    counts = torch.bincount(counts, minlength=NUM_EXPERTS).to(torch.int32)

    # Avoid zero-count experts so every expert's dense APEX4 path is exercised.
    for expert in range(NUM_EXPERTS):
        if counts[expert].item() == 0:
            donor = int(torch.argmax(counts).item())
            counts[donor] -= 1
            counts[expert] += 1

    offsets = torch.empty(NUM_EXPERTS, dtype=torch.int32)
    running = 0
    for expert in range(NUM_EXPERTS):
        offsets[expert] = running
        running += int(counts[expert].item())
    return counts.to(device), offsets.to(device)


def _quantize_w4_symmetric(weight: torch.Tensor, group_size: int):
    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError(f"group_size={group_size} must divide K={in_features}")

    groups = weight.float().view(out_features, in_features // group_size, group_size)
    scale = groups.abs().amax(dim=2, keepdim=True).mul_(2.0 / 15.0).clamp_(min=1e-6)
    q = torch.round(groups / scale).clamp_(-8, 7)
    qdq = (q * scale).view(out_features, in_features).to(torch.float16).contiguous()
    scales = scale.squeeze(2).to(torch.float16).contiguous()
    return qdq, scales


def _quantize_w4_uniform(weight: torch.Tensor, group_size: int, scale: torch.Tensor):
    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError(f"group_size={group_size} must divide K={in_features}")
    scale = scale.to(device=weight.device, dtype=torch.float32).clamp_min(1e-6)
    q = torch.round(weight.float() / scale).clamp_(-8, 7)
    qdq = (q * scale).to(torch.float16).contiguous()
    scales = torch.full(
        (out_features, in_features // group_size),
        float(scale.item()),
        device=weight.device,
        dtype=torch.float16,
    )
    return qdq, scales


def _pack_simple_w4_weights_uniform(w2: torch.Tensor, scale: float):
    q = torch.round(w2.float() / scale).clamp_(-8, 7).to(torch.int32)
    q = q & 0xF
    packed = torch.zeros(
        (NUM_EXPERTS, OUT_DIM, IN_DIM // 8),
        dtype=torch.int32,
        device=w2.device,
    )
    for i in range(8):
        packed |= q[:, :, i::8] << (4 * i)
    return packed.contiguous()


def _make_padded_route_pool(h_route, counts, offsets):
    padded_offsets = torch.empty((NUM_EXPERTS + 1,), dtype=torch.int32, device=h_route.device)
    padded_total = 0
    for expert in range(NUM_EXPERTS):
        padded_offsets[expert] = padded_total
        padded_total += _padded_expert_rows(int(counts[expert].item()))
    padded_offsets[NUM_EXPERTS] = padded_total

    h_padded = torch.zeros((padded_total, IN_DIM), dtype=h_route.dtype, device=h_route.device)
    for expert in range(NUM_EXPERTS):
        count = int(counts[expert].item())
        src_offset = int(offsets[expert].item())
        dst_offset = int(padded_offsets[expert].item())
        if count > 0:
            h_padded[dst_offset : dst_offset + count].copy_(h_route[src_offset : src_offset + count])
    return h_padded, padded_offsets


def _compact_from_padded(y_padded, counts, compact_offsets, padded_offsets):
    total_routes = int(counts.sum().item())
    compact = torch.empty((total_routes, OUT_DIM), dtype=y_padded.dtype, device=y_padded.device)
    for expert in range(NUM_EXPERTS):
        count = int(counts[expert].item())
        if count > 0:
            src = int(padded_offsets[expert].item())
            dst = int(compact_offsets[expert].item())
            compact[dst : dst + count].copy_(y_padded[src : src + count])
    return compact


def _run_grouped_route_pool_int4_probe(
    h_route,
    w2,
    bias,
    counts,
    offsets,
    act_scale: float,
    weight_scale: float,
    warmup: int,
    iters: int,
):
    ext = _load_grouped_int4_ext()
    h_padded, padded_offsets = _make_padded_route_pool(h_route, counts, offsets)
    w_pack = _pack_simple_w4_weights_uniform(w2, weight_scale)
    a_pack = torch.empty((h_padded.size(0), IN_DIM // 8), dtype=torch.int32, device=h_route.device)
    y_padded = torch.empty((h_padded.size(0), OUT_DIM), dtype=torch.float16, device=h_route.device)
    bias_arg = bias if bias is not None else torch.zeros(
        (NUM_EXPERTS, OUT_DIM),
        dtype=torch.float16,
        device=h_route.device,
    )

    ext.pack_routes_uniform_int4(h_padded, a_pack, counts, padded_offsets, act_scale)
    ext.grouped_fc2_int4_wmma(
        a_pack,
        w_pack,
        bias_arg,
        counts,
        padded_offsets,
        y_padded,
        act_scale * weight_scale,
    )
    torch.cuda.synchronize()

    def gemm_only():
        ext.grouped_fc2_int4_wmma(
            a_pack,
            w_pack,
            bias_arg,
            counts,
            padded_offsets,
            y_padded,
            act_scale * weight_scale,
        )

    def pack_plus_gemm():
        ext.pack_routes_uniform_int4(h_padded, a_pack, counts, padded_offsets, act_scale)
        ext.grouped_fc2_int4_wmma(
            a_pack,
            w_pack,
            bias_arg,
            counts,
            padded_offsets,
            y_padded,
            act_scale * weight_scale,
        )

    gemm_ms = _benchmark_ms(gemm_only, warmup, iters)
    total_ms = _benchmark_ms(pack_plus_gemm, warmup, iters)
    compact = _compact_from_padded(y_padded, counts, offsets, padded_offsets)
    return compact, total_ms, gemm_ms


def _pack_w4a4_fc2_weights(
    csrc,
    w2: torch.Tensor,
    group_size: int,
    uniform_scale: bool = False,
    weight_scale: torch.Tensor | None = None,
):
    experts: list[ExpertW4A4] = []
    for expert in range(NUM_EXPERTS):
        if uniform_scale:
            if weight_scale is None:
                raise ValueError("weight_scale is required for uniform weight quantization")
            qdq, scales = _quantize_w4_uniform(w2[expert], group_size, weight_scale)
        else:
            qdq, scales = _quantize_w4_symmetric(w2[expert], group_size)
        linear = nn.Linear(IN_DIM, OUT_DIM, bias=False, device=w2.device, dtype=torch.float16)
        linear.weight.data.copy_(qdq)

        layer = csrc.W4A4Layer(IN_DIM, OUT_DIM, groupsize=group_size).to(w2.device)
        layer.pack(linear, scales)
        experts.append(ExpertW4A4(layer=layer, qdq_weight=qdq))
    return experts


def _unpack_signed_int4(packed: torch.Tensor, k: int):
    packed_u = packed.to(torch.int64) & 0xFFFFFFFF
    vals = []
    for idx in range(8):
        q = ((packed_u >> (4 * idx)) & 0xF).to(torch.int16)
        q = torch.where(q >= 8, q - 16, q)
        vals.append(q)
    return torch.stack(vals, dim=-1).reshape(packed.size(0), -1)[:, :k]


def _dequantize_activation_from_packed(
    packed: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
):
    q = _unpack_signed_int4(packed, IN_DIM).float()
    return (
        q.view(packed.size(0), IN_DIM // group_size, group_size)
        * scales.float().unsqueeze(2)
    ).view(packed.size(0), IN_DIM).to(torch.float16)


def _prequantize_activations(csrc, h_route, counts, offsets, group_size: int):
    packed = []
    scales = []
    for expert in range(NUM_EXPERTS):
        count = int(counts[expert].item())
        offset = int(offsets[expert].item())
        padded_count = _padded_expert_rows(count)
        a = torch.empty((padded_count, IN_DIM // 8), dtype=torch.int32, device=h_route.device)
        s1 = torch.empty((padded_count, IN_DIM // group_size), dtype=torch.float16, device=h_route.device)
        if count > 0:
            x_pad = torch.zeros((padded_count, IN_DIM), dtype=h_route.dtype, device=h_route.device)
            x_pad[:count].copy_(h_route[offset : offset + count])
            csrc.quantize_compress(
                x_pad,
                a,
                s1,
                group_size,
            )
        packed.append(a)
        scales.append(s1)
    return packed, scales


def _make_padded_inputs(counts: torch.Tensor):
    inputs = []
    for expert in range(NUM_EXPERTS):
        count = int(counts[expert].item())
        inputs.append(
            torch.empty(
                (_padded_expert_rows(count), IN_DIM),
                dtype=torch.float16,
                device=counts.device,
            )
        )
    return inputs


def _prequantize_activations_uniform(
    pack_ext,
    h_route,
    counts,
    offsets,
    group_size: int,
    act_scale: float,
    x_pad_cache: list[torch.Tensor] | None = None,
):
    packed = []
    scales = []
    if x_pad_cache is None:
        x_pad_cache = _make_padded_inputs(counts)
    for expert in range(NUM_EXPERTS):
        count = int(counts[expert].item())
        offset = int(offsets[expert].item())
        padded_count = _padded_expert_rows(count)
        a = torch.empty((padded_count, IN_DIM // 8), dtype=torch.int32, device=h_route.device)
        s1 = torch.empty((padded_count, IN_DIM // group_size), dtype=torch.float16, device=h_route.device)
        if count > 0:
            x_pad = x_pad_cache[expert]
            x_pad.zero_()
            x_pad[:count].copy_(h_route[offset : offset + count])
            pack_ext.uniform_pack_w4a4(x_pad, a, s1, act_scale)
        packed.append(a)
        scales.append(s1)
    return packed, scales


def _padded_expert_rows(count: int):
    if count <= 0:
        return 0
    return max(32, ((count + 15) // 16) * 16)


def _make_padded_outputs(counts: torch.Tensor):
    outputs = []
    for expert in range(NUM_EXPERTS):
        count = int(counts[expert].item())
        outputs.append(
            torch.empty(
                (_padded_expert_rows(count), OUT_DIM),
                dtype=torch.float16,
                device=counts.device,
            )
        )
    return outputs


def _w4a4_fc2_once(
    csrc,
    h_route: torch.Tensor,
    counts: torch.Tensor,
    offsets: torch.Tensor,
    experts: list[ExpertW4A4],
    bias: torch.Tensor | None,
    group_size: int,
    out: torch.Tensor,
    packed_cache: list[torch.Tensor] | None = None,
    scale_cache: list[torch.Tensor] | None = None,
    padded_out_cache: list[torch.Tensor] | None = None,
    copy_to_route: bool = True,
):
    for expert in range(NUM_EXPERTS):
        count = int(counts[expert].item())
        offset = int(offsets[expert].item())
        if count == 0:
            continue

        if packed_cache is None or scale_cache is None:
            padded_count = _padded_expert_rows(count)
            a = torch.empty((padded_count, IN_DIM // 8), dtype=torch.int32, device=h_route.device)
            s1 = torch.empty((padded_count, IN_DIM // group_size), dtype=torch.float16, device=h_route.device)
            x_pad = torch.zeros((padded_count, IN_DIM), dtype=h_route.dtype, device=h_route.device)
            x_pad[:count].copy_(h_route[offset : offset + count])
            csrc.quantize_compress(
                x_pad,
                a,
                s1,
                group_size,
            )
        else:
            a = packed_cache[expert]
            s1 = scale_cache[expert]

        if padded_out_cache is None:
            dst = torch.empty((a.size(0), OUT_DIM), dtype=torch.float16, device=h_route.device)
        else:
            dst = padded_out_cache[expert]
        layer = experts[expert].layer
        csrc.w4a4_mul_G(
            a,
            layer.B,
            layer.reduce_buffer,
            dst,
            s1,
            layer.s_group,
            layer.workspace,
            max_par=layer.max_par,
        )
        if copy_to_route:
            out_slice = out[offset : offset + count]
            out_slice.copy_(dst[:count])
            if bias is not None:
                out_slice.add_(bias[expert])


def _w4a4_fc2_once_uniform(
    csrc,
    pack_ext,
    h_route: torch.Tensor,
    counts: torch.Tensor,
    offsets: torch.Tensor,
    experts: list[ExpertW4A4],
    bias: torch.Tensor | None,
    group_size: int,
    act_scale: float,
    out: torch.Tensor,
    packed_cache: list[torch.Tensor],
    scale_cache: list[torch.Tensor],
    padded_out_cache: list[torch.Tensor],
    x_pad_cache: list[torch.Tensor] | None = None,
    quant_pack: bool = True,
    copy_to_route: bool = True,
):
    for expert in range(NUM_EXPERTS):
        count = int(counts[expert].item())
        offset = int(offsets[expert].item())
        if count == 0:
            continue

        a = packed_cache[expert]
        s1 = scale_cache[expert]
        if quant_pack:
            if x_pad_cache is None:
                raise ValueError("x_pad_cache is required when quant_pack=True")
            x_pad = x_pad_cache[expert]
            x_pad.zero_()
            x_pad[:count].copy_(h_route[offset : offset + count])
            pack_ext.uniform_pack_w4a4(x_pad, a, s1, act_scale)

        dst = padded_out_cache[expert]
        layer = experts[expert].layer
        csrc.w4a4_mul_G(
            a,
            layer.B,
            layer.reduce_buffer,
            dst,
            s1,
            layer.s_group,
            layer.workspace,
            max_par=layer.max_par,
        )
        if copy_to_route:
            out_slice = out[offset : offset + count]
            out_slice.copy_(dst[:count])
            if bias is not None:
                out_slice.add_(bias[expert])


def _w4a4_reference_from_packed(
    packed_cache: list[torch.Tensor],
    scale_cache: list[torch.Tensor],
    counts: torch.Tensor,
    offsets: torch.Tensor,
    experts: list[ExpertW4A4],
    bias: torch.Tensor | None,
    group_size: int,
):
    out = torch.empty((int(counts.sum().item()), OUT_DIM), dtype=torch.float16, device=counts.device)
    for expert in range(NUM_EXPERTS):
        count = int(counts[expert].item())
        offset = int(offsets[expert].item())
        if count == 0:
            continue
        a_qdq = _dequantize_activation_from_packed(
            packed_cache[expert],
            scale_cache[expert],
            group_size=group_size,
        )[:count]
        out[offset : offset + count] = F.linear(
            a_qdq,
            experts[expert].qdq_weight,
            bias[expert] if bias is not None else None,
        )
    return out


def _benchmark_ms(fn: Callable[[], None], warmup: int, iters: int):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _try_custom_fp16_fc2(h_route, w2, bias, counts, offsets, warmup, iters):
    try:
        sys.path.insert(0, str(ROOT))
        import infer  # type: ignore

        ext = infer._get_routed_smoe_ext()
        if not hasattr(ext, "smoe_grouped_fc2"):
            return None
    except Exception as exc:
        print(f"[WARN] custom fp16 smoe_grouped_fc2 unavailable: {exc}")
        return None

    padded_offsets = torch.empty((NUM_EXPERTS,), dtype=offsets.dtype, device=offsets.device)
    padded_total = 0
    for expert in range(NUM_EXPERTS):
        padded_offsets[expert] = padded_total
        padded_total += _padded_expert_rows(int(counts[expert].item()))

    h_padded = torch.zeros((padded_total, IN_DIM), dtype=h_route.dtype, device=h_route.device)
    for expert in range(NUM_EXPERTS):
        count = int(counts[expert].item())
        src_offset = int(offsets[expert].item())
        dst_offset = int(padded_offsets[expert].item())
        if count > 0:
            h_padded[dst_offset : dst_offset + count].copy_(h_route[src_offset : src_offset + count])

    max_routes_per_expert = max(_padded_expert_rows(int(count.item())) for count in counts)
    ext_offsets = torch.empty((NUM_EXPERTS + 1,), dtype=offsets.dtype, device=offsets.device)
    ext_offsets[:NUM_EXPERTS].copy_(padded_offsets)
    ext_offsets[NUM_EXPERTS] = padded_total
    out = ext.smoe_grouped_fc2(
        h_padded,
        w2,
        bias,
        counts,
        ext_offsets,
        max_routes_per_expert,
    )
    torch.cuda.synchronize()

    def run():
        ext.smoe_grouped_fc2(
            h_padded,
            w2,
            bias,
            counts,
            ext_offsets,
            max_routes_per_expert,
        )

    compact = torch.empty_like(h_route[:, :OUT_DIM])
    for expert in range(NUM_EXPERTS):
        count = int(counts[expert].item())
        src_offset = int(padded_offsets[expert].item())
        dst_offset = int(offsets[expert].item())
        if count > 0:
            compact[dst_offset : dst_offset + count].copy_(out[src_offset : src_offset + count])
    torch.cuda.synchronize()

    return compact, _benchmark_ms(run, warmup, iters)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes", type=int, default=8192)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--build-apex4", action="store_true")
    parser.add_argument("--no-bias", action="store_true")
    parser.add_argument("--compare-custom-fp16", action="store_true")
    parser.add_argument("--uniform-scale", action="store_true")
    parser.add_argument("--grouped-route-pool-int4", action="store_true")
    parser.add_argument("--act-scale", type=float, default=0.0)
    parser.add_argument("--weight-scale", type=float, default=0.0)
    args = parser.parse_args()

    _require_experiment_enabled()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.group_size not in (32, 64, 128, 256, 512, 1024):
        raise ValueError("--group-size must be one of 32,64,128,256,512,1024")

    csrc = None if args.grouped_route_pool_int4 else _load_apex4(args.build_apex4)
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    counts, offsets = _make_route_counts(args.routes, args.seed, device)
    total_routes = int(counts.sum().item())
    h_route = torch.randn((total_routes, IN_DIM), device=device, dtype=torch.float16)
    w2 = torch.randn((NUM_EXPERTS, OUT_DIM, IN_DIM), device=device, dtype=torch.float16) / 32.0
    bias = None if args.no_bias else torch.randn((NUM_EXPERTS, OUT_DIM), device=device, dtype=torch.float16) / 32.0

    if args.grouped_route_pool_int4:
        args.uniform_scale = True
    pack_ext = _load_uniform_pack_ext() if args.uniform_scale and not args.grouped_route_pool_int4 else None
    act_scale = args.act_scale
    weight_scale = args.weight_scale
    if args.uniform_scale:
        if act_scale <= 0.0:
            act_scale = float((h_route.float().abs().max() * (2.0 / 15.0)).clamp_min(1e-6).item())
        if weight_scale <= 0.0:
            weight_scale = float((w2.float().abs().max() * (2.0 / 15.0)).clamp_min(1e-6).item())

    if args.grouped_route_pool_int4:
        out, total_ms, gemm_only_ms = _run_grouped_route_pool_int4_probe(
            h_route,
            w2,
            bias,
            counts,
            offsets,
            act_scale,
            weight_scale,
            args.warmup,
            args.iters,
        )
        ref = F.linear(h_route, w2[0], bias[0] if bias is not None else None)
        diff = (out[: min(out.size(0), ref.size(0))].float() - ref[: min(out.size(0), ref.size(0))].float()).abs()
        rel = diff.mean() / ref.float().abs().mean().clamp_min(1e-6)
        print("[INFO] W4A4 SMoE fc2 isolated experiment")
        print(f"[INFO] routes={total_routes}, counts={counts.detach().cpu().tolist()}")
        print(f"[INFO] group_size={args.group_size}, bias={bias is not None}, uniform_scale=True, grouped_route_pool_int4=True, kernel=wmma_s4_cta16x128_pack2")
        print(f"[INFO] uniform act_scale={act_scale:.8e}, weight_scale={weight_scale:.8e}")
        print(f"[RESULT] coarse_w4a4_vs_expert0_fp16 max_abs={diff.max().item():.6e} mean_abs={diff.mean().item():.6e} rel_mean={rel.item():.6e}")
        print(f"[RESULT] grouped_route_pool_int4_quant_pack_plus_gemm_ms={total_ms:.6f}")
        print(f"[RESULT] grouped_route_pool_int4_prepacked_gemm_only_ms={gemm_only_ms:.6f}")
        if args.compare_custom_fp16:
            fp16 = _try_custom_fp16_fc2(
                h_route,
                w2.contiguous(),
                bias if bias is not None else torch.zeros((NUM_EXPERTS, OUT_DIM), device=device, dtype=torch.float16),
                counts,
                offsets,
                args.warmup,
                args.iters,
            )
            if fp16 is not None:
                fp16_out, fp16_ms = fp16
                w4a4_vs_fp16 = (out.float() - fp16_out.float()).abs()
                print(f"[RESULT] custom_fp16_grouped_fc2_ms={fp16_ms:.6f}")
                print(
                    "[RESULT] grouped_int4_vs_fp16 "
                    f"max_abs={w4a4_vs_fp16.max().item():.6e} "
                    f"mean_abs={w4a4_vs_fp16.mean().item():.6e}"
                )
        return

    experts = _pack_w4a4_fc2_weights(
        csrc,
        w2,
        args.group_size,
        uniform_scale=args.uniform_scale,
        weight_scale=torch.tensor(weight_scale, device=device) if args.uniform_scale else None,
    )
    x_pad_cache = _make_padded_inputs(counts) if args.uniform_scale else None
    if args.uniform_scale:
        packed_cache, scale_cache = _prequantize_activations_uniform(
            pack_ext,
            h_route,
            counts,
            offsets,
            args.group_size,
            act_scale,
            x_pad_cache,
        )
    else:
        packed_cache, scale_cache = _prequantize_activations(
            csrc,
            h_route,
            counts,
            offsets,
            args.group_size,
        )
    padded_outputs = _make_padded_outputs(counts)
    torch.cuda.synchronize()

    out = torch.empty((total_routes, OUT_DIM), device=device, dtype=torch.float16)
    if args.uniform_scale:
        _w4a4_fc2_once_uniform(
            csrc,
            pack_ext,
            h_route,
            counts,
            offsets,
            experts,
            bias,
            args.group_size,
            act_scale,
            out,
            packed_cache,
            scale_cache,
            padded_outputs,
            x_pad_cache,
            quant_pack=False,
        )
    else:
        _w4a4_fc2_once(
            csrc,
            h_route,
            counts,
            offsets,
            experts,
            bias,
            args.group_size,
            out,
            packed_cache,
            scale_cache,
            padded_outputs,
        )
    ref = _w4a4_reference_from_packed(
        packed_cache,
        scale_cache,
        counts,
        offsets,
        experts,
        bias,
        args.group_size,
    )
    torch.cuda.synchronize()

    diff = (out.float() - ref.float()).abs()
    rel = diff.mean() / ref.float().abs().mean().clamp_min(1e-6)

    out_total = torch.empty_like(out)
    out_gemm = torch.empty_like(out)
    if args.uniform_scale:
        total_ms = _benchmark_ms(
            lambda: _w4a4_fc2_once_uniform(
                csrc,
                pack_ext,
                h_route,
                counts,
                offsets,
                experts,
                bias,
                args.group_size,
                act_scale,
                out_total,
                packed_cache,
                scale_cache,
                padded_outputs,
                x_pad_cache,
                quant_pack=True,
            ),
            args.warmup,
            args.iters,
        )
        gemm_only_ms = _benchmark_ms(
            lambda: _w4a4_fc2_once_uniform(
                csrc,
                pack_ext,
                h_route,
                counts,
                offsets,
                experts,
                bias,
                args.group_size,
                act_scale,
                out_gemm,
                packed_cache,
                scale_cache,
                padded_outputs,
                x_pad_cache,
                quant_pack=False,
                copy_to_route=False,
            ),
            args.warmup,
            args.iters,
        )
    else:
        total_ms = _benchmark_ms(
            lambda: _w4a4_fc2_once(
                csrc,
                h_route,
                counts,
                offsets,
                experts,
                bias,
                args.group_size,
                out_total,
                padded_out_cache=padded_outputs,
            ),
            args.warmup,
            args.iters,
        )
        gemm_only_ms = _benchmark_ms(
            lambda: _w4a4_fc2_once(
                csrc,
                h_route,
                counts,
                offsets,
                experts,
                bias,
                args.group_size,
                out_gemm,
                packed_cache,
                scale_cache,
                padded_outputs,
                copy_to_route=False,
            ),
            args.warmup,
            args.iters,
        )

    print("[INFO] W4A4 SMoE fc2 isolated experiment")
    print(f"[INFO] routes={total_routes}, counts={counts.detach().cpu().tolist()}")
    print(f"[INFO] group_size={args.group_size}, bias={bias is not None}, uniform_scale={args.uniform_scale}")
    if args.uniform_scale:
        print(f"[INFO] uniform act_scale={act_scale:.8e}, weight_scale={weight_scale:.8e}")
    print(f"[RESULT] correctness max_abs={diff.max().item():.6e} mean_abs={diff.mean().item():.6e} rel_mean={rel.item():.6e}")
    if args.uniform_scale:
        print(f"[RESULT] uniform_w4a4_quant_pack_plus_gemm_ms={total_ms:.6f}")
        print(f"[RESULT] uniform_w4a4_prepacked_gemm_only_ms={gemm_only_ms:.6f}")
    else:
        print(f"[RESULT] apex4_w4a4_quant_gemm_copy_ms={total_ms:.6f}")
        print(f"[RESULT] apex4_w4a4_gemm_only_ms={gemm_only_ms:.6f}")

    if args.compare_custom_fp16:
        fp16 = _try_custom_fp16_fc2(
            h_route,
            w2.contiguous(),
            bias if bias is not None else torch.zeros((NUM_EXPERTS, OUT_DIM), device=device, dtype=torch.float16),
            counts,
            offsets,
            args.warmup,
            args.iters,
        )
        if fp16 is not None:
            fp16_out, fp16_ms = fp16
            w4a4_vs_fp16 = (out.float() - fp16_out.float()).abs()
            print(f"[RESULT] custom_fp16_grouped_fc2_ms={fp16_ms:.6f}")
            print(
                "[RESULT] w4a4_vs_fp16 "
                f"max_abs={w4a4_vs_fp16.max().item():.6e} "
                f"mean_abs={w4a4_vs_fp16.mean().item():.6e}"
            )


if __name__ == "__main__":
    main()
