// CUDA LayerNorm and residual-add kernels.
//
// Implement normalization and elementwise fusion for:
// - RepEncoder input LayerNorm over width 28 * 512 = 14336.
// - Transformer LayerNorm over width 512.
// - residual_add_kernel for x = residual + projection.
// - add_layernorm_512_kernel for fused residual add + LayerNorm.
//
// Accumulation should use fp32 for mean/variance, with fp16 output.
//
// Reference ideas:
// - LeetCUDA/kernels/layer-norm
// - LeetCUDA/kernels/reduce
// - LeetCUDA/kernels/elementwise

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace {

constexpr int kLayerNormWidth = 512;
constexpr int kThreads = 256;
constexpr int kWarpSize = 32;

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int mask = kWarpSize >> 1; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
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

void check_same_dtype(const torch::Tensor& a, const torch::Tensor& b, const char* a_name, const char* b_name) {
    TORCH_CHECK(a.scalar_type() == b.scalar_type(), a_name, " and ", b_name, " must have the same dtype");
}

__global__ void layernorm_512_scalar_kernel(
    const c10::Half* __restrict__ x,
    const c10::Half* __restrict__ weight,
    const c10::Half* __restrict__ bias,
    c10::Half* __restrict__ out,
    int64_t rows,
    float eps
) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const int tid = threadIdx.x;
    const int64_t base = static_cast<int64_t>(row) * kLayerNormWidth;

    __shared__ float mean_shared;
    __shared__ float inv_std_shared;

    const int col0 = tid;
    const int col1 = tid + kThreads;
    const float val0 = static_cast<float>(x[base + col0]);
    const float val1 = static_cast<float>(x[base + col1]);
    const float sum = val0 + val1;
    const float row_sum = block_reduce_sum<kThreads>(sum);

    if (tid == 0) {
        mean_shared = row_sum / static_cast<float>(kLayerNormWidth);
    }
    __syncthreads();

    const float mean = mean_shared;

    const float diff0 = val0 - mean;
    const float diff1 = val1 - mean;
    const float var_sum = diff0 * diff0 + diff1 * diff1;
    const float row_var_sum = block_reduce_sum<kThreads>(var_sum);

    if (tid == 0) {
        inv_std_shared = rsqrtf(row_var_sum / static_cast<float>(kLayerNormWidth) + eps);
    }
    __syncthreads();

    const float inv_std = inv_std_shared;

    const int64_t idx0 = base + col0;
    const int64_t idx1 = base + col1;
    out[idx0] = static_cast<c10::Half>(diff0 * inv_std * static_cast<float>(weight[col0]) + static_cast<float>(bias[col0]));
    out[idx1] = static_cast<c10::Half>(diff1 * inv_std * static_cast<float>(weight[col1]) + static_cast<float>(bias[col1]));
}

__global__ void add_layernorm_512_scalar_kernel(
    const c10::Half* __restrict__ residual,
    const c10::Half* __restrict__ x,
    const c10::Half* __restrict__ weight,
    const c10::Half* __restrict__ bias,
    c10::Half* __restrict__ residual_out,
    c10::Half* __restrict__ out,
    int64_t rows,
    float eps
) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const int tid = threadIdx.x;
    const int64_t base = static_cast<int64_t>(row) * kLayerNormWidth;

    __shared__ float mean_shared;
    __shared__ float inv_std_shared;

    const int col0 = tid;
    const int col1 = tid + kThreads;
    const int64_t idx0 = base + col0;
    const int64_t idx1 = base + col1;
    const float val0 = static_cast<float>(residual[idx0]) + static_cast<float>(x[idx0]);
    const float val1 = static_cast<float>(residual[idx1]) + static_cast<float>(x[idx1]);
    const float sum = val0 + val1;
    const float row_sum = block_reduce_sum<kThreads>(sum);

    if (tid == 0) {
        mean_shared = row_sum / static_cast<float>(kLayerNormWidth);
    }
    __syncthreads();

    const float mean = mean_shared;

    const float diff0 = val0 - mean;
    const float diff1 = val1 - mean;
    const float var_sum = diff0 * diff0 + diff1 * diff1;
    const float row_var_sum = block_reduce_sum<kThreads>(var_sum);

    if (tid == 0) {
        inv_std_shared = rsqrtf(row_var_sum / static_cast<float>(kLayerNormWidth) + eps);
    }
    __syncthreads();

    const float inv_std = inv_std_shared;

    if (residual_out != nullptr) {
        residual_out[idx0] = static_cast<c10::Half>(val0);
        residual_out[idx1] = static_cast<c10::Half>(val1);
    }
    out[idx0] = static_cast<c10::Half>(diff0 * inv_std * static_cast<float>(weight[col0]) + static_cast<float>(bias[col0]));
    out[idx1] = static_cast<c10::Half>(diff1 * inv_std * static_cast<float>(weight[col1]) + static_cast<float>(bias[col1]));
}

}  // namespace

torch::Tensor layernorm_512(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kHalf, "x must be float16");
    check_same_dtype(x, weight, "x", "weight");
    check_same_dtype(x, bias, "x", "bias");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(x.dim() >= 1, "x must have at least 1 dimension");
    TORCH_CHECK(x.size(-1) == kLayerNormWidth, "x last dimension must be 512");
    TORCH_CHECK(weight.numel() == kLayerNormWidth, "weight must have 512 elements");
    TORCH_CHECK(bias.numel() == kLayerNormWidth, "bias must have 512 elements");
    TORCH_CHECK(x.device() == weight.device(), "x and weight must be on the same CUDA device");
    TORCH_CHECK(x.device() == bias.device(), "x and bias must be on the same CUDA device");
    TORCH_CHECK(eps > 0.0, "eps must be positive");

    auto out = torch::empty_like(x);
    const int64_t rows = x.numel() / kLayerNormWidth;
    if (rows == 0) {
        return out;
    }

    layernorm_512_scalar_kernel<<<static_cast<int>(rows), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<c10::Half>(),
        weight.data_ptr<c10::Half>(),
        bias.data_ptr<c10::Half>(),
        out.data_ptr<c10::Half>(),
        rows,
        static_cast<float>(eps)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

torch::Tensor add_layernorm_512(
    torch::Tensor residual,
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps
) {
    TORCH_CHECK(residual.is_cuda(), "residual must be a CUDA tensor");
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(residual.scalar_type() == at::kHalf, "residual must be float16");
    check_same_dtype(residual, x, "residual", "x");
    check_same_dtype(residual, weight, "residual", "weight");
    check_same_dtype(residual, bias, "residual", "bias");
    TORCH_CHECK(residual.is_contiguous(), "residual must be contiguous");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(residual.sizes() == x.sizes(), "residual and x must have the same shape");
    TORCH_CHECK(residual.dim() >= 1, "residual must have at least 1 dimension");
    TORCH_CHECK(residual.size(-1) == kLayerNormWidth, "residual last dimension must be 512");
    TORCH_CHECK(weight.numel() == kLayerNormWidth, "weight must have 512 elements");
    TORCH_CHECK(bias.numel() == kLayerNormWidth, "bias must have 512 elements");
    TORCH_CHECK(residual.device() == x.device(), "residual and x must be on the same CUDA device");
    TORCH_CHECK(residual.device() == weight.device(), "residual and weight must be on the same CUDA device");
    TORCH_CHECK(residual.device() == bias.device(), "residual and bias must be on the same CUDA device");
    TORCH_CHECK(eps > 0.0, "eps must be positive");

    auto out = torch::empty_like(x);
    const int64_t rows = x.numel() / kLayerNormWidth;
    if (rows == 0) {
        return out;
    }

    add_layernorm_512_scalar_kernel<<<static_cast<int>(rows), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        residual.data_ptr<c10::Half>(),
        x.data_ptr<c10::Half>(),
        weight.data_ptr<c10::Half>(),
        bias.data_ptr<c10::Half>(),
        nullptr,
        out.data_ptr<c10::Half>(),
        rows,
        static_cast<float>(eps)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

std::vector<torch::Tensor> add_layernorm_512_with_residual(
    torch::Tensor residual,
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps
) {
    TORCH_CHECK(residual.is_cuda(), "residual must be a CUDA tensor");
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(residual.scalar_type() == at::kHalf, "residual must be float16");
    check_same_dtype(residual, x, "residual", "x");
    check_same_dtype(residual, weight, "residual", "weight");
    check_same_dtype(residual, bias, "residual", "bias");
    TORCH_CHECK(residual.is_contiguous(), "residual must be contiguous");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(residual.sizes() == x.sizes(), "residual and x must have the same shape");
    TORCH_CHECK(residual.dim() >= 1, "residual must have at least 1 dimension");
    TORCH_CHECK(residual.size(-1) == kLayerNormWidth, "residual last dimension must be 512");
    TORCH_CHECK(weight.numel() == kLayerNormWidth, "weight must have 512 elements");
    TORCH_CHECK(bias.numel() == kLayerNormWidth, "bias must have 512 elements");
    TORCH_CHECK(residual.device() == x.device(), "residual and x must be on the same CUDA device");
    TORCH_CHECK(residual.device() == weight.device(), "residual and weight must be on the same CUDA device");
    TORCH_CHECK(residual.device() == bias.device(), "residual and bias must be on the same CUDA device");
    TORCH_CHECK(eps > 0.0, "eps must be positive");

    auto residual_out = torch::empty_like(x);
    auto norm_out = torch::empty_like(x);
    const int64_t rows = x.numel() / kLayerNormWidth;
    if (rows == 0) {
        return {residual_out, norm_out};
    }

    add_layernorm_512_scalar_kernel<<<static_cast<int>(rows), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        residual.data_ptr<c10::Half>(),
        x.data_ptr<c10::Half>(),
        weight.data_ptr<c10::Half>(),
        bias.data_ptr<c10::Half>(),
        residual_out.data_ptr<c10::Half>(),
        norm_out.data_ptr<c10::Half>(),
        rows,
        static_cast<float>(eps)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {residual_out, norm_out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("layernorm_512", &layernorm_512, "LayerNorm over the last dimension fixed at 512 (CUDA, float16)");
    m.def(
        "add_layernorm_512",
        &add_layernorm_512,
        "Fused residual add plus LayerNorm over the last dimension fixed at 512 (CUDA, float16)"
    );
    m.def(
        "add_layernorm_512_with_residual",
        &add_layernorm_512_with_residual,
        "Fused residual add plus LayerNorm returning both the residual sum and normalized output (CUDA, float16)"
    );
}
