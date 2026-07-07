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

#include <cstdint>

namespace {

constexpr int kLayerNormWidth = 512;
constexpr int kThreads = 256;

__global__ void layernorm_512_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int64_t rows,
    float eps
) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const int tid = threadIdx.x;
    const int64_t base = static_cast<int64_t>(row) * kLayerNormWidth;

    float sum = 0.0f;
    if (tid < kLayerNormWidth) {
        sum += x[base + tid];
    }
    if (tid + kThreads < kLayerNormWidth) {
        sum += x[base + tid + kThreads];
    }

    __shared__ float shared[kThreads];
    __shared__ float mean_shared;
    __shared__ float inv_std_shared;

    shared[tid] = sum;
    __syncthreads();

    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        mean_shared = shared[0] / static_cast<float>(kLayerNormWidth);
    }
    __syncthreads();

    const float mean = mean_shared;

    float var_sum = 0.0f;
    if (tid < kLayerNormWidth) {
        const float diff = x[base + tid] - mean;
        var_sum += diff * diff;
    }
    if (tid + kThreads < kLayerNormWidth) {
        const float diff = x[base + tid + kThreads] - mean;
        var_sum += diff * diff;
    }

    shared[tid] = var_sum;
    __syncthreads();

    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        inv_std_shared = rsqrtf(shared[0] / static_cast<float>(kLayerNormWidth) + eps);
    }
    __syncthreads();

    const float inv_std = inv_std_shared;

    if (tid < kLayerNormWidth) {
        const int64_t idx = base + tid;
        out[idx] = (x[idx] - mean) * inv_std * weight[tid] + bias[tid];
    }
    if (tid + kThreads < kLayerNormWidth) {
        const int col = tid + kThreads;
        const int64_t idx = base + col;
        out[idx] = (x[idx] - mean) * inv_std * weight[col] + bias[col];
    }
}

}  // namespace

torch::Tensor layernorm_512(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");
    TORCH_CHECK(bias.scalar_type() == at::kFloat, "bias must be float32");
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

    layernorm_512_kernel<<<static_cast<int>(rows), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        rows,
        static_cast<float>(eps)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("layernorm_512", &layernorm_512, "LayerNorm over the last dimension fixed at 512 (CUDA, float32)");
}
