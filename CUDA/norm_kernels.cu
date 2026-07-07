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
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <cstdint>
#include <vector>

namespace {

constexpr int kLayerNormWidth = 512;
constexpr int kRepLayerNormWidth = 28 * kLayerNormWidth;
constexpr int kThreads = 256;
constexpr int kVecWidth = 8;
constexpr int kLayerNormVecs = kLayerNormWidth / kVecWidth;
constexpr int kRepLayerNormVecs = kRepLayerNormWidth / kVecWidth;
constexpr int kWarpSize = 32;

static_assert(kLayerNormWidth % kVecWidth == 0, "LayerNorm width must be divisible by vector width");
static_assert(kRepLayerNormWidth % kVecWidth == 0, "Rep LayerNorm width must be divisible by vector width");

union Half8 {
    uint4 u4;
    __half h[kVecWidth];
};

bool use_layernorm_512_vec8() {
    const char* flag = std::getenv("USE_LAYERNORM_512_VEC8");
    return flag != nullptr && !(flag[0] == '\0' || (flag[0] == '0' && flag[1] == '\0'));
}

bool use_layernorm_14336_vec8() {
    const char* flag = std::getenv("USE_REP_LAYERNORM_14336_VEC8");
    return flag == nullptr || !(flag[0] == '\0' || (flag[0] == '0' && flag[1] == '\0'));
}

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

__global__ void layernorm_512_vec8_kernel(
    const c10::Half* __restrict__ x,
    const c10::Half* __restrict__ weight,
    const c10::Half* __restrict__ bias,
    c10::Half* __restrict__ out,
    int64_t rows,
    float eps
) {
    const int row = blockIdx.x;
    const int vec = threadIdx.x;
    if (row >= rows || vec >= kLayerNormVecs) {
        return;
    }

    const int dim_base = vec * kVecWidth;
    const int64_t row_base = static_cast<int64_t>(row) * kLayerNormWidth;
    const int64_t idx_base = row_base + dim_base;

    Half8 x_pack;
    x_pack.u4 = *reinterpret_cast<const uint4*>(x + idx_base);

    float vals[kVecWidth];
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < kVecWidth; ++i) {
        vals[i] = __half2float(x_pack.h[i]);
        sum += vals[i];
    }

    __shared__ float mean_shared;
    __shared__ float inv_std_shared;
    const float row_sum = block_reduce_sum<kLayerNormVecs>(sum);

    if (vec == 0) {
        mean_shared = row_sum / static_cast<float>(kLayerNormWidth);
    }
    __syncthreads();

    const float mean = mean_shared;
    float var_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < kVecWidth; ++i) {
        const float diff = vals[i] - mean;
        var_sum += diff * diff;
    }
    const float row_var_sum = block_reduce_sum<kLayerNormVecs>(var_sum);

    if (vec == 0) {
        inv_std_shared = rsqrtf(row_var_sum / static_cast<float>(kLayerNormWidth) + eps);
    }
    __syncthreads();

    Half8 weight_pack;
    Half8 bias_pack;
    Half8 out_pack;
    weight_pack.u4 = *reinterpret_cast<const uint4*>(weight + dim_base);
    bias_pack.u4 = *reinterpret_cast<const uint4*>(bias + dim_base);

    const float inv_std = inv_std_shared;
#pragma unroll
    for (int i = 0; i < kVecWidth; ++i) {
        const float diff = vals[i] - mean;
        const float value =
            diff * inv_std * __half2float(weight_pack.h[i]) + __half2float(bias_pack.h[i]);
        out_pack.h[i] = __float2half_rn(value);
    }
    *reinterpret_cast<uint4*>(out + idx_base) = out_pack.u4;
}

__global__ void layernorm_14336_scalar_kernel(
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
    const int64_t base = static_cast<int64_t>(row) * kRepLayerNormWidth;

    __shared__ float mean_shared;
    __shared__ float inv_std_shared;

    float sum = 0.0f;
    for (int col = tid; col < kRepLayerNormWidth; col += kThreads) {
        sum += static_cast<float>(x[base + col]);
    }
    const float row_sum = block_reduce_sum<kThreads>(sum);

    if (tid == 0) {
        mean_shared = row_sum / static_cast<float>(kRepLayerNormWidth);
    }
    __syncthreads();

    const float mean = mean_shared;

    float var_sum = 0.0f;
    for (int col = tid; col < kRepLayerNormWidth; col += kThreads) {
        const float diff = static_cast<float>(x[base + col]) - mean;
        var_sum += diff * diff;
    }
    const float row_var_sum = block_reduce_sum<kThreads>(var_sum);

    if (tid == 0) {
        inv_std_shared = rsqrtf(row_var_sum / static_cast<float>(kRepLayerNormWidth) + eps);
    }
    __syncthreads();

    const float inv_std = inv_std_shared;

    for (int col = tid; col < kRepLayerNormWidth; col += kThreads) {
        const int64_t idx = base + col;
        const float diff = static_cast<float>(x[idx]) - mean;
        out[idx] = static_cast<c10::Half>(
            diff * inv_std * static_cast<float>(weight[col]) + static_cast<float>(bias[col])
        );
    }
}

__global__ void layernorm_14336_vec8_kernel(
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
    const int64_t row_base = static_cast<int64_t>(row) * kRepLayerNormWidth;

    __shared__ float mean_shared;
    __shared__ float inv_std_shared;

    float sum = 0.0f;
    for (int vec = tid; vec < kRepLayerNormVecs; vec += kThreads) {
        const int dim_base = vec * kVecWidth;
        Half8 x_pack;
        x_pack.u4 = *reinterpret_cast<const uint4*>(x + row_base + dim_base);
#pragma unroll
        for (int i = 0; i < kVecWidth; ++i) {
            sum += __half2float(x_pack.h[i]);
        }
    }
    const float row_sum = block_reduce_sum<kThreads>(sum);

    if (tid == 0) {
        mean_shared = row_sum / static_cast<float>(kRepLayerNormWidth);
    }
    __syncthreads();

    const float mean = mean_shared;

    float var_sum = 0.0f;
    for (int vec = tid; vec < kRepLayerNormVecs; vec += kThreads) {
        const int dim_base = vec * kVecWidth;
        Half8 x_pack;
        x_pack.u4 = *reinterpret_cast<const uint4*>(x + row_base + dim_base);
#pragma unroll
        for (int i = 0; i < kVecWidth; ++i) {
            const float diff = __half2float(x_pack.h[i]) - mean;
            var_sum += diff * diff;
        }
    }
    const float row_var_sum = block_reduce_sum<kThreads>(var_sum);

    if (tid == 0) {
        inv_std_shared = rsqrtf(row_var_sum / static_cast<float>(kRepLayerNormWidth) + eps);
    }
    __syncthreads();

    const float inv_std = inv_std_shared;

    for (int vec = tid; vec < kRepLayerNormVecs; vec += kThreads) {
        const int dim_base = vec * kVecWidth;
        const int64_t idx_base = row_base + dim_base;
        Half8 x_pack;
        Half8 weight_pack;
        Half8 bias_pack;
        Half8 out_pack;
        x_pack.u4 = *reinterpret_cast<const uint4*>(x + idx_base);
        weight_pack.u4 = *reinterpret_cast<const uint4*>(weight + dim_base);
        bias_pack.u4 = *reinterpret_cast<const uint4*>(bias + dim_base);
#pragma unroll
        for (int i = 0; i < kVecWidth; ++i) {
            const float diff = __half2float(x_pack.h[i]) - mean;
            const float value =
                diff * inv_std * __half2float(weight_pack.h[i]) + __half2float(bias_pack.h[i]);
            out_pack.h[i] = __float2half_rn(value);
        }
        *reinterpret_cast<uint4*>(out + idx_base) = out_pack.u4;
    }
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

__global__ void add_layernorm_512_vec8_kernel(
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
    const int vec = threadIdx.x;
    if (row >= rows || vec >= kLayerNormVecs) {
        return;
    }

    const int dim_base = vec * kVecWidth;
    const int64_t row_base = static_cast<int64_t>(row) * kLayerNormWidth;
    const int64_t idx_base = row_base + dim_base;

    Half8 residual_pack;
    Half8 x_pack;
    residual_pack.u4 = *reinterpret_cast<const uint4*>(residual + idx_base);
    x_pack.u4 = *reinterpret_cast<const uint4*>(x + idx_base);

    float vals[kVecWidth];
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < kVecWidth; ++i) {
        vals[i] = __half2float(residual_pack.h[i]) + __half2float(x_pack.h[i]);
        sum += vals[i];
    }

    __shared__ float mean_shared;
    __shared__ float inv_std_shared;
    const float row_sum = block_reduce_sum<kLayerNormVecs>(sum);

    if (vec == 0) {
        mean_shared = row_sum / static_cast<float>(kLayerNormWidth);
    }
    __syncthreads();

    const float mean = mean_shared;
    float var_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < kVecWidth; ++i) {
        const float diff = vals[i] - mean;
        var_sum += diff * diff;
    }
    const float row_var_sum = block_reduce_sum<kLayerNormVecs>(var_sum);

    if (vec == 0) {
        inv_std_shared = rsqrtf(row_var_sum / static_cast<float>(kLayerNormWidth) + eps);
    }
    __syncthreads();

    if (residual_out != nullptr) {
        Half8 residual_out_pack;
#pragma unroll
        for (int i = 0; i < kVecWidth; ++i) {
            residual_out_pack.h[i] = __float2half_rn(vals[i]);
        }
        *reinterpret_cast<uint4*>(residual_out + idx_base) = residual_out_pack.u4;
    }

    Half8 weight_pack;
    Half8 bias_pack;
    Half8 out_pack;
    weight_pack.u4 = *reinterpret_cast<const uint4*>(weight + dim_base);
    bias_pack.u4 = *reinterpret_cast<const uint4*>(bias + dim_base);

    const float inv_std = inv_std_shared;
#pragma unroll
    for (int i = 0; i < kVecWidth; ++i) {
        const float diff = vals[i] - mean;
        const float value =
            diff * inv_std * __half2float(weight_pack.h[i]) + __half2float(bias_pack.h[i]);
        out_pack.h[i] = __float2half_rn(value);
    }
    *reinterpret_cast<uint4*>(out + idx_base) = out_pack.u4;
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

    if (use_layernorm_512_vec8()) {
        layernorm_512_vec8_kernel<<<static_cast<int>(rows), kLayerNormVecs, 0, at::cuda::getCurrentCUDAStream()>>>(
            x.data_ptr<c10::Half>(),
            weight.data_ptr<c10::Half>(),
            bias.data_ptr<c10::Half>(),
            out.data_ptr<c10::Half>(),
            rows,
            static_cast<float>(eps)
        );
    } else {
        layernorm_512_scalar_kernel<<<static_cast<int>(rows), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
            x.data_ptr<c10::Half>(),
            weight.data_ptr<c10::Half>(),
            bias.data_ptr<c10::Half>(),
            out.data_ptr<c10::Half>(),
            rows,
            static_cast<float>(eps)
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

torch::Tensor layernorm_14336(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps) {
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
    TORCH_CHECK(x.size(-1) == kRepLayerNormWidth, "x last dimension must be 14336");
    TORCH_CHECK(weight.numel() == kRepLayerNormWidth, "weight must have 14336 elements");
    TORCH_CHECK(bias.numel() == kRepLayerNormWidth, "bias must have 14336 elements");
    TORCH_CHECK(x.device() == weight.device(), "x and weight must be on the same CUDA device");
    TORCH_CHECK(x.device() == bias.device(), "x and bias must be on the same CUDA device");
    TORCH_CHECK(eps > 0.0, "eps must be positive");

    auto out = torch::empty_like(x);
    const int64_t rows = x.numel() / kRepLayerNormWidth;
    if (rows == 0) {
        return out;
    }

    if (use_layernorm_14336_vec8()) {
        layernorm_14336_vec8_kernel<<<static_cast<int>(rows), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
            x.data_ptr<c10::Half>(),
            weight.data_ptr<c10::Half>(),
            bias.data_ptr<c10::Half>(),
            out.data_ptr<c10::Half>(),
            rows,
            static_cast<float>(eps)
        );
    } else {
        layernorm_14336_scalar_kernel<<<static_cast<int>(rows), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
            x.data_ptr<c10::Half>(),
            weight.data_ptr<c10::Half>(),
            bias.data_ptr<c10::Half>(),
            out.data_ptr<c10::Half>(),
            rows,
            static_cast<float>(eps)
        );
    }
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

    if (use_layernorm_512_vec8()) {
        add_layernorm_512_vec8_kernel<<<static_cast<int>(rows), kLayerNormVecs, 0, at::cuda::getCurrentCUDAStream()>>>(
            residual.data_ptr<c10::Half>(),
            x.data_ptr<c10::Half>(),
            weight.data_ptr<c10::Half>(),
            bias.data_ptr<c10::Half>(),
            nullptr,
            out.data_ptr<c10::Half>(),
            rows,
            static_cast<float>(eps)
        );
    } else {
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
    }
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

    if (use_layernorm_512_vec8()) {
        add_layernorm_512_vec8_kernel<<<static_cast<int>(rows), kLayerNormVecs, 0, at::cuda::getCurrentCUDAStream()>>>(
            residual.data_ptr<c10::Half>(),
            x.data_ptr<c10::Half>(),
            weight.data_ptr<c10::Half>(),
            bias.data_ptr<c10::Half>(),
            residual_out.data_ptr<c10::Half>(),
            norm_out.data_ptr<c10::Half>(),
            rows,
            static_cast<float>(eps)
        );
    } else {
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
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {residual_out, norm_out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("layernorm_512", &layernorm_512, "LayerNorm over the last dimension fixed at 512 (CUDA, float16)");
    m.def("layernorm_14336", &layernorm_14336, "LayerNorm over the last dimension fixed at 14336 (CUDA, float16)");
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
