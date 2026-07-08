// CUDA fp16 top-2 softmax kernel for SMoE gate post-processing.
//
// In inference, TopKGate only needs:
//   logits [*, 8] -> topk_idx [*, 2], topk_score [*, 2]
//
// The model runs in fp16 by default, so this extension is intentionally
// half-only. Non-fp16 and training paths should use the PyTorch fallback.

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

constexpr int kExperts = 8;
constexpr int kTopK = 2;
constexpr int kThreads = 256;
constexpr float kNegInfinity = -3.4028234663852886e+38F;

void check_logits(const torch::Tensor& logits) {
    TORCH_CHECK(logits.is_cuda(), "logits must be a CUDA tensor");
    TORCH_CHECK(logits.scalar_type() == at::kHalf, "logits must be float16");
    TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous");
    TORCH_CHECK(logits.dim() >= 1, "logits must have shape [...,8]");
    TORCH_CHECK(logits.size(-1) == kExperts, "logits last dimension must be 8");
    TORCH_CHECK(
        logits.numel() / kExperts <= static_cast<int64_t>(std::numeric_limits<int>::max()),
        "logits has too many rows"
    );
}

__device__ __forceinline__ bool better_topk(float value, int idx, float best_value, int best_idx) {
    return value > best_value || (value == best_value && idx < best_idx);
}

__global__ void top2_softmax_8_kernel(
    const __half* __restrict__ logits,
    int64_t* __restrict__ topk_idx,
    __half* __restrict__ topk_score,
    int64_t n_rows
) {
    const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= n_rows) {
        return;
    }

    float vals[kExperts];
    float max_val = kNegInfinity;
    float top1 = kNegInfinity;
    float top2 = kNegInfinity;
    int top1_idx = 0;
    int top2_idx = 1;

    const int64_t base = row * kExperts;
#pragma unroll
    for (int i = 0; i < kExperts; ++i) {
        const float value = __half2float(logits[base + i]);
        vals[i] = value;
        max_val = fmaxf(max_val, value);

        if (better_topk(value, i, top1, top1_idx)) {
            top2 = top1;
            top2_idx = top1_idx;
            top1 = value;
            top1_idx = i;
        } else if (better_topk(value, i, top2, top2_idx)) {
            top2 = value;
            top2_idx = i;
        }
    }

    float denom = 0.0f;
#pragma unroll
    for (int i = 0; i < kExperts; ++i) {
        denom += expf(vals[i] - max_val);
    }
    const float inv_denom = 1.0f / denom;

    const int64_t out_base = row * kTopK;
    topk_idx[out_base + 0] = static_cast<int64_t>(top1_idx);
    topk_idx[out_base + 1] = static_cast<int64_t>(top2_idx);
    topk_score[out_base + 0] = __float2half_rn(expf(top1 - max_val) * inv_denom);
    topk_score[out_base + 1] = __float2half_rn(expf(top2 - max_val) * inv_denom);
}

}  // namespace

std::vector<torch::Tensor> top2_softmax_8(torch::Tensor logits) {
    check_logits(logits);

    const int64_t n_rows = logits.numel() / kExperts;
    std::vector<int64_t> out_shape = logits.sizes().vec();
    out_shape.back() = kTopK;

    auto topk_idx = torch::empty(out_shape, logits.options().dtype(torch::kLong));
    auto topk_score = torch::empty(out_shape, logits.options());

    if (n_rows == 0) {
        return {topk_idx, topk_score};
    }

    const int blocks = static_cast<int>((n_rows + kThreads - 1) / kThreads);
    top2_softmax_8_kernel<<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(logits.data_ptr<c10::Half>()),
        topk_idx.data_ptr<int64_t>(),
        reinterpret_cast<__half*>(topk_score.data_ptr<c10::Half>()),
        n_rows
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {topk_idx, topk_score};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("top2_softmax_8", &top2_softmax_8, "fp16 top-2 softmax over last dim 8");
}
