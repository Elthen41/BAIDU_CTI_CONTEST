// CUDA output kernels for the final CTR prediction head.
//
// head_logits_all keeps CTRModel.forward semantics:
//   encoder_output [S,512] -> head Linear(512 -> 1) -> clamp [-15, 15]
//   returns full logits [S,1].
//
// head_sigmoid_gather_positions is the optimized local-runner helper:
//   encoder_output[pred_positions] [P,512] -> head -> clamp -> sigmoid
//   -> compact logid/prob buffers.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

constexpr int kHiddenDim = 512;
constexpr int kThreads = 256;
constexpr int kUint4Width = 8;
constexpr int kUint4Threads = kHiddenDim / kUint4Width;

static_assert(kHiddenDim == kThreads * 2, "half2 final head assumes two hidden values per thread");
static_assert(kHiddenDim % kUint4Width == 0, "uint4 final head assumes hidden dim is divisible by 8");

union Half8 {
    uint4 u4;
    __half h[kUint4Width];
};

bool use_output_head_half2() {
    const char* flag = std::getenv("USE_OUTPUT_HEAD_HALF2");
    return flag != nullptr && !(flag[0] == '\0' || (flag[0] == '0' && flag[1] == '\0'));
}

bool use_output_head_uint4() {
    const char* flag = std::getenv("USE_OUTPUT_HEAD_UINT4");
    return flag != nullptr && !(flag[0] == '\0' || (flag[0] == '0' && flag[1] == '\0'));
}

__device__ __forceinline__ float half2_dot(const __half2 x, const __half2 w) {
    return __half2float(__low2half(x)) * __half2float(__low2half(w)) +
           __half2float(__high2half(x)) * __half2float(__high2half(w));
}

__device__ __forceinline__ float half8_dot(const Half8 x, const Half8 w) {
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < kUint4Width; ++i) {
        sum += __half2float(x.h[i]) * __half2float(w.h[i]);
    }
    return sum;
}

void check_same_device(const torch::Tensor& a, const torch::Tensor& b, const char* a_name, const char* b_name) {
    TORCH_CHECK(a.device() == b.device(), a_name, " and ", b_name, " must be on the same CUDA device");
}

void check_inputs(
    const torch::Tensor& encoder_output,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& logids,
    const torch::Tensor& pred_positions
) {
    TORCH_CHECK(encoder_output.is_cuda(), "encoder_output must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(logids.is_cuda(), "logids must be a CUDA tensor");
    TORCH_CHECK(pred_positions.is_cuda(), "pred_positions must be a CUDA tensor");

    check_same_device(encoder_output, weight, "encoder_output", "weight");
    check_same_device(encoder_output, bias, "encoder_output", "bias");
    check_same_device(encoder_output, logids, "encoder_output", "logids");
    check_same_device(encoder_output, pred_positions, "encoder_output", "pred_positions");

    TORCH_CHECK(encoder_output.scalar_type() == at::kHalf, "encoder_output must be float16");
    TORCH_CHECK(weight.scalar_type() == at::kHalf, "weight must be float16");
    TORCH_CHECK(bias.scalar_type() == at::kHalf, "bias must be float16");
    TORCH_CHECK(logids.scalar_type() == at::kLong, "logids must be int64");
    TORCH_CHECK(pred_positions.scalar_type() == at::kLong, "pred_positions must be int64");

    TORCH_CHECK(encoder_output.is_contiguous(), "encoder_output must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(logids.is_contiguous(), "logids must be contiguous");
    TORCH_CHECK(pred_positions.is_contiguous(), "pred_positions must be contiguous");

    TORCH_CHECK(encoder_output.dim() == 2, "encoder_output must have shape [S,512]");
    TORCH_CHECK(encoder_output.size(1) == kHiddenDim, "encoder_output must have shape [S,512]");
    TORCH_CHECK(weight.dim() == 2, "weight must have shape [1,512]");
    TORCH_CHECK(weight.size(0) == 1, "weight must have shape [1,512]");
    TORCH_CHECK(weight.size(1) == kHiddenDim, "weight must have shape [1,512]");
    TORCH_CHECK(bias.dim() == 1, "bias must have shape [1]");
    TORCH_CHECK(bias.size(0) == 1, "bias must have shape [1]");
    TORCH_CHECK(logids.dim() == 1, "logids must have shape [S]");
    TORCH_CHECK(logids.size(0) == encoder_output.size(0), "logids must have shape [S]");
    TORCH_CHECK(pred_positions.dim() == 1, "pred_positions must have shape [P]");
    TORCH_CHECK(
        pred_positions.size(0) <= static_cast<int64_t>(std::numeric_limits<int>::max()),
        "pred_positions has too many elements"
    );
}

void check_head_logits_inputs(
    const torch::Tensor& encoder_output,
    const torch::Tensor& weight,
    const torch::Tensor& bias
) {
    TORCH_CHECK(encoder_output.is_cuda(), "encoder_output must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");

    check_same_device(encoder_output, weight, "encoder_output", "weight");
    check_same_device(encoder_output, bias, "encoder_output", "bias");

    TORCH_CHECK(encoder_output.scalar_type() == at::kHalf, "encoder_output must be float16");
    TORCH_CHECK(weight.scalar_type() == at::kHalf, "weight must be float16");
    TORCH_CHECK(bias.scalar_type() == at::kHalf, "bias must be float16");

    TORCH_CHECK(encoder_output.is_contiguous(), "encoder_output must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");

    TORCH_CHECK(encoder_output.dim() == 2, "encoder_output must have shape [S,512]");
    TORCH_CHECK(encoder_output.size(1) == kHiddenDim, "encoder_output must have shape [S,512]");
    TORCH_CHECK(weight.dim() == 2, "weight must have shape [1,512]");
    TORCH_CHECK(weight.size(0) == 1, "weight must have shape [1,512]");
    TORCH_CHECK(weight.size(1) == kHiddenDim, "weight must have shape [1,512]");
    TORCH_CHECK(bias.dim() == 1, "bias must have shape [1]");
    TORCH_CHECK(bias.size(0) == 1, "bias must have shape [1]");
    TORCH_CHECK(
        encoder_output.size(0) <= static_cast<int64_t>(std::numeric_limits<int>::max()),
        "encoder_output has too many tokens"
    );
}

__global__ void head_logits_all_kernel(
    const __half* __restrict__ encoder_output,
    const __half* __restrict__ weight,
    const __half* __restrict__ bias,
    __half* __restrict__ out_logits,
    int64_t n_tokens
) {
    const int64_t token_idx = static_cast<int64_t>(blockIdx.x);
    if (token_idx >= n_tokens) {
        return;
    }

    const int tid = threadIdx.x;
    float sum = 0.0f;
    const int64_t token_base = token_idx * kHiddenDim;
    if (tid < kHiddenDim) {
        sum += __half2float(encoder_output[token_base + tid]) * __half2float(weight[tid]);
    }
    const int dim1 = tid + kThreads;
    if (dim1 < kHiddenDim) {
        sum += __half2float(encoder_output[token_base + dim1]) * __half2float(weight[dim1]);
    }

    __shared__ float partial[kThreads];
    partial[tid] = sum;
    __syncthreads();

    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float logit = __half2float(__float2half_rn(partial[0] + __half2float(bias[0])));
        logit = fminf(fmaxf(logit, -15.0f), 15.0f);
        out_logits[token_idx] = __float2half_rn(logit);
    }
}

__global__ void head_logits_all_half2_kernel(
    const __half* __restrict__ encoder_output,
    const __half* __restrict__ weight,
    const __half* __restrict__ bias,
    __half* __restrict__ out_logits,
    int64_t n_tokens
) {
    const int64_t token_idx = static_cast<int64_t>(blockIdx.x);
    if (token_idx >= n_tokens) {
        return;
    }

    const int tid = threadIdx.x;
    const int dim = tid * 2;
    const int64_t token_base = token_idx * kHiddenDim;
    const __half2 x2 = *reinterpret_cast<const __half2*>(encoder_output + token_base + dim);
    const __half2 w2 = *reinterpret_cast<const __half2*>(weight + dim);
    const float sum = half2_dot(x2, w2);

    __shared__ float partial[kThreads];
    partial[tid] = sum;
    __syncthreads();

    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float logit = __half2float(__float2half_rn(partial[0] + __half2float(bias[0])));
        logit = fminf(fmaxf(logit, -15.0f), 15.0f);
        out_logits[token_idx] = __float2half_rn(logit);
    }
}

__global__ void head_logits_all_uint4_kernel(
    const __half* __restrict__ encoder_output,
    const __half* __restrict__ weight,
    const __half* __restrict__ bias,
    __half* __restrict__ out_logits,
    int64_t n_tokens
) {
    const int64_t token_idx = static_cast<int64_t>(blockIdx.x);
    if (token_idx >= n_tokens) {
        return;
    }

    const int tid = threadIdx.x;
    const int dim = tid * kUint4Width;
    const int64_t token_base = token_idx * kHiddenDim;

    Half8 x_pack;
    Half8 w_pack;
    x_pack.u4 = *reinterpret_cast<const uint4*>(encoder_output + token_base + dim);
    w_pack.u4 = *reinterpret_cast<const uint4*>(weight + dim);
    const float sum = half8_dot(x_pack, w_pack);

    __shared__ float partial[kUint4Threads];
    partial[tid] = sum;
    __syncthreads();

    for (int stride = kUint4Threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float logit = __half2float(__float2half_rn(partial[0] + __half2float(bias[0])));
        logit = fminf(fmaxf(logit, -15.0f), 15.0f);
        out_logits[token_idx] = __float2half_rn(logit);
    }
}

__global__ void head_sigmoid_gather_positions_kernel(
    const __half* __restrict__ encoder_output,
    const __half* __restrict__ weight,
    const __half* __restrict__ bias,
    const int64_t* __restrict__ logids,
    const int64_t* __restrict__ pred_positions,
    int64_t* __restrict__ out_logids,
    float* __restrict__ out_probs,
    int64_t n_tokens,
    int64_t n_pred
) {
    const int64_t pred_idx = static_cast<int64_t>(blockIdx.x);
    if (pred_idx >= n_pred) {
        return;
    }

    const int tid = threadIdx.x;
    const int64_t token_idx = pred_positions[pred_idx];
    if (token_idx < 0 || token_idx >= n_tokens) {
        if (tid == 0) {
            out_logids[pred_idx] = 0;
            out_probs[pred_idx] = 0.0f;
        }
        return;
    }

    float sum = 0.0f;
    const int64_t token_base = token_idx * kHiddenDim;
    if (tid < kHiddenDim) {
        sum += __half2float(encoder_output[token_base + tid]) * __half2float(weight[tid]);
    }
    const int dim1 = tid + kThreads;
    if (dim1 < kHiddenDim) {
        sum += __half2float(encoder_output[token_base + dim1]) * __half2float(weight[dim1]);
    }

    __shared__ float partial[kThreads];
    partial[tid] = sum;
    __syncthreads();

    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float logit = __half2float(__float2half_rn(partial[0] + __half2float(bias[0])));
        logit = fminf(fmaxf(logit, -15.0f), 15.0f);
        out_logids[pred_idx] = logids[token_idx];
        out_probs[pred_idx] = 1.0f / (1.0f + expf(-logit));
    }
}

__global__ void head_sigmoid_gather_positions_uint4_kernel(
    const __half* __restrict__ encoder_output,
    const __half* __restrict__ weight,
    const __half* __restrict__ bias,
    const int64_t* __restrict__ logids,
    const int64_t* __restrict__ pred_positions,
    int64_t* __restrict__ out_logids,
    float* __restrict__ out_probs,
    int64_t n_tokens,
    int64_t n_pred
) {
    const int64_t pred_idx = static_cast<int64_t>(blockIdx.x);
    if (pred_idx >= n_pred) {
        return;
    }

    const int tid = threadIdx.x;
    const int64_t token_idx = pred_positions[pred_idx];
    if (token_idx < 0 || token_idx >= n_tokens) {
        if (tid == 0) {
            out_logids[pred_idx] = 0;
            out_probs[pred_idx] = 0.0f;
        }
        return;
    }

    const int dim = tid * kUint4Width;
    const int64_t token_base = token_idx * kHiddenDim;

    Half8 x_pack;
    Half8 w_pack;
    x_pack.u4 = *reinterpret_cast<const uint4*>(encoder_output + token_base + dim);
    w_pack.u4 = *reinterpret_cast<const uint4*>(weight + dim);
    const float sum = half8_dot(x_pack, w_pack);

    __shared__ float partial[kUint4Threads];
    partial[tid] = sum;
    __syncthreads();

    for (int stride = kUint4Threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float logit = __half2float(__float2half_rn(partial[0] + __half2float(bias[0])));
        logit = fminf(fmaxf(logit, -15.0f), 15.0f);
        out_logids[pred_idx] = logids[token_idx];
        out_probs[pred_idx] = 1.0f / (1.0f + expf(-logit));
    }
}

__global__ void head_sigmoid_gather_positions_half2_kernel(
    const __half* __restrict__ encoder_output,
    const __half* __restrict__ weight,
    const __half* __restrict__ bias,
    const int64_t* __restrict__ logids,
    const int64_t* __restrict__ pred_positions,
    int64_t* __restrict__ out_logids,
    float* __restrict__ out_probs,
    int64_t n_tokens,
    int64_t n_pred
) {
    const int64_t pred_idx = static_cast<int64_t>(blockIdx.x);
    if (pred_idx >= n_pred) {
        return;
    }

    const int tid = threadIdx.x;
    const int64_t token_idx = pred_positions[pred_idx];
    if (token_idx < 0 || token_idx >= n_tokens) {
        if (tid == 0) {
            out_logids[pred_idx] = 0;
            out_probs[pred_idx] = 0.0f;
        }
        return;
    }

    const int dim = tid * 2;
    const int64_t token_base = token_idx * kHiddenDim;
    const __half2 x2 = *reinterpret_cast<const __half2*>(encoder_output + token_base + dim);
    const __half2 w2 = *reinterpret_cast<const __half2*>(weight + dim);
    const float sum = half2_dot(x2, w2);

    __shared__ float partial[kThreads];
    partial[tid] = sum;
    __syncthreads();

    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float logit = __half2float(__float2half_rn(partial[0] + __half2float(bias[0])));
        logit = fminf(fmaxf(logit, -15.0f), 15.0f);
        out_logids[pred_idx] = logids[token_idx];
        out_probs[pred_idx] = 1.0f / (1.0f + expf(-logit));
    }
}

}  // namespace

torch::Tensor head_logits_all(
    torch::Tensor encoder_output,
    torch::Tensor weight,
    torch::Tensor bias
) {
    check_head_logits_inputs(encoder_output, weight, bias);

    const int64_t n_tokens = encoder_output.size(0);
    auto out_logits = torch::empty({n_tokens, 1}, encoder_output.options());

    if (n_tokens == 0) {
        return out_logits;
    }

    if (use_output_head_uint4()) {
        head_logits_all_uint4_kernel<<<static_cast<int>(n_tokens), kUint4Threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __half*>(encoder_output.data_ptr<c10::Half>()),
            reinterpret_cast<const __half*>(weight.data_ptr<c10::Half>()),
            reinterpret_cast<const __half*>(bias.data_ptr<c10::Half>()),
            reinterpret_cast<__half*>(out_logits.data_ptr<c10::Half>()),
            n_tokens
        );
    } else if (use_output_head_half2()) {
        head_logits_all_half2_kernel<<<static_cast<int>(n_tokens), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __half*>(encoder_output.data_ptr<c10::Half>()),
            reinterpret_cast<const __half*>(weight.data_ptr<c10::Half>()),
            reinterpret_cast<const __half*>(bias.data_ptr<c10::Half>()),
            reinterpret_cast<__half*>(out_logits.data_ptr<c10::Half>()),
            n_tokens
        );
    } else {
        head_logits_all_kernel<<<static_cast<int>(n_tokens), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __half*>(encoder_output.data_ptr<c10::Half>()),
            reinterpret_cast<const __half*>(weight.data_ptr<c10::Half>()),
            reinterpret_cast<const __half*>(bias.data_ptr<c10::Half>()),
            reinterpret_cast<__half*>(out_logits.data_ptr<c10::Half>()),
            n_tokens
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out_logits;
}

std::vector<torch::Tensor> head_sigmoid_gather_positions(
    torch::Tensor encoder_output,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor logids,
    torch::Tensor pred_positions
) {
    check_inputs(encoder_output, weight, bias, logids, pred_positions);

    const int64_t n_tokens = encoder_output.size(0);
    const int64_t n_pred = pred_positions.size(0);
    auto out_logids = torch::empty({n_pred}, logids.options());
    auto out_probs = torch::empty({n_pred}, encoder_output.options().dtype(torch::kFloat));

    if (n_pred == 0) {
        return {out_logids, out_probs};
    }

    if (use_output_head_uint4()) {
        head_sigmoid_gather_positions_uint4_kernel<<<static_cast<int>(n_pred), kUint4Threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __half*>(encoder_output.data_ptr<c10::Half>()),
            reinterpret_cast<const __half*>(weight.data_ptr<c10::Half>()),
            reinterpret_cast<const __half*>(bias.data_ptr<c10::Half>()),
            logids.data_ptr<int64_t>(),
            pred_positions.data_ptr<int64_t>(),
            out_logids.data_ptr<int64_t>(),
            out_probs.data_ptr<float>(),
            n_tokens,
            n_pred
        );
    } else if (use_output_head_half2()) {
        head_sigmoid_gather_positions_half2_kernel<<<static_cast<int>(n_pred), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __half*>(encoder_output.data_ptr<c10::Half>()),
            reinterpret_cast<const __half*>(weight.data_ptr<c10::Half>()),
            reinterpret_cast<const __half*>(bias.data_ptr<c10::Half>()),
            logids.data_ptr<int64_t>(),
            pred_positions.data_ptr<int64_t>(),
            out_logids.data_ptr<int64_t>(),
            out_probs.data_ptr<float>(),
            n_tokens,
            n_pred
        );
    } else {
        head_sigmoid_gather_positions_kernel<<<static_cast<int>(n_pred), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __half*>(encoder_output.data_ptr<c10::Half>()),
            reinterpret_cast<const __half*>(weight.data_ptr<c10::Half>()),
            reinterpret_cast<const __half*>(bias.data_ptr<c10::Half>()),
            logids.data_ptr<int64_t>(),
            pred_positions.data_ptr<int64_t>(),
            out_logids.data_ptr<int64_t>(),
            out_probs.data_ptr<float>(),
            n_tokens,
            n_pred
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {out_logids, out_probs};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "head_logits_all",
        &head_logits_all,
        "Final head Linear(512->1) + clamp for all tokens"
    );
    m.def(
        "head_sigmoid_gather_positions",
        &head_sigmoid_gather_positions,
        "Final head Linear(512->1) + sigmoid + pred position gather"
    );
}
