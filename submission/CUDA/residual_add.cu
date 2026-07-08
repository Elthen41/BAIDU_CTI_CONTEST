#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cstdint>

__global__ void residual_add_kernel(
    const float* a,
    const float* b,
    float* out,
    int64_t n
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

torch::Tensor residual_add(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(a.scalar_type() == at::kFloat, "a must be float32");
    TORCH_CHECK(b.scalar_type() == at::kFloat, "b must be float32");
    TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape");
    TORCH_CHECK(a.device() == b.device(), "a and b must be on the same CUDA device");
    TORCH_CHECK(a.is_contiguous(), "a must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "b must be contiguous");

    auto out = torch::empty_like(a);
    const int64_t n = a.numel();
    if (n == 0) {
        return out;
    }

    constexpr int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);

    residual_add_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        n
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("residual_add", &residual_add, "Residual add kernel: out = a + b (CUDA, float32)");
}
