// CUDA RepEncoder embedding-bag kernels.
//
// Implement the feature side of RepEncoder:
// - Clamp feature ids into [0, vocab_size).
// - Sum embedding vectors for offsets-defined bags.
// - Support a single-slot kernel and a 28-slot fused kernel.
// - Embedding dimension is fixed at 512 fp16 values.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <pybind11/stl.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace {

constexpr int kNumSlots = 28;
constexpr int kEmbDim = 512;
constexpr int kVecWidth = 8;
constexpr int kVecsPerRow = kEmbDim / kVecWidth;
constexpr int kThreads = kVecsPerRow;

union Half8 {
    uint4 u4;
    __half h[kVecWidth];
};

void check_half_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kHalf, name, " must be float16");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_i64_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kLong, name, " must be int64");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_same_device(const torch::Tensor& a, const torch::Tensor& b, const char* a_name, const char* b_name) {
    TORCH_CHECK(a.device() == b.device(), a_name, " and ", b_name, " must be on the same CUDA device");
}

void check_embedding_weight(const torch::Tensor& weight) {
    check_half_cuda_contiguous(weight, "embedding_weight");
    TORCH_CHECK(weight.dim() == 2, "embedding_weight must have shape [vocab_size,512]");
    TORCH_CHECK(weight.size(1) == kEmbDim, "embedding_weight must have shape [vocab_size,512]");
    TORCH_CHECK(weight.size(0) > 0, "embedding_weight vocab_size must be positive");
}

void check_slot_inputs(
    const torch::Tensor& values,
    const torch::Tensor& offsets,
    const torch::Tensor& weight,
    const char* values_name,
    const char* offsets_name
) {
    check_i64_cuda_contiguous(values, values_name);
    check_i64_cuda_contiguous(offsets, offsets_name);
    check_same_device(values, weight, values_name, "embedding_weight");
    check_same_device(offsets, weight, offsets_name, "embedding_weight");
    TORCH_CHECK(values.dim() == 1, values_name, " must be 1D");
    TORCH_CHECK(offsets.dim() == 1, offsets_name, " must be 1D");
    TORCH_CHECK(offsets.size(0) >= 1, offsets_name, " must contain at least the final offset");
}

__device__ __forceinline__ int64_t clamp_feature_id(int64_t id, int64_t vocab_size) {
    if (id < 0) {
        return 0;
    }
    if (id >= vocab_size) {
        return vocab_size - 1;
    }
    return id;
}

__device__ __forceinline__ void sum_embedding_vec8(
    const int64_t* __restrict__ values,
    const int64_t* __restrict__ offsets,
    const __half* __restrict__ weight,
    __half* __restrict__ output,
    int64_t row,
    int vec,
    int64_t vocab_size
) {
    float acc[kVecWidth];
#pragma unroll
    for (int i = 0; i < kVecWidth; ++i) {
        acc[i] = 0.0f;
    }

    const int64_t start = offsets[row];
    const int64_t end = offsets[row + 1];
    const int dim_base = vec * kVecWidth;

    for (int64_t pos = start; pos < end; ++pos) {
        const int64_t feature_id = clamp_feature_id(values[pos], vocab_size);
        const __half* __restrict__ weight_vec =
            weight + feature_id * kEmbDim + dim_base;

        Half8 pack;
        pack.u4 = *reinterpret_cast<const uint4*>(weight_vec);
#pragma unroll
        for (int i = 0; i < kVecWidth; ++i) {
            acc[i] += __half2float(pack.h[i]);
        }
    }

    Half8 out_pack;
#pragma unroll
    for (int i = 0; i < kVecWidth; ++i) {
        out_pack.h[i] = __float2half_rn(acc[i]);
    }
    *reinterpret_cast<uint4*>(output + dim_base) = out_pack.u4;
}

__global__ void embedding_bag_slot_sum_kernel(
    const int64_t* __restrict__ values,
    const int64_t* __restrict__ offsets,
    const __half* __restrict__ weight,
    __half* __restrict__ output,
    int64_t n_rows,
    int64_t vocab_size
) {
    const int64_t row = blockIdx.x;
    const int vec = threadIdx.x;
    if (row >= n_rows || vec >= kVecsPerRow) {
        return;
    }

    sum_embedding_vec8(
        values,
        offsets,
        weight,
        output + row * kEmbDim,
        row,
        vec,
        vocab_size
    );
}

__global__ void embedding_bag_28slot_fused_kernel(
    const int64_t* __restrict__ value_ptrs,
    const int64_t* __restrict__ offset_ptrs,
    const __half* __restrict__ weight,
    __half* __restrict__ output,
    int64_t n_rows,
    const int64_t* __restrict__ active_rows,
    int64_t vocab_size
) {
    const int64_t row = blockIdx.x;
    const int slot = blockIdx.y;
    const int vec = threadIdx.x;
    const int64_t active = active_rows == nullptr ? n_rows : active_rows[0];
    if (row >= n_rows || row >= active || slot >= kNumSlots || vec >= kVecsPerRow) {
        return;
    }

    const int64_t* __restrict__ values =
        reinterpret_cast<const int64_t*>(value_ptrs[slot]);
    const int64_t* __restrict__ offsets =
        reinterpret_cast<const int64_t*>(offset_ptrs[slot]);
    __half* __restrict__ slot_output =
        output + row * (kNumSlots * kEmbDim) + slot * kEmbDim;

    sum_embedding_vec8(
        values,
        offsets,
        weight,
        slot_output,
        row,
        vec,
        vocab_size
    );
}

torch::Tensor make_pointer_tensor(
    const std::vector<torch::Tensor>& tensors,
    const torch::Tensor& reference
) {
    std::vector<int64_t> host_ptrs;
    host_ptrs.reserve(tensors.size());
    for (const auto& tensor : tensors) {
        host_ptrs.push_back(static_cast<int64_t>(
            reinterpret_cast<std::uintptr_t>(tensor.data_ptr<int64_t>())
        ));
    }

    auto ptrs = torch::empty(
        {static_cast<int64_t>(tensors.size())},
        reference.options().dtype(torch::kLong)
    );
    auto stream = at::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaMemcpyAsync(
        ptrs.data_ptr<int64_t>(),
        host_ptrs.data(),
        static_cast<size_t>(host_ptrs.size()) * sizeof(int64_t),
        cudaMemcpyHostToDevice,
        stream.stream()
    ));
    return ptrs;
}

void check_pointer_tensor(const torch::Tensor& ptrs, const char* name, const torch::Tensor& reference) {
    check_i64_cuda_contiguous(ptrs, name);
    check_same_device(ptrs, reference, name, "embedding_weight");
    TORCH_CHECK(ptrs.dim() == 1, name, " must be 1D");
    TORCH_CHECK(ptrs.size(0) == kNumSlots, name, " must contain 28 pointers");
}

}  // namespace

torch::Tensor embedding_bag_slot_sum(
    torch::Tensor values,
    torch::Tensor offsets,
    torch::Tensor embedding_weight
) {
    check_embedding_weight(embedding_weight);
    check_slot_inputs(values, offsets, embedding_weight, "values", "offsets");

    const int64_t n_rows = offsets.size(0) - 1;
    auto output = torch::empty({n_rows, kEmbDim}, embedding_weight.options());
    if (n_rows == 0) {
        return output;
    }

    const dim3 block(kThreads);
    const dim3 grid(static_cast<unsigned int>(n_rows));
    embedding_bag_slot_sum_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        values.data_ptr<int64_t>(),
        offsets.data_ptr<int64_t>(),
        reinterpret_cast<const __half*>(embedding_weight.data_ptr<c10::Half>()),
        reinterpret_cast<__half*>(output.data_ptr<c10::Half>()),
        n_rows,
        embedding_weight.size(0)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}

torch::Tensor embedding_bag_28slot_fused(
    torch::Tensor embedding_weight,
    std::vector<torch::Tensor> slot_values,
    std::vector<torch::Tensor> slot_offsets
) {
    check_embedding_weight(embedding_weight);
    TORCH_CHECK(slot_values.size() == kNumSlots, "slot_values must contain 28 tensors");
    TORCH_CHECK(slot_offsets.size() == kNumSlots, "slot_offsets must contain 28 tensors");

    int64_t n_rows = -1;
    for (int slot = 0; slot < kNumSlots; ++slot) {
        const std::string values_name = "slot_values[" + std::to_string(slot) + "]";
        const std::string offsets_name = "slot_offsets[" + std::to_string(slot) + "]";
        check_slot_inputs(
            slot_values[slot],
            slot_offsets[slot],
            embedding_weight,
            values_name.c_str(),
            offsets_name.c_str()
        );

        const int64_t slot_rows = slot_offsets[slot].size(0) - 1;
        if (slot == 0) {
            n_rows = slot_rows;
        } else {
            TORCH_CHECK(slot_rows == n_rows, "all slot_offsets tensors must have the same N + 1 length");
        }
    }

    auto output = torch::empty({n_rows, kNumSlots * kEmbDim}, embedding_weight.options());
    if (n_rows == 0) {
        return output;
    }

    auto value_ptrs = make_pointer_tensor(slot_values, embedding_weight);
    auto offset_ptrs = make_pointer_tensor(slot_offsets, embedding_weight);

    const dim3 block(kThreads);
    const dim3 grid(static_cast<unsigned int>(n_rows), kNumSlots);
    embedding_bag_28slot_fused_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        value_ptrs.data_ptr<int64_t>(),
        offset_ptrs.data_ptr<int64_t>(),
        reinterpret_cast<const __half*>(embedding_weight.data_ptr<c10::Half>()),
        reinterpret_cast<__half*>(output.data_ptr<c10::Half>()),
        n_rows,
        nullptr,
        embedding_weight.size(0)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}

torch::Tensor embedding_bag_28slot_fused_with_ptrs(
    torch::Tensor embedding_weight,
    torch::Tensor value_ptrs,
    torch::Tensor offset_ptrs,
    int64_t n_rows
) {
    check_embedding_weight(embedding_weight);
    check_pointer_tensor(value_ptrs, "value_ptrs", embedding_weight);
    check_pointer_tensor(offset_ptrs, "offset_ptrs", embedding_weight);
    TORCH_CHECK(n_rows >= 0, "n_rows must be non-negative");

    auto output = torch::empty({n_rows, kNumSlots * kEmbDim}, embedding_weight.options());
    if (n_rows == 0) {
        return output;
    }

    const dim3 block(kThreads);
    const dim3 grid(static_cast<unsigned int>(n_rows), kNumSlots);
    embedding_bag_28slot_fused_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        value_ptrs.data_ptr<int64_t>(),
        offset_ptrs.data_ptr<int64_t>(),
        reinterpret_cast<const __half*>(embedding_weight.data_ptr<c10::Half>()),
        reinterpret_cast<__half*>(output.data_ptr<c10::Half>()),
        n_rows,
        nullptr,
        embedding_weight.size(0)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}

torch::Tensor embedding_bag_28slot_fused_with_ptrs_active(
    torch::Tensor embedding_weight,
    torch::Tensor value_ptrs,
    torch::Tensor offset_ptrs,
    int64_t output_rows,
    torch::Tensor active_rows
) {
    check_embedding_weight(embedding_weight);
    check_pointer_tensor(value_ptrs, "value_ptrs", embedding_weight);
    check_pointer_tensor(offset_ptrs, "offset_ptrs", embedding_weight);
    check_i64_cuda_contiguous(active_rows, "active_rows");
    check_same_device(active_rows, embedding_weight, "active_rows", "embedding_weight");
    TORCH_CHECK(active_rows.dim() == 1 && active_rows.numel() == 1, "active_rows must have shape [1]");
    TORCH_CHECK(output_rows >= 0, "output_rows must be non-negative");

    auto output = torch::empty({output_rows, kNumSlots * kEmbDim}, embedding_weight.options());
    if (output_rows == 0) {
        return output;
    }

    const dim3 block(kThreads);
    const dim3 grid(static_cast<unsigned int>(output_rows), kNumSlots);
    embedding_bag_28slot_fused_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        value_ptrs.data_ptr<int64_t>(),
        offset_ptrs.data_ptr<int64_t>(),
        reinterpret_cast<const __half*>(embedding_weight.data_ptr<c10::Half>()),
        reinterpret_cast<__half*>(output.data_ptr<c10::Half>()),
        output_rows,
        active_rows.data_ptr<int64_t>(),
        embedding_weight.size(0)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("embedding_bag_slot_sum", &embedding_bag_slot_sum, "Single-slot fp16 embedding bag sum");
    m.def("embedding_bag_28slot_fused", &embedding_bag_28slot_fused, "Fused 28-slot fp16 embedding bag sum");
    m.def(
        "embedding_bag_28slot_fused_with_ptrs",
        &embedding_bag_28slot_fused_with_ptrs,
        "Graph-safe fused 28-slot fp16 embedding bag sum using prebuilt CUDA pointer tensors"
    );
    m.def(
        "embedding_bag_28slot_fused_with_ptrs_active",
        &embedding_bag_28slot_fused_with_ptrs_active,
        "Graph-safe fused 28-slot fp16 embedding bag sum with fixed output rows and device active-row count"
    );
}
