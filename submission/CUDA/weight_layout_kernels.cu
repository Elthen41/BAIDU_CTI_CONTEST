// CUDA weight layout preparation kernels.
//
// Implement one-time preprocessing for checkpoint weights before inference:
// - Convert and pack PyTorch checkpoint tensors into CUDA-friendly layouts.
// - Transpose Linear weights into the layout expected by GEMM kernels.
// - Pack per-layer Transformer weights contiguously.
// - Pack per-layer/per-expert SMoE weights contiguously.
// - Align fp16 weight storage for vectorized 16-byte loads.
// - Optionally convert fp32 checkpoint weights to fp16 buffers.
//
// This file should not contain model math kernels. It should only prepare
// immutable weights and metadata used by the runtime runner.
