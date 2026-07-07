// CUDA full inference runner.
//
// Implement orchestration for the complete CUDA inference path:
// - Own persistent workspace allocation for all intermediate tensors.
// - Launch RepEncoder kernels.
// - Launch 8 Transformer layers in order.
// - Launch Head/output kernels.
// - Manage CUDA streams and CUDA events for profiling.
// - Optionally capture fixed-shape buckets with CUDA Graph.
//
// Minimal execution order:
// 1. embedding_bag -> rep LayerNorm -> rep Linear
// 2. For each Transformer layer:
//    norm1 -> qkv Linear -> varlen causal attention -> out Linear + residual
//    norm2 -> gate Linear -> top2 softmax -> SMoE -> residual
// 3. head Linear -> clamp -> sigmoid -> pred_mask gather
//
// This file should become the replacement for Python-level model.forward
// once individual kernels are correct.
