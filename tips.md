# 推理优化简版
1. `sign id clamp` 前移到 `collate/cache` 阶段，避免每次 forward 重复 clamp。
2. Gate 推理时不构造完整 `softmax probs`，用 `topk_logits + logsumexp` 算 top-k score。
3. `ATTN_MODE` 可做 `full/block/hybrid`：full 少 launch，block 少无效计算，hybrid 按成本选。

# 个人想法
1. 对于有clik的log让其q\k分数更高？
2. all_userids可以减少重复？
3. mask_attention和mask_softmax是不是可以减少计算量进行改进？
4. SMoE的实现目前是使用每个专家会推理哪些token，这样似乎不如每个token分别去处理的L2-cache好？但是其可以进行批量处理，同时减少kernel调用次数？我们的优化版本里dense-SMoE就是直接做一个大矩阵乘法最后取前k个专家的结果，直接使用一个更大的batch进行调用，同时也避免了计算每个专家需要对哪些token进行计算的代价？但总感觉这里还能够进行一些优化。
5. 关于mask的实现似乎能使用flashmask进行优化

# 已实现
1. 推理主循环用 `torch.inference_mode()`，比 `no_grad()` 更适合纯推理。
2. CPU batch tensor 尽量 `pin_memory()`，搬到 GPU 时用 `to(device, non_blocking=True)`。
3. `RepEncoder` 优先用 `F.embedding_bag(..., mode="sum")` 替代 `embedding + segment_reduce`。
4. CPU batch tensor 尽量 `pin_memory()`，搬到 GPU 时用 `to(device, non_blocking=True)`。
5. H2D prefetch 用 pinned memory、`non_blocking=True` 和 copy stream 预取下一 batch。