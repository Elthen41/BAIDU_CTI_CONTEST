看完了，结论是：这条路线是对的，而且仓库里最值得借鉴的顺序是 **DeepGEMM > LeetCUDA > FlashMLA**。

**可借鉴点**

DeepGEMM 最相关。它的 contiguous grouped GEMM 正是 “不同 expert 有不同 M，N/K 固定” 的 MoE 场景，README 里也明确说 expert tokens 拼成 contiguous layout，并且每个 expert segment 要按 GEMM `block_m` 对齐：[DeepGEMM/README.md](/home/ada/github/BAIDU_CTI/DeepGEMM/README.md:78)。

它的 MegaMoE 设计也很有用：

- workspace 里有 token pool、expert count、token source metadata 的概念：[mega_moe.cuh](/home/ada/github/BAIDU_CTI/DeepGEMM/deep_gemm/include/deep_gemm/layout/mega_moe.cuh:8)
- `TokenSrcMetadata` 的思想可以简化成我们需要的 `{token_idx, topk_slot}`，用于最后 scatter/reduce。
- block_m heuristic 可参考：[mega_moe.hpp](/home/ada/github/BAIDU_CTI/DeepGEMM/csrc/jit_kernels/heuristics/mega_moe.hpp:64)。它按 expected tokens/expert 选 `block_m`，这对 routed sparse 很重要。
- scheduler 里按 expert 的 token count 分配 L1/L2 blocks：[scheduler/mega_moe.cuh](/home/ada/github/BAIDU_CTI/DeepGEMM/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh:13)。第一版不用照搬 persistent scheduler，但后续优化有价值。

LeetCUDA 最适合借低层实现。`hgemm_mma_stage_tn.cu` 已经是 SM80/A800 可用的 fp16 Tensor Core 路线：128x128x16 tile、`cp.async`、`ldmatrix`、`mma.sync.m16n8k16`：[hgemm_mma_stage_tn.cu](/home/ada/github/BAIDU_CTI/LeetCUDA/kernels/hgemm/mma/basic/hgemm_mma_stage_tn.cu:121)。这比 DeepGEMM 的 SM90/SM100 FP8/FP4 kernel 更适合直接迁移。

FlashMLA 不太适合直接搬 kernel。它主要是 SM90/SM100 attention，很多 TMA/GMMA/WGMMA 对 A800 不适用。但它的 scheduler metadata 思路可参考：先算调度元数据，再让主 kernel 按 metadata 做紧凑工作分配：[get_decoding_sched_meta.cu](/home/ada/github/BAIDU_CTI/FlashMLA/csrc/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu:11)。

**建议实现路线**

仓库里其实已经有对应骨架：[CUDA/smoe_kernels.cu](/home/ada/github/BAIDU_CTI/CUDA/smoe_kernels.cu:3)，而且 [CUDA/cuda-rewrite-plan.md](/home/ada/github/BAIDU_CTI/CUDA/cuda-rewrite-plan.md:270) 写的路线和你现在说的一致。建议下一步就填这个文件，而不是继续改 dense-all。

第一版建议这样做：

1. `route_count + prefix + pack`
   - `counts[8]`
   - `offsets[9]`
   - `route_token[2N]`
   - `route_slot[2N]`
   - `route_score[2N]`
   - `route_pos_by_token_slot[N,2]`
   - `x_route[total_routes_aligned,512]`

2. `grouped_fc1_relu`
   - A: `x_route[total_routes,512]`
   - W1: `[8,1024,512]`
   - output: `h_route[total_routes,1024]`
   - 先用 LeetCUDA TN MMA 改成 grouped 版本。

3. `grouped_fc2`
   - A: `h_route[total_routes,1024]`
   - W2: `[8,512,1024]`
   - output: `y_route[total_routes,512]`

4. `route_reduce`
   - 不建议 fp16 atomic 作为主路径。
   - 用 `route_pos_by_token_slot[N,2]` 直接做：
     ```text
     out[n,d] = score0 * y_route[pos0,d] + score1 * y_route[pos1,d]
     ```

第一版的 grouped GEMM scheduler 可以先简单一点：`grid = (n_block, m_block, expert)`，每个 expert 根据 `counts[e]` 判断当前 M block 是否有效。后续再做 compact block list 或 DeepGEMM 风格 scheduler，减少空 block 和提升负载均衡。

**关键判断**

不要直接搬 DeepGEMM kernel，因为它偏 SM90/SM100、FP8/FP4、多卡 symmetric memory；但它的 **contiguous routed layout、expert count metadata、block_m heuristic、L1/L2 scheduler** 很值得照着设计。

真正能直接复用到 A800 fp16 的，是 LeetCUDA 的 SM80 HGEMM TN MMA 微内核。下一步最合理的是：以 `CUDA/smoe_kernels.cu` 为主文件，先实现 routed metadata + correctness-first grouped GEMM，再逐步替换成 LeetCUDA 风格 MMA。





可以优化，而且最优先的不是 MMA 细节，而是 **scheduler/metadata**。当前 `CUDA/smoe_kernels.cu` 第一版能跑通路线，但会启动大量空 CTA。

**最优先**
1. **改 grouped GEMM 的 block scheduler**
   
   现在 `launch_grouped_linear()` 用的是：

   ```text
   grid = (OutDim/128, ceil(2N/128), 8 experts)
   ```

   这是假设每个 expert 都可能吃到 `2N` routes。实际 balanced top2 下，每个 expert 大概只有 `N/4` routes，所以大约会有 7/8 的 M blocks 直接 `return`。

   更好的方式是参考 DeepGEMM contiguous grouped layout：

   ```text
   offsets[e] 已经按 BlockM padding
   total_pool_blocks = offsets[8] / BlockM
   grid = (OutDim/128, total_pool_blocks)
   ```

   kernel 内用 `pool_block_idx` 扫 8 个 offsets 找 expert：

   ```text
   pool_m_base = blockIdx.y * BlockM
   expert = find e where offsets[e] <= pool_m_base < offsets[e+1]
   local_m = pool_m_base - offsets[e]
   ```

   这一步能直接减少大量空 CTA，是当前最值得做的。

2. **route_count 改成 block-local histogram**

   当前 `smoe_route_count_kernel` 对 8 个 expert counter 做全局 atomic。因为 expert 只有 8 个，冲突会比较集中。

   可改成：

   ```text
   每个 block shared counts[8]
   block 内累加
   block 结束后每个 expert 只做一次 global atomicAdd
   ```

   这比 LeetCUDA histogram 里的简单 global atomic 更适合这里。

3. **route_pack 降低 atomic 和 block 开销**

   当前是“一条 route 一个 CTA”，每个 CTA 拷贝 512 half。正确但开销偏粗。

   后续可以做两种优化：

   - 一个 CTA pack 多条 route，比如 2 或 4 条 route。
   - warp/block aggregated atomic：同一个 block 内按 expert 聚合后，一次 atomic 申请一段位置，再用局部 prefix 分配 route position。

   DeepGEMM 的 dispatch metadata 思路更接近这个方向。

**MMA 侧优化**
4. **K_STAGE sweep：2/3/4 stage**

   当前是 2-stage cp.async。LeetCUDA 有 stage kernel 结构，可以尝试 3 或 4 stage。对 `fc2 K=1024` 可能更有收益；对 `fc1 K=512` 要实测。

5. **block_m heuristic**

   DeepGEMM 会按 expected tokens/expert 选 `block_m`。我们现在固定 `128`。如果某些 batch route 很少，`64` 可能更好；如果 batch 很大，`128` 仍可能最好。

   但这需要写第二套 kernel variant，不是第一优先级。

6. **block swizzle / expert wave**

   当前 block 调度不保证同一 expert 的 weight reuse。后面可以参考 LeetCUDA block swizzle 或 DeepGEMM experts-per-wave，让同 expert 的 N/M blocks 更集中，提升 L2 weight locality。

**内存与融合**
7. **去掉 forward 里不用的 metadata 写入**

   `route_token/route_slot/route_score` 当前在 `smoe_forward` 主路径里没有用到，只有 debug/未来融合可能用。可以拆一个 lightweight pack，只保留：

   ```text
   x_route
   route_pos
   counts
   offsets
   ```

8. **route_reduce vectorize**

   当前 route_reduce 是每个 half 一个线程。可以改成 `uint4`/8 half 一组，按 token + 8 dims 处理，减少线程和指令开销。

9. **fc2 + reduce 的非 atomic 融合**

   不建议直接 `atomicAdd(half*)` 到最终 out。更稳的是 fc2 epilogue 写：

   ```text
   partial_out[slot, token, dim] = score * y
   ```

   然后一个很轻的 vectorized kernel 做：

   ```text
   out = partial_out[0] + partial_out[1]
   ```

**建议下一步**

我建议先做第 1 个：把 grouped GEMM 从 `(expert, worst-case m_block)` 改成基于 `offsets[8]` 的 pooled-M scheduler。它最像 DeepGEMM contiguous grouped GEMM 的核心思想，而且改动相对集中，收益也最可能明显。