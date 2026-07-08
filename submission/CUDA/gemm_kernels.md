我同意可以借，但我不建议把 `CUDA/gemm_kernels.cu` 做成“通用 PyTorch Linear 替代品”。更现实的路线是：用 LeetCUDA/DeepGEMM 的思想做少数固定 shape、带融合逻辑的 kernel。

**最值得借的是 LeetCUDA**
LeetCUDA 的 SM80 fp16 HGEMM 最贴近 A800：

- [hgemm_mma_stage_tn.cu](/home/ada/github/BAIDU_CTI/LeetCUDA/kernels/hgemm/mma/basic/hgemm_mma_stage_tn.cu)
- [hgemm_mma_stage_tn_swizzle.cu](/home/ada/github/BAIDU_CTI/LeetCUDA/kernels/hgemm/mma/swizzle/hgemm_mma_stage_tn_swizzle.cu)

这些能借：
- `cp.async`
- shared memory double buffer
- `ldmatrix`
- `mma.sync.m16n8k16`
- TN layout
- swizzle tile scheduler
- epilogue 写回布局控制

我们现在的 `smoe_kernels.cu`、`dense_all_smoe_kernels.cu` 本质上已经是这条路线。

**DeepGEMM 更适合借 scheduler，不适合直接搬 kernel**
DeepGEMM 重点是 grouped GEMM / MoE / scheduler，尤其是：

- [scheduler/gemm.cuh](/home/ada/github/BAIDU_CTI/DeepGEMM/deep_gemm/include/deep_gemm/scheduler/gemm.cuh)
- [scheduler/mega_moe.cuh](/home/ada/github/BAIDU_CTI/DeepGEMM/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh)

但它很多 kernel 偏 SM90/SM100、FP8/FP4/BF16，不适合直接搬到 A800 fp16。适合借的是：
- grouped GEMM metadata 设计
- workload scheduler
- MoE block 调度思想
- JIT/heuristic 调参思路

**FlashMLA 不适合用来做普通 GEMM**
FlashMLA 主要是 attention/decode，且很多是 SM90/SM100 思路。对 `gemm_kernels.cu` 直接价值不大。它适合借调度和 pipeline 观念，不适合搬 Linear kernel。

我认为 `gemm_kernels.cu` 后续优先级应该这样排：

1. **QKV projection + layout fusion**
   当前：
   ```python
   qkv = self.qkv_proj[i](x)
   qkv = qkv.view(...).permute(...)
   q, k, v = torch.split(...)
   ```
   如果自定义 GEMM 能直接输出 attention kernel 需要的 `[1, heads, S, 64]` Q/K/V layout，就能省掉中间 `qkv [S,1536]` 的落地、view/permute/split 相关开销。这个比单纯替换 `qkv_proj` 更有意义。

2. **out_proj + AddLayerNorm 融合调研**
   当前：
   ```python
   attn_proj = self.out_proj[i](attn_out)
   residual, x = self.norm2[i].add_layernorm_with_residual(residual, attn_proj)
   ```
   理想上是 GEMM epilogue 直接产生 `residual + attn_proj`，再接 LayerNorm。但这比 QKV layout fusion 难，因为 LayerNorm 要做 row-wise mean/var reduce。

3. **Gate linear 不建议马上继续**
   `w_g: [S,512] -> [S,8]` 是 skinny GEMM。我们试过 `linear + top2` 融合变慢，说明一行一个 CTA 不行。要继续做，需要专门的 warp-level GEMV/skiny GEMM，而不是普通 HGEMM。收益不确定。

4. **RepEncoder.linear 不建议先动**
   `[S,14336] -> [S,512]` 是大 K GEMM，cuBLAS 很强。除非做 `input_norm + linear` 大融合，否则单独替换很可能收益不稳定。

5. **Final head 已经放到 output_kernels.cu 是对的**
   `[S,512] -> [S,1]` 只有结合 `pred_positions + sigmoid + gather` 才有意义。现在已经验证有收益。

所以我的建议是：`gemm_kernels.cu` 如果要开始写，第一目标不要做通用 GEMM，而是做 **`qkv_proj_fused_layout_kernel`**，参考 LeetCUDA SM80 HGEMM TN，把输出直接写成 Q/K/V attention layout。这个最符合“针对性替换”的原则，也比硬刚 cuBLAS 普通 Linear 更有机会带来稳定收益。