# BAIDU_CTI

本仓库定位为 CTR 推理任务的高性能 CUDA 算子优化工程。核心目标是在保持模型行为一致的前提下，将 PyTorch 中开销较高、形状相对固定的推理路径逐步替换为自定义 CUDA kernel，并通过 `build_env.sh` 中的 CMake 构建为 PyTorch extension 接入 `infer.py`。

当前主要面向最终 A800 测试环境，默认以 fp16 推理为优化基准：

```python
if USE_MODEL_HALF and dev.type == "cuda":
    model.half()
```

因此自定义 CUDA 算子优先支持 `float16` 输入，并在需要数值稳定性的中间状态中使用 fp32 累加。

## 当前优化范围

- `CUDA/norm_kernels.cu`：Transformer 中固定宽度 LayerNorm，以及 residual add + LayerNorm 融合。
- `CUDA/attention_kernels.cu`：基于 `user_offsets` 的变长 causal attention，保证不同用户序列之间不互相 attend。
- `CUDA/attention_opt.md`：attention kernel 后续 tile 化、online softmax、调度策略等优化记录。

## Attention 方向

现有 attention fallback 是 one-query-per-block 的正确性优先版本。下一阶段优化重点是 tile 级 attention：一个 CTA 处理多个 query token，复用同一批 K/V tile，并使用 fp32 的 online softmax 状态维护 `m/l/O`。

第一版 tiled kernel 暂不使用 Tensor Core MMA、`ldmatrix`、persistent scheduler 或 fp8，优先验证共享 K/V tile 复用是否能在真实序列长度分布下带来收益。当前 A800 独立 benchmark 已显示 tiled 版本快于 fallback，因此 `infer.py` 默认启用 tiled attention，并保留环境变量回退路径。

## 独立验证入口

attention kernel 可通过测试脚本单独构建和 benchmark：

```bash
python code/test_cuda_attention.py --bench-len 226 --iters 50
python code/test_cuda_attention.py --bench-len 401 --iters 50
```

端到端推理仍以 `infer.py` 为主入口，自定义算子通过环境变量控制开关。

默认启用 tiled attention；若云端端到端结果异常或需要对比旧实现，可回退到 one-query-per-block kernel：

```bash
USE_TILED_CUDA_ATTENTION=0 python infer.py
```
