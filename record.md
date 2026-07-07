# 推理优化记录

这个文件记录已经验证过的推理优化、默认开关、负优化结论和后续路线。`tips.md` 保持简版想法和待办；这里记录实验事实。

## 当前主路径

- 模型默认 `model.half()`，CUDA 主路径走 fp16。
- cached batch shards 直接加载，补齐 `attention_tile_starts` / `attention_tile_starts_mma` / `pred_positions`。
- CPU batch tensor 使用 pinned memory，并用 `non_blocking=True` 搬到 GPU。
- H2D prefetch 使用 copy stream 预取下一 batch。
- `RepEncoder` 走 CUDA fused 28-slot embedding bag，并用 CUDA LayerNorm(14336) 做 input_norm。
- Transformer attention 走 CUDA varlen causal attention，默认 MMA QK/PV 路径。
- Gate 推理走 CUDA `top2_softmax_8`。
- SMoE 走 CUDA routed sparse path，并融合最后的 residual add。
- 输出阶段走 CUDA final head + clamp + sigmoid + `pred_positions` gather。
- 除 attention/norm 仍保留调试开关外，主路径不再提供低性能降级路径；环境或 shape 不满足时直接 fail-fast。

## 已完成且保留

| 优化 | 主要文件 | 状态 | 结果 | 备注 |
| --- | --- | --- | --- | --- |
| fp16 推理 | `infer.py` | 固定主路径 | 正收益 | 权重和输入固定为 fp16。 |
| CUDA LayerNorm / AddLayerNorm | `CUDA/norm_kernels.cu`, `infer.py` | `USE_CUSTOM_CUDA_NORM=1` | 保留 | attention 后 residual add + LayerNorm 已融合。 |
| CUDA RepEncoder LayerNorm(14336) | `CUDA/norm_kernels.cu`, `infer.py` | `USE_CUSTOM_CUDA_REP_NORM=1` | 待测试 | 替换 `RepEncoder.input_norm` 的 PyTorch LayerNorm，可单独关闭对比。 |
| CUDA varlen attention | `CUDA/attention_kernels.cu`, `infer.py` | `USE_CUSTOM_CUDA_ATTENTION=1` | 保留 | 避免完整 `[S,S]` mask。 |
| CUDA fused embedding bag | `CUDA/embedding_bag_kernels.cu`, `infer.py` | 固定主路径 | `30s -> 24s` 左右 | 比 PyTorch `F.embedding_bag` 快；不满足 CUDA/fp16/shape 时直接报错。 |
| CUDA gate top2 softmax | `CUDA/softmax_topk_kernels.cu`, `infer.py` | 固定主路径 | `24.2s -> 23.1654s` | 避免完整 softmax/probs 后处理；不满足条件时直接报错。 |
| CUDA routed sparse SMoE | `CUDA/smoe_kernels.cu`, `infer.py` | 固定主路径 | 明显正收益 | 比 dense-all 路径更快，且正确率正常；不满足条件时直接报错。 |
| routed SMoE residual add fusion | `CUDA/smoe_kernels.cu`, `infer.py` | 固定主路径 | 小正收益 | 要求 `smoe_forward_with_residual` 存在，否则直接报错。 |
| CUDA final head + sigmoid + gather | `CUDA/output_kernels.cu`, `infer.py` | 固定主路径 | `23.16s -> 22.592s` | 只对 `pred_positions` 算 final head；不满足条件时直接报错。 |
| pinned CPU batch memory | `infer.py` | 固定主路径 | `22.5s -> 17.8243s` | 大收益，说明 H2D 是主要瓶颈之一；pin 失败时直接报错。 |
| H2D prefetch copy stream | `infer.py` | 固定主路径 | `17.8s -> 17.2s` | 小正收益，依赖 pinned batches。 |

## 已尝试但默认关闭或回退

| 优化 | 主要文件 | 结果 | 结论 |
| --- | --- | --- | --- |
| QKV fused HGEMM layout | `CUDA/gemm_kernels.cu`, `infer.py` | `22.592s -> 24.5132s` | 负优化，`infer.py` 接入已移除。 |
| Gate linear + top2 softmax 融合 | `CUDA/softmax_topk_kernels.cu` / `CUDA/gemm_kernels.cu` | 变慢 | 已回退，skinny GEMM 不能简单一行一个 CTA。 |
| embedding len 0/1/2 fast path | `CUDA/embedding_bag_kernels.cu` | `24s -> 25s` | 负优化，已回退。 |
| SMoE pooled-M scheduler | `CUDA/smoe_kernels.cu` | `43s -> 52s` | 负优化，已回退到旧 `(expert, m_block)` scheduler。 |
| CUDA dense-all SMoE 替代 PyTorch dense-all | `CUDA/dense_all_smoe_kernels.cu` | 不如 PyTorch dense-all | `infer.py` 接入已移除；主路径改用 routed sparse。 |

## 2026-07-06 W4A4 SMoE 实验

测试环境：aistudio，`/home/aistudio/infer.py`，清理并重编译 `routed_smoe_ext` 后测试。粗暴版 W4A4 使用 uniform scale；新增 `USE_SIMPLE_W4A4_FC1_SMOE` 后可让 fc1/fc2 都走 simple W4A4。

| 配置 | 命令关键环境变量 | AUC | PCOC | Latency | score_all | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| fc1+fc2 W4A4 默认 scale | `USE_SIMPLE_W4A4_SMOE=1 USE_SIMPLE_W4A4_FC1_SMOE=1 USE_W4A16_SMOE=0` | 0.722499 | 2.166075 | 21.4349s | 64.998525 | 编译和运行通过，但 PCOC 严重超上界。 |
| fc2-only W4A4 默认 scale | `USE_SIMPLE_W4A4_SMOE=1 USE_SIMPLE_W4A4_FC1_SMOE=0 USE_W4A16_SMOE=0` | 0.762088 | 1.195336 | 20.6437s | 65.183125 | AUC 较好，PCOC 略高于 1.15。 |
| fc1+fc2 W4A4 调 fc1 输出 | `SIMPLE_W4A4_FC1_OUTPUT_SCALE=1.6 USE_SIMPLE_W4A4_SMOE=1 USE_SIMPLE_W4A4_FC1_SMOE=1 USE_W4A16_SMOE=0` | 0.710048 | 1.034639 | 22.3796s | 70.422958 | PCOC 回到区间，但 AUC 损失明显。 |
| fc2-only W4A4 调 fc2 输出 | `SIMPLE_W4A4_FC2_OUTPUT_SCALE=1.1 USE_SIMPLE_W4A4_SMOE=1 USE_SIMPLE_W4A4_FC1_SMOE=0 USE_W4A16_SMOE=0` | 0.760598 | 1.063852 | 21.2365s | 74.739930 | 当前更优方向：轻微调 fc2 输出即可满足 PCOC，AUC 保留更多。 |
| fc2-only W4A4 + SMoE memory path 优化 | `SIMPLE_W4A4_FC2_OUTPUT_SCALE=1.1 CUDA_EXT_VERBOSE=1 USE_SIMPLE_W4A4_SMOE=1 USE_SIMPLE_W4A4_FC1_SMOE=0 USE_W4A16_SMOE=0` | 0.760598 | 1.063852 | 20.5225s | 74.906533 | fc1 epilogue 直接输出 fc2 activation pack，并把 route reduce 改成 vec8 `uint4` 访问；指标不变，latency 下降。 |
| fc2-only W4A4 + fc2 epilogue `uint4` half 写 | 同上 | 0.760598 | 1.063852 | 21.3183s | 74.720850 | 负优化，已回退；减少 store 指令没有抵消 shuffle/epilogue 重排开销。 |
| fc2-only W4A4 + epilogue `uint4` 写 + token-block reduce | 同上 | 0.760598 | 1.063852 | 20.8556s | 74.828813 | 负优化，已回退；token-block 少读 metadata，但 block 数增加，整体变慢。 |
| fc2-only W4A4 + LayerNorm512 vec8 `uint4` 访问 | `USE_LAYERNORM_512_VEC8=1`，其余同上 | 0.760742 | 1.063921 | 20.9470s | 74.819092 | 当前单次测试偏慢，默认关闭但保留开关；512 维 LayerNorm 改成 64 线程 vec8 后访存指令减少，但可能受 reduction/寄存器压力和测试波动影响。 |
| fc2-only W4A4 + final head scalar 复测 | 同 memory path 环境 | 0.760598 | 1.063852 | 20.8242s / 20.7211s | 74.836126 / 74.860201 | 作为 final head A/B baseline；同一环境仍有约 0.1s 波动。 |
| fc2-only W4A4 + final head half2 | `USE_OUTPUT_HEAD_HALF2=1`，其余同上 | 0.760598 | 1.063852 | 20.8796s / 21.7690s | 74.823216 / 74.615685 | 不赚，第二轮明显变慢；不建议默认打开。 |
| fc2-only W4A4 + final head `uint4` vec8 | `USE_OUTPUT_HEAD_UINT4=1`，其余同上 | 0.760598 | 1.063852 | 20.6817s / 20.1321s | 74.869381 / 74.997623 | 两轮均优于相邻 scalar/half2，当前更值得继续复测；默认仍关闭，避免被单次波动误导。 |
| fc2-only W4A4 + attention interleaved qkv + token-major out | `USE_INTERLEAVED_QKV_ATTENTION=1 USE_INTERLEAVED_QKV_ATTENTION_TOKEN_MAJOR_OUT=1`，其余同上 | 0.760598 | 1.063852 | 20.6521s | 74.876297 | 新 attention entry 直接读 `[1,S,H,192]` qkv，输出 `[1,S,H,64]`，省三份 q/k/v contiguous 和后续输出 permute copy；correctness smoke 与旧路径 max diff `0.0`。 |
| fc2-only W4A4 + attention interleaved qkv + head-major out | `USE_INTERLEAVED_QKV_ATTENTION=1 USE_INTERLEAVED_QKV_ATTENTION_TOKEN_MAJOR_OUT=0`，其余同上 | 0.760598 | 1.063852 | 21.5029s | 74.677762 | 单独只省 q/k/v contiguous 不稳定且这轮偏慢；如果继续用 interleaved attention，优先测 token-major 输出路径。 |
| fc2-only W4A4 + attention interleaved token-major + Rep LN14336 scalar A/B | `USE_REP_LAYERNORM_14336_VEC8=0 USE_INTERLEAVED_QKV_ATTENTION=1 USE_INTERLEAVED_QKV_ATTENTION_TOKEN_MAJOR_OUT=1`，其余同上 | 0.760598 | 1.063852 | 20.4435s / 24.2995s / 21.2484s | 74.924957 / 74.025225 / 74.737162 | 3 轮交错 A/B；第二轮 24.2995s 是明显慢尾。作为 vec8 默认开启前的相邻 baseline。 |
| fc2-only W4A4 + attention interleaved token-major + Rep LN14336 vec8 | `USE_REP_LAYERNORM_14336_VEC8=1 USE_INTERLEAVED_QKV_ATTENTION=1 USE_INTERLEAVED_QKV_ATTENTION_TOKEN_MAJOR_OUT=1`，其余同上 | 0.761060 | 1.063562 | 20.5977s / 20.2786s / 19.7122s | 74.929114 / 75.003554 / 75.135709 | 新增 14336 LayerNorm vec8/uint4 路径；custom scalar-vs-vec8 smoke max diff `4.77e-07`，对 PyTorch reference max diff `0.001953`。交错多轮稳定，没有 scalar 的慢尾，且 AUC/PCOC 稳定略优；这套已成为 `python infer.py` 默认配置。 |

补充观察：

- `SIMPLE_W4A4_FC1_OUTPUT_SCALE=0.5` 会把 PCOC 推到 `5.245232`；`2.0` 会压到 `0.688696`，说明 fc1+fc2 路径对该尺度很敏感，不宜只靠粗暴输出缩放作为最终方案。
- 当前 `python infer.py` 默认走 fc2-only W4A4：`SIMPLE_W4A4_FC2_OUTPUT_SCALE=1.1`、`USE_SIMPLE_W4A4_SMOE=1`、`USE_SIMPLE_W4A4_FC1_SMOE=0`、`USE_W4A16_SMOE=0`、`USE_INTERLEAVED_QKV_ATTENTION=1`、`USE_INTERLEAVED_QKV_ATTENTION_TOKEN_MAJOR_OUT=1`、`USE_REP_LAYERNORM_14336_VEC8=1`、`CUDA_EXT_VERBOSE=0`。
- fc1 介入后的主要问题是精度损失，而不是编译或性能；短期可优先沿当前默认的 fc2-only W4A4 继续做。
- 2026-07-06 的 memory path 优化已在 aistudio 编译通过：fc2-only 路径不再写 `h_route` 再读回 pack，而是 fp16 fc1 epilogue 直接写 int4 packed activation；route reduce 从标量 half 改成 vec8 `uint4` 读写。
- 后续不要优先重复尝试 simple W4A4 fc2 epilogue `uint4` half store 或 token-block route reduce；LayerNorm512 vec8 `uint4` 版本已保留为 `USE_LAYERNORM_512_VEC8=1`，可在多轮 A/B 时复测。
- final head dot 的 half2 路径当前不赚；`USE_OUTPUT_HEAD_UINT4=1` 两轮更快，但仍建议再做交错多轮 A/B 后再决定是否默认开启。
- attention interleaved qkv 新路径已在 aistudio 编译通过；旧 q/k/v、interleaved head-major、interleaved token-major 三者 correctness smoke 最大 diff 都是 `0.0`。单轮全量 A/B 里 token-major 输出更有希望，head-major 输出这轮为负。
- RepEncoder LayerNorm(14336) vec8/uint4 已默认开启；它只优化 LayerNorm 自身访存，不改变 embedding bag 或后续 `14336 -> 512` linear。若远端出现异常，可用 `USE_REP_LAYERNORM_14336_VEC8=0` 回退。embedding bag + LayerNorm 融合仍建议等更大的优化点做完后再评估。
- 后续如果继续做 fc1 W4A4，应尝试 per-layer/per-expert/per-channel scale 或校准激活 scale，而不是所有层共用一个 uniform scale。

## 待尝试

1. `sign id clamp` 前移到 `collate/cache` 阶段，避免 forward 中重复 clamp。
2. cached batches 合并或重排，减少 2039 次 Python loop 和小 batch 调度开销。
3. GPU batch object reuse，减少每轮 `.to()` 产生的小 tensor allocation。
4. 评估 `RepEncoder input_norm + linear` 融合是否值得继续做；普通大 GEMM 不应直接硬刚 cuBLAS。
5. routed SMoE route metadata 轻量化，但要谨慎，之前 scheduler 和 fast path 都出现过负优化。
6. W4A4 短期优先验证 fc2-only + `SIMPLE_W4A4_FC2_OUTPUT_SCALE=1.1` 的稳定性；fc1 W4A4 需要更细粒度 scale 后再继续。
7. final head `USE_OUTPUT_HEAD_UINT4=1` 可继续做交错多轮 A/B；若稳定收益，再考虑设为默认路径。
8. attention `USE_INTERLEAVED_QKV_ATTENTION=1` + token-major 输出建议做交错多轮 A/B；如果稳定快于默认路径，再考虑默认开启。
9. RepEncoder LayerNorm(14336) vec8 已默认开启；后续可继续评估 embedding bag + LayerNorm 融合实验开关。

## 常用对比命令

```bash
# 当前最佳默认路径：fc2-only W4A4 + interleaved attention + Rep LN14336 vec8
python infer.py
USE_CUSTOM_CUDA_ATTENTION=0 python infer.py
USE_CUSTOM_CUDA_NORM=0 python infer.py
USE_CUSTOM_CUDA_REP_NORM=0 python infer.py
CHECK_CUSTOM_CUDA_NORM=1 python infer.py
```

## 提交打包清单

当前最佳路径只需要 `submission_manifest.txt` 中列出的 CUDA 源文件。不要整包打入 `CUDA/` 目录，否则会包含已经验证为负优化或仅作规划的实验文件，例如：

- `CUDA/gemm_kernels.cu`
- `CUDA/dense_all_smoe_kernels.cu`
- `CUDA/residual_add.cu`
- `CUDA/weight_layout_kernels.cu`
- `CUDA/runtime_runner.cu`

重新编译某个 extension 时先清对应缓存，例如：

```bash
rm -rf .torch_extensions/routed_smoe_ext*
CUDA_EXT_VERBOSE=1 python infer.py
```

## 经验原则

- 普通大 GEMM 优先相信 cuBLAS；只有融合明显减少 layout/copy/launch 时才值得自写。
- 小 kernel 优化优先看 launch 数、H2D、Python loop 和 CPU/GPU 同步点。
- 每次只改一个变量做 A/B，对比全量 `AUC / PCOC / Latency / score_all`。
- 负优化也要记录，避免后续重复实现相同方向。
