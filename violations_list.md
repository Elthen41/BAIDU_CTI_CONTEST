# Rules

## 一、违规清单

**规则 1 — 禁止在推理路径对输入采样 / 截断**

- 不得对 batch 做 sampling、subset、截断。
- 禁止 `BAIDU_*MAX_BATCHES`、`limit_batches`、跳过 batch。
- 禁止在 infer 中把 `max_ctx_len` 改小。

**规则 2 — 禁止结构性裁剪组网（命中任一即违规）**

- (a) Transformer 层数 < 8（如 ZeroLayer / <8-layer / 跳层不跑）。
- (b) 删除或跳过任一层的 Attention（`qkv_proj` / `out_proj` / `scaled_dot_product`）。
- (c) 删除或跳过任一层的 SMoE（expert / gate）。
- (d) MoE 有效 expert 数 < 8 或 Top-K 中 K < 2（合并/共享权重导致有效数下降视为违规；纯算子融合 / 分组 GEMM 保持数学等价的实现不违规）。
- (e) RepEncoder 缺失 `Embedding(5M, 512)` / `LayerNorm(14336)` / `Linear(14336, 512)` 任一层。
- (f) 跳过 / 常数化某层输出（如 `if layer_idx == 1: return x`）。

**规则 3 — 禁止破坏 forward 拓扑 / 输入完整性**

- 剪枝/压缩必须数学等价或近似等价（fp16/量化误差范围内），且不删除非线性已有的层或模块种类。
- SKIP_SLOTS / slot_mask / 把 28 个 slot 中某些 slot 池化结果置零 → 违规（改变 RepEncoder 输入语义）。
- Embedding 表某些 row 未被访问是合法的（数据驱动），但主动 mask/zero slot 不合法。

**规则 4 — 禁止 hack / 欺骗评测脚本（cheating）**

评测语义是“在计时区内真实执行完整 forward”。任何计时数字变小、但没真正跑等量计算的手段都算作弊。命中任一即违规：

- 量度计时：替换 / monkey-patch `time`、`perf_counter`、`torch.cuda.Event`、`cuda.synchronize`。
- 计算搬家：把真实 forward / prefill / H2D 拷贝从计时区搬到 `load_model` 或 import 阶段。（`load_model` 只允许：权重加载与融合、小规模 dummy 预热、`torch.compile`、cache 预分配；不允许跑全量真实 batch。）
- 结果预算 / 查表：提前跑数据集生成 `logit_cache` / `pred_cache` / `logid→prob` 字典，forward 内退化成查表 + sigmoid。
- 异步逃逸：计时区外提前发射 kernel / 用非默认 stream 且不 sync，让计时区只等空 sync。
- 结果伪造：输出常量、随机值、读磁盘上预算好的 `predict.txt`，或用小模型替换 forward 输出。

---

## 二、允许的合规优化（鼓励使用）

本次比赛的核心是“工程优化推理性能”——在不改变模型结构、不改变输入规模、不改变评测语义的前提下，通过工程手段（更快的 kernel、更好的内存布局、更高效的调度、更低的精度冗余等）把同一份 forward 跑得更快。

鼓励大家从系统 / 编译 / 硬件 视角切入，比拼的是把基线模型“原封不动地跑得更快”的能力，而不是“少跑一点”或“换个网络”。

- 精度/量化：fp16 / bf16 / int8 / fp8 / W8A8 / W4A16。
- 算子层：自定义 Triton/CUDA kernel、FlashAttention、PagedAttention、fused gate / MoE / MLP / LayerNorm。
- 推理图：`torch.compile`、CUDA Graph、多 stream 并行。
- 内存：KV cache 复用、预分配、pinned memory、零拷贝。
- 调度：batch grouping、padding 优化、pack / varlen attention、动态 batch。
- MoE：expert 分组 GEMM、topk 优化、expert 并行（保持有效 expert 数 = 8，K = 2）。
- I/O：异步加载、prefetch、predict batch 重排。
- 代数等价重写：A·B 提前 fuse 成单个 weight，等等。
