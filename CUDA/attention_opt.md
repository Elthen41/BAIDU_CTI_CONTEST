# Attention 优化路线

本文档记录当前 `CUDA/attention_kernels.cu` 的后续优化方向。重点只参考：

- `LeetCUDA/kernels/flash-attn`：SM80 / fp16 / MMA 微内核实现。
- `FlashMLA`：scheduler 和 workload 分配思路。

暂不考虑 DeepGEMM。DeepGEMM 的 attention 相关实现主要偏 SM90/SM100、FP8/FP4、MQA/MLA，和当前 A800/SM80/fp16/MHA 场景不直接匹配。

## 目录

1. [当前基线](#当前基线)
2. [本项目 attention 的硬约束](#本项目-attention-的硬约束)
3. [LeetCUDA 可借鉴点](#leetcuda-可借鉴点)
4. [FlashMLA 可借鉴点](#flashmla-可借鉴点)
5. [不能直接照搬的原因](#不能直接照搬的原因)
6. [后续优化路线](#后续优化路线)
7. [实现建议](#实现建议)
8. [验证方式](#验证方式)
9. [风险和非目标](#风险和非目标)

## 当前基线

当前 attention 已经是自定义 CUDA 路径，不是官方 FlashAttention：

```text
q/k/v/out: [B=1, heads=8, S, head_dim=64]
dtype: fp16
attention: user_offsets 分段 causal attention
softmax state: fp32 m/l/O
```

主要 CUDA 入口在 `CUDA/attention_kernels.cu`：

```text
varlen_causal_attention
    one-query-per-block 稳定路径。

varlen_causal_attention_tiled
    CUDA-core tiled online-softmax 路径。

varlen_causal_attention_tiled_compact
    使用 tile_starts 的 compact tiled 路径。

varlen_causal_attention_mma_qk
    QK 使用 m16n8k16 MMA，P@V 仍走 CUDA core。

varlen_causal_attention_mma_qk_pv
    QK 和 P@V 都使用 m16n8k16 MMA。
```

`infer.py` 默认优先使用 `varlen_causal_attention_mma_qk_pv`，如果扩展里没有该入口再退到 `varlen_causal_attention_mma_qk`。调试开关仍保留：

```bash
USE_MMA_CUDA_ATTENTION=0 python infer.py
USE_CUSTOM_CUDA_ATTENTION=0 python infer.py
```

当前已有的关键技术：

```text
128-bit half8 load/store
cp.async K/V gmem -> smem
K/V 2-stage double buffering
ldmatrix
mma.sync.aligned.m16n8k16
fp32 online softmax
compact tile_starts
```

因此后续优化不应重新写一个普通 attention kernel，而应该在现有 `mma-qk-pv` 路径上继续吸收 LeetCUDA 和 FlashMLA 的局部设计。

## 本项目 Attention 的硬约束

本项目 attention 和标准 dense FlashAttention 不一样：

```text
1. B 固定为 1，heads 固定为 8，head_dim 固定为 64。
2. seq_len 是 batch 内所有用户拼接后的长度。
3. causal 边界在每个 user segment 内，跨用户不能 attend。
4. 每个 user_len 不同，需要处理 ragged tail。
5. q/k/v layout 来自 PyTorch linear 后的 [1, heads, S, 64]。
6. 端到端性能比单 kernel TFLOPS 更重要。
```

这意味着优化重点是：

```text
SM80 fp16 Tensor Core 利用率
K/V tile 复用
shared memory bank conflict
ragged tile 调度
减少空 CTA / 负载不均衡
保持 fp32 online softmax 的数值稳定
```

## LeetCUDA 可借鉴点

重点参考：

```text
LeetCUDA/kernels/flash-attn/README.md
LeetCUDA/kernels/flash-attn/mma/basic/flash_attn_mma_tiling_qk_F32F16F16F32.cu
LeetCUDA/kernels/flash-attn/mma/basic/flash_attn_mma_share_kv_F32F16F16F32.cu
LeetCUDA/kernels/flash-attn/mma/basic/flash_attn_mma_share_qkv_F32F16F16F32.cu
LeetCUDA/kernels/flash-attn/mma/swizzle/*
```

`README.md` 里明确列出的 SM80 FlashAttention-2 MMA 技术点包括：

```text
split-Q
shared KV / shared QKV smem
128-bit LD/ST
cp.async
SMEM padding / swizzle
m16n8k16
QK 和 P@V 都走 Tensor Core
Q smem -> register prefetch
K/V gmem -> smem prefetch
QK fine-grained tiling
```

最相关的实现是：

```text
LeetCUDA/kernels/flash-attn/mma/basic/flash_attn_mma_tiling_qk_F32F16F16F32.cu
```

它的关键形态：

```text
Q/K/V/O: [B, H, N, D]
D=64 时使用 Br=64, Bc=64 的 split-Q MMA 结构
QK: mma.sync.m16n8k16, fp16 x fp16 -> fp32
softmax: fp32 online softmax
P@V: P 转 half 后继续用 MMA
store: 128-bit vectorized store
```

对我们最有价值的不是完整搬运代码，而是这几类局部结构。

### 1. Split-Q warp layout

LeetCUDA 的 split-Q 思路是让不同 warp 负责不同 query rows，同时共享 K/V tile，从而减少 warp 间通信和 shared memory 往返。

当前本项目 `mma-qk-pv` 固定：

```text
Br=16
Bc=64
threads=128
```

后续可以尝试更接近 LeetCUDA 的布局：

```text
Br=32, Bc=64, threads=128/256
Br=64, Bc=64, threads=128/256
```

但不能直接把 LeetCUDA 的 `Br=64` 设为默认。原因是本项目 user_len 分布偏 ragged，短用户和 tail tile 比较多，过大的 Br 可能造成无效行和寄存器压力。

### 2. Shared KV / Shared QKV smem

LeetCUDA 里有两个重要优化：

```text
shared KV:
    K 和 V 分阶段复用同一块 smem，降低 smem footprint。

shared QKV:
    Q/K/V 进一步复用 smem，并把 Q 提前放到寄存器，提升 occupancy。
```

当前本项目为了实现简单和 pipeline，仍然有独立的：

```text
q_shared
k_shared[2 stages]
v_shared[2 stages]
p_shared
scores_shared
out_shared
```

后续可以做两个实验：

```text
实验 A: shared KV
    QK 阶段只加载 K。
    softmax 得到 P 后，再把同一块 smem 复用给 V。
    优点是降低 smem，提高 occupancy。
    缺点是 V load 不能完全和 QK 重叠，需要实测。

实验 B: shared QKV
    Q tile 先从 smem 预取到 register fragment。
    后续 Q smem 可以被 K/V/P 复用。
    优点是进一步降低 smem。
    风险是寄存器压力上升。
```

优先级：先做 shared KV，再考虑 shared QKV。

### 3. SMEM padding / swizzle

当前实现已经使用 `ldmatrix` 和 MMA，但 shared memory 还是比较直接的 row-major 数组。LeetCUDA swizzle 版本可以用来优化：

```text
Q/K/V smem bank conflict
P smem bank conflict
ldmatrix address pattern
```

第一步不建议直接引入完整 swizzle。可以先做 padding：

```text
Q stride: 64 -> 72
K stride: 64 -> 72
V stride: 64 -> 72
P stride: Bc -> Bc + 8
```

如果 padding 明确有收益，再考虑 LeetCUDA `mma/swizzle/*` 里的 swizzled layout。

### 4. 128-bit LD/ST 审计

当前已经有 128-bit half8 copy，但仍应检查：

```text
Q/K/V gmem -> smem 是否所有主路径都走 half8
out store 是否可以按 half8 写回
tail row 是否避免越界
alignment 是否始终满足 16 bytes
```

head_dim=64，每行 half 正好 128 bytes，非常适合以 8 half 为单位搬运。

### 5. cp.async pipeline 扩展

当前 K/V 已经有 2-stage `cp.async`。后续可评估：

```text
Q tile gmem -> smem 也改成 cp.async
K/V load 分得更细，让 QK/PV 与 copy overlap 更充分
不同 Br/Bc 下重新评估 stage 数
```

Q tile 每个 output tile 只加载一次，收益可能不如 K/V，但实现成本较低，可以作为小实验。

## FlashMLA 可借鉴点

FlashMLA 的 kernel 本体不适合直接移植：

```text
主要目标: SM90/SM100
核心机制: GMMA / TMA / CuTe / named barrier
数据形态: MLA/MQA、BF16/FP8、paged KV cache
```

但它的 scheduler 思想很有价值。

重点参考：

```text
FlashMLA/csrc/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu
FlashMLA/csrc/sm100/prefill/dense/kernel/fmha_tile_scheduler.hpp
```

### 1. 按实际 seqlen/block 分配 workload

FlashMLA 的 `get_decoding_sched_meta` 会按每个请求的 seqlen 计算 block 数，再把 block payload 分给不同 SM part。核心思想是：

```text
不要让 grid 只按 batch/request 均匀切。
要按真实 KV block 数或 estimated work 切。
长序列应该拿到更多调度份额。
```

本项目可以借鉴为：

```text
每个 attention tile 记录 estimated_work。
estimated_work 可以近似为该 q_tile 需要访问的 KV tile 数。
按 work 排序或按 work prefix 切分 tile。
```

当前 `tile_starts` 只有 q_tile_start，一个 tile 一个 CTA。它能减少空 CTA，但不能解决长用户后段 tile 更重的问题。

### 2. Persistent tile scheduler

FlashMLA/CUTLASS 的 persistent scheduler 用较小 grid 覆盖所有 work：

```text
grid = min(num_blocks, sm_count)
每个 CTA 循环处理多个 tile
block_idx += gridDim.x
```

本项目可以做一个 SM80 版本：

```text
grid.x = min(num_tiles * heads, sm_count * blocks_per_sm)
for linear_tile = blockIdx.x; linear_tile < total_tiles; linear_tile += gridDim.x:
    decode head 和 tile_id
    执行一个 varlen causal attention tile
```

优点：

```text
减少 launch 后大量 tiny CTA 的调度压力
长尾 tile 更容易均衡
便于按 work-sorted tile list 调度
```

风险：

```text
每个 CTA 循环多个 tile 会增加寄存器生命周期和分支复杂度
如果 tile 数本身足够多，收益可能有限
```

### 3. richer tile metadata

当前 Python 侧只构造：

```text
attention_tile_starts: int64[num_tiles]
attention_tile_starts_mma: int64[num_tiles]
```

后续可以扩展为：

```text
attention_tile_meta: int32[num_tiles, 4]
    user_start
    user_end
    q_tile_start
    q_tile_len
```

第一版 meta 只放 kernel 立刻需要的边界信息，用来减少 `find_user_for_token`。后续如果做 persistent scheduler，再扩展为 `[user_start, user_end, q_tile_start, q_tile_len, estimated_work]` 或额外 work array。

## 不能直接照搬的原因

### LeetCUDA 不能直接搬

LeetCUDA FlashAttention 默认：

```text
Q/K/V/O = [B,H,N,D]
所有 batch/head 使用相同 N
launcher 假设 QKV_seqlen 是 Br/Bc 的整倍数
没有 user_offsets
没有跨用户隔离
```

而本项目：

```text
S 是多个用户拼接后的总长度
每个用户单独 causal
每个 user_len 不同
每个 user tail 都要正确处理
```

所以 LeetCUDA 的 MMA fragment mapping、shared memory layout、pipeline 可以借，但入口、scheduler、mask/tail 逻辑必须保留本项目自己的 varlen 结构。

### FlashMLA kernel 本体不能直接搬

FlashMLA 主要面向：

```text
SM90 / SM100
GMMA / TMA
MLA / MQA
FP8 KV cache
```

A800 是 SM80，不支持这些 Hopper/Blackwell 专用路径。因此只借 scheduler，不借 GMMA/TMA kernel。

## 后续优化路线

建议按收益和风险分阶段推进。

### 阶段 1：LeetCUDA 风格 shared memory 改造

目标：在当前 `varlen_causal_attention_mma_qk_pv` 上降低 bank conflict 和 smem footprint。

优先顺序：

```text
1. Q/K/V/P smem padding（已完成，默认开启 padded MMA attention）
2. out store half8 vectorized audit
3. shared KV smem 实验（已完成，默认开启）
4. shared QKV smem 实验
```

注意：在当前 `Br=16, Bc=64, threads=128`、`Q/K/V/P + scores/out` 都使用静态 shared memory 的结构下，一次性把 Q/K/V/P stride 都从 64 pad 到 72 会超过默认 48KB 静态 shared memory 限制。因此 `varlen_causal_attention_mma_qk_pv_padded` 应把 Q/K/V/P half buffers 放到动态 shared memory，并在 launch 前通过 `cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize, ...)` opt-in 更大的 per-block shared memory。`scores_shared/out_shared/m/l` 等 fp32 状态仍可继续用静态 shared。

建议先新增编译期分支，而不是直接覆盖当前默认路径：

```text
varlen_causal_attention_mma_qk_pv_padded                    已完成，infer.py 默认启用
varlen_causal_attention_mma_qk_pv_padded_shared_kv          已完成，infer.py 默认启用
varlen_causal_attention_mma_qk_pv_padded_shared_kv_meta     已完成，infer.py 默认启用
```

独立 benchmark 确认后，再调整 `infer.py` 默认优先级。

`padded_shared_kv` 的实验策略：

```text
QK 前只把 K tile 放入 shared memory。
QK 完成后复用当前 K stage 重新加载当前 tile 的 V。
V load 尽量和 row max / softmax / P 写入重叠。
PV 前 wait 当前 V load 完成。
```

预期收益是减少动态 shared memory footprint，可能提高 occupancy。风险是 K/V 不再同时预取，会削弱原有 copy/compute overlap，所以必须用 correctness 和 latency 实测决定是否保留。

端到端 A/B 开关：

```bash
USE_SHARED_KV_MMA_ATTENTION=1 python infer.py
```

该开关默认开启；需要 A/B 回退时使用：

```bash
USE_SHARED_KV_MMA_ATTENTION=0 python infer.py
```

tile meta 入口用于减少 kernel 内二分查找用户边界，当前默认启用：

```bash
USE_META_MMA_ATTENTION=1 python infer.py
```

需要 A/B 回退时使用：

```bash
USE_META_MMA_ATTENTION=0 python infer.py
```

### 阶段 2：Br/Bc 与 split-Q 变体

当前默认 MMA 配置：

```text
Br=16
Bc=64
threads=128
```

建议测试：

```text
Br=16, Bc=64,  threads=128  当前默认
Br=32, Bc=64,  threads=128
Br=32, Bc=64,  threads=256
Br=64, Bc=64,  threads=128/256
Br=32, Bc=128, threads=256
```

判断原则：

```text
mean user_len 附近不能变慢太多
p90/p99 user_len 应该有明显收益
端到端 latency 优先于独立 kernel TFLOPS
```

如果 `Br=32/64` 独立 attention 更快但端到端无收益，说明 batch 内调度或 tail waste 抵消了收益，应先做 scheduler。

### 阶段 3：FlashMLA 风格 scheduler

目标：把当前 `tile_starts` 从“减少空 CTA”升级为“按 workload 分配”。

路线：

```text
1. 保留当前 attention_tile_starts_mma 作为 baseline。
2. 新增 attention_tile_meta_mma，包含 user_start/user_end/q_tile_start/q_tile_len。
3. 后续扩展 estimated_work，或另建 work array。
4. tile list 按 estimated_work 降序或反向 causal work 排序。
5. 新增 persistent kernel 入口，CTA 循环处理 tile。
6. 对比普通 compact grid 和 persistent grid。
```

estimated_work 可以先用简单模型：

```text
kv_tokens = q_tile_end - user_start + 1
kv_tiles = ceil(kv_tokens / Bc)
work = q_tile_len * kv_tiles
```

后续如果需要更精细，再用：

```text
work = q_tile_len * kv_tokens
```

### 阶段 4：LeetCUDA swizzle

只有在 padding/shared KV 明确有收益后，再考虑完整 swizzle。

原因：

```text
swizzle 会改 ldmatrix address mapping
容易引入 correctness bug
对 ragged tail 的 debug 成本较高
```

建议先只对 K/V/P 做局部 swizzle，不要一次性重写所有 shared memory layout。

## 实现建议

### 修改文件

主要修改：

```text
CUDA/attention_kernels.cu
infer.py
```

可选新增测试/脚本：

```text
code/test_cuda_attention.py
code/sweep_attention_params.py
```

如果要引入 richer metadata，还需要改：

```text
infer.py:
    make_attention_tile_starts
    ensure_attention_tile_starts
    collate/cache 补齐逻辑
```

### 建议的代码组织

`attention_kernels.cu` 里尽量保留当前稳定入口，把新实验做成模板参数或新入口：

```text
UseMmaQk
UseMmaPv
UsePaddedSmem
UseSharedKv
UsePersistentScheduler
```

不要一开始删除旧入口。attention 对端到端影响明显，而且之前 QKV fused linear 已经出现过负优化，A/B 开关要保留。

### 接入优先级

`infer.py` 的 `_get_attention_kernel()` 可以保持如下优先级：

```text
1. 新优化入口，例如 mma-qk-pv-padded/shared-kv/persistent
2. 当前 varlen_causal_attention_mma_qk_pv
3. varlen_causal_attention_mma_qk
4. compact tiled CUDA-core
5. one-query fallback
```

只有当独立 benchmark 和端到端都确认更快时，才把新入口放到最高优先级。

## 验证方式

### 正确性

每个新 kernel 先跑独立 attention correctness：

```bash
python code/test_cuda_attention.py --bench-len 226 --iters 50
python code/test_cuda_attention.py --bench-len 401 --iters 50
```

推荐补测长尾：

```bash
python code/test_cuda_attention.py --bench-len 857 --iters 30
python code/test_cuda_attention.py --bench-len 3350 --iters 10
```

正确性目标：

```text
相对 PyTorch fp16 reference:
max_abs_err <= 3e-3
mean_abs_err 不异常放大

端到端:
AUC/PCOC 不明显劣化
```

### 性能

独立 benchmark 至少覆盖：

```text
mean user_len: 226
p90 user_len: 401
p99 user_len: 857
max long-tail: 3350
```

端到端 benchmark：

```bash
rm -rf .torch_extensions/varlen_attention_ext*
CUDA_EXT_VERBOSE=1 python infer.py
```

A/B 对照：

```bash
USE_MMA_CUDA_ATTENTION=0 python infer.py
USE_CUSTOM_CUDA_ATTENTION=0 python infer.py
```

如果新增入口有独立开关，建议记录：

```text
kernel name
compile flags
attention independent latency
end-to-end latency
AUC
PCOC
```

### 判断标准

优先级从高到低：

```text
1. 端到端 latency 下降
2. AUC/PCOC 正常
3. 独立 attention latency 下降
4. kernel 实现可维护
```

如果独立 attention 更快但端到端变慢，默认视为负优化，不进入主路径。

## 风险和非目标

### 风险

```text
1. 更大的 Br 可能提升 Tensor Core 利用率，但增加 tail waste。
2. shared KV 可能降低 smem footprint，但破坏 K/V copy 与 compute overlap。
3. Q smem -> register prefetch 会增加寄存器压力。
4. swizzle 容易引入 ldmatrix 地址错误。
5. persistent scheduler 可能减少 CTA 调度开销，但增加 kernel 分支复杂度。
```

### 非目标

```text
1. 不移植 FlashMLA 的 GMMA/TMA kernel。
2. 不引入 DeepGEMM attention 路径。
3. 不把固定长度 LeetCUDA kernel 直接接到 infer.py。
4. 不为了单 kernel TFLOPS 牺牲端到端 latency。
5. 不删除当前稳定 attention fallback 和 A/B 开关。
```

## 后续 Checklist

```text
[ ] 对当前 mma-qk-pv 做 Nsight/profile，确认瓶颈是 smem、copy、softmax 还是 occupancy。
[x] 实现 Q/K/V/P 动态 shared padding 分支。
[x] benchmark padded 分支的 correctness 和 latency。
[x] 端到端 A/B padded 分支，并在 infer.py 默认启用。
[x] 实现 shared KV 分支。
[x] benchmark shared KV 并在 infer.py 默认启用。
[ ] 测试 Br=32/64 MMA 变体。
[x] 设计并实现 attention_tile_meta_mma，减少 find_user_for_token。
[x] benchmark attention_tile_meta_mma 并在 infer.py 默认启用。
[ ] 实现 FlashMLA 风格 persistent tile scheduler 实验入口。
[ ] 端到端 A/B 后再调整 infer.py 默认优先级。
```
