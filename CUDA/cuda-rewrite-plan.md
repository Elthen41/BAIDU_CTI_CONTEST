# CUDA 全量推理替换执行计划

目标：将当前 `infer.py` 中依赖 PyTorch 的推理路径全量替换为自定义 CUDA 实现，便于手动控制 block/thread、warp、shared memory、寄存器、访存布局、Tensor Core/MMA、kernel fusion 和调度顺序。

本文只描述执行步骤和需要实现的算子。具体 CUDA 代码按 `CUDA/*.cu` 文件中的注释逐步补齐。

## 0. 固定模型结构和张量形状

当前模型尺寸需要先按常量固化，后续 kernel 可以围绕这些固定尺寸做特化：

```text
vocab_size = 5000000
slot_num   = 28
emb_dim    = 512
d_model    = 512
n_heads    = 8
head_dim   = 64
num_layers = 8
dim_ff     = 1024
num_expert = 8
topk       = 2
```

动态维度：

```text
N = 当前 batch 合并后的 token/log 数
U = 当前 batch 的 user 数
slot_values[i].numel() 每个 slot 不固定
user_offsets 长度为 U + 1
```

需要优先统一权重布局：

```text
embedding.weight              [vocab_size, 512]
rep input layernorm           [28 * 512]
rep linear                    [28 * 512, 512]
transformer qkv linear        每层 [512, 1536]
transformer out linear        每层 [512, 512]
transformer norm1/norm2       每层 [512]
gate linear                   每层 [512, 8]
expert fc1                    每层每 expert [512, 1024]
expert fc2                    每层每 expert [1024, 512]
head linear                   [512, 1]
```

## 1. CUDA 文件划分

建议先按功能写以下文件：

```text
CUDA/
  cuda-rewrite-plan.md
  weight_layout_kernels.cu
  embedding_bag_kernels.cu
  norm_kernels.cu
  gemm_kernels.cu
  attention_kernels.cu
  softmax_topk_kernels.cu
  smoe_kernels.cu
  output_kernels.cu
  runtime_runner.cu
```

这些文件的职责是：

| 文件 | 需要实现的内容 |
| --- | --- |
| `weight_layout_kernels.cu` | 权重转置、打包、对齐、fp32/fp16 转换、一次性预处理 |
| `embedding_bag_kernels.cu` | 28 个 slot 的 embedding lookup、segment sum、拼接成 `[N, 14336]` |
| `norm_kernels.cu` | LayerNorm、Residual Add、Add + Norm 融合 |
| `gemm_kernels.cu` | fp16 GEMM/GEMV，覆盖 rep linear、qkv、out、gate、expert fc、head |
| `attention_kernels.cu` | 基于 `user_offsets` 的 varlen causal attention |
| `softmax_topk_kernels.cu` | gate logits 的 softmax/logsumexp/top2 |
| `smoe_kernels.cu` | route count/pack、expert fc1+relu、expert fc2、route reduce |
| `output_kernels.cu` | logits clamp、sigmoid、pred_mask gather、输出 buffer 写入 |
| `runtime_runner.cu` | 全推理调度、workspace 管理、kernel launch 顺序、CUDA Graph |

## 2. 执行步骤

### Step 1：权重预处理

先实现 `weight_layout_kernels.cu`。

需要做：

1. 将 PyTorch checkpoint 中的权重整理成 CUDA 友好的 contiguous layout。
2. 对 Linear 权重统一成 kernel 需要的布局，例如 row-major 或 Tensor Core tile-friendly layout。
3. 对 expert 权重按 `[layer, expert, ...]` 连续存储，避免推理时分散访问。
4. 对需要 vectorized load 的权重做 16B 对齐。

验收：

```text
所有权重指针、shape、stride 固定；
后续 kernel 不再依赖 PyTorch module 结构。
```

### Step 2：替换 RepEncoder

实现 `embedding_bag_kernels.cu`、`norm_kernels.cu`、`gemm_kernels.cu` 中 RepEncoder 相关算子。

当前 PyTorch 路径：

```text
for slot in 1..28:
  values, offsets = batch[slot]
  slot_emb = embedding_bag(values, embedding.weight, offsets, mode=sum)

fused = concat(slot_emb[1..28])       # [N, 14336]
norm  = LayerNorm(fused)              # [N, 14336]
rep   = Linear(norm, 14336 -> 512)    # [N, 512]
```

需要写的算子：

```text
embedding_bag_slot_sum_kernel
embedding_bag_28slot_fused_kernel
rep_input_layernorm_kernel
rep_linear_kernel
```

实现顺序：

1. 先每个 slot 一个 kernel，保证正确。
2. 再写 28 slot fused kernel，减少 launch。
3. 再考虑将 embedding sum 的输出直接写入 rep linear 需要的布局。

参考：

```text
LeetCUDA/kernels/embedding
LeetCUDA/kernels/reduce
LeetCUDA/kernels/layer-norm
LeetCUDA/kernels/hgemm
```

### Step 3：替换 Transformer 基础算子

实现 `norm_kernels.cu` 和 `gemm_kernels.cu` 中 Transformer 相关算子。

每层结构：

```text
residual = x
x        = norm1(x)
qkv      = linear(x, 512 -> 1536)
q,k,v    = split/reshape(qkv) to [8, N, 64]
attn     = varlen causal attention(q,k,v,user_offsets)
x        = residual + out_linear(attn)
residual = x
x        = norm2(x)
moe_out  = SMoE(x)
x        = residual + moe_out
```

需要写的基础算子：

```text
layernorm_512_kernel
qkv_linear_kernel
qkv_split_transpose_kernel
out_linear_kernel
residual_add_kernel
add_layernorm_512_kernel
```

优先融合：

```text
residual_add + layernorm
qkv linear + qkv layout transform
attention output layout transform + out linear
```

参考：

```text
LeetCUDA/kernels/layer-norm
LeetCUDA/kernels/hgemm
LeetCUDA/kernels/hgemv
LeetCUDA/kernels/elementwise
```

### Step 4：替换 Attention

实现 `attention_kernels.cu`。

当前注意力逻辑是按用户序列切块的 causal attention：

```text
for each user range [start, end):
  q_i, k_i, v_i = q[:, :, start:end, :]
  out_i = causal_attention(q_i, k_i, v_i)
```

需要写的算子：

```text
varlen_causal_attention_forward_kernel
attention_softmax_kernel
attention_score_value_kernel
```

建议实现顺序：

1. 先写朴素版本：每个 user/head/block 处理一个局部 causal attention。
2. 再写 tiled 版本：使用 shared memory 缓存 K/V tile。
3. 最后参考 FlashAttention 思路，做 online softmax，减少 score matrix 写读。

关键约束：

```text
head_dim = 64
n_heads = 8
user_offsets 决定每段序列长度
attention 必须严格 causal，不能跨 user 看历史
```

参考：

```text
LeetCUDA/kernels/flash-attn
LeetCUDA/kernels/softmax
LeetCUDA/kernels/reduce
```

### Step 5：替换 Gate 和 TopK

实现 `softmax_topk_kernels.cu`。

当前 gate：

```text
logits = linear(x, 512 -> 8)
topk_logits, topk_idx = topk(logits, k=2)
topk_score = exp(topk_logits - logsumexp(logits))
```

需要写的算子：

```text
gate_linear_kernel
top2_softmax_8_kernel
```

因为 expert 数固定为 8，可以每个 token 使用一个 warp 或一个小 block：

```text
读取 8 个 logits
求 max
求 exp sum
找 top1/top2
输出 topk_idx[2], topk_score[2]
```

参考：

```text
LeetCUDA/kernels/softmax
LeetCUDA/kernels/reduce
```

### Step 6：替换 SMoE

实现 `smoe_kernels.cu` 和 `gemm_kernels.cu` 中 expert GEMM。

推荐主线是 routed sparse，不再计算全部 8 个 expert：

```text
route_count(topk_idx)                       -> counts[8]
route_pack(x, topk_idx, topk_score)         -> x_route, route_score, route_token
grouped_fc1_relu(x_route, expert_w1/b1)     -> h_route
grouped_fc2(h_route, expert_w2/b2)          -> y_route
route_reduce(y_route, route_score, token)   -> moe_out[N, 512]
```

需要写的算子：

```text
smoe_route_count_kernel
smoe_route_prefix_kernel
smoe_route_pack_kernel
smoe_grouped_fc1_relu_kernel
smoe_grouped_fc2_kernel
smoe_route_reduce_kernel
```

优化顺序：

1. 先用普通 tiled fp16 matmul 实现 expert fc1/fc2。
2. 再切换 Tensor Core/MMA。
3. 再尝试 fc2 + route_reduce 融合。
4. 再尝试 route_pack + fc1 局部融合。

注意：

```text
atomic 只建议用于 route position 分配；
不要把 fp16 输出累加 atomic 作为主路径；
PCOC 异常时优先检查 topk_score、route_token、route_reduce 顺序。
```

参考：

```text
LeetCUDA/kernels/hgemm
LeetCUDA/kernels/swizzle
LeetCUDA/kernels/relu
LeetCUDA/kernels/reduce
```

### Step 7：替换 Head 和输出

实现 `output_kernels.cu`。

当前 PyTorch 路径：

```text
pred = linear(encoder_output, 512 -> 1)
logits = clamp(pred, -15, 15)
probs = sigmoid(logits)
masked_probs = probs[pred_mask]
masked_logids = logid[pred_mask]
```

需要写的算子：

```text
head_linear_kernel
clamp_sigmoid_kernel
pred_mask_gather_kernel
```

目标：

```text
避免每个 batch 频繁 cpu().tolist()
将待写 predict.txt 的概率先放入连续 host/device buffer
最后统一按 logid 顺序写出
```

参考：

```text
LeetCUDA/kernels/sigmoid
LeetCUDA/kernels/elementwise
LeetCUDA/kernels/hgemv
```

### Step 8：实现全量 CUDA Runner

实现 `runtime_runner.cu`。

Runner 负责：

```text
加载/接收 batch tensor 指针
管理所有中间 workspace
按 8 层 Transformer 顺序 launch kernel
记录 CUDA event latency
可选启用 CUDA Graph
统一输出 prediction buffer
```

最小 launch 顺序：

```text
embedding bag -> rep norm -> rep linear
for layer in 0..7:
  norm1 -> qkv linear -> attention -> out linear + residual
  norm2 -> gate top2 -> smoe -> residual
head linear -> clamp sigmoid -> mask gather
```

后续融合目标：

```text
embedding bag + concat
residual add + layernorm
qkv linear + layout transform
attention output transform + out linear
fc2 + route_reduce
head linear + clamp + sigmoid
```

## 3. 正确性检查顺序

每写完一类算子，就和当前 `infer.py` 的 PyTorch 输出对齐：

```text
1. 单 kernel 小 shape 对齐
2. RepEncoder 输出对齐
3. 单层 Transformer 输出对齐
4. 单层 SMoE 输出对齐
5. 8 层 seq_encoder 输出对齐
6. logits 对齐
7. sigmoid 概率对齐
8. predict.txt 数量、顺序、AUC、PCOC 对齐
```

数值容忍建议：

```text
fp16 局部 kernel: max_abs_diff <= 1e-2 先接受
最终 PCOC:       0.85 <= PCOC <= 1.15
AUC:             不低于当前 PyTorch baseline 明显范围
```

## 4. Profiling 顺序

先只记录每个大模块耗时：

```text
RepEncoder
Transformer attention
Transformer linear/norm
SMoE gate
SMoE expert
Head/output
```

再用 Nsight Compute 看具体 kernel：

```text
global load/store throughput
shared memory bank conflict
register count
achieved occupancy
warp stall reason
Tensor Core utilization
kernel launch count
```

参考：

```text
LeetCUDA/kernels/nvidia-nsight
LeetCUDA/kernels/hgemm
LeetCUDA/kernels/flash-attn
```

## 5. 推荐实际编写顺序

```text
1. weight_layout_kernels.cu
2. embedding_bag_kernels.cu
3. norm_kernels.cu
4. gemm_kernels.cu
5. softmax_topk_kernels.cu
6. smoe_kernels.cu
7. attention_kernels.cu
8. output_kernels.cu
9. runtime_runner.cu
```

如果以比赛分数为导向，可以优先写：

```text
embedding_bag_kernels.cu
gemm_kernels.cu
smoe_kernels.cu
output_kernels.cu
```

如果以全量替换完整性为导向，则按上面的 1 到 9 顺序推进。
