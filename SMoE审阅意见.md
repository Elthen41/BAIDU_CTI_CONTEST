看了你的 `infer.py` 和 `smoe_kernels.cu` 后，我的判断更明确了：

**在 A800 + prefill + `E=8, top_k=2, D=512, dim_ff=1024` 这个设定下，你应该继续走“按 expert 整理后做 grouped GEMM”的路线。你的 CUDA 文件其实已经是这个方向，不应该改回纯 token-centric。** 你现在真正的问题不是路线选错，而是 **padding 粒度、route pack 开销、临时 workspace、以及默认 W4A4 实验路径** 这几个点。`infer.py` 里 TopKGate 固定为 `Linear(512->8) + top2_softmax_8`，SMoE 固定 `num_experts=8, k=2, D=512`，并且会把 expert 权重 stack 成 `[8,1024,512]` 和 `[8,512,1024]` 传给 CUDA extension。 

---

## 1. 你现在不是 token-centric，而是 expert-centric

`smoe_kernels.cu` 文件头已经写得很清楚：

```text
topk_idx/topk_score
  -> route count + padded expert offsets
  -> pack x[N,512] by expert into x_route[pool,512]
  -> grouped fc1 + bias + ReLU
  -> grouped fc2 + bias
  -> explicit top-2 reduce back to out[N,512]
```

也就是说，你当前实现是：

```text
topk
  -> count 每个 expert 有多少 route
  -> prefix/pad offsets
  -> 把 token hidden 按 expert pack 成 x_route
  -> grouped fc1
  -> grouped fc2
  -> top-2 weighted reduce + residual
```

这正是我之前建议的主路径。对于 prefill，这个方向是对的。

为什么？你的每个 route 要做：

```text
fc1: [1,512]  @ [512,1024]
fc2: [1,1024] @ [1024,512]
```

如果按 token-centric 做，就是大量小 GEMV。
但按 expert 分桶后，每个 expert 的平均 token 数是：

```text
avg_routes_per_expert = N * top_k / E = N * 2 / 8 = N / 4
```

只要一个 prefill batch 里 `N >= 512`，平均每个 expert 大约 128 个 route，正好匹配你当前 `kBlockM=128` 的 Tensor Core CTA tile。A800 是 Ampere 路线，你的 kernel 也在用 `cp.async`、`ldmatrix`、`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` 这类 SM80 风格写法；NVIDIA PTX 文档把 `cp.async`、`commit_group/wait_group` 作为异步 copy 指令列出，CUTLASS 的 SM80 MMA 实现也对应这个 `m16n8k16` FP16→FP32 accumulation 形态。([NVIDIA Docs][1])

---

## 2. 你现在最需要查的是 padding tax

你的 `smoe_route_prefix_kernel` 固定按 `kBlockM=128` padding：

```cpp
const int32_t padded = ((count + kBlockM - 1) / kBlockM) * kBlockM;
```

这意味着：

```text
actual_routes = N * 2
padded_routes = sum_e ceil(count[e] / 128) * 128
```

如果 batch 里 token 数不够大，padding 会非常痛。例如路由均匀时：

| N tokens | 平均每 expert routes | actual routes | padded routes, BLOCK_M=128 | padding 倍率 |
| -------: | ----------------: | ------------: | -------------------------: | ---------: |
|      128 |                32 |           256 |                       1024 |       4.0× |
|      256 |                64 |           512 |                       1024 |       2.0× |
|      512 |               128 |          1024 |                       1024 |       1.0× |
|     1024 |               256 |          2048 |                       2048 |       1.0× |

所以你的关键阈值非常简单：

```text
如果 N >= 512，当前 128 tile 大概率合理。
如果 N 在 128~512，当前实现可能被 padding 吃掉很多性能。
如果 N < 128，那 grouped GEMM 仍可能比 token-centric 好，但必须改小 M tile 或低延迟路径。
```

你说是 prefill，所以我猜大多数 batch 应该能到 `N >= 512`。但你的 DataLoader 是按 `batch_size=50` 个用户组 batch，不是固定 token 数；每个用户历史长度变化会导致 `N` 波动。`infer.py` 里确实是 `DataLoader(..., batch_size=50, ...)`，CUDA Graph 又按 token bucket 捕获。

建议你先加这个统计：

```python
@torch.no_grad()
def print_moe_route_stats(moe, x):
    # x: [1, S, 512] or [B, S, 512]
    topk_idx, topk_score, _ = moe.gate(x)
    flat = topk_idx.reshape(-1).to(torch.int64)
    counts = torch.bincount(flat, minlength=8).cpu()
    actual = int(flat.numel())

    pad128 = int(((counts + 127) // 128 * 128).sum())
    pad64 = int(((counts + 63) // 64 * 64).sum())
    pad32 = int(((counts + 31) // 32 * 32).sum())

    print("N tokens:", topk_idx.numel() // moe.k)
    print("counts:", counts.tolist())
    print("actual routes:", actual)
    print("pad128:", pad128, "ratio:", round(pad128 / max(actual, 1), 3))
    print("pad64 :", pad64,  "ratio:", round(pad64  / max(actual, 1), 3))
    print("pad32 :", pad32,  "ratio:", round(pad32  / max(actual, 1), 3))
```

我的经验判断：

```text
pad128 / actual <= 1.15  -> 当前 128 tile 可以继续
1.15 ~ 1.6              -> 值得做 M64
> 1.6                   -> M64/M32 或 no-pack 小 batch path 值得做
```

---

## 3. 你已经写了 M64 kernel，但它现在没有真正减少 padding

这是我看代码时发现的一个重要点。

你有：

```cpp
constexpr int kBlockM64 = 64;
smoe_grouped_linear_mma_tn_m64_kernel(...)
smoe_forward_m64(...)
smoe_forward_m64_with_residual(...)
```

但是 `build_route_metadata()` 仍然调用同一个 `smoe_route_prefix_kernel`，而这个 prefix kernel 固定用 `kBlockM=128` 做 padding。也就是说，`smoe_forward_m64` 目前只是 **用 64-row GEMM tile 去跑一个已经按 128 padding 的 route pool**，它没有把 `padded_routes` 从 128 对齐降到 64 对齐。

这会导致：

```text
你以为 M64 减少 padding；
实际上 offsets 还是 128-aligned；
M64 只改变 GEMM CTA 形状，没有改变 route pool 长度。
```

建议把 prefix kernel 改成可传 `pad_multiple`：

```cpp
__global__ void smoe_route_prefix_kernel(
    const int32_t* __restrict__ counts,
    int32_t* __restrict__ offsets,
    int32_t* __restrict__ cursors,
    int32_t pad_multiple
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    int32_t running = 0;
#pragma unroll
    for (int expert = 0; expert < kNumExperts; ++expert) {
        offsets[expert] = running;
        cursors[expert] = running;
        const int32_t count = counts[expert];
        const int32_t padded =
            ((count + pad_multiple - 1) / pad_multiple) * pad_multiple;
        running += padded;
    }
    offsets[kNumExperts] = running;
}
```

然后：

```cpp
build_route_metadata(x, topk_idx, topk_score, /*pad_multiple=*/128)
build_route_metadata(x, topk_idx, topk_score, /*pad_multiple=*/64)
```

这样 `smoe_forward_m64_with_residual` 才真正有意义。

---

## 4. 你当前 pack kernel 可以明显优化

现在 `smoe_route_pack_kernel` 是：

```cpp
grid = total_routes blocks
blockDim = 256
每个 block 处理 1 个 route
每个 route 拷贝 x[token, 512] 到 x_route[pos, 512]
```

但是每行 `512 half = 1024 bytes = 64 个 uint4`。你的代码里：

```cpp
constexpr int kVecsPerRow = kHiddenDim / 8;  // 64
for (int vec = threadIdx.x; vec < kVecsPerRow; vec += blockDim.x) {
    route_vec[vec] = x_vec[vec];
}
```

`blockDim=256` 时，只有 64 个线程真正拷贝，另外 192 个线程基本闲着。对于 prefill，route 数多时还可以忍；但如果 batch token 不大，CTA launch/scheduling 开销会比较明显。

更合理的是：

```text
一个 CTA 处理 4 个 route，每个 warp 处理 1 个 route；
每个 warp 32 lane，每 lane 拷贝 2 个 uint4；
blockDim = 128；
每个 CTA 完成 4 行 x_route copy。
```

伪代码方向：

```cpp
template <typename IndexT, typename ScoreT>
__global__ void smoe_route_pack_4routes_per_cta_kernel(...) {
    int warp = threadIdx.x >> 5;      // 0..3
    int lane = threadIdx.x & 31;
    int64_t route_id = int64_t(blockIdx.x) * 4 + warp;
    if (route_id >= n_tokens * kTopK) return;

    int token = route_id >> 1;
    int slot  = route_id & 1;
    int expert = int(topk_idx[route_id]);

    __shared__ int32_t pos_s[4];

    if (lane == 0) {
        int32_t pos = atomicAdd(cursors + expert, 1);
        pos_s[warp] = pos;
        route_pos[token * kTopK + slot] = pos;
    }
    int32_t pos = __shfl_sync(0xffffffff, pos_s[warp], 0);

    const uint4* x_vec = reinterpret_cast<const uint4*>(x + token * 512);
    uint4* r_vec = reinterpret_cast<uint4*>(x_route + int64_t(pos) * 512);

    // 64 uint4 per row, 32 lanes -> each lane copies 2
    r_vec[lane]      = x_vec[lane];
    r_vec[lane + 32] = x_vec[lane + 32];
}
```

这不会改变数学结果，但能减少 `route_pack` 的 CTA 数，并提高线程利用率。

---

## 5. `route_token / route_slot / route_score` 在 forward 主路径里是浪费的

`build_route_metadata()` 里分配并写了：

```cpp
route_token
route_slot
route_score
```

但你的 `smoe_forward()` / `smoe_forward_with_residual()` 主路径后面只真正需要：

```cpp
x_route
route_pos
counts
offsets
```

reduce 阶段用的是原始 `topk_score`，不是 `route_score`。所以 forward 主路径里 `route_token / route_slot / route_score` 可以删掉，或者拆成两个版本：

```cpp
build_route_metadata_forward_only()
build_route_metadata_debug_or_pack_api()
```

forward-only 版本只做：

```text
counts
offsets
cursors
route_pos
x_route
```

这会减少临时显存、少写一些 global memory，也能让 CUDA Graph 多 bucket 时少占显存。

---

## 6. 你默认开了 W4A4 fc2，这和“普通全连接 FFN”不是一回事

`infer.py` 里：

```python
USE_SIMPLE_W4A4_SMOE = os.environ.get("USE_SIMPLE_W4A4_SMOE", "1") != "0"
```

所以默认会走：

```python
smoe_forward_simple_w4a4_fc2_with_residual
```

这条路径是：

```text
fc1: FP16 grouped GEMM + ReLU + pack activation to int4
fc2: W4A4 experimental kernel
```

而不是普通 FP16 FC。`prepare_simple_w4a4_weights()` 只给 `fc2` 做了一个全局 uniform scale 的 int4 pack；这对 latency 可能有帮助，但很容易影响 AUC/PCOC，尤其是 CTR 模型这种对数值校准敏感的任务。相关路径在 `infer.py` 里默认启用，并在 load_model 时准备 W4A4 packed weights。

如果你现在是在验证 SMoE kernel 本身，我建议先跑：

```bash
USE_SIMPLE_W4A4_SMOE=0 python infer.py
```

先把 FP16 expert-centric 路径 benchmark 清楚。等 FP16 grouped GEMM 路径稳定后，再单独评估 W4A4 fc2 是否值得。

---

## 7. CUDA Graph 下还有一个隐藏浪费：SMoE 可能在算 padded token

你的 `CudaGraphBatchRunner` 会按 `token_cap = round_up(tokens, CUDA_GRAPH_TOKEN_BUCKET)` 捕获静态 batch。静态 batch 里有 `_graph_active_rows` 和 `_graph_n_rows`，embedding bag kernel 看起来已经支持 active rows；但是 SMoE 里是：

```python
x_flat = x.reshape(-1, D).contiguous()
n_tokens = x_flat.shape[0]
```

也就是说，SMoE 看到的是 `token_cap`，不是实际 `active_tokens`。CUDA Graph bucket 默认是 128，所以单 batch 最多浪费 127 个 token；如果你的 batch token 本身很大，这无所谓。如果很多 batch 只有几百 token，这个浪费会进一步放大 MoE padding tax。CUDA Graph 分桶逻辑和 static batch 逻辑在 `infer.py` 里是按 token bucket 构造的。

如果你要继续深挖，可以加一个 active-row 版本：

```cpp
smoe_forward_with_residual_active(
    x, residual, w1, b1, w2, b2,
    topk_idx, topk_score,
    active_n_tokens
)
```

内部只 route `active_n_tokens`，但输出仍然分配 `[token_cap, 512]`。这样 graph shape 不变，MoE 不算 padding rows。

---

## 8. 我建议的优化优先级

### P0：先确认 token 分布和 padding 倍率

先不要猜。对真实 prefill batch 统计：

```text
N tokens
counts[8]
actual_routes = N*2
padded_routes_128
padded_routes_64
padded_routes_32
```

如果大多数 batch：

```text
N >= 512
pad128 / actual <= 1.15
```

那你当前 `kBlockM=128` 是合理的，重点放在 pack/workspace。

如果大量 batch：

```text
N < 512
pad128 / actual > 1.3
```

优先做真正的 M64 metadata。

### P1：把 M64 route metadata 做真

现在 `smoe_forward_m64` 不会减少 padding，因为 prefix 还是 128。把 `pad_multiple` 参数打通后再比较：

```text
smoe_forward_with_residual, pad=128
smoe_forward_m64_with_residual, pad=64
```

我预计在 `N=128~512` 区间，M64 会更明显；在 `N>=1024` 区间，M128 可能仍然更快。

### P2：优化 route_pack

把 “1 CTA / route / 256 threads” 改成 “1 CTA / 4 routes / 128 threads” 或 “1 CTA / 8 routes / 256 threads”。

这对中小 prefill batch 和 CUDA Graph token padding 场景会更有帮助。

### P3：删 forward 不用的 metadata

主 forward 不要写：

```text
route_token
route_slot
route_score
```

保留 debug API 或 standalone `smoe_route_pack()` 时再写。

### P4：workspace 预分配

当前 `build_route_metadata()` 每次创建：

```text
counts
cursors
offsets
route_pos
route_token
route_slot
x_route
route_score
h_route
y_route
out
```

在 CUDA Graph capture 下，分配会被捕获进各 bucket 的 graph pool；bucket 多时显存会膨胀。建议 extension 接收预分配 workspace，或者在 Python module 里按最大 token_cap 给每层/全模型复用 buffer。

### P5：再考虑 CUTLASS/cuBLASLt grouped GEMM baseline

你手写的 SM80 MMA kernel方向是对的，但生产上最好拿 CUTLASS grouped GEMM 或 cuBLASLt grouped matmul 做 baseline。你代码里已经有 `BAIDU_CTI_ENABLE_CUTLASS_SMOE`，但目前只包了 fc2，而且 Python 编译 flag 没有默认打开。可以先做一个 “CUTLASS fc1+ReLU epilogue + CUTLASS fc2” 对照，不一定最后用，但能知道手写 kernel 离库实现差多少。

---

## 9. 最终建议

对你这个具体模型，我会这样定路线：

```text
主路径：
  expert-centric route count/prefix/pack + grouped GEMM

不要改成：
  token-centric 每 token 找 top2 expert 后直接算 GEMV

需要马上改：
  1. 统计真实 N 和 counts
  2. 修正 M64 让 prefix 真的按 64 pad
  3. 优化 route_pack 的 CTA 粒度
  4. 普通 FP16 FC 先关闭 USE_SIMPLE_W4A4_SMOE
  5. 删掉 forward 不用的 route_token/slot/score
```

一句话：**你的大方向是对的；A800 prefill 下继续 expert-centric grouped GEMM。当前最可能的性能坑是 `BLOCK_M=128` padding 和 pack kernel 粒度，而不是 MoE 应该 token-centric。**

[1]: https://docs.nvidia.com/cuda/parallel-thread-execution/contents.html "Contents — PTX ISA 9.3 documentation"
