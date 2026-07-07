# SMoE W4A16 Quantization Plan

## Direction

The quantization path is now centered on the BitDecoding-style A100 implementation:

```text
4-bit packed weights in memory
-> tile-local unpack/dequant to fp16
-> fp16 Tensor Core MMA on SM80
```

For SMoE this maps to W4A16:

```text
activation: fp16
weight:     4-bit packed
compute:    fp16 Tensor Core MMA
output:     fp16, then existing bias/ReLU/top-2 reduce
```

The existing fp16 routed SMoE path remains the baseline and the implementation scaffold.

## BitDecoding Notes

Relevant implementation ideas from `BitDecoding`:

- A100 path is compiled for `sm_80`.
- Packed low-bit data is stored compactly in memory.
- 4-bit values are unpacked and dequantized inside the tile path.
- Dequantized fragments are fp16/bf16.
- MMA uses `SM80_16x8x16_F32F16F16F32_TN`, not a separate low-bit GEMM atom.
- Active low-bit paths are specialized, mainly head-dim 128 with group sizes 32 or 128.

The most important idea to port is not the full attention kernel, but this local sequence:

```text
packed 4-bit tile
-> load scale/zero params
-> dequantize to fp16 tile/fragment
-> call fp16 MMA
```

## SMoE Mapping

Current fp16 SMoE flow:

```text
route count
route pack x -> x_route fp16
grouped fc1 fp16 MMA + bias + ReLU
grouped fc2 fp16 MMA + bias
top-2 reduce / residual reduce
```

Target W4A16 flow:

```text
route count
route pack x -> x_route fp16
grouped fc1 W4A16 MMA + bias + ReLU
grouped fc2 W4A16 MMA + bias
top-2 reduce / residual reduce
```

Only expert weights change format. Routing, packed activations, output buffers, and reduce stay fp16.

## Weight Format

First target format:

```text
w1_pack   int16 bits [8, 1024, 512 / 4]
w1_scale  fp16/fp32  [8, 1024, 512 / group_size]
w1_zero   fp16/fp32  [8, 1024, 512 / group_size]
w2_pack   int16 bits [8, 512, 1024 / 4]
w2_scale  fp16/fp32  [8, 512, 1024 / group_size]
w2_zero   fp16/fp32  [8, 512, 1024 / group_size]
```

Use `group_size=128` first, because it matches an active BitDecoding path and keeps scale overhead low.
Try `group_size=32` if model quality drops.

Packing rule:

```text
four 4-bit values -> one 16-bit storage element
```

The stored tensor can use PyTorch `int16`; CUDA treats the bits as unsigned 16-bit payload.

## Implementation Phases

### Phase 1: W4A16 Reference

Add a Python-only W4A16 reference for SMoE expert weights.

Purpose:

- Verify AUC/PCOC before CUDA work.
- Compare W4A16 with current fp16 custom CUDA baseline.
- Decide between group size 128 and 32.

Reference semantics:

```text
weight -> group-wise 4-bit quant -> dequant weight
activation remains fp16/fp32 matmul input
F.linear(x, weight_dequant, bias)
```

Implemented switch:

```text
USE_W4A16_SMOE=1
CHECK_W4A16_SMOE=1
W4A16_GROUP_SIZE=128
```

Phase 1 keeps dequantized reference weights as non-persistent buffers:

```text
_w4_w1_qdq
_w4_w1_scale
_w4_w1_zero
_w4_b1
_w4_w2_qdq
_w4_w2_scale
_w4_w2_zero
_w4_b2
```

Run first with `W4A16_GROUP_SIZE=128`. If AUC/PCOC drops too much, rerun with `W4A16_GROUP_SIZE=32`.

Cloud result with `W4A16_GROUP_SIZE=128`:

```text
weight check:
  fc1_max_abs_err  = 5.493164e-03
  fc1_mean_abs_err = 1.722571e-03
  fc2_max_abs_err  = 3.997803e-03
  fc2_mean_abs_err = 1.249923e-03

quality:
  AUC     = 0.758982
  PCOC    = 1.102514
  latency = 50.0753s
  score   = 67.661432
```

Compared with the fp16 custom CUDA baseline from the same cloud setup:

```text
fp16 baseline:
  AUC     = 0.759609
  PCOC    = 1.110132
  latency = 22.5246s
  score   = 74.099831

delta:
  AUC     = -0.000627
  PCOC    = -0.007618
  latency = +27.5507s
```

Interpretation: `group_size=128` is a quality pass. The latency regression is expected for the Python reference path and should not block CUDA implementation.

### Phase 2: Weight Packing Buffers

Add model-side preparation:

```text
prepare_w4a16_weights()
```

Register non-persistent buffers for packed weights and params.

Expected buffers:

```text
_w4_w1_pack
_w4_w1_scale
_w4_w1_zero
_w4_b1
_w4_w2_pack
_w4_w2_scale
_w4_w2_zero
_w4_b2
```

Implemented packing rule:

```text
packed_int16 = q0 | (q1 << 4) | (q2 << 8) | (q3 << 12)
```

The stored PyTorch tensor dtype is `int16`; values whose unsigned payload is above `0x7fff` are stored with the same 16-bit bit pattern as signed `int16`. CUDA should reinterpret the data as an unsigned 16-bit payload before unpacking.

Phase 2 keeps the Python reference path on `_w4_w*_qdq`, while also preparing `_w4_w*_pack` for CUDA. This keeps Phase 1 quality behavior unchanged and isolates the data-layout change.

`CHECK_W4A16_SMOE=1` now verifies:

```text
original weight -> qdq error
packed weight -> unpack -> dequant -> qdq reconstruction error
```

Expected reconstruction error from packed weight to qdq should be exactly zero after dtype conversion. Any nonzero value means bit order, signed storage handling, or group scale/zero indexing is wrong.

Cloud pack-check result with `W4A16_GROUP_SIZE=128`:

```text
pack shape:
  _w4_w1_pack = [8, 1024, 128]
  _w4_w2_pack = [8, 512, 256]

weight qdq error:
  fc1_max_abs_err  = 5.493164e-03
  fc1_mean_abs_err = 1.722571e-03
  fc2_max_abs_err  = 3.997803e-03
  fc2_mean_abs_err = 1.249923e-03

pack reconstruction:
  fc1_pack_recon_max_abs_err = 0.000000e+00
  fc2_pack_recon_max_abs_err = 0.000000e+00

quality:
  AUC     = 0.758982
  PCOC    = 1.102514
  latency = 49.1511s
  score   = 67.877069
```

Interpretation: Phase 2 is a pass. Packed buffers reconstruct the qdq reference exactly, and full-model quality matches the Phase 1 reference result.

### Phase 3: CUDA Interface Skeleton

Add extension entries:

```text
smoe_forward_w4a16(...)
smoe_forward_w4a16_with_residual(...)
```

Python switch:

```text
USE_W4A16_SMOE=1
USE_CUDA_W4A16_SMOE=1
REQUIRE_CUDA_W4A16_SMOE=0/1
CHECK_W4A16_SMOE=1
```

Default remains fp16 routed SMoE.

Interface parameter order:

```text
smoe_forward_w4a16(
  x,
  w1_pack, w1_scale, w1_zero, b1,
  w2_pack, w2_scale, w2_zero, b2,
  topk_idx, topk_score,
  group_size
)

smoe_forward_w4a16_with_residual(
  x, residual,
  w1_pack, w1_scale, w1_zero, b1,
  w2_pack, w2_scale, w2_zero, b2,
  topk_idx, topk_score,
  group_size
)
```

Phase 3 skeleton behavior:

- CUDA extension exports both W4A16 symbols.
- CUDA side validates dtype, device, contiguity, shape, and `group_size`.
- CUDA side then raises a fixed not-implemented error.
- Python side attempts the CUDA path only when `USE_CUDA_W4A16_SMOE=1`.
- With `REQUIRE_CUDA_W4A16_SMOE=0`, that fixed not-implemented error falls back to the Python W4A16 reference.
- With `REQUIRE_CUDA_W4A16_SMOE=1`, the error is raised, which is useful for checking the extension symbol and argument path.
- Extension loading is warmed in `load_model()` so compilation is not pushed into the timed inference loop.

Cloud interface-check result:

```text
fallback run:
  command = USE_W4A16_SMOE=1 USE_CUDA_W4A16_SMOE=1 CHECK_W4A16_SMOE=1 W4A16_GROUP_SIZE=128 python infer.py
  extension = Loaded SMoE CUDA W4A16 extension skeleton
  fallback  = CUDA W4A16 residual path unavailable -> Python W4A16 reference
  AUC       = 0.758982
  PCOC      = 1.102514
  latency   = 49.2136s
  score     = 67.862489

strict run:
  command = USE_W4A16_SMOE=1 USE_CUDA_W4A16_SMOE=1 REQUIRE_CUDA_W4A16_SMOE=1 W4A16_GROUP_SIZE=128 python infer.py
  result  = expected RuntimeError at smoe_forward_w4a16_with_residual
  message = CUDA kernel is not implemented yet; Phase 3 only exports the interface skeleton
```

Interpretation: Phase 3 is a pass. The extension compiles, pybind symbols are visible, Python argument wiring reaches CUDA, fallback preserves Phase 2 quality, and strict mode confirms the skeleton boundary.

### Phase 4: First CUDA Kernel

Modify the current grouped fp16 MMA path minimally:

```text
load activation tile as fp16
load packed 4-bit weight tile
dequantize weight tile into fp16 shared-memory tile
reuse existing compute_grouped_mma_stage()
store fp16 output with existing bias/ReLU handling
```

This is the fastest path to correctness because the current fp16 MMA kernel already handles:

- expert routing offsets
- pooled M scheduling
- fc1/fc2 shapes
- bias/ReLU
- fp16 output layout

First implementation status:

- Added `smoe_grouped_linear_w4a16_mma_tn_kernel`.
- Reuses existing route metadata, route packing, reduce, and residual reduce.
- Loads activation tiles as fp16.
- Loads packed weight tiles by scalar int16 reads.
- Unpacks q4 values with simple shift/mask.
- Dequantizes each weight value with `q * scale + zero`.
- Writes dequantized fp16 weight tile into shared memory.
- Reuses existing `compute_grouped_mma_stage()`.
- Supports both fc1 `[512 -> 1024]` with ReLU and fc2 `[1024 -> 512]`.

This is intentionally not optimized. It should be used to check correctness against the Python W4A16 reference before adding BitDecoding-style dequant optimizations.

Cloud result with `USE_CUDA_W4A16_SMOE=1`, `REQUIRE_CUDA_W4A16_SMOE=1`, `W4A16_GROUP_SIZE=128`:

```text
weight check:
  w1_pack_shape = [8, 1024, 128]
  w2_pack_shape = [8, 512, 256]
  fc1_pack_recon_max_abs_err = 0.000000e+00
  fc2_pack_recon_max_abs_err = 0.000000e+00

quality:
  AUC     = 0.758269
  PCOC    = 1.102424
  latency = 31.5436s
  score   = 71.926523
```

Compared with the Python W4A16 reference:

```text
Python W4A16:
  AUC     = 0.758982
  PCOC    = 1.102514
  latency = 49.1511s
  score   = 67.877069

CUDA W4A16 delta:
  AUC     = -0.000713
  PCOC    = -0.000090
  latency = -17.6075s
  score   = +4.049454
```

Compared with the fp16 custom CUDA baseline:

```text
fp16 baseline:
  AUC     = 0.759609
  PCOC    = 1.110132
  latency = 22.5246s
  score   = 74.099831

CUDA W4A16 delta:
  AUC     = -0.001340
  PCOC    = -0.007708
  latency = +9.0190s
  score   = -2.173308
```

Interpretation: Phase 4 first kernel is a quality pass and a functional CUDA pass. It is faster than the Python W4A16 reference, but still slower than the fp16 routed SMoE baseline because the current W4A16 path uses scalar unpack/dequant into shared memory.

### Phase 5: BitDecoding-Style Optimization

After correctness:

- Replace simple unpack/dequant with `lop3`/half2-style dequant.
- Consider dequantizing directly into register fragments instead of shared-memory tiles.
- Tune packed weight layout for coalesced loads.
- Compare `group_size=128` and `group_size=32`.
- Profile with Nsight Compute on A100.

Phase 5.1 first low-risk optimization:

```text
Before:
  for each q4 element:
    load packed int16
    load scale
    load zero
    extract one q4
    dequant one fp16

After:
  for each packed int16:
    load packed int16 once
    load scale once
    load zero once
    extract four q4 values
    dequant/write four fp16 values into b_shared
```

This keeps the shared-memory tile and MMA path unchanged. It should reduce redundant packed-weight and scale/zero loads by roughly 4x inside the weight tile load path, while preserving exactly the same `q * scale + zero` semantics.

Cloud result after Phase 5.1:

```text
quality:
  AUC     = 0.758269
  PCOC    = 1.102424
  latency = 25.1283s
  score   = 73.423422
```

Compared with Phase 4 first CUDA kernel:

```text
Phase 4:
  AUC     = 0.758269
  PCOC    = 1.102424
  latency = 31.5436s
  score   = 71.926523

Phase 5.1 delta:
  AUC     = 0.000000
  PCOC    = 0.000000
  latency = -6.4153s
  score   = +1.496899
```

Compared with the fp16 custom CUDA baseline:

```text
fp16 baseline:
  AUC     = 0.759609
  PCOC    = 1.110132
  latency = 22.5246s
  score   = 74.099831

Phase 5.1 CUDA W4A16 delta:
  AUC     = -0.001340
  PCOC    = -0.007708
  latency = +2.6037s
  score   = -0.676409
```

Interpretation: Phase 5.1 is a strong pass. The packed-load optimization preserved quality exactly and recovered most of the latency gap to the fp16 baseline. Remaining optimization should focus on shared-memory dequant overhead, reducing per-tile scalar conversion cost, and/or using a faster W4A16 weight tile load path.

Phase 5.2 second low-risk optimization:

```text
Before:
  activation tile load:
    2048 scalar half loads/stores per CTA per K tile

After:
  activation tile load:
    256 uint4 vector loads/stores per CTA per K tile
```

This only changes how the fp16 activation tile is copied into shared memory. It does not change routing, packed weight unpack/dequant, MMA, bias, ReLU, reduce, or residual math.

Expected effect:

- Preserve AUC/PCOC exactly relative to Phase 5.1.
- Reduce activation tile load overhead after the Phase 5.1 packed-weight fix.
- Potentially recover part of the remaining `+2.6037s` latency gap to the fp16 baseline.

Cloud validation command:

```bash
USE_W4A16_SMOE=1 USE_CUDA_W4A16_SMOE=1 REQUIRE_CUDA_W4A16_SMOE=1 CHECK_W4A16_SMOE=1 W4A16_GROUP_SIZE=128 python infer.py
```

Cloud result after Phase 5.2:

```text
quality:
  AUC     = 0.758269
  PCOC    = 1.102424
  latency = 24.7354s
  score   = 73.515104
```

Compared with Phase 5.1:

```text
Phase 5.1:
  AUC     = 0.758269
  PCOC    = 1.102424
  latency = 25.1283s
  score   = 73.423422

Phase 5.2 delta:
  AUC     = 0.000000
  PCOC    = 0.000000
  latency = -0.3929s
  score   = +0.091682
```

Compared with the fp16 custom CUDA baseline:

```text
fp16 baseline:
  AUC     = 0.759609
  PCOC    = 1.110132
  latency = 22.5246s
  score   = 74.099831

Phase 5.2 CUDA W4A16 delta:
  AUC     = -0.001340
  PCOC    = -0.007708
  latency = +2.2108s
  score   = -0.584727
```

Interpretation: Phase 5.2 is a small pass. The vector activation tile load preserved quality exactly and recovered another 0.3929s. Because the gain is much smaller than Phase 5.1, the remaining bottleneck is likely the W4 weight dequant path, scale/zero loads, and/or shared-memory dequant staging rather than activation tile load.

### Phase 6: BitDecoding-Style W4A16 Rewrite

Add a separate experimental CUDA entry instead of replacing the stable Phase 5.2 path:

```text
smoe_forward_w4a16_reg(...)
smoe_forward_w4a16_reg_with_residual(...)
```

Python switch:

```text
USE_REG_DEQUANT_W4A16_SMOE=1
```

Stable path remains:

```text
USE_W4A16_SMOE=1
USE_CUDA_W4A16_SMOE=1
USE_REG_DEQUANT_W4A16_SMOE=0
```

Experimental path:

```text
USE_W4A16_SMOE=1
USE_CUDA_W4A16_SMOE=1
USE_REG_DEQUANT_W4A16_SMOE=1
```

Implementation boundary:

- Keep existing route count, route pack, top-2 reduce, and residual reduce.
- Keep the existing Phase 5.2 W4A16 kernel as fallback and A/B baseline.
- Add a new grouped linear instantiation using a BitDecoding-style `lop3 + half2` q4 dequant helper.
- New path loads four packed int16 values for one output row and one 16-wide K tile, dequantizes them through two `lop3_dequant_u4x8` calls, and writes the dequantized half2 pairs into the MMA-compatible B shared tile.

This is not yet the final CUTE-style register-fragment pipeline from BitDecoding. It is the first isolated rewrite step that moves the W4 dequant math from scalar float operations to `lop3 + half2`, while keeping the current hand-written `mma.sync.m16n8k16` path intact.

Cloud validation command:

```bash
USE_W4A16_SMOE=1 USE_CUDA_W4A16_SMOE=1 USE_REG_DEQUANT_W4A16_SMOE=1 REQUIRE_CUDA_W4A16_SMOE=1 CHECK_W4A16_SMOE=1 W4A16_GROUP_SIZE=128 python infer.py
```

Expected checks:

- Extension exports `smoe_forward_w4a16_reg` and `smoe_forward_w4a16_reg_with_residual`.
- Full inference reaches the new log line with `reg_dequant=True`.
- AUC/PCOC should be checked against Phase 5.2 because scale/zero are now applied through half2 arithmetic.
- If quality passes, compare latency against Phase 5.2 `24.7354s`.

Cloud result:

```text
quality:
  AUC     = 0.758190
  PCOC    = 1.102736
  latency = 26.1143s
  score   = 73.185057
```

Compared with Phase 5.2:

```text
Phase 5.2:
  AUC     = 0.758269
  PCOC    = 1.102424
  latency = 24.7354s
  score   = 73.515104

Phase 6 delta:
  AUC     = -0.000079
  PCOC    = +0.000312
  latency = +1.3789s
  score   = -0.330047
```

Interpretation: Phase 6 is a functional CUDA pass but not a performance pass. It proves the new extension symbols and switch work, but the current implementation still writes dequantized weights into the shared-memory B tile and then uses the old `ldmatrix` path. The added `lop3 + half2` work does not remove the shared-memory staging or B-fragment load cost, and it also reduces B tile load parallelism. The small AUC drift is expected because scale/zero are applied through half2 arithmetic instead of the Phase 5.2 float scalar dequant path.

### Phase 7: Direct B-Fragment Dequant

Add a separate true fragment-dequant CUDA entry:

```text
smoe_forward_w4a16_frag(...)
smoe_forward_w4a16_frag_with_residual(...)
```

Python switch:

```text
USE_FRAG_DEQUANT_W4A16_SMOE=1
```

This path is different from Phase 6:

- Phase 6 still dequantized W4 values into a half B shared-memory tile, then used `ldmatrix`.
- Phase 7 removes the half B shared-memory tile.
- For each `mma.sync.m16n8k16` step, the kernel directly constructs `b_frag[j][0..1]` from packed W4 global memory and scale/zero.
- A still uses the existing shared-memory tile and `ldmatrix_x4`.
- Routing, route pack, top-2 reduce, and residual reduce remain unchanged.

Cloud validation command:

```bash
USE_W4A16_SMOE=1 USE_CUDA_W4A16_SMOE=1 USE_FRAG_DEQUANT_W4A16_SMOE=1 REQUIRE_CUDA_W4A16_SMOE=1 CHECK_W4A16_SMOE=1 W4A16_GROUP_SIZE=128 python infer.py
```

Expected checks:

- Extension exports `smoe_forward_w4a16_frag` and `smoe_forward_w4a16_frag_with_residual`.
- Full inference reaches the new log line with `frag_dequant=True`.
- If B fragment lane mapping is correct, AUC/PCOC should be near Phase 6/Phase 5.2.
- If B fragment lane mapping is wrong, quality will fail clearly; keep Phase 5.2 as the stable fallback.
- If quality passes, compare latency against Phase 5.2 `24.7354s`.

Cloud result:

```text
Phase 7:
  AUC     = 0.733237
  PCOC    = 5.091662
  latency = 23.5610s
  score   = 64.502436

Compared with Phase 5.2:
  AUC     = -0.025032
  PCOC    = +3.989238
  latency = -1.1744s
  score   = -9.012668
```

Interpretation: Phase 7 is a latency pass but a quality failure. The 23.5610s timing proves that removing the half B shared-memory tile can recover real time, but the AUC/PCOC collapse means the current direct B-fragment construction does not match the `ldmatrix_x2` B operand layout expected by `mma.sync.m16n8k16.row.col`. This is not a submit path yet. The next step is a small B-fragment layout diagnostic that compares registers produced by the stable shared-memory `ldmatrix_x2` path against registers produced by the direct W4 dequant path for the same deterministic tile.

Micro diagnostic command:

```bash
W4A16_GROUP_SIZE=128 python infer.py --debug-w4a16-bfrag
```

This command does not load data or run full inference. It builds deterministic packed W4 fc1/fc2 tiles, then dumps:

```text
[warp_id, lane, j, ref0/ref1/direct0/direct1]
```

where `ref0/ref1` come from the stable `b_shared -> ldmatrix_x2` path and `direct0/direct1` come from the current direct W4 dequant mapping. The expected first target is `mismatched_pairs=0`; any mismatch means the B operand lane/register mapping still needs correction before rerunning AUC/PCOC.

First diagnostic result:

```text
fc1 mismatched_pairs = 928/1024
fc2 mismatched_pairs = 928/1024
```

The mismatches showed that the direct path used the `ldmatrix_x2` address-provider lane mapping:

```text
old:
  lane_n      = lane & 7
  lane_k_pair = (lane >> 3) * 2
```

but `mma.sync` consumes the `ldmatrix_x2` output fragment mapping:

```text
fixed:
  lane_n      = lane >> 2
  lane_k_pair = (lane & 3) * 2
```

This fix has been applied to both the direct fragment-dequant kernel and the diagnostic direct path. Re-run the micro diagnostic first; only if it reaches `mismatched_pairs=0/1024` should full AUC/PCOC be rerun.

Second diagnostic result after the mapping fix:

```text
fc1 mismatched_pairs = 0/1024
fc2 mismatched_pairs = 0/1024
```

Interpretation: the direct W4 dequant B registers now match the stable `ldmatrix_x2` B registers for both fc1 and fc2 deterministic tiles. The next validation step is full inference with `USE_FRAG_DEQUANT_W4A16_SMOE=1`.

Full Phase 7 validation after the mapping fix:

```text
AUC      = 0.758190
PCOC     = 1.102736
Latency  = 23.1877s
score    = 73.867949
```

Interpretation: direct fragment dequant is now a quality pass and faster than the Phase 5.2 shared-B W4A16 path. It is still slightly slower than the fp16 routed SMoE baseline, so Phase 8 focuses on measured micro diagnostics and low-risk direct-fragment cleanup.

## Phase 8 Forward Diagnostic

Phase 8 adds a deterministic forward micro diagnostic:

```bash
W4A16_GROUP_SIZE=128 python infer.py --debug-w4a16-forward
```

This constructs a standalone SMoE, fixed synthetic input, fixed top-2 routing, and fixed scores. It compares:

```text
Python W4A16 qdq reference
CUDA W4A16 shared-B path
CUDA W4A16 reg/lop3 shared-B path
CUDA W4A16 direct-frag path
```

The default `--debug-w4a16-forward-tokens=640` gives each expert both a full 128-route tile and a partial tile, so it exercises the important grouped-M edge cases without running the full dataset.

Phase 8 also changes the direct-frag B operand load from two independent pair dequant calls to one paired helper:

```text
same output row + same quant group
  -> load scale/zero once
  -> load packed K pair and packed K+8 pair
  -> produce b_frag[j][0] and b_frag[j][1]
```

This preserves the current arithmetic path while reducing repeated scale/zero work. Re-run `--debug-w4a16-bfrag` after this change; it should remain `mismatched_pairs=0/1024` before any full inference run.

Phase 8 cloud validation:

```text
B-fragment diagnostic:
  fc1 mismatched_pairs = 0/1024
  fc2 mismatched_pairs = 0/1024

Forward diagnostic:
  shared forward max_abs_err   = 0.000000e+00
  shared residual max_abs_err  = 0.000000e+00
  reg forward max_abs_err      = 3.051758e-04
  reg residual max_abs_err     = 3.662109e-04
  frag forward max_abs_err     = 3.051758e-04
  frag residual max_abs_err    = 3.662109e-04

Full inference:
  AUC      = 0.758190
  PCOC     = 1.102736
  Latency  = 22.7517s
  score    = 73.969681
```

Interpretation: paired direct-fragment dequant is a correctness pass and improves latency from the Phase 7 post-fix `23.1877s` to `22.7517s`, while preserving AUC/PCOC exactly. It is now within about `0.23s` of the fp16 routed SMoE baseline latency.

## Phase 9 Half Scale/Zero Buffers

Phase 9 adds CUDA-only fp16 scale/zero buffers:

```text
_w4_w1_scale_h
_w4_w1_zero_h
_w4_w2_scale_h
_w4_w2_zero_h
```

The float32 scale/zero buffers remain the source of truth for the Python qdq reference and pack reconstruction checks:

```text
_w4_w1_scale / _w4_w1_zero
_w4_w2_scale / _w4_w2_zero
```

The switch is:

```bash
USE_HALF_SCALE_W4A16_SMOE=1
```

It defaults to enabled. It is only applied to the reg-dequant and frag-dequant CUDA entries; the shared-B W4A16 path keeps float32 scale/zero so it can remain a zero-error diagnostic reference against Python qdq.

Expected behavior:

```text
shared path:
  float32 scale/zero, exact against Python qdq reference

reg/frag path:
  fp16 scale/zero, same half2 dequant semantics as the current CUDA path
```

Validation order:

```bash
W4A16_GROUP_SIZE=128 python infer.py --debug-w4a16-bfrag
W4A16_GROUP_SIZE=128 python infer.py --debug-w4a16-forward
USE_W4A16_SMOE=1 USE_CUDA_W4A16_SMOE=1 USE_FRAG_DEQUANT_W4A16_SMOE=1 REQUIRE_CUDA_W4A16_SMOE=1 CHECK_W4A16_SMOE=1 W4A16_GROUP_SIZE=128 python infer.py
```

Use this fallback if half scale/zero unexpectedly regresses:

```bash
USE_HALF_SCALE_W4A16_SMOE=0
```

Phase 9 cloud validation:

```text
B-fragment diagnostic:
  fc1 mismatched_pairs = 0/1024
  fc2 mismatched_pairs = 0/1024

Forward diagnostic:
  shared forward max_abs_err   = 0.000000e+00
  shared residual max_abs_err  = 0.000000e+00
  reg forward max_abs_err      = 3.051758e-04
  reg residual max_abs_err     = 3.662109e-04
  frag forward max_abs_err     = 3.051758e-04
  frag residual max_abs_err    = 3.662109e-04

Weight check:
  fc1_scale_h_max_abs_err = 3.561378e-06
  fc1_zero_h_max_abs_err  = 0.000000e+00
  fc2_scale_h_max_abs_err = 3.560446e-06
  fc2_zero_h_max_abs_err  = 0.000000e+00

Full inference:
  AUC      = 0.758190
  PCOC     = 1.102736
  Latency  = 22.6213s
  score    = 74.000093
```

Interpretation: half scale/zero buffers are a correctness pass and improve latency from Phase 8 `22.7517s` to `22.6213s`, while preserving AUC/PCOC exactly. The remaining gap to the fp16 routed SMoE baseline is now about `0.10s`.

## Phase 10 Direct Half2 Scale Broadcast

Phase 10 removes an unnecessary conversion in the W4A16 CUDA dequant path.

Before:

```text
c10::Half scale/zero
-> scalar_to_float()
-> __float2half2_rn()
-> half2 dequant
```

After:

```text
c10::Half scale/zero
-> __ushort_as_half(16-bit payload)
-> __halves2half2(h, h)
-> half2 dequant
```

The change is implemented as a typed helper:

```text
scalar_to_half2(float)     -> __float2half2_rn(...)
scalar_to_half2(c10::Half) -> direct half2 broadcast
```

Touched CUDA paths:

```text
dequant_w4_pair_bits_from_global()
dequant_w4_two_pair_bits_from_global()
load_grouped_tile_w4a16_lop3()
```

Expected behavior:

```text
float32 scale/zero shared diagnostic path:
  unchanged

fp16 scale/zero reg/frag path:
  same numerical result, fewer conversion instructions
```

Validation order:

```bash
W4A16_GROUP_SIZE=128 python infer.py --debug-w4a16-bfrag
W4A16_GROUP_SIZE=128 python infer.py --debug-w4a16-forward
USE_W4A16_SMOE=1 USE_CUDA_W4A16_SMOE=1 USE_FRAG_DEQUANT_W4A16_SMOE=1 REQUIRE_CUDA_W4A16_SMOE=1 CHECK_W4A16_SMOE=1 W4A16_GROUP_SIZE=128 python infer.py
```

Phase 10 cloud validation:

```text
B-fragment diagnostic:
  fc1 mismatched_pairs = 0/1024
  fc2 mismatched_pairs = 0/1024

Forward diagnostic:
  shared forward max_abs_err   = 0.000000e+00
  shared residual max_abs_err  = 0.000000e+00
  reg forward max_abs_err      = 3.051758e-04
  reg residual max_abs_err     = 3.662109e-04
  frag forward max_abs_err     = 3.051758e-04
  frag residual max_abs_err    = 3.662109e-04

Full inference:
  AUC      = 0.758190
  PCOC     = 1.102736
  Latency  = 23.5079s
  score    = 73.793237
```

Interpretation: direct half2 broadcast is a correctness pass but a performance regression. It worsened latency from Phase 9 `22.6213s` to `23.5079s`, so the CUDA default was reverted to the Phase 9 `__float2half2_rn(scalar_to_float(...))` path. Keep this as a measured dead end, not a submission path.

## Phase 11 Lop3 Direct-Fragment Q Dequant

Phase 11 tested the remaining q-value conversion in the default direct-frag path.

Before, default frag used scalar nibble extraction:

```text
packed int16
-> shift/mask q0/q1
-> __float2half_rn(q)
-> __hfma2(q, scale, zero)
```

The attempted Phase 11 change made `dequant_w4_two_pair_bits_from_global()` use the BitDecoding-style lop3 helper:

```text
packed0 for k/k+1
packed1 for k+8/k+9
-> combine into one uint32 q0..q7 payload
-> lop3_dequant_u4x8()
-> low half lanes  -> b_frag[j][0]
-> high half lanes -> b_frag[j][1]
-> __hfma2(q, scale, zero)
```

This keeps the already-validated ldmatrix output mapping:

```text
lane_n      = lane >> 2
lane_k_pair = (lane & 3) * 2
```

Cloud validation:

B-fragment diagnostic:
  fc1/fc2 mismatched_pairs = 0/1024

Forward diagnostic:
  frag forward max_abs_err   = 3.051758e-04
  frag residual max_abs_err  = 3.662109e-04

Full inference:
  AUC      = 0.758190
  PCOC     = 1.102736
  Latency  = 23.5777s
  score    = 73.776939
```

Interpretation: the lop3 direct-fragment q-value dequant is a correctness pass but a performance regression. It worsened latency from Phase 9 `22.6213s` to `23.5777s`, so the CUDA default was reverted to the Phase 9 shift/mask + `__float2half_rn` direct-frag path. Keep this as a measured dead end, not a submission path.

## Phase 12 Atomic Fused Fc2 Reduce Experiment

Phase 12 adds an experimental direct-frag path that removes the separate `y_route -> top2 reduce (+ residual)` kernel after fc2.

The new path is behind `USE_FUSED_FC2_REDUCE_W4A16_SMOE=1` and only applies when `USE_FRAG_DEQUANT_W4A16_SMOE=1`.

```text
route count/prefix/pack
-> fc1 direct-frag W4A16
-> fc2 direct-frag W4A16
   -> atomicAdd(score * value) into final out[token, dim]
```

For residual mode, the final output buffer is initialized by a device-to-device residual copy before the fc2 atomic scatter. For non-residual mode, it is zeroed first.

Important caveat: this is an experiment, not the default path. The old reduce computes both top-2 products in fp32 and casts once to fp16. Phase 12 uses global fp16 atomic accumulation, so it can introduce extra rounding and nondeterministic accumulation order. Validate both diagnostics and full inference before treating it as a candidate.

Validation commands:

```bash
USE_W4A16_SMOE=1 USE_CUDA_W4A16_SMOE=1 USE_FRAG_DEQUANT_W4A16_SMOE=1 USE_FUSED_FC2_REDUCE_W4A16_SMOE=1 W4A16_GROUP_SIZE=128 python infer.py --debug-w4a16-forward

USE_W4A16_SMOE=1 USE_CUDA_W4A16_SMOE=1 USE_FRAG_DEQUANT_W4A16_SMOE=1 USE_FUSED_FC2_REDUCE_W4A16_SMOE=1 REQUIRE_CUDA_W4A16_SMOE=1 CHECK_W4A16_SMOE=1 W4A16_GROUP_SIZE=128 python infer.py
```

Forward diagnostic result:

```text
compile: pass

shared forward max_abs_err       = 0.000000e+00
shared residual max_abs_err      = 0.000000e+00
reg forward max_abs_err          = 3.051758e-04
reg residual max_abs_err         = 3.662109e-04
frag forward max_abs_err         = 3.051758e-04
frag residual max_abs_err        = 3.662109e-04
frag_atomic forward max_abs_err  = 3.356934e-04
frag_atomic residual max_abs_err = 3.662109e-04

frag_atomic forward mean_abs_err  = 5.653043e-05
frag_atomic forward rmse          = 7.117497e-05
frag_atomic residual mean_abs_err = 5.763895e-05
frag_atomic residual rmse         = 7.487988e-05
```

Interpretation: Phase 12 passes the deterministic forward diagnostic. The atomic path adds only a small extra rounding delta relative to Phase 9 direct-frag, mainly visible in forward max error and residual mean/RMSE. Continue to full inference before deciding whether the saved reduce kernel offsets the fp16 atomic overhead.

Full inference result:

```text
AUC      = 0.758683
PCOC     = 1.102687
Latency  = 26.9335s
score    = 73.035276
```

Compared with Phase 9:

```text
Phase 9 latency  = 22.6213s
Phase 12 latency = 26.9335s
delta            = +4.3122s

Phase 9 score    = 74.000093
Phase 12 score   = 73.035276
delta            = -0.964817
```

Interpretation: Phase 12 is a correctness pass but a performance regression. The separate reduce kernel is cheaper than replacing the fc2 store with fp16 global atomics. Keep the code as an opt-in experiment, but do not use `USE_FUSED_FC2_REDUCE_W4A16_SMOE=1` for submission.

## Phase 13 W4A16 Param-Cache Experiment

Phase 13 targets the direct-frag W4A16 path's repeated scale/zero loads.

Current Phase 9 direct-frag path:

```text
for each K tile:
  load A tile
  dequant B fragment from global packed weight
  load scale/zero from global for every B fragment lane
  MMA
```

With `group_size=128`, one scale/zero group is reused across eight `kMmaK=16` tiles. Phase 13 adds an opt-in `frag_param` path:

```text
for each K group:
  load scale_shared[128], zero_shared[128] once per CTA
  for each K tile in the group:
    load A tile
    dequant B fragment from global packed weight using shared scale/zero
    MMA
```

This keeps packed weights direct-from-global and preserves the existing coalesced fc2 store + separate reduce path. It does not combine with the Phase 12 atomic reduce experiment.

Implemented switch:

```text
USE_PARAM_CACHE_W4A16_SMOE=1
```

Validation commands:

```bash
USE_W4A16_SMOE=1 USE_CUDA_W4A16_SMOE=1 USE_FRAG_DEQUANT_W4A16_SMOE=1 USE_PARAM_CACHE_W4A16_SMOE=1 W4A16_GROUP_SIZE=128 python infer.py --debug-w4a16-forward

USE_W4A16_SMOE=1 USE_CUDA_W4A16_SMOE=1 USE_FRAG_DEQUANT_W4A16_SMOE=1 USE_PARAM_CACHE_W4A16_SMOE=1 REQUIRE_CUDA_W4A16_SMOE=1 CHECK_W4A16_SMOE=1 W4A16_GROUP_SIZE=128 python infer.py
```

Forward diagnostic result:

```text
frag_param forward max_abs_err  = 3.051758e-04
frag_param forward mean_abs_err = 5.654232e-05
frag_param forward rmse         = 7.115609e-05

frag_param residual max_abs_err  = 3.662109e-04
frag_param residual mean_abs_err = 5.652193e-05
frag_param residual rmse         = 7.348951e-05
```

This matches the Phase 9 direct-frag diagnostic profile.

Full inference result:

```text
AUC      = 0.758190
PCOC     = 1.102736
Latency  = 23.6426s
score    = 73.761806
```

Compared with Phase 9:

```text
Phase 9 latency  = 22.6213s
Phase 13 latency = 23.6426s
delta            = +1.0213s

Phase 9 score    = 74.000093
Phase 13 score   = 73.761806
delta            = -0.238287
```

Interpretation: Phase 13 preserves correctness but regresses latency. The scale/zero global loads were likely not the dominant bottleneck, or were already served well by cache. The shared-memory param cache adds shared-memory traffic, synchronization, and possible occupancy/register-pressure effects without addressing packed-weight load overlap. Keep this path opt-in, but do not use `USE_PARAM_CACHE_W4A16_SMOE=1` for submission in its current form.

## Phase 14 W4A16 A/W4-Pack Cp.Async Pipeline Experiment

Phase 14 targets the current direct-frag W4A16 loop:

```text
for each K tile:
  load A tile into shared memory with ordinary vector loads
  load W4 packed values directly from global during B-fragment dequant
  MMA
  sync
```

The new opt-in `frag_pipe` path adds two-stage shared-memory buffering for full M tiles:

```text
stage buffers:
  A tile:          [2, 128, 16] fp16
  W4 packed tile: [2, 128, 8] int16

prologue:
  cp.async A tile 0
  cp.async aligned W4-pack window 0

main loop:
  cp.async next A tile
  cp.async next aligned W4-pack window
  compute current tile
  wait/sync

epilogue:
  compute final loaded tile
```

Packed W4 rows are loaded as 16-byte aligned windows. Each K tile needs four packed `int16` values per output row, but the cp.async path loads eight packed `int16` values per row so both even and odd K tiles can use aligned 16-byte global addresses:

```text
k_base = 0   -> use shared packed entries 0..3
k_base = 16  -> use shared packed entries 4..7
k_base = 32  -> use shared packed entries 0..3 from next aligned window
```

Partial M tiles fall back to the Phase 9 direct-frag loop to keep route-tail correctness simple.

Implemented switch:

```text
USE_PIPELINED_W4A16_SMOE=1
```

This switch requires `USE_FRAG_DEQUANT_W4A16_SMOE=1` and does not combine with the Phase 12 atomic reduce or Phase 13 param-cache experiments.

Validation commands:

```bash
USE_W4A16_SMOE=1 USE_CUDA_W4A16_SMOE=1 USE_FRAG_DEQUANT_W4A16_SMOE=1 USE_PIPELINED_W4A16_SMOE=1 W4A16_GROUP_SIZE=128 python infer.py --debug-w4a16-forward

USE_W4A16_SMOE=1 USE_CUDA_W4A16_SMOE=1 USE_FRAG_DEQUANT_W4A16_SMOE=1 USE_PIPELINED_W4A16_SMOE=1 REQUIRE_CUDA_W4A16_SMOE=1 CHECK_W4A16_SMOE=1 W4A16_GROUP_SIZE=128 python infer.py
```

Forward diagnostic result:

```text
frag_pipe forward max_abs_err  = 3.051758e-04
frag_pipe forward mean_abs_err = 5.654232e-05
frag_pipe forward rmse         = 7.115609e-05

frag_pipe residual max_abs_err  = 3.662109e-04
frag_pipe residual mean_abs_err = 5.652193e-05
frag_pipe residual rmse         = 7.348951e-05
```

This matches the Phase 9 direct-frag diagnostic profile.

Full inference result:

```text
AUC      = 0.758190
PCOC     = 1.102736
Latency  = 22.6844s
score    = 73.985380
```

Compared with Phase 9:

```text
Phase 9 latency  = 22.6213s
Phase 14 latency = 22.6844s
delta            = +0.0631s

Phase 9 score    = 74.000093
Phase 14 score   = 73.985380
delta            = -0.014713
```

Interpretation: Phase 14 is a correctness pass and near performance parity with Phase 9, but it is not a measured win. Loading W4 pack through cp.async shared buffers does not expose a clear bottleneck; the added shared-memory traffic and larger shared footprint roughly cancel any global-load overlap benefit. Keep this path opt-in, but do not replace the Phase 9 default unless repeated same-session A/B runs show the small delta is noise or a win.

## Correctness Checks

Run in this order:

```text
fp16 baseline vs Python W4A16 reference
Python W4A16 reference vs CUDA W4A16
full inference AUC/PCOC
cloud latency
```

Per-stage checks:

- packed weight dequant max/mean error
- fc1 output max/mean error
- fc2 output max/mean error
- final SMoE output max/mean error

## Current Status

```text
Status: Phase 14 A/W4-pack cp.async pipeline was measured near parity but not faster; Phase 9 remains the current W4A16 submission candidate

Done:
  - previous SMoE quantization branch removed from test7_plus
  - fp16 routed SMoE remains as baseline
  - BitDecoding/A100 implementation direction documented
  - Phase 1 Python W4A16 reference implemented in test7_plus/infer.py
  - W4A16 weight error check added behind CHECK_W4A16_SMOE=1
  - Phase 1 cloud quality checked with group_size=128
  - W4A16 AUC drop is small enough to continue CUDA work
  - Phase 2 W4 packed buffers added to prepare_w4a16_weights()
  - Phase 2 pack/unpack/dequant reconstruction check added
  - Phase 2 cloud pack check passed with zero reconstruction error
  - Phase 2 full-model AUC/PCOC matches Phase 1 reference
  - Phase 3 W4A16 CUDA extension symbols added
  - Phase 3 Python CUDA-attempt switch and reference fallback added
  - Phase 3 CUDA-side parameter validation added
  - Phase 3 cloud fallback compile check passed
  - Phase 3 cloud strict interface check reached expected not-implemented boundary
  - Phase 4 first W4A16 grouped linear CUDA kernel added
  - Phase 4 W4A16 forward now uses CUDA fc1/fc2 and existing reduce/residual
  - Phase 4 cloud run completed with strict CUDA W4A16 path
  - Phase 4 CUDA W4A16 is faster than Python W4A16 reference
  - Phase 4 CUDA W4A16 is still slower than fp16 baseline
  - Phase 5.1 W4A16 B tile load now unpacks four q4 values per packed int16 load
  - Phase 5.1 cloud timing improved latency from 31.5436s to 25.1283s
  - Phase 5.1 preserved AUC/PCOC exactly relative to Phase 4
  - Phase 5.1 score is within 0.676409 of the fp16 baseline score
  - Phase 5.2 W4A16 A tile load now uses uint4 vector copies into shared memory
  - Phase 5.2 cloud timing improved latency from 25.1283s to 24.7354s
  - Phase 5.2 preserved AUC/PCOC exactly relative to Phase 5.1
  - Phase 5.2 score is within 0.584727 of the fp16 baseline score
  - Phase 6 added separate smoe_forward_w4a16_reg CUDA extension entries
  - Phase 6 added USE_REG_DEQUANT_W4A16_SMOE Python switch
  - Phase 6 added BitDecoding-style lop3 + half2 q4 dequant helper
  - Phase 6 cloud validation compiled and ran with reg_dequant=True
  - Phase 6 latency regressed from 24.7354s to 26.1143s
  - Phase 6 quality changed slightly because dequant arithmetic moved from float scalar to half2
  - Phase 7 added separate smoe_forward_w4a16_frag CUDA extension entries
  - Phase 7 added USE_FRAG_DEQUANT_W4A16_SMOE Python switch
  - Phase 7 removes the half B shared-memory tile and constructs B MMA fragments directly
  - Phase 7 cloud validation compiled and ran with frag_dequant=True
  - Phase 7 improved latency from 24.7354s to 23.5610s
  - Phase 7 quality failed with PCOC=5.091662, indicating a B fragment layout mismatch
  - W4A16 B-fragment micro diagnostic added behind --debug-w4a16-bfrag
  - B-fragment diagnostic found 928/1024 mismatches in both fc1 and fc2
  - Direct fragment-dequant mapping fixed from address-lane mapping to ldmatrix output mapping
  - B-fragment diagnostic now passes with 0/1024 mismatches in both fc1 and fc2
  - Phase 7 full inference after mapping fix passed quality
  - Phase 7 latency improved to 23.1877s with score_all=73.867949
  - Phase 8 added --debug-w4a16-forward deterministic forward diagnostic
  - Phase 8 added exact Python W4A16 residual reference helper
  - Phase 8 direct-frag path now produces b_frag[j][0/1] with one paired scale/zero load
  - Phase 8 B-fragment diagnostic still passes with 0/1024 mismatches in both fc1 and fc2
  - Phase 8 forward diagnostic shows frag/reg differ from qdq reference only by small half2 dequant rounding
  - Phase 8 full inference preserves AUC=0.758190 and PCOC=1.102736
  - Phase 8 latency improved to 22.7517s with score_all=73.969681
  - Phase 9 added USE_HALF_SCALE_W4A16_SMOE switch, default enabled
  - Phase 9 added fp16 scale/zero buffers for CUDA reg/frag paths
  - Phase 9 keeps shared-B W4A16 on float32 scale/zero for exact diagnostic comparison
  - Phase 9 B-fragment diagnostic still passes with 0/1024 mismatches in both fc1 and fc2
  - Phase 9 forward diagnostic matches Phase 8 error profile
  - Phase 9 full inference preserves AUC=0.758190 and PCOC=1.102736
  - Phase 9 latency improved to 22.6213s with score_all=74.000093
  - Phase 10 direct c10::Half -> __half2 broadcast compiled and passed diagnostics
  - Phase 10 regressed latency to 23.5079s with score_all=73.793237
  - Phase 10 CUDA default was reverted to Phase 9 behavior
  - Phase 11 lop3 direct-fragment q-value dequant compiled and passed diagnostics
  - Phase 11 B-fragment diagnostic passed with 0/1024 mismatches in both fc1 and fc2
  - Phase 11 forward diagnostic matched the Phase 9 error profile
  - Phase 11 regressed latency to 23.5777s with score_all=73.776939
  - Phase 11 CUDA default was reverted to Phase 9 behavior
  - Phase 12 added smoe_forward_w4a16_frag_atomic CUDA extension entries
  - Phase 12 added USE_FUSED_FC2_REDUCE_W4A16_SMOE Python switch
  - Phase 12 fuses fc2 route output with top-2 reduce by atomic scatter into final out
  - Phase 12 keeps the atomic path opt-in and leaves Phase 9 as default
  - Phase 12 compiled on cloud and passed the W4A16 forward diagnostic
  - Phase 12 full inference preserved AUC/PCOC but regressed latency to 26.9335s
  - Phase 12 is not a submission candidate in its current atomic form
  - Phase 13 added smoe_forward_w4a16_frag_param CUDA extension entries
  - Phase 13 added USE_PARAM_CACHE_W4A16_SMOE Python switch
  - Phase 13 caches per-output-tile W4A16 scale/zero in shared memory for direct-frag dequant
  - Phase 13 compiled on cloud and matched the Phase 9 forward diagnostic profile
  - Phase 13 full inference preserved AUC/PCOC but regressed latency to 23.6426s
  - Phase 13 is not a submission candidate in its current shared-param-cache form
  - Phase 14 added smoe_forward_w4a16_frag_pipe CUDA extension entries
  - Phase 14 added USE_PIPELINED_W4A16_SMOE Python switch
  - Phase 14 loads A tile and aligned W4 packed windows through two-stage cp.async shared buffers
  - Phase 14 keeps partial route tiles on the Phase 9 direct-frag fallback path
  - Phase 14 compiled on cloud and matched the Phase 9 forward diagnostic profile
  - Phase 14 full inference preserved AUC/PCOC but measured latency at 22.6844s
  - Phase 14 is near parity with Phase 9 but is not a measured improvement

Next:
  - keep Phase 8 frag-dequant as fallback via USE_HALF_SCALE_W4A16_SMOE=0
  - do not use USE_REG_DEQUANT_W4A16_SMOE=1 for submission in its current form
  - keep Phase 9 frag-dequant with half scale/zero as the current W4A16 submission candidate
  - compare Phase 9 against a fresh fp16 baseline on the same run environment
  - do not use USE_FUSED_FC2_REDUCE_W4A16_SMOE=1 for submission in its current form
  - do not use USE_PARAM_CACHE_W4A16_SMOE=1 for submission in its current form
  - do not use USE_PIPELINED_W4A16_SMOE=1 for submission unless repeated same-session A/B shows a real latency win
  - profile before any further W4A16 kernel experiment; prefer proving the bottleneck before adding another shared-memory cache
  - keep group_size=32 as a fallback only if CUDA W4A16 introduces extra quality drift
```
