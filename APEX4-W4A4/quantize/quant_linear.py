import gc
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from quantize.matmul_had import *
from quantize.quantizer import UniformActivationQuantizer
# W4A4 GEMM kernels (the only kernels in this release).
from kernels.APEX4._CUDA import apex4_gemm_w4a4_group    # activation/weight per-group, half
from kernels.APEX4._CUDA import apex4_gemm_w4a4_channel  # activation/weight per-channel, half
from kernels.APEX4._CUDA import quantize_compress_A      # fused activation quant + 4-bit pack


# =====================================================================
#  Triton fused kernels — JIT compiled, no build step required
# =====================================================================
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False
    print("[quant_linear] triton not found, falling back to PyTorch ops "
          "(pip install triton for ~2-3x decode speedup)")

if _HAS_TRITON:
    @triton.jit
    def _fused_channel_quant_compress_kernel(
        x_ptr,      # (M, K) fp16 input
        A_ptr,      # (M, K//8) int32 output — packed 4-bit
        scale_ptr,  # (M,) fp16 output — per-row quant scale
        M, K,
        stride_xm,
        stride_am,
        BLOCK_K: tl.constexpr,
        BLOCK_GROUPS: tl.constexpr,
    ):
        """
        Fused channel-wise activation quantization + int4 compression.
        1 Triton launch 替换 ~26 PyTorch kernel launches.
        """
        row = tl.program_id(0)

        # ── Pass 1: row-wise abs max ─────────────────────────────────
        max_val = 0.0
        for k_start in range(0, K, BLOCK_K):
            k_offs = k_start + tl.arange(0, BLOCK_K)
            mask = k_offs < K
            x = tl.load(x_ptr + row * stride_xm + k_offs,
                        mask=mask, other=0.0).to(tl.float32)
            block_max = tl.max(tl.abs(x))
            max_val = tl.maximum(max_val, block_max)

        max_val = tl.maximum(max_val, 1e-6)
        scale_f16 = (max_val * (2.0 / 15.0)).to(tl.float16)
        tl.store(scale_ptr + row, scale_f16)
        inv_scale = 1.0 / scale_f16.to(tl.float32)

        # ── Pass 2: quantize int4 + pack 8 nibbles → int32 ──────────
        n_groups = (K + 7) // 8
        for g_start in range(0, n_groups, BLOCK_GROUPS):
            g_ids = g_start + tl.arange(0, BLOCK_GROUPS)
            g_mask = g_ids < n_groups

            packed = tl.zeros([BLOCK_GROUPS], dtype=tl.int32)
            for i in range(8):  # compile-time unrolled
                k_ids = g_ids * 8 + i
                k_mask = g_mask & (k_ids < K)
                x = tl.load(x_ptr + row * stride_xm + k_ids,
                            mask=k_mask, other=0.0).to(tl.float32)

                x_scaled = x * inv_scale
                q_int = tl.where(x_scaled >= 0,
                                 (x_scaled + 0.5).to(tl.int32),
                                 (x_scaled - 0.5).to(tl.int32))
                q_int = tl.minimum(tl.maximum(q_int, -8), 7) & 0xF
                packed = packed | (q_int << (i * 4))

            tl.store(A_ptr + row * stride_am + g_ids, packed, mask=g_mask)

    @triton.jit
    def _fused_dynamic_quant_int8_kernel(
        x_ptr,      # (M, K) fp16 input
        out_ptr,    # (M, K) int8 output
        scale_ptr,  # (M,) fp32 output — per-token scale
        M, K,
        stride_xm,
        stride_outm,
        BLOCK_K: tl.constexpr,
    ):
        """
        Fused per-token int8 dynamic quantization (legacy W4A8 path; unused in W4A4).
        1 Triton launch replaces ~10 PyTorch kernel launches.
        """
        row = tl.program_id(0)

        max_val = 0.0
        for k_start in range(0, K, BLOCK_K):
            k_offs = k_start + tl.arange(0, BLOCK_K)
            mask = k_offs < K
            x = tl.load(x_ptr + row * stride_xm + k_offs,
                        mask=mask, other=0.0).to(tl.float32)
            block_max = tl.max(tl.abs(x))
            max_val = tl.maximum(max_val, block_max)

        max_val = tl.maximum(max_val, 1e-6)
        scale = max_val / 127.0
        tl.store(scale_ptr + row, scale)
        inv_scale = 127.0 / max_val

        for k_start in range(0, K, BLOCK_K):
            k_offs = k_start + tl.arange(0, BLOCK_K)
            mask = k_offs < K
            x = tl.load(x_ptr + row * stride_xm + k_offs,
                        mask=mask, other=0.0).to(tl.float32)

            x_scaled = x * inv_scale
            q = tl.where(x_scaled >= 0,
                         (x_scaled + 0.5).to(tl.int32),
                         (x_scaled - 0.5).to(tl.int32))
            q = tl.minimum(tl.maximum(q, -128), 127)
            tl.store(out_ptr + row * stride_outm + k_offs,
                     q.to(tl.int8), mask=mask)


# ── Python wrappers ──────────────────────────────────────────────────

def fused_channel_quant_compress(x: torch.Tensor):
    m, k = x.shape
    compressed_k = (k + 7) // 8
    A = torch.empty((m, compressed_k), dtype=torch.int32, device=x.device)
    quant_scale = torch.empty((m,), dtype=torch.float16, device=x.device)

    BLOCK_K = min(triton.next_power_of_2(k), 1024)
    BLOCK_GROUPS = min(triton.next_power_of_2(compressed_k), 128)

    _fused_channel_quant_compress_kernel[(m,)](
        x, A, quant_scale, m, k,
        x.stride(0), A.stride(0),
        BLOCK_K=BLOCK_K, BLOCK_GROUPS=BLOCK_GROUPS,
    )
    return A, quant_scale.unsqueeze(1)


def fused_dynamic_quant_int8(x: torch.Tensor):
    m, k = x.shape
    A = torch.empty((m, k), dtype=torch.int8, device=x.device)
    s1 = torch.empty((m,), dtype=torch.float32, device=x.device)

    BLOCK_K = min(triton.next_power_of_2(k), 1024)

    _fused_dynamic_quant_int8_kernel[(m,)](
        x, A, s1, m, k,
        x.stride(0), A.stride(0),
        BLOCK_K=BLOCK_K,
    )
    return A, s1


# ── Fallback ─────────────────────────────────────────────────────────

def _channel_quant_compress_fallback(x: torch.Tensor):
    abs_max = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
    quant_scale = (abs_max * (2.0 / 15.0)).to(torch.half)
    x_q = (x.float() / quant_scale.float()).round().clamp(-8, 7).to(torch.int8)
    A = compress_4bit_to_int32(x_q)
    return A, quant_scale


def _dynamic_quant_int8_fallback(x: torch.Tensor):
    abs_max = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
    s1 = (abs_max / 127.0).to(torch.float32)
    A = (x.float() / s1).round().clamp(-128, 127).to(torch.int8)
    s1 = s1.squeeze(-1).contiguous()
    return A, s1


# =====================================================================

def preprocess_to_int4(matrix: torch.Tensor):
    H, W = matrix.shape
    matrix_t = matrix.t().contiguous()
    matrix_int4 = matrix_t.view(W, H // 4, 4)
    matrix_final = matrix_int4.transpose(0, 1).contiguous()
    return matrix_final


def quantize_compress(x_input, x_compressed, quant_scales, group_size):
    quantize_compress_A(x_input, x_compressed, quant_scales, group_size)


def compress_4bit_to_int32(x: torch.Tensor) -> torch.Tensor:
    """Fallback: Python int4 compression."""
    m, k = x.shape
    full_groups = k // 8
    remainder = k % 8
    compressed_k = (k + 7) // 8

    compressed = torch.zeros((m, compressed_k), dtype=torch.int32, device=x.device)

    if full_groups > 0:
        vals = (x[:, :full_groups * 8].reshape(m, full_groups, 8) & 0xF).to(torch.int32)
        compressed[:, :full_groups] = (
            vals[:, :, 0]        |
            (vals[:, :, 1] << 4)  |
            (vals[:, :, 2] << 8)  |
            (vals[:, :, 3] << 12) |
            (vals[:, :, 4] << 16) |
            (vals[:, :, 5] << 20) |
            (vals[:, :, 6] << 24) |
            (vals[:, :, 7] << 28)
        )

    if remainder > 0:
        tail = (x[:, -remainder:] & 0xF).to(torch.int32)
        last_col = torch.zeros(m, dtype=torch.int32, device=x.device)
        for i in range(remainder):
            last_col |= (tail[:, i] << (4 * i))
        compressed[:, -1] = last_col

    return compressed


def w4a4_mul_C(A, B, C, D, s1, s2, workspace, thread_k=-1, thread_n=-1, sms=-1, max_par=16):
    apex4_gemm_w4a4_channel(A, B, C, D, s1, s2, workspace, thread_k, thread_n, sms, max_par)

def w4a4_mul_G(A, B, C, D, s1, s2, workspace, thread_k=-1, thread_n=-1, sms=-1, max_par=16):
    apex4_gemm_w4a4_group(A, B, C, D, s1, s2, workspace, thread_k, thread_n, sms, max_par)


class RoundSTEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, qmax, qmin):
        return x.round().clamp(qmin, qmax)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


def round_ste(x: torch.Tensor, qmax, qmin):
    return RoundSTEFunction.apply(x, qmax, qmin)


class QuantLinearTorch(nn.Module):

    def __init__(self, qparams, aparams, infeatures, outfeatures, bias, symmetric, real_pack=True,
                 cuda_optimized=False):
        super().__init__()
        self.infeatures = infeatures
        self.outfeatures = outfeatures

        self.n_bits = qparams.bits
        self.group_size = qparams.group_size
        self.real_pack = real_pack
        self.symmetric = symmetric
        self.cuda_optimized = cuda_optimized

        if real_pack:
            if not cuda_optimized:
                self.register_buffer('qweight', torch.zeros((outfeatures, (infeatures // (32 // self.n_bits[0]))),
                                                            dtype=torch.int32))
            else:
                compressed_infeatures = infeatures // (32 // self.n_bits[0])
                final_h = compressed_infeatures // 4
                final_w = outfeatures
                self.register_buffer('qweight', torch.zeros((final_h, final_w, 4), dtype=torch.int32))
        else:
            self.qweight = nn.Parameter(torch.zeros((infeatures, outfeatures), dtype=torch.half))

        scale_zero_dims = self._calculate_scale_zero_dimensions(infeatures, outfeatures)

        self.scales = nn.Parameter(torch.zeros(scale_zero_dims, dtype=torch.half))
        self.zeros = nn.Parameter(torch.zeros(scale_zero_dims, dtype=torch.half))

        if bias:
            self.register_buffer('bias', torch.zeros((outfeatures), dtype=torch.half))
        else:
            self.bias = None

        self.act_quantizer = UniformActivationQuantizer(aparams)
        self.act_group_size = list(self.act_quantizer.group_size.values())[0]
        self.register_buffer("qmax_list", torch.ones(1) * 15)
        self.register_buffer("enable_qat", torch.zeros(1) > 0)
        self.scale_perm_single = []
        for i in range(4):
            self.scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])

        # 标记: buffer 是否已预分配
        self._buffers_ready = False

    def _calculate_scale_zero_dimensions(self, infeatures, outfeatures):
        group_size_list = list(self.group_size.values())
        total_scale_groups = 0
        for i, group_size in enumerate(group_size_list):
            if group_size == -1:
                total_scale_groups += 1
            else:
                total_scale_groups += infeatures // group_size
        return (total_scale_groups, outfeatures)

    def _get_tensor_reshape_and_groups(self, tensor_per_bits, group_size_val):
        tensor_shape = tensor_per_bits.size()
        if group_size_val == -1:
            reshaped_tensor = tensor_per_bits
            num_groups_per_channel = 1
        else:
            num_groups_per_channel = tensor_shape[1] // group_size_val
            reshaped_tensor = tensor_per_bits.view(-1, group_size_val)
        return reshaped_tensor, num_groups_per_channel

    def _compute_quantization_parameters(self, tensor, group_size_val, bit_width, symmetric):
        if group_size_val == -1:
            dim = 1
        else:
            dim = 1

        if bit_width > 1:
            if symmetric:
                qmax = (1 << (bit_width - 1)) - 1
                qmin = -(1 << (bit_width - 1))
                abs_max = tensor.abs().amax(dim=dim, keepdim=True)
                scale = abs_max / qmax
                zero_point = torch.full_like(scale, qmax)
            else:
                qmax = (1 << bit_width) - 1
                xmin = tensor.amin(dim=dim, keepdim=True)
                xmax = tensor.amax(dim=dim, keepdim=True)
                scale = (xmax - xmin) / qmax
                zero_point = -xmin
        else:
            scale = tensor.norm(p=1, dim=dim, keepdim=True) / tensor.size(dim)
            zero_point = -tensor.mean(dim=dim, keepdim=True)
        return scale, zero_point

    # ================================================================
    #  Pre-allocate inference buffers — 在权重加载后、推理前调用一次
    # ================================================================
    def prepare_inference_buffers(self):
        """
        预分配 forward 中反复使用的 scratch buffer.
        必须在权重加载到真实 device 后调用 (不能在 meta device 上).

        预分配的 buffer:
          - _C_buf:  int32 reduce buffer, shape 固定 (max_par*64, n)
          - _ws_buf: int32 workspace, shape 固定 (n//128 * max_par)
        这两个 shape 只依赖 outfeatures, 不依赖 batch size.

        D (output) 不预分配, 因为 m 随 batch 变化, 且 torch.empty 本身很快.
        """
        dev = self.qweight.device
        n = self.outfeatures
        max_par = 16

        # W4A4 reduce buffer + workspace
        self._C_buf = torch.zeros((16 * 4 * max_par, n), dtype=torch.int32, device=dev)
        self._ws_buf = torch.zeros(n // 128 * max_par, dtype=torch.int32, device=dev)

        self._buffers_ready = True

    def reinitialize(self):
        assert self.real_pack is False, "reinitialization is only available when real_pack is False"
        qscale_post = self.scales.view(self.scales.size(0), -1).half()
        qzeros_post = self.zeros.view(self.zeros.size(0), -1).half() if self.zeros is not None else None
        group_size_val = list(self.group_size.values())[0]
        if group_size_val == -1:
            scale = qscale_post.expand(self.infeatures, -1)
            zeros = qzeros_post.expand(self.infeatures, -1) if qzeros_post is not None else None
        else:
            scale = qscale_post.unsqueeze(1).repeat(1, group_size_val, 1).view(-1, self.qweight.size(-1))
            zeros = qzeros_post.unsqueeze(1).repeat(1, group_size_val, 1).view(-1, self.qweight.size(
                -1)) if qzeros_post is not None else None
        weights = self.qweight.mul(scale)
        if zeros is not None and not self.symmetric:
            weights = weights - zeros
        self.weights = nn.Parameter(weights)
        del self.qweight, self.scales, self.zeros
        self.enable_qat[0] = True

    def pack(self, linear, scale_post, zeros_post, bits_row_end_index=None, qmax_list=None):
        if linear.bias is not None:
            self.bias.data = linear.bias.data.clone()

        if hasattr(linear, 'act_quantizer') and linear.act_quantizer is not None:
            self.act_quantizer = linear.act_quantizer
        else:
            self.act_quantizer = None

        tensors = []
        quantized = []
        self.register_buffer("qmax_list", torch.Tensor(qmax_list))

        weight = linear.weight.data
        target_device = weight.device

        for i in range(len(bits_row_end_index)):
            if i == 0:
                tensors.append(weight[:, :bits_row_end_index[i]])
            else:
                tensors.append(weight[:, bits_row_end_index[i - 1]:bits_row_end_index[i]])

        for i, (tensor_per_bits, scale_per_bits, zeros_per_bits) in enumerate(zip(tensors, scale_post, zeros_post)):
            scale_per_bits = scale_per_bits.to(target_device)
            zeros_per_bits = zeros_per_bits.to(target_device)
            tensor_per_bits = tensor_per_bits.to(target_device)

            tensor_shape_per_bits = tensor_per_bits.size()
            group_size_val = list(self.group_size.values())[i]

            tensor_reshaped, num_groups_per_channel = self._get_tensor_reshape_and_groups(
                tensor_per_bits, group_size_val)

            if self.n_bits[i] > 1:
                if self.symmetric:
                    tensor_quantized = torch.round(tensor_reshaped / scale_per_bits)
                    qmin = -(1 << (self.n_bits[i] - 1))
                    qmax = (1 << (self.n_bits[i] - 1)) - 1
                    tensor_quantized = tensor_quantized.clamp(qmin, qmax)
                else:
                    tensor_quantized = tensor_reshaped + zeros_per_bits
                    tensor_quantized = torch.round(tensor_quantized / scale_per_bits)
                    tensor_quantized = tensor_quantized.clamp(0, self.qmax_list[i].item())
            else:
                tensor_quantized = tensor_reshaped + zeros_per_bits
                tensor_quantized = tensor_quantized.sign().add(1).div(2).clamp(0, 1)

            tensor_quantized = tensor_quantized.cpu()
            tensor_quantized = tensor_quantized.reshape(tensor_shape_per_bits).t().cpu().numpy().astype(np.int32)

            if self.symmetric and self.n_bits[i] > 1:
                n_bits_ceil = math.ceil(self.n_bits[i])
                mask = (1 << n_bits_ceil) - 1
                tensor_quantized = tensor_quantized & mask

            tensor_quantized = tensor_quantized.astype(np.uint32)
            quantized_tensor_per_bits = np.zeros(
                (tensor_shape_per_bits[1] // 32 * math.ceil(self.n_bits[i]), tensor_shape_per_bits[0]),
                dtype=np.int32)

            j = 0
            row = 0
            while row < quantized_tensor_per_bits.shape[0]:
                if self.n_bits[i] in [1, 1.5, 2, 4, 8]:
                    for k in range(j, j + (32 // math.ceil(self.n_bits[i]))):
                        quantized_tensor_per_bits[row] |= tensor_quantized[k] << (
                                    math.ceil(self.n_bits[i]) * (k - j))
                    j += 32 // math.ceil(self.n_bits[i])
                    row += 1
                else:
                    raise NotImplementedError("Only 1, 1.5, 2, 4, 8 bits are supported.")

            if self.real_pack:
                quantized_tensor_per_bits = quantized_tensor_per_bits.astype(np.int32)
                quantized.append(torch.from_numpy(quantized_tensor_per_bits))
            else:
                quantized.append(torch.from_numpy(tensor_quantized.astype(np.float16)))

        qweight = torch.cat(quantized, dim=0)
        self.qweight.data = qweight.data.T

        qscale_list = []
        qzeros_list = []
        for i, (scale_post_per_bits, zeros_post_per_bits) in enumerate(zip(scale_post, zeros_post)):
            group_size_val = list(self.group_size.values())[i]
            scale_post_per_bits = scale_post_per_bits.cpu()
            zeros_post_per_bits = zeros_post_per_bits.cpu()

            if group_size_val == -1:
                if scale_post_per_bits.dim() == 2:
                    scale_reshaped = scale_post_per_bits.t().contiguous()
                else:
                    scale_reshaped = scale_post_per_bits.unsqueeze(0)
                qscale_list.append(scale_reshaped.half().cpu())
                if self.symmetric:
                    qmax_val = self.qmax_list[i].item()
                    symmetric_zeros = torch.full_like(scale_reshaped, qmax_val // 2)
                    qzeros_list.append(symmetric_zeros.half().cpu())
                else:
                    if zeros_post_per_bits.dim() == 2:
                        zeros_reshaped = zeros_post_per_bits.t().contiguous()
                    else:
                        zeros_reshaped = zeros_post_per_bits.unsqueeze(0)
                    qzeros_list.append(zeros_reshaped.half().cpu())
            else:
                scale_reshaped = scale_post_per_bits.reshape(self.outfeatures, -1).t().contiguous()
                qscale_list.append(scale_reshaped.half().cpu())
                if self.symmetric:
                    qmax_val = self.qmax_list[i].item()
                    symmetric_zeros = torch.full_like(zeros_post_per_bits, qmax_val // 2)
                    zeros_reshaped = symmetric_zeros.reshape(self.outfeatures, -1).t().contiguous()
                    qzeros_list.append(zeros_reshaped.half().cpu())
                else:
                    zeros_reshaped = zeros_post_per_bits.reshape(self.outfeatures, -1).t().contiguous()
                    qzeros_list.append(zeros_reshaped.half().cpu())

        self.scales.data = torch.cat(qscale_list, dim=0).half()
        self.zeros.data = torch.cat(qzeros_list, dim=0).half()

        self.scales.data = self.scales.data.reshape((-1, len(self.scale_perm_single)))[:,
                           self.scale_perm_single].reshape(-1, self.outfeatures).contiguous()
        self.qweight.data = preprocess_to_int4(self.qweight.data.t().contiguous())

        linear = linear.cpu()
        del linear, scale_post, zeros_post, weight
        torch.cuda.empty_cache()
        gc.collect()

    def dynamic_quant(self, x: torch.Tensor):
        quant_scale = x.abs().max(dim=-1, keepdim=True)[0].div(127.0).to(torch.float32)
        x = (x / quant_scale).round().clamp(-128, 127).to(torch.int8)
        return x, quant_scale

    def _get_scale_zero_expansion(self, qscale_post, qzeros_post, group_size_val):
        if group_size_val == -1:
            scale = qscale_post
            zeros = qzeros_post if qzeros_post is not None else None
        else:
            scale = qscale_post.unsqueeze(1).repeat(1, group_size_val, 1).view(-1, qscale_post.size(-1))
            zeros = qzeros_post.unsqueeze(1).repeat(1, group_size_val, 1).view(-1, qzeros_post.size(
                -1)) if qzeros_post is not None else None
        return scale, zeros

    # ================================================================
    #  W4A4 forward  ★ Triton fused + pre-allocated buffers
    # ================================================================
    def forward_w4a4(self, x):
        size = x.size()
        x = x.view(-1, size[-1])
        m_orig = x.size(0)

        groupsize = self.act_group_size
        is_channel_wise = (groupsize == -1)

        # ---- Pad m ----
        ALIGN_QC = 64 if is_channel_wise else groupsize
        ALIGN_GEMM = 64
        ALIGN = (ALIGN_QC * ALIGN_GEMM) // math.gcd(ALIGN_QC, ALIGN_GEMM)

        if m_orig % ALIGN != 0:
            pad_rows = ALIGN - (m_orig % ALIGN)
            x = F.pad(x, (0, 0, 0, pad_rows))

        m = x.size(0)
        k = x.size(-1)

        # ---- Activation quantization + compression ----
        if is_channel_wise:
            if _HAS_TRITON:
                A, quant_scale = fused_channel_quant_compress(x)
            else:
                A, quant_scale = _channel_quant_compress_fallback(x)
        else:
            A = torch.zeros((m, k // 8), device=x.device, dtype=torch.int)
            quant_scale = torch.zeros((m, k // groupsize), device=x.device, dtype=torch.half)
            quantize_compress(x, A, quant_scale, groupsize)

        max_par = 16
        scales = self.scales
        B = self.qweight
        n = B.size(-2)

        # ---- 使用预分配 buffer 或 fallback ----
        if self._buffers_ready:
            C = self._C_buf
            workspace = self._ws_buf
        else:
            C = torch.zeros((16 * 4 * max_par, n), dtype=torch.int32, device=A.device)
            workspace = torch.zeros(n // 128 * 16, device=A.device)

        D = torch.empty((m, n), dtype=torch.half, device=A.device)

        if list(self.group_size.values())[0] == -1:
            w4a4_mul_C(A, B, C, D, quant_scale, scales, workspace,
                        thread_k=-1, thread_n=-1, sms=-1, max_par=max_par)
        else:
            w4a4_mul_G(A, B, C, D, quant_scale, scales, workspace,
                        thread_k=-1, thread_n=-1, sms=-1, max_par=max_par)

        D = D[:m_orig, :]
        D = D + self.bias if self.bias is not None else D
        D = D.view(*size[:-1], -1)
        return D

    def forward(self, x):
        return self.forward_w4a4(x)
