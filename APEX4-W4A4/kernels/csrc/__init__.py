# Modified by HandH1998
# Copyright (C) Marlin.2024 Elias Frantar (elias.frantar@ist.ac.at)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# W4A4 kernel Python wrappers. This package exposes exactly the two W4A4 GEMM
# variants we use (per-group and per-channel, half scales) plus the fused
# activation quantization/compression kernel.

import numpy as np
import torch
import torch.nn as nn

from APEX4._CUDA import apex4_gemm_w4a4_group    # W4A4 per-group, half
from APEX4._CUDA import apex4_gemm_w4a4_channel  # W4A4 per-channel, half
from APEX4._CUDA import quantize_compress_A      # fused activation quant + pack


def preprocess_to_int4(matrix: torch.Tensor):
    """Pack an [H, W] int matrix (H divisible by 4) into the kernel's
    [H//4, W, 4] int4-vector layout."""
    H, W = matrix.shape
    matrix_t = matrix.t().contiguous()           # [W, H]
    matrix_int4 = matrix_t.view(W, H // 4, 4)     # [W, H//4, 4]
    matrix_final = matrix_int4.transpose(0, 1).contiguous()  # [H//4, W, 4]
    return matrix_final


def quantize_compress(x_input, x_compressed, quant_scales, group_size):
    """Fused per-row 4-bit activation quantization + int32 packing (CUDA)."""
    quantize_compress_A(x_input, x_compressed, quant_scales, group_size)


def w4a4_mul_G(A, B, C, D, s1, s2, workspace,
               thread_k=-1, thread_n=-1, sms=-1, max_par=16):
    """W4A4 GEMM — activation per-group, weight per-group, half scales."""
    apex4_gemm_w4a4_group(A, B, C, D, s1, s2, workspace,
                          thread_k, thread_n, sms, max_par)


def w4a4_mul_CC_half(A, B, C, D, s1, s2, workspace,
                     thread_k=-1, thread_n=-1, sms=-1, max_par=16):
    """W4A4 GEMM — activation per-channel, weight per-channel, half scales."""
    apex4_gemm_w4a4_channel(A, B, C, D, s1, s2, workspace,
                            thread_k, thread_n, sms, max_par)


class W4A4Layer(nn.Module):
    """4-bit weight / 4-bit activation linear layer (no bias).

    Group mode (groupsize > 0) uses w4a4_mul_G with `s_group`;
    channel mode (groupsize == -1) uses w4a4_mul_CC_half with `s22_channel`.
    """

    def __init__(self, infeatures, outfeatures, groupsize=-1):
        super().__init__()
        if infeatures % 128 != 0 or outfeatures % 256 != 0:
            raise ValueError('`infeatures` must be divisible by 128 and `outfeatures` by 256.')
        if groupsize == -1:
            groupsize = infeatures
        self.k = infeatures
        self.n = outfeatures
        self.groupsize = groupsize
        self.max_par = 16
        self.register_buffer('B', torch.empty((self.k // 32, self.n, 4), dtype=torch.int))
        self.register_buffer(
            "s22_channel", torch.empty((1, self.n), dtype=torch.half))
        self.register_buffer(
            "s_group", torch.empty((self.k // self.groupsize, self.n), dtype=torch.half))
        self.register_buffer(
            "reduce_buffer",
            torch.zeros((self.max_par * 16 * 4, self.n), dtype=torch.int),
            persistent=False)
        # 128 is the minimum tile_n (max workspace); 16 is the default max_par.
        self.register_buffer(
            'workspace', torch.zeros(self.n // 128 * 16, dtype=torch.int), persistent=False)
        self._perm, self._scale_perm, self._scale_perm_single = self._get_perms()

    # activation 4-bit quantization
    def dynamic_quant(self, x: torch.Tensor):
        quant_scale = x.abs().max(dim=-1, keepdim=True)[0].div(127.0).to(torch.float)
        x = (x / quant_scale).round().clamp(-8, 7).to(torch.int8)
        return x, quant_scale

    def forward(self, A):
        out_shape = A.shape[:-1] + (self.n,)
        A = A.reshape(-1, A.shape[-1]).half()
        quant_A, s1 = self.dynamic_quant(A)
        D = torch.empty(A.shape[0], self.n, dtype=A.dtype, device=A.device)
        if self.groupsize == self.k:   # channel mode (groupsize was -1)
            w4a4_mul_CC_half(quant_A, self.B, self.reduce_buffer, D,
                             s1, self.s22_channel, self.workspace, max_par=self.max_par)
        else:                          # group mode
            w4a4_mul_G(quant_A, self.B, self.reduce_buffer, D,
                       s1, self.s_group, self.workspace, max_par=self.max_par)
        return D.reshape(out_shape)

    def _get_perms(self):
        perm = []
        for i in range(32):
            perm1 = []
            col = i // 4
            for block in [0, 1]:
                for row in [4 * (i % 4), 4 * (i % 4) + 1,
                            4 * (i % 4) + 2, 4 * (i % 4) + 3]:
                    perm1.append(16 * row + col + 8 * block)
            for j in range(4):
                perm.extend([p + 256 * j for p in perm1])
        perm = np.array(perm)
        interleave = np.array([4, 0, 5, 1, 6, 2, 7, 3])
        perm = perm.reshape((-1, 8))[:, interleave].ravel()
        perm = torch.from_numpy(perm)
        scale_perm = []
        for i in range(8):
            scale_perm.extend([i + 8 * j for j in range(8)])
        scale_perm_single = []
        for i in range(4):
            scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
        return perm, scale_perm, scale_perm_single

    def pack(self, linear, scales):
        """Pack a fake-quantized `torch.half` nn.Linear (+ scales of shape
        (infeatures, groups)) into this layer's W4A4 kernel format."""
        if linear.weight.dtype != torch.half:
            raise ValueError('Only `torch.half` weights are supported.')
        s = scales.t()                       # groups, n  (half)
        w = linear.weight.data.t()           # k, n
        if self.groupsize != -1:
            w = w.reshape((-1, self.groupsize, self.n))
            w = w.permute(1, 0, 2)
            w = w.reshape((self.groupsize, -1))
            s = s.reshape((1, -1))
        w = torch.round(w / s).int()         # quantize weights to 4-bit

        if self.groupsize != -1:
            w = w.reshape((self.groupsize, -1, self.n))
            w = w.permute(1, 0, 2)
            w = w.reshape((self.k, self.n)).contiguous()
        s2 = s.reshape((-1, len(self._scale_perm_single)))[:, self._scale_perm_single]
        s2 = s2.reshape((-1, self.n)).contiguous()

        res = w.t()                          # n, k
        q = np.zeros((res.shape[0], res.shape[1] // 8), dtype=np.uint32)
        res = res.cpu().numpy().astype(np.uint32)
        for i in range(8):
            q |= (res[:, i::8] & 0xF) << 4 * i
        q = torch.from_numpy(q.astype(np.int32)).to(w.device)
        q = q.t()                            # k//8, n
        q1 = preprocess_to_int4(q)
        self.B[:, :] = q1.to(self.B.device)

        if self.groupsize != -1:
            self.s_group[:, :] = s2.to(self.s_group.device)
        else:
            self.s22_channel[:, :] = s2.to(self.s22_channel.device)
