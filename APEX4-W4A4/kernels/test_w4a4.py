import unittest

import numpy as np
import torch
import torch.nn as nn
import csrc
import time

import csv
import os
from datetime import datetime
import pandas as pd
seed = 0
np.random.seed(seed)
torch.random.manual_seed(seed)

DEV = torch.device('cuda:0')
# W4A4 kernel correctness/perf test. Weights are offline-preprocessed into the
# int4-vector layout. Only two kernels are supported in this release:
#   - w4a4_mul_G        : activation per-group,   weight per-group,   half scales
#   - w4a4_mul_CC_half  : activation per-channel, weight per-channel, half scales

CSV_FILE = 'A100_w4a4_results.csv'
GROUPSIZE_VALUES = [-1, 32, 64, 128, 256, 512, 1024]


def generate_csv_headers():
    base_headers = ['m', 'n', 'k', 'thread_k', 'thread_n']
    groupsize_headers = [f'groupsize={gs}（ms）' for gs in GROUPSIZE_VALUES]
    return base_headers + groupsize_headers


CSV_HEADERS = generate_csv_headers()

def init_csv_file():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)
        print(f"Created CSV file: {CSV_FILE}")
    else:
        print(f"Using existing CSV file: {CSV_FILE}")


def save_to_csv(m, n, k, thread_k, thread_n, groupsize, avg_time_ms):
    try:
        if os.path.exists(CSV_FILE):
            df = pd.read_csv(CSV_FILE)
        else:
            df = pd.DataFrame(columns=CSV_HEADERS)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        df = pd.DataFrame(columns=CSV_HEADERS)

    if len(df) > 0:
        mask = (
                (df['m'] == m) &
                (df['n'] == n) &
                (df['k'] == k) &
                (df['thread_k'] == thread_k) &
                (df['thread_n'] == thread_n)
        )
        matching_rows = df[mask]

        if len(matching_rows) > 0:
            row_idx = matching_rows.index[0]
            groupsize_col = f'groupsize={groupsize}（ms）'
            df.loc[row_idx, groupsize_col] = round(avg_time_ms, 3)
        else:
            new_row = create_new_row(m, n, k, thread_k, thread_n, groupsize, avg_time_ms)
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        new_row = create_new_row(m, n, k, thread_k, thread_n, groupsize, avg_time_ms)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_csv(CSV_FILE, index=False, encoding='utf-8')
    print(f"Results saved to {CSV_FILE} - groupsize={groupsize}: {avg_time_ms:.3f}ms")


def create_new_row(m, n, k, thread_k, thread_n, groupsize, avg_time_ms):
    new_row = {
        'm': m,
        'n': n,
        'k': k,
        'thread_k': thread_k,
        'thread_n': thread_n
    }

    for gs in GROUPSIZE_VALUES:
        groupsize_col = f'groupsize={gs}（ms）'
        new_row[groupsize_col] = ''
    current_groupsize_col = f'groupsize={groupsize}（ms）'
    new_row[current_groupsize_col] = round(avg_time_ms, 3)

    return new_row

def compress_4bit_to_int32(A):
    m, k = A.shape
    compressed_k = (k + 7) // 8

    compressed = torch.zeros((m, compressed_k), dtype=torch.int32, device=A.device)

    full_groups = k // 8
    if full_groups > 0:

        A_reshaped = A[:, :full_groups * 8].reshape(m, full_groups, 8) & 0xF

        A_int32 = A_reshaped.to(torch.int32)

        compressed[:, :full_groups] = (
                A_int32[:, :, 0] |
                (A_int32[:, :, 1] << 4) |
                (A_int32[:, :, 2] << 8) |
                (A_int32[:, :, 3] << 12) |
                (A_int32[:, :, 4] << 16) |
                (A_int32[:, :, 5] << 20) |
                (A_int32[:, :, 6] << 24) |
                (A_int32[:, :, 7] << 28)
        )

    remainder = k % 8
    if remainder > 0:
        last_chunk = (A[:, -remainder:] & 0xF).to(torch.int32)

        last_col = torch.zeros((m,), dtype=torch.int32, device=A.device)

        for i in range(remainder):
            last_col |= (last_chunk[:, i] << (4 * i))

        compressed[:, -1] = last_col

    return compressed
def group_scale_matrix(A, s1):
    m, k = A.shape
    _, group = s1.shape

    assert k % group == 0, f"k ({k}) 必须是 group ({group}) 的整数倍"

    group_size = k // group

    A_reshaped = A.reshape(m, group, group_size)

    s1_expanded = s1.unsqueeze(-1)

    scaled_A_reshaped = A_reshaped * s1_expanded

    scaled_A = scaled_A_reshaped.reshape(m, k)
    scaled_A = scaled_A.to(torch.half)

    return scaled_A
def gen_quant4(m, n, groupsize=-1):
    tile = 16
    maxq = 2 ** 4 - 1                                                       # 0-15
    w = torch.randn((m, n), dtype=torch.half, device=DEV)                  # k n

    if groupsize != -1:
        w = w.reshape((-1, groupsize, n))                                   # k n -> k//groupsize groupsize n
        w = w.permute(1, 0, 2)                                              # groupsize, k//groupsize, n
        w = w.reshape((groupsize, -1))                                      # groupsize, k*n//groupsize
    s = torch.max(torch.abs(w), 0, keepdim=True)[0]                         # 1, k*n//groupsize      [1, k//groupsize, n] -> [k//groupsize, 1, n]
    # print('s',s)
    s *= 2 / maxq                                                           # -8, 7
    # print('s',s)
    w = torch.round(w / s).int()
    w += (maxq + 1) // 2
    # print('w', w)
    w = torch.clamp(w, 0, maxq)
    # print('w', w)
    ref = (w - (maxq + 1) // 2).half() * s

    if groupsize != -1:
        def reshape(w):
            w = w.reshape((groupsize, -1, n))
            w = w.permute(1, 0, 2)
            w = w.reshape((m, n)).contiguous()
            return w
        ref = reshape(ref)

    s = s.reshape((-1, n)).contiguous()             #k//groupsize, n

    linear = nn.Linear(m, n)
    linear.weight.data = ref.t()            #n k

    fake_quant_ref = ref

    layer = csrc.W4A4Layer(256, 256, groupsize=groupsize)

    layer.k = m
    layer.n = n
    layer.groupsize = groupsize
    layer.B = torch.empty((m//32, n, 4), dtype=torch.int, device=DEV)
    if groupsize != -1:
        layer.s_group = torch.empty((m // groupsize, n), dtype=torch.half, device=DEV)
    else:
        layer.s22_channel = torch.empty((1, n), dtype=torch.half, device=DEV)

    layer.pack(linear, s.t())

    q = layer.B
    if groupsize != -1:
        s2 = layer.s_group
    else:
        s2 = layer.s22_channel

    print()
    return ref, q, fake_quant_ref, s2
class Test(unittest.TestCase):

    def setUp(self):
        """Initialize the CSV file before the start of each test."""
        init_csv_file()

    def run_problem(self, m, n, k, thread_k, thread_n, groupsize=-1):
        print('% 5d % 6d % 6d % 4d % 4d % 4d' % (m, n, k, thread_k, thread_n, groupsize))

        A = torch.randint(-8, 8, (m, k), dtype=torch.int8, device=DEV) #one int stores eight 4-bit values.
        A_compressed = compress_4bit_to_int32(A)

        if groupsize == -1:
            s1_ref = torch.randn((m, 1), dtype=torch.half, device=DEV)  # channel s1 half
        else:
            s1_ref = torch.randn((m, k // groupsize), dtype=torch.half, device=DEV)  # group s1 half

        s1 = s1_ref
        # print('s1_ref:', s1_ref)
        B_ref, B, fake_quant_B, s22 = gen_quant4(k, n, groupsize=groupsize)
        max_par = 16
        C = torch.zeros((16 * 4 * max_par, n), dtype=torch.int32, device=DEV)
        D = torch.zeros((m, n), dtype=torch.half, device=DEV)

        result = group_scale_matrix(A, s1_ref)
        # print('A*s1 result', result)

        D_ref = torch.matmul(result, fake_quant_B)

        workspace = torch.zeros(n // 128 * 16, device=DEV)

        # Warm-up run (to avoid the overhead of the first invocation)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        print("Warm-up run...")
        for _ in range(5):  #
            csrc.w4a4_mul_G(A_compressed, B, C, D, s1, s22, workspace, thread_k, thread_n, -1, max_par=max_par)  # a group, w group, half
            # csrc.w4a4_mul_CC_half(A_compressed, B, C, D, s1, s22, workspace, thread_k, thread_n, -1, max_par=max_par)  # a channel, w channel, half

            if torch.cuda.is_available():
                torch.cuda.synchronize()

        # Begin the official test.
        print("Begin the official test....")
        num_runs = 100
        total_time = 0.0
        times = []

        for i in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()

            csrc.w4a4_mul_G(A_compressed, B, C, D, s1, s22, workspace, thread_k, thread_n, -1, max_par=max_par) # w4a4 a group, w group, half
            # csrc.w4a4_mul_CC_half(A_compressed, B, C, D, s1, s22, workspace, thread_k, thread_n, -1, max_par=max_par)  # w4a4 a channel, w channel, half

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end_time = time.time()

            elapsed_time_ms = (end_time - start_time) * 1000
            times.append(elapsed_time_ms)
            total_time += elapsed_time_ms

            if (i + 1) % 20 == 0:
                print(f"finish {i + 1}/{num_runs} test")

        # 计算统计信息
        avg_time_ms = total_time / num_runs
        min_time_ms = min(times)
        max_time_ms = max(times)
        std_time_ms = (sum((t - avg_time_ms) ** 2 for t in times) / num_runs) ** 0.5

        print(f"\n=== W4A4 Matrix Multiplication Performance Test Results ===")
        print(f"Number of tests: {num_runs}")
        print(f"Average time: {avg_time_ms:.3f} ms")
        print(f"Minimum time: {min_time_ms:.3f} ms")
        print(f"Maximum time: {max_time_ms:.3f} ms")
        print(f"Standard deviation:   {std_time_ms:.3f} ms")
        print(f"Coefficient of variation: {(std_time_ms / avg_time_ms * 100):.2f}%")


        save_to_csv(m, n, k, thread_k, thread_n, groupsize, avg_time_ms)
        torch.cuda.synchronize()
        self.assertLess(torch.mean(torch.abs(D - D_ref)) / torch.mean(torch.abs(D_ref)), 0.003)

    def test_tiles(self):
        print()
        for m in [1, 2, 3, 4, 8, 12, 16, 24, 32, 48, 64, 118, 128, 152, 768, 1024]:
            for thread_k, thread_n in [(64, 256), (128, 128)]:
                if m > 16 and thread_k == 128:
                    continue
                self.run_problem(m, 2 * 256, 1024, thread_k, thread_n)

    def test_k_stages_divisibility(self):
        print()
        for k in [3 * 64 + 64 * 4 * 2 + 64 * i for i in range(1, 4)]:
            self.run_problem(16, 2 * 256, k, 64, 256)

    def test_very_few_stages(self):
        print()
        for k in [64, 128, 192]:
            self.run_problem(16, 2 * 256, k, 64, 256)

    def test_llama_shapes(self):
        print()
        MODELS = {
            ' 7B': [
                (4096, 3 * 4096),
                (4096, 4096),
                (4096, 2 * 10752),
                (10752, 4096)
            ],
            '13B': [
                (5120, 3 * 5120),
                (5120, 5120),
                (5120, 2 * 13568),
                (13568, 5120)
            ],
            '33B': [
                (6656, 3 * 6656),
                (6656, 6656),
                (6656, 2 * 17664),
                (17664, 6656)
            ],
            '70B': [
                (8192, 3 * 8192),
                (8192, 8192),
                (8192, 2 * 21760),
                (21760, 8192)
            ]
        }
        for _, layers in MODELS.items():
            for layer in layers:
                for thread_k, thread_n in [(128, 128)]:
                    for batch in [1, 16]:
                        self.run_problem(batch, layer[1], layer[0], thread_k, thread_n, 128)

    def test_groups(self):
        print()
        for m in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
            for groupsize in [32,64,128,256,512,1024]: #-1 32,64,128,256,512,1024
                for n, k in [(8192,8192)]:
                    for thread_shape in [(128, 256)]:
                        self.run_problem(m, n, k, *thread_shape, groupsize)

if __name__ == '__main__':
    unittest.main()
