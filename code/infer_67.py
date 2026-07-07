import sys
import time
import os

# 获取当前环境脚本所在目录或指定绝对路径
# if os.path.exists("../libraries"):
#     lib_path = os.path.abspath("../libraries")
#     sys.path.append(lib_path)

from pathlib import Path

cur_dir = Path(__file__).resolve().parent       # /home/aistudio/code
lib_path = cur_dir.parent / "libraries"         # /home/aistudio/libraries

if lib_path.exists():
    lib_path = str(lib_path)
    if lib_path not in sys.path:
        sys.path.insert(0, lib_path)

import math
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

DEFAULT_VOCAB_SIZE = 5000000
USE_EMBEDDING_BAG_REP_ENCODER = os.environ.get("DISABLE_EMBEDDING_BAG_REP_ENCODER") != "1"
PIN_BATCH_IN_COLLATE = os.environ.get("DISABLE_PIN_BATCH") != "1"
USE_DENSE_ALL_SMOE = True

# ============================================================
# 数据加载（来自 train/dataset.py）
# ============================================================

def _detect_has_clk(file_path):
    """检测 CSV 文件是否包含 clk 列（5列 vs 4列格式）。
    5列格式: logid,userid,adid,clk,timestamp,sign:slot...
    4列格式: logid,userid,adid,timestamp,sign:slot...
    通过第5个字段是否包含 ':' 来判断：有 ':' 说明已经是 sign:slot，即无 clk 列。
    """
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) >= 5:
                return ':' not in parts[4]
            return False
    return False


def load_sample_files(sample_files_list):
    """加载 CSV sample 文件，返回 item_dict 和 user_seq。
    自动检测每个文件是 5列（含clk）还是 4列（无clk）格式。
    """
    sample_files = sorted([Path(f) for f in sample_files_list])
    print(f'[INFO] loading {len(sample_files)} files: {[str(f) for f in sample_files]}')

    item_dict = {}
    user_logs = defaultdict(list)

    for sample_file in tqdm(sample_files, desc='Loading sample files'):
        has_clk = _detect_has_clk(sample_file)
        min_parts = 5 if has_clk else 4
        print(f'  {sample_file.name}: has_clk={has_clk}')

        with open(sample_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) < min_parts:
                    continue

                logid = int(parts[0])
                userid = int(parts[1])
                adid = int(parts[2])

                if has_clk:
                    clk = int(parts[3])
                    timestamp = int(parts[4])
                    feat_start = 5
                else:
                    clk = 0
                    timestamp = int(parts[3])
                    feat_start = 4

                signs = []
                slots = []
                for pair in parts[feat_start:]:
                    if ':' in pair:
                        s, sl = pair.split(':', 1)
                        signs.append(int(s))
                        slots.append(int(sl))

                item_dict[logid] = {
                    'logid': logid,
                    'userid': userid,
                    'adid': adid,
                    'clk': clk,
                    'timestamp': timestamp,
                    'signs': np.array(signs, dtype=np.int64),
                    'slots': np.array(slots, dtype=np.int64),
                }
                user_logs[userid].append((timestamp, logid))

    user_seq = {}
    for userid, logs in user_logs.items():
        logs.sort(key=lambda x: x[0])
        user_seq[userid] = [logid for _, logid in logs]

    print(f'[INFO] loaded {len(item_dict)} records, {len(user_seq)} users')
    return item_dict, user_seq


def load_logids_from_file(file_path):
    """快速读取一个 sample 文件中的所有 logid"""
    logids = set()
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            comma = line.index(',')
            logids.add(int(line[:comma]))
    return logids


class CTRUserDataset(Dataset):
    """按用户组织的 CTR 数据集"""

    def __init__(self, item_dict, user_seq=None, max_feasign_per_slot=None, pred_logids=None):
        super().__init__()
        self.item_dict = item_dict
        self.user_seq = user_seq if user_seq else {}
        self.max_feasign_per_slot = max_feasign_per_slot
        self.pred_logids = pred_logids if pred_logids is not None else set()

        self.user_items = defaultdict(list)
        for logid, rec in item_dict.items():
            userid = rec['userid']
            feasign = defaultdict(list)
            for slot, sign in zip(rec['slots'].tolist(), rec['signs'].tolist()):
                feasign[slot].append(sign)
            if max_feasign_per_slot is not None:
                feasign = {slot: signs[:max_feasign_per_slot[slot]]
                           if max_feasign_per_slot.get(slot, -1) != -1 else signs
                           for slot, signs in feasign.items()}
            feasign = dict(feasign)
            label = rec['clk']
            self.user_items[userid].append((logid, feasign, label))

        self.user_ids = sorted(self.user_items.keys())
        self.num_users = len(self.user_ids)
        self.total_samples = len(item_dict)

        all_signs = set()
        for rec in item_dict.values():
            all_signs.update(rec['signs'].tolist())
        self.max_slot_id = 28
        self.max_sign_id = max(all_signs) if all_signs else 0

    def __len__(self):
        return self.num_users

    def __getitem__(self, index):
        userid = self.user_ids[index]
        items = self.user_items[userid]

        if self.user_seq and userid in self.user_seq:
            seq_order = {logid: i for i, logid in enumerate(self.user_seq[userid])}
            items.sort(key=lambda x: seq_order.get(x[0], x[0]))
        else:
            items.sort(key=lambda x: x[0])

        feasigns = []
        labels = []
        logids = []
        for logid, feasign, label in items:
            logids.append(logid)
            feasigns.append(feasign)
            labels.append(label)

        return {
            'userid': userid,
            'logids': logids,
            'feasigns': feasigns,
            'labels': labels,
            'pred_mask': [1 if logid in self.pred_logids else 0 for logid in logids],
        }


def make_collate_fn(max_slot_id):
    def collate_user_batch(batch):
        all_userids = []
        all_logids = []
        all_labels = []
        all_pred_masks = []
        all_feasigns = []
        user_offsets = [0]

        for item in batch:
            for i, logid in enumerate(item['logids']):
                all_userids.append(item['userid'])
                all_logids.append(logid)
                all_labels.append(item['labels'][i])
                all_pred_masks.append(item['pred_mask'][i])
                all_feasigns.append(item['feasigns'][i])
            user_offsets.append(len(all_labels))

        slot_data = {}
        for slot in range(1, max_slot_id + 1):
            values = []
            offsets = [0]
            for feasign in all_feasigns:
                if slot in feasign:
                    values.extend(feasign[slot])
                offsets.append(len(values))
            value_tensor = torch.tensor(values, dtype=torch.long)
            if value_tensor.numel() > 0:
                value_tensor.clamp_(0, DEFAULT_VOCAB_SIZE - 1)
            slot_data[slot] = (
                value_tensor,
                torch.tensor(offsets, dtype=torch.long),
            )

        result = {
            'userid': torch.tensor(all_userids, dtype=torch.long),
            'logid': torch.tensor(all_logids, dtype=torch.long),
            'label': torch.tensor(all_labels, dtype=torch.float32),
            'pred_mask': torch.tensor(all_pred_masks, dtype=torch.bool),
            'user_offsets': torch.tensor(user_offsets, dtype=torch.long),
            '_values_preclamped': True,
        }
        result.update(slot_data)
        if PIN_BATCH_IN_COLLATE:
            result = pin_batch_memory(result)
        return result

    return collate_user_batch


# ============================================================
# 模型定义（来自 main.py）
# ============================================================

def pin_batch_memory(batch):
    """递归 pin CPU tensor，配合 non_blocking H2D 拷贝使用。"""
    if not torch.cuda.is_available():
        return batch

    if isinstance(batch, dict):
        return {k: pin_batch_memory(v) for k, v in batch.items()}
    elif isinstance(batch, tuple):
        return tuple(pin_batch_memory(v) for v in batch)
    elif isinstance(batch, list):
        return [pin_batch_memory(v) for v in batch]
    elif torch.is_tensor(batch):
        if batch.device.type == "cpu" and not batch.is_pinned():
            try:
                return batch.pin_memory()
            except RuntimeError:
                return batch
        return batch
    else:
        return batch


def move_batch_to_device(batch, device):
    if isinstance(batch, dict):
        moved = {k: move_batch_to_device(v, device) for k, v in batch.items()}

        values_preclamped = bool(moved.get("_values_preclamped", False))
        if not values_preclamped:
            for slot in range(1, 29):
                slot_value = moved.get(slot)
                if slot_value is not None and slot_value[0].numel() > 0:
                    slot_value[0].clamp_(0, DEFAULT_VOCAB_SIZE - 1)
            moved["_values_preclamped"] = True

        return moved
    elif isinstance(batch, tuple):
        return tuple(move_batch_to_device(x, device) for x in batch)
    elif isinstance(batch, list):
        return [move_batch_to_device(x, device) for x in batch]
    elif torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    else:
        return batch


def prepare_cached_batch_for_current_collate(batch):
    """兼容旧 batch cache：补齐 sign clamp 和 pinned memory。"""
    if not isinstance(batch, dict):
        return batch

    if not batch.get("_values_preclamped", False):
        for slot in range(1, 29):
            slot_value = batch.get(slot)
            if slot_value is not None and slot_value[0].numel() > 0:
                slot_value[0].clamp_(0, DEFAULT_VOCAB_SIZE - 1)
        batch["_values_preclamped"] = True

    if PIN_BATCH_IN_COLLATE:
        batch = pin_batch_memory(batch)

    return batch


# RepEncoder 把每条 log 的 28 个 slot 稀疏 sign 特征，
# 通过 embedding lookup 和 slot 内 sum pooling，
# 变成一个固定的 d_model 维 dense 向量。

# 每个 sign 查成 512 维
# 每个 slot 内多个 sign embedding 求和
# 28 个 slot 拼成 14336 维
# LayerNorm
# Linear 压缩到 512 维
class RepEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, padding_idx=0, slot_num=0, d_model=0):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim, padding_idx=padding_idx)
        self.emb_dim = emb_dim
        self.slot_num = slot_num
        self.input_norm = nn.LayerNorm(slot_num * emb_dim)
        self.linear = nn.Linear(in_features=slot_num * emb_dim, out_features=d_model)

    # def forward(self, batch):
    #     pooled_embs = []
    #     max_idx = self.emb.num_embeddings - 1
    #     for i in range(self.slot_num):
    #         values, offsets = batch[i + 1]
    #         offsets = offsets.to(values.device)
    #         values = values.clamp(0, max_idx)  # 超出 vocab_size 的 sign id 截断，避免越界
    #         sign_emb = self.emb(values)
    #         res = torch.segment_reduce(sign_emb, reduce='sum', offsets=offsets, initial=0)
    #         pooled_embs.append(res)
    #     fused_embs = torch.cat(pooled_embs, dim=1)
    #     norm_emb = self.input_norm(fused_embs)
    #     rep_emb = self.linear(norm_emb)
    #     return rep_emb
    def forward(self, batch):
        pooled_embs = []
        max_idx = self.emb.num_embeddings - 1
        values_preclamped = batch.get("_values_preclamped", False)

        for i in range(self.slot_num):
            values, offsets = batch[i + 1]
            offsets = offsets.to(values.device)

            if not values_preclamped:
                values = values.clamp(0, max_idx)

            if USE_EMBEDDING_BAG_REP_ENCODER:
                if values.numel() > 0:
                    res = F.embedding_bag(
                        values,
                        self.emb.weight,
                        offsets,
                        mode='sum',
                        include_last_offset=True,
                    )
                else:
                    res = torch.zeros(
                        offsets.numel() - 1,
                        self.emb_dim,
                        device=offsets.device,
                        dtype=self.emb.weight.dtype,
                    )
            else:
                if values.numel() > 0:
                    sign_emb = self.emb(values)
                    res = torch.segment_reduce(
                        sign_emb,
                        reduce='sum',
                        offsets=offsets,
                        initial=0,
                    )
                else:
                    res = torch.zeros(
                        offsets.numel() - 1,
                        self.emb_dim,
                        device=offsets.device,
                        dtype=self.emb.weight.dtype,
                    )

            pooled_embs.append(res)

        fused_embs = torch.cat(pooled_embs, dim=1)
        norm_emb = self.input_norm(fused_embs)
        rep_emb = self.linear(norm_emb)
        return rep_emb


# def scaled_dot_product(q, k, v, extension):
#     d = q.size(-1)
#     scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
#     if extension is not None and "mask" in extension:
#         mask = extension["mask"]
#         scores = scores.masked_fill(mask == 0, float("-inf"))
#     attn = torch.softmax(scores, dim=-1)
#     out = torch.matmul(attn, v)
#     return out

def scaled_dot_product(q, k, v, extension):
    """
    SDPA 版本的 dense attention fallback。

    q, k, v: [B, H, S, Dh]
    extension["mask"]: [B, H, S, S] 或可 broadcast 到该形状
    """
    attn_mask = None

    if extension is not None and "mask" in extension:
        # 原代码里 mask == 1 表示允许看，mask == 0 表示禁止看。
        # PyTorch SDPA 的 bool mask 里 True 表示允许参与 attention。
        attn_mask = extension["mask"].bool()

    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=False,
    )


def block_causal_attention(q, k, v, extension):
    """
    q, k, v: [B, H, S, Dh]
    extension["user_offsets"]: 每个用户在展平序列中的边界，例如 [0, 3, 5, 10]

    作用：
    按用户切块做 causal attention，避免构造完整 [S, S] mask，
    也避免计算不同用户之间的无效 attention。
    """
    user_offsets = None
    if extension is not None:
        user_offsets = extension.get("user_offsets", None)

    # 如果没传 user_offsets，就退回原始 dense attention
    if user_offsets is None:
        return scaled_dot_product(q, k, v, extension)

    out = torch.empty_like(q)

    for start, end in zip(user_offsets[:-1], user_offsets[1:]):
        if end <= start:
            continue

        q_i = q[:, :, start:end, :]
        k_i = k[:, :, start:end, :]
        v_i = v[:, :, start:end, :]

        out[:, :, start:end, :] = F.scaled_dot_product_attention(
            q_i,
            k_i,
            v_i,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        )
        
    if not hasattr(block_causal_attention, "_printed"):
        print("[INFO] SDPA q dtype:", q.dtype)
        print("[INFO] SDPA q shape:", q.shape)
        block_causal_attention._printed = True
    return out


class Expert(nn.Module):
    def __init__(self, d_model, dim_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, dim_ff)
        self.fc2 = nn.Linear(dim_ff, d_model)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class TopKGate(nn.Module):
    def __init__(self, d_model, num_experts, k=2, noisy_gating=True):
        super().__init__()
        self.w_g = nn.Linear(d_model, num_experts)
        self.num_experts = num_experts
        self.k = k
        self.noisy_gating = noisy_gating

    def forward(self, x):
        # x: [B,S,D]
        logits = self.w_g(x)  # [B,S,E]

        if self.noisy_gating and self.training:
            logits = logits + torch.randn_like(logits) * 0.1

        if self.training:
            probs = torch.softmax(logits, dim=-1)  # [B,S,E]
            topk_score, topk_idx = torch.topk(probs, self.k, dim=-1)
            return topk_idx, topk_score, probs
        else:
            # 推理路径：不构造完整 probs
            topk_logits, topk_idx = torch.topk(logits, self.k, dim=-1)
            log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
            topk_score = torch.exp(topk_logits - log_z)
            return topk_idx, topk_score, None

class SMoE(nn.Module):
    def __init__(self, d_model, dim_ff, num_experts, k=2):
        super().__init__()
        self.num_experts = num_experts
        self.k = k

        self.experts = nn.ModuleList([
            Expert(d_model, dim_ff) for _ in range(num_experts)
        ])

        self.gate = TopKGate(d_model, num_experts, k=k)
    def prepare_dense_all(self):
        """
        将 8 个 expert 的 fc1/fc2 权重堆叠起来，供推理阶段 dense-all 路径使用。

        原 expert:
        fc1: D -> F
        fc2: F -> D

        堆叠后:
        _dense_all_w1: [E, D, F]
        _dense_all_b1: [E, F]
        _dense_all_w2: [E, F, D]
        _dense_all_b2: [E, D]
        """
        w1 = torch.stack(
            [expert.fc1.weight.detach().t().contiguous() for expert in self.experts],
            dim=0,
        )
        b1 = torch.stack(
            [expert.fc1.bias.detach() for expert in self.experts],
            dim=0,
        )

        w2 = torch.stack(
            [expert.fc2.weight.detach().t().contiguous() for expert in self.experts],
            dim=0,
        )
        b2 = torch.stack(
            [expert.fc2.bias.detach() for expert in self.experts],
            dim=0,
        )

        self.register_buffer("_dense_all_w1", w1, persistent=False)
        self.register_buffer("_dense_all_b1", b1, persistent=False)
        self.register_buffer("_dense_all_w2", w2, persistent=False)
        self.register_buffer("_dense_all_b2", b2, persistent=False)

    def _forward_dense_all(self, x, topk_idx, topk_score):
        """
        x:          [B, S, D]
        topk_idx:   [B, S, k]
        topk_score: [B, S, k]

        返回:
        out: [B, S, D]
        """
        B, S, D = x.shape
        x_flat = x.reshape(-1, D)       # [N, D]
        n_tokens = x_flat.shape[0]      # N = B * S

        # 第一层 expert MLP:
        # x_flat: [N, D]
        # expand 后: [E, N, D]
        # _dense_all_w1: [E, D, F]
        # 输出 h: [E, N, F]
        h = torch.baddbmm(
            self._dense_all_b1[:, None, :].expand(-1, n_tokens, -1),
            x_flat.unsqueeze(0).expand(self.num_experts, -1, -1),
            self._dense_all_w1,
        )

        h = F.relu(h)

        # 第二层 expert MLP:
        # h: [E, N, F]
        # _dense_all_w2: [E, F, D]
        # 输出 y: [E, N, D]
        y = torch.baddbmm(
            self._dense_all_b2[:, None, :].expand(-1, n_tokens, -1),
            h,
            self._dense_all_w2,
        )

        # [E, N, D] -> [N, E, D]
        y = y.permute(1, 0, 2).contiguous()

        # 根据 topk_idx 取出每个 token 的 top-k expert 输出
        # topk_idx.reshape(n_tokens, k): [N, k]
        # gather index: [N, k, D]
        selected = y.gather(
            1,
            topk_idx.reshape(n_tokens, self.k).unsqueeze(-1).expand(-1, -1, D),
        )  # [N, k, D]

        # top-k expert 输出加权求和
        out = (
            selected * topk_score.reshape(n_tokens, self.k).unsqueeze(-1)
        ).sum(dim=1)  # [N, D]

        return out.reshape(B, S, D)

    def forward(self, x):
        # x: [B, S, D]
        B, S, D = x.shape

        topk_idx, topk_score, probs = self.gate(x)

        # dense-all 推理路径
        if (
            USE_DENSE_ALL_SMOE
            and not self.training
            and hasattr(self, "_dense_all_w1")
        ):
            if not hasattr(self, "_printed_dense_all"):
                print("[INFO] SMoE using dense-all path")
                self._printed_dense_all = True
            out = self._forward_dense_all(x, topk_idx, topk_score)
            moe_loss = x.new_zeros(())
            return out, moe_loss

        # fallback：原来的 sparse MoE 路径
        out = torch.zeros_like(x)

        x_flat = x.reshape(-1, D)                # [B*S, D]
        idx_flat = topk_idx.reshape(-1, self.k)  # [B*S, k]
        score_flat = topk_score.reshape(-1, self.k)
        out_flat = out.reshape(-1, D)

        for i in range(self.num_experts):
            mask = (idx_flat == i)  # [B*S, k]

            # 不要用 if not mask.any()，这个可能触发同步
            token_idx, k_idx = mask.nonzero(as_tuple=True)
            if token_idx.numel() == 0:
                continue

            selected_x = x_flat[token_idx]  # [N_i, D]
            expert_out = self.experts[i](selected_x)  # [N_i, D]
            weight = score_flat[token_idx, k_idx].unsqueeze(-1)

            # 这里保留原写法，不用 index_add_，因为专家记录里 index_add_ 全量变慢
            out_flat[token_idx] += expert_out * weight

        if self.training:
            importance = probs.sum(dim=(0, 1))
            moe_loss = importance.std() / (importance.mean() + 1e-6)
        else:
            moe_loss = x.new_zeros(())

        return out, moe_loss


    #profile:

    # def forward(self, x):
    #     profile = True

    #     if profile and torch.cuda.is_available():
    #         torch.cuda.synchronize()
    #         t0 = time.time()

    #     B, S, D = x.shape
    #     topk_idx, topk_score, probs = self.gate(x)

    #     if profile and torch.cuda.is_available():
    #         torch.cuda.synchronize()
    #         t1 = time.time()

    #     out = torch.zeros_like(x)

    #     x_flat = x.reshape(-1, D)
    #     idx_flat = topk_idx.reshape(-1, self.k)
    #     score_flat = topk_score.reshape(-1, self.k)
    #     out_flat = out.reshape(-1, D)

    #     if profile and torch.cuda.is_available():
    #         torch.cuda.synchronize()
    #         t2 = time.time()

    #     route_time = 0.0
    #     expert_time = 0.0
    #     scatter_time = 0.0
    #     token_counts = []

    #     for i in range(self.num_experts):
    #         if profile and torch.cuda.is_available():
    #             torch.cuda.synchronize()
    #             r0 = time.time()

    #         mask = (idx_flat == i)
    #         token_idx, k_idx = mask.nonzero(as_tuple=True)

    #         if token_idx.numel() == 0:
    #             continue

    #         selected_x = x_flat[token_idx]
    #         weight = score_flat[token_idx, k_idx].unsqueeze(-1)

    #         if profile and torch.cuda.is_available():
    #             torch.cuda.synchronize()
    #             r1 = time.time()

    #         expert_out = self.experts[i](selected_x)

    #         if profile and torch.cuda.is_available():
    #             torch.cuda.synchronize()
    #             r2 = time.time()

    #         # 这里先保留你当前版本
    #         out_flat.index_add_(0, token_idx, expert_out * weight)
    #         # 或者测试原版：
    #         # out_flat[token_idx] += expert_out * weight

    #         if profile and torch.cuda.is_available():
    #             torch.cuda.synchronize()
    #             r3 = time.time()

    #             route_time += r1 - r0
    #             expert_time += r2 - r1
    #             scatter_time += r3 - r2
    #             token_counts.append(int(token_idx.numel()))

    #     if self.training and probs is not None:
    #         importance = probs.sum(dim=(0, 1))
    #         moe_loss = importance.std() / (importance.mean() + 1e-6)
    #     else:
    #         moe_loss = x.new_zeros(())

    #     if profile and torch.cuda.is_available():
    #         torch.cuda.synchronize()
    #         t3 = time.time()
    #         print(
    #             f"[PROFILE][MoE] "
    #             f"gate={t1 - t0:.4f}s, "
    #             f"prep={t2 - t1:.4f}s, "
    #             f"route={route_time:.4f}s, "
    #             f"expert={expert_time:.4f}s, "
    #             f"scatter={scatter_time:.4f}s, "
    #             f"total={t3 - t0:.4f}s, "
    #             f"counts={token_counts}"
    #         )

    #     return out, moe_loss


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, num_layers, dim_ff, act="relu",
                #  attention_fn=scaled_dot_product):
                 attention_fn=block_causal_attention):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.num_layers = num_layers
        assert d_model % n_heads == 0

        self.qkv_proj = nn.ModuleList([nn.Linear(d_model, 3 * d_model) for _ in range(num_layers)])
        self.out_proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers)])
        self.ffn1 = nn.ModuleList([nn.Linear(d_model, dim_ff) for _ in range(num_layers)])
        self.ffn2 = nn.ModuleList([nn.Linear(dim_ff, d_model) for _ in range(num_layers)])
        self.norm1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.act = getattr(F, act)
        self.attention_fn = attention_fn
        self.moe = nn.ModuleList([
            SMoE(d_model, dim_ff, num_experts=8, k=2)
            for _ in range(num_layers)
        ])

    def prepare_dense_all_smoe(self):
        for moe in self.moe:
            moe.prepare_dense_all()

    def forward(self, x, extension):
        x = x.unsqueeze(0)
        B, S, D = x.shape

        moe_loss_total = 0.0
        for i in range(self.num_layers):
            residual = x
            x = self.norm1[i](x)
            qkv = self.qkv_proj[i](x)
            qkv = qkv.view(B, S, self.n_heads, 3 * self.head_dim)
            qkv = qkv.permute(0, 2, 1, 3)
            q, k, v = torch.split(qkv, self.head_dim, dim=-1)
            attn_out = self.attention_fn(q, k, v, extension)
            attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, S, D)
            x = residual + self.out_proj[i](attn_out)
            residual = x
            x = self.norm2[i](x)

            moe_out, moe_loss = self.moe[i](x)

            x = residual + moe_out
            if moe_loss is not None:
                moe_loss_total = moe_loss_total + moe_loss

        return x, moe_loss_total

    #profile版本

    # def forward(self, x, extension):
    #     x = x.unsqueeze(0)
    #     B, S, D = x.shape

    #     moe_loss_total = 0.0

    #     profile = True
    #     qkv_time = 0.0
    #     attn_time = 0.0
    #     out_time = 0.0
    #     moe_time = 0.0

    #     for i in range(self.num_layers):
    #         if profile and torch.cuda.is_available():
    #             torch.cuda.synchronize()
    #             t0 = time.time()

    #         residual = x
    #         x = self.norm1[i](x)
    #         qkv = self.qkv_proj[i](x)
    #         qkv = qkv.view(B, S, self.n_heads, 3 * self.head_dim)
    #         qkv = qkv.permute(0, 2, 1, 3)
    #         q, k, v = torch.split(qkv, self.head_dim, dim=-1)

    #         if profile and torch.cuda.is_available():
    #             torch.cuda.synchronize()
    #             t1 = time.time()

    #         attn_out = self.attention_fn(q, k, v, extension)

    #         if profile and torch.cuda.is_available():
    #             torch.cuda.synchronize()
    #             t2 = time.time()

    #         attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, S, D)
    #         x = residual + self.out_proj[i](attn_out)

    #         residual = x
    #         x = self.norm2[i](x)

    #         if profile and torch.cuda.is_available():
    #             torch.cuda.synchronize()
    #             t3 = time.time()

    #         moe_out, moe_loss = self.moe[i](x)

    #         if profile and torch.cuda.is_available():
    #             torch.cuda.synchronize()
    #             t4 = time.time()

    #         x = residual + moe_out
    #         if moe_loss is not None:
    #             moe_loss_total = moe_loss_total + moe_loss

    #         if profile:
    #             qkv_time += t1 - t0
    #             attn_time += t2 - t1
    #             out_time += t3 - t2
    #             moe_time += t4 - t3

    #     if profile:
    #         print(
    #             f"[PROFILE][Seq] "
    #             f"qkv_norm={qkv_time:.4f}s, "
    #             f"attn={attn_time:.4f}s, "
    #             f"out_norm={out_time:.4f}s, "
    #             f"moe={moe_time:.4f}s"
    #         )

    #     return x, moe_loss_total


class CTRModel(nn.Module):
    def __init__(self, rep_encoder, seq_encoder, d_model):
        super().__init__()
        self.rep_encoder = rep_encoder
        self.seq_encoder = seq_encoder
        self.d_model = d_model
        self.linear = nn.Linear(d_model, 1)

    def get_sequence_causal_mask(self, seq_info):
        lengths = seq_info[1:] - seq_info[:-1]
        lengths = lengths.view(-1)
        indices = torch.cumsum(torch.ones_like(lengths), dim=0) - 1
        result = torch.repeat_interleave(indices, lengths)
        a = result.view(1, -1) - result.view(-1, 1)
        out_mask = torch.tril((a == 0).to(torch.int32)).bool()
        return out_mask

    # def forward(self, batch):
    #     seq_input = self.rep_encoder(batch)
    #     seq_mask = self.get_sequence_causal_mask(batch["user_offsets"])
    #     encoder_output, moe_loss = self.seq_encoder(
    #         x=seq_input,
    #         extension={"mask": seq_mask.unsqueeze(0).unsqueeze(0)},
    #     )
    #     encoder_output_dim = encoder_output.shape[-1]
    #     encoder_output = encoder_output.reshape(1, -1, encoder_output_dim).squeeze(0)
    #     pred = self.linear(encoder_output)
    #     pred_logits = torch.clamp(pred, min=-15.0, max=15.0)
    #     return pred_logits, moe_loss

    # def forward(self, batch):
    #     seq_input = self.rep_encoder(batch)

    #     # batch["user_offsets"] 形如 tensor([0, 3, 5, ...])
    #     # 这里转成 Python list，用于在 block_causal_attention 里切片。
    #     user_offsets = batch["user_offsets"].detach().cpu().tolist()

    #     encoder_output, moe_loss = self.seq_encoder(
    #         x=seq_input,
    #         extension={"user_offsets": user_offsets},
    #     )

    #     encoder_output_dim = encoder_output.shape[-1]
    #     encoder_output = encoder_output.reshape(1, -1, encoder_output_dim).squeeze(0)
    #     pred = self.linear(encoder_output)
    #     pred_logits = torch.clamp(pred, min=-15.0, max=15.0)
    #     return pred_logits, moe_loss

    def forward(self, batch):
        profile = False

        if profile and torch.cuda.is_available():
            torch.cuda.synchronize()
            t0 = time.time()

        seq_input = self.rep_encoder(batch)

        if profile and torch.cuda.is_available():
            torch.cuda.synchronize()
            t1 = time.time()

        user_offsets = batch["user_offsets"].detach().cpu().tolist()

        encoder_output, moe_loss = self.seq_encoder(
            x=seq_input,
            extension={"user_offsets": user_offsets},
        )

        if profile and torch.cuda.is_available():
            torch.cuda.synchronize()
            t2 = time.time()

        encoder_output_dim = encoder_output.shape[-1]
        encoder_output = encoder_output.reshape(1, -1, encoder_output_dim).squeeze(0)
        pred = self.linear(encoder_output)
        pred_logits = torch.clamp(pred, min=-15.0, max=15.0)

        if profile and torch.cuda.is_available():
            torch.cuda.synchronize()
            t3 = time.time()
            print(
                f"[PROFILE] rep_encoder={t1 - t0:.4f}s, "
                f"seq_encoder={t2 - t1:.4f}s, "
                f"head={t3 - t2:.4f}s"
            )

        return pred_logits, moe_loss


# ============================================================
# 模型加载入口
# ============================================================

def load_model(device='cuda:0', ckpt_path=None):
    """加载模型并返回，供 evaluation.py 调用。

    Args:
        device: 推理设备（默认 'cuda:0'）
        ckpt_path: checkpoint 文件路径，默认使用 infer.py 同目录下的 ckpt.pt

    Returns:
        (model, device) 元组
    """
    emb_dim = 512
    slot_num = 28
    vocab_size = DEFAULT_VOCAB_SIZE
    d_model = 512
    n_heads = 8
    num_layers = 8
    dim_ff = 1024

    rep_encoder = RepEncoder(
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        padding_idx=0,
        slot_num=slot_num,
        d_model=d_model,
    )
    seq_encoder = TransformerEncoder(
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        dim_ff=dim_ff,
        act="relu",
    )
    model = CTRModel(rep_encoder, seq_encoder, d_model=d_model)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    # 加载 checkpoint
    # 若需要加载自定义修改的权重，请修改 479-488行逻辑，强制使用你文件夹中的权重
    # 测评系统默认使用原始官方权重
    if ckpt_path is None:
        ckpt_path = Path(__file__).parent / 'ckpt.pt'
    else:
        ckpt_path = Path(ckpt_path)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"[INFO] Loaded checkpoint from {ckpt_path} (epoch={ckpt.get('epoch', '?')})")
    else:
        print(f"[WARNING] Checkpoint {ckpt_path} not found, using random weights")

    model.half()

    model.to(dev)

    if dev.type == "cuda":
        model.half()
        print("[INFO] Converted model weights to float16")

    if USE_DENSE_ALL_SMOE:
        model.seq_encoder.prepare_dense_all_smoe()
        print("[INFO] Prepared dense-all SMoE weights")

    model.eval()

    print(f"[INFO] Model ready. Device: {dev}")
    print(f"[INFO] Model param dtype: {next(model.parameters()).dtype}")

    return model, dev


# ============================================================
# 打分工具（与 evaluation.py 保持一致）
# ============================================================

def _read_predict(file_path):
    predictions = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(float(line))
    import numpy as np
    return np.array(predictions)


def _read_label(file_path):
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(',')
                if len(parts) >= 4:
                    labels.append(float(parts[3]))
                else:
                    labels.append(float(line))
    import numpy as np
    return np.array(labels)


def _cal_score(predict_file, label_file, default_latency=0.0):
    import numpy as np
    from sklearn.metrics import roc_auc_score

    predictions = _read_predict(predict_file)
    labels = _read_label(label_file)

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        print('[WARNING] only one class present in labels, AUC is not defined, returning 0.5')
        auc = 0.5
    else:
        auc = roc_auc_score(labels, predictions)

    mean_pred = np.mean(predictions)
    mean_label = np.mean(labels)
    if mean_label == 0:
        pcoc = 1.0 if mean_pred == 0 else float('inf')
    else:
        pcoc = float(mean_pred / mean_label)

    latency = default_latency
    base_latency = 300
    score_latency = max(0.0, (base_latency - latency) / base_latency) if latency < base_latency else 0.0

    if pcoc < 0.85 or pcoc > 1.15:
        score_model = 0.0
    else:
        score_model = ((auc - 0.65) * 1000 + (0.15 - abs(pcoc - 1)) / 0.15 * 10) / 360

    score_all = score_latency * 70 + score_model * 30

    return {
        'auc': auc,
        'pcoc': pcoc,
        'latency': latency,
        'score_latency': score_latency,
        'score_model': score_model,
        'score_all': score_all,
    }



def merge_two_batches(b1, b2, max_slot_id=28):
    """
    把两个 collate 后的 batch 合并成一个更大的 batch。
    适用于已经缓存好的 batch。
    注意：两个 batch 应该都在 CPU 上合并，然后再 move_batch_to_device。
    """
    merged = {}

    # log-level 字段：直接拼接
    for key in ["userid", "logid", "label", "pred_mask"]:
        merged[key] = torch.cat([b1[key], b2[key]], dim=0)

    # user_offsets 需要平移第二个 batch
    n1 = b1["logid"].numel()
    merged["user_offsets"] = torch.cat([
        b1["user_offsets"],
        b2["user_offsets"][1:] + n1
    ], dim=0)

    # 每个 slot 的 values 直接拼，offsets 需要平移
    for slot in range(1, max_slot_id + 1):
        v1, o1 = b1[slot]
        v2, o2 = b2[slot]

        merged_values = torch.cat([v1, v2], dim=0)
        merged_offsets = torch.cat([
            o1,
            o2[1:] + v1.numel()
        ], dim=0)

        merged[slot] = (merged_values, merged_offsets)

    merged["_values_preclamped"] = bool(
        b1.get("_values_preclamped", False)
        and b2.get("_values_preclamped", False)
    )

    return merged

def merge_cached_batches(all_batches, group_size=2, max_slot_id=28):
    """
    group_size=2 表示把两个 batch_size=50 合成一个 batch_size≈100。
    group_size=4 表示四个合成一个 batch_size≈200。
    """
    merged_batches = []

    for i in range(0, len(all_batches), group_size):
        group = all_batches[i:i + group_size]

        cur = group[0]
        for nxt in group[1:]:
            cur = merge_two_batches(cur, nxt, max_slot_id=max_slot_id)

        merged_batches.append(cur)

    return merged_batches


# ============================================================
# main：直接运行 infer.py 进行测试
# ============================================================

def main():
    import io
    import time
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint 文件路径，默认使用同目录下的 ckpt.pt')
    args = parser.parse_args()

    cur_path = Path(__file__).parent.absolute()
    ref_dir = cur_path / 'dataset'
    history_dir = ref_dir / 'history'
    input_file = ref_dir / 'test.csv'
    output_file = Path('predict.txt')
    label_file = ref_dir / 'label_data.txt'

    # ----- 数据加载，优先从缓存读取 -----
    MAX_SHARD_BYTES = 2 * 1024 * 1024 * 1024  # 2GB per shard
    batches_cache_dir = ref_dir / 'cached_batches'

    if batches_cache_dir.exists() and any(batches_cache_dir.glob('shard_*.pt')):
        print(f'[INFO] loading cached batch shards from {batches_cache_dir}')
        all_batches = []
        shard_files = sorted(batches_cache_dir.glob('shard_*.pt'),
                             key=lambda p: int(p.stem.split('_')[1]))
        for sf in shard_files:
            shard_batches = torch.load(sf, weights_only=False)
            all_batches.extend(shard_batches)
            print(f'[INFO] loaded {len(shard_batches)} batches from {sf.name}')
        print(f'[INFO] loaded {len(all_batches)} cached batches total from {len(shard_files)} shards')
    else:
        print('[INFO] start loading data from CSV')
        history_files = sorted(history_dir.glob('*.csv')) if history_dir.exists() else []
        all_files = history_files + [input_file]

        item_dict, user_seq = load_sample_files(sample_files_list=all_files)
        test_pred_logids = load_logids_from_file(input_file)
        print(f'[INFO] Test pred logids count: {len(test_pred_logids)}')

        max_feasign_per_slot = {1: 2}
        test_dataset = CTRUserDataset(
            item_dict, user_seq,
            max_feasign_per_slot=max_feasign_per_slot,
            pred_logids=test_pred_logids,
        )
        print(f'[INFO] num_users={test_dataset.num_users}, '
              f'total_samples={test_dataset.total_samples}, '
              f'pred_samples={len(test_pred_logids)}, '
              f'max_sign_id={test_dataset.max_sign_id}')

        test_loader = DataLoader(
            test_dataset,
            batch_size=50,
            shuffle=False,
            num_workers=0,
            collate_fn=make_collate_fn(test_dataset.max_slot_id),
        )

        # 收集 batches 并按分片缓存
        print('[INFO] collecting batches and saving sharded cache...')
        all_batches = [batch for batch in test_loader]

        batches_cache_dir.mkdir(parents=True, exist_ok=True)
        shard_idx = 0
        current_shard = []
        current_size = 0
        for batch in all_batches:
            buf = io.BytesIO()
            torch.save(batch, buf)
            batch_size_bytes = buf.tell()
            if current_shard and current_size + batch_size_bytes > MAX_SHARD_BYTES:
                shard_path = batches_cache_dir / f'shard_{shard_idx:04d}.pt'
                torch.save(current_shard, shard_path)
                print(f'[INFO] saved shard {shard_path.name}: {len(current_shard)} batches, '
                      f'~{current_size / 1024**3:.2f}GB')
                shard_idx += 1
                current_shard = []
                current_size = 0
            current_shard.append(batch)
            current_size += batch_size_bytes
        if current_shard:
            shard_path = batches_cache_dir / f'shard_{shard_idx:04d}.pt'
            torch.save(current_shard, shard_path)
            print(f'[INFO] saved shard {shard_path.name}: {len(current_shard)} batches, '
                  f'~{current_size / 1024**3:.2f}GB')
            shard_idx += 1
        print(f'[INFO] saved {len(all_batches)} batches to {shard_idx} shards in {batches_cache_dir}')

    print('[INFO] data loading done')


    print('[INFO] data loading done')

    MERGE_CACHED_BATCHES = True
    MERGE_GROUP_SIZE = 32

    if MERGE_CACHED_BATCHES:
        old_num_batches = len(all_batches)
        all_batches = merge_cached_batches(all_batches, MERGE_GROUP_SIZE, max_slot_id=28)
        print(f"[INFO] merged cached batches: {old_num_batches} -> {len(all_batches)} ")
    
    # ----- 加载模型 -----
    model, dev = load_model(ckpt_path=args.ckpt)

    # ----- 推理 -----
    print('*' * 20 + ' start inference ' + '*' * 20)
    all_logids = []
    all_probs = []
    time_sum = 0.0
    t_start = time.time()
    with torch.inference_mode():
        for batch in tqdm(all_batches, desc="Inference"):
            batch = prepare_cached_batch_for_current_collate(batch)
            batch = move_batch_to_device(batch, dev)
            pred_mask = batch["pred_mask"].bool()

            
            logits, moe_loss = model(batch)
            logits = logits.squeeze(-1)
            probs = torch.sigmoid(logits)
            # time_sum += time.time() - t_start

            masked_logids = batch["logid"][pred_mask].cpu().tolist()
            masked_probs = probs[pred_mask].cpu().tolist()
            all_logids.extend(masked_logids)
            all_probs.extend(masked_probs)
    time_sum += time.time() - t_start

    print(f'[INFO] inference time: {round(time_sum, 4)}s')
    print('*' * 20 + ' end inference ' + '*' * 20)

    # ----- 按 test.csv 顺序写预测文件 -----
    logid_to_prob = dict(zip(all_logids, all_probs))
    test_logids_in_order = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                test_logids_in_order.append(int(line.split(',')[0]))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        for logid in test_logids_in_order:
            f.write(f"{logid_to_prob[logid]}\n")
    print(f'[INFO] predictions written to {output_file}, total: {len(test_logids_in_order)}')

    # ----- 打分 -----
    if label_file.exists():
        result = _cal_score(output_file, label_file, default_latency=time_sum)
        print(f'[INFO] AUC:            {result["auc"]:.6f}')
        print(f'[INFO] PCOC:           {result["pcoc"]:.6f}')
        print(f'[INFO] Latency:        {result["latency"]:.4f}s')
        print(f'[INFO] score_latency:  {result["score_latency"]:.6f}')
        print(f'[INFO] score_model:    {result["score_model"]:.6f}')
        print(f'[INFO] score_all:      {result["score_all"]:.6f}')
        return result
    else:
        print(f'[WARNING] label file {label_file} not found, skipping scoring')
        return None


if __name__ == '__main__':
    main()
