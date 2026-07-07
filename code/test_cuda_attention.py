from pathlib import Path
import argparse
import math
import os
import shutil
import stat
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_LIBRARIES = (REPO_ROOT / "libraries").resolve()

if PROJECT_LIBRARIES.exists():
    sys.path.insert(0, str(PROJECT_LIBRARIES))

try:
    import torch
    from torch.utils.cpp_extension import load
except ImportError as exc:
    raise SystemExit("Failed to import PyTorch from the active Python environment.") from exc


CUDA_SRC = REPO_ROOT / "CUDA" / "attention_kernels.cu"
KERNEL_NAMES = (
    "varlen_causal_attention",
    "varlen_causal_attention_tiled",
    "varlen_causal_attention_tiled_compact",
    "varlen_causal_attention_mma_qk",
    "varlen_causal_attention_mma_qk_pv",
    "varlen_causal_attention_mma_qk_pv_padded",
    "varlen_causal_attention_mma_qk_pv_padded_shared_kv",
    "varlen_causal_attention_mma_qk_pv_padded_shared_kv_meta",
    "varlen_causal_attention_mma_qk_pv_padded_shared_qkv_meta",
    "varlen_causal_attention_mma_fa2_split_q_meta",
    "varlen_causal_attention_mma_qk_pv_padded_shared_kv_meta_regpipe",
)

MMA_KERNEL_NAMES = {
    "varlen_causal_attention_mma_qk",
    "varlen_causal_attention_mma_qk_pv",
    "varlen_causal_attention_mma_qk_pv_padded",
    "varlen_causal_attention_mma_qk_pv_padded_shared_kv",
    "varlen_causal_attention_mma_qk_pv_padded_shared_kv_meta",
    "varlen_causal_attention_mma_qk_pv_padded_shared_qkv_meta",
    "varlen_causal_attention_mma_fa2_split_q_meta",
    "varlen_causal_attention_mma_qk_pv_padded_shared_kv_meta_regpipe",
}

META_KERNEL_NAMES = {
    "varlen_causal_attention_mma_qk_pv_padded_shared_kv_meta",
    "varlen_causal_attention_mma_qk_pv_padded_shared_qkv_meta",
    "varlen_causal_attention_mma_fa2_split_q_meta",
    "varlen_causal_attention_mma_qk_pv_padded_shared_kv_meta_regpipe",
}

FA2_SPLIT_Q_KERNEL_NAMES = {
    "varlen_causal_attention_mma_fa2_split_q_meta",
}


def configure_ninja():
    if shutil.which("ninja") is not None:
        return

    candidates = [
        PROJECT_LIBRARIES / "bin" / "ninja",
        PROJECT_LIBRARIES / "ninja" / "data" / "bin" / "ninja",
    ]

    try:
        import ninja
        bin_dir = getattr(ninja, "BIN_DIR", None)
        if bin_dir is not None:
            candidates.append(Path(bin_dir) / "ninja")
    except ImportError:
        pass

    if PROJECT_LIBRARIES.exists():
        candidates.extend(path for path in PROJECT_LIBRARIES.rglob("ninja") if path.is_file())

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            mode = candidate.stat().st_mode
            if not mode & stat.S_IXUSR:
                candidate.chmod(mode | stat.S_IXUSR)
            os.environ["PATH"] = str(candidate.parent) + os.pathsep + os.environ.get("PATH", "")
            if shutil.which("ninja") is not None:
                print(f"using local ninja: {candidate}")
                return

    raise RuntimeError("Ninja is required to build this PyTorch CUDA extension.")


def configure_cuda_arch():
    if "TORCH_CUDA_ARCH_LIST" in os.environ:
        return

    major, minor = torch.cuda.get_device_capability()
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    print(f"using TORCH_CUDA_ARCH_LIST={os.environ['TORCH_CUDA_ARCH_LIST']}")


def attention_tiled_cuda_flags():
    env_to_macro = (
        ("ATTENTION_TILED_BR", "ATTENTION_TILED_BR"),
        ("ATTENTION_TILED_BC", "ATTENTION_TILED_BC"),
        ("ATTENTION_TILED_THREADS", "ATTENTION_TILED_THREADS"),
    )

    flags = []
    config = {}
    for env_name, macro_name in env_to_macro:
        value = os.environ.get(env_name)
        if value is None:
            continue
        try:
            parsed = int(value)
        except ValueError as exc:
            raise RuntimeError(f"{env_name} must be an integer, got {value!r}") from exc
        if parsed <= 0:
            raise RuntimeError(f"{env_name} must be positive, got {parsed}")
        flags.append(f"-D{macro_name}={parsed}")
        config[env_name] = parsed

    if os.environ.get("USE_FAST_ATTENTION_EXP", "0") != "0":
        flags.append("-DUSE_FAST_ATTENTION_EXP=1")
        config["USE_FAST_ATTENTION_EXP"] = 1

    if config:
        summary = ", ".join(f"{key}={value}" for key, value in config.items())
        print(f"using tiled attention compile config: {summary}")
    return flags


def build_extension():
    configure_ninja()
    configure_cuda_arch()
    return load(
        name="varlen_attention_ext",
        sources=[str(CUDA_SRC)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"] + attention_tiled_cuda_flags(),
        verbose=True,
    )


def make_offsets(lengths, device):
    offsets = [0]
    total = 0
    for length in lengths:
        total += int(length)
        offsets.append(total)
    return torch.tensor(offsets, device=device, dtype=torch.long)


def attention_tiled_br():
    return int(os.environ.get("ATTENTION_TILED_BR", "8"))


def make_attention_tile_starts(user_offsets, br=None):
    if br is None:
        br = attention_tiled_br()
    offsets = user_offsets.detach().cpu().tolist()
    starts = []
    for start, end in zip(offsets[:-1], offsets[1:]):
        starts.extend(range(int(start), int(end), br))
    starts.reverse()
    return torch.tensor(starts, device=user_offsets.device, dtype=torch.long)


def make_attention_tile_meta(user_offsets, br):
    offsets = user_offsets.detach().cpu().tolist()
    meta = []
    for start, end in zip(offsets[:-1], offsets[1:]):
        start = int(start)
        end = int(end)
        for tile_start in range(start, end, br):
            tile_len = min(end - tile_start, br)
            meta.append((start, end, tile_start, tile_len))
    meta.reverse()
    if not meta:
        return torch.empty((0, 4), device=user_offsets.device, dtype=torch.int32)
    return torch.tensor(meta, device=user_offsets.device, dtype=torch.int32)


def tile_br_for_kernel(kernel_name):
    if kernel_name in FA2_SPLIT_Q_KERNEL_NAMES:
        return 64
    return 16 if kernel_name in MMA_KERNEL_NAMES else None


def call_attention_kernel(ext, kernel_name, q, k, v, offsets, tile_starts=None, tile_meta=None):
    fn = getattr(ext, kernel_name)
    if kernel_name in META_KERNEL_NAMES:
        if tile_meta is None:
            tile_meta = make_attention_tile_meta(offsets, br=tile_br_for_kernel(kernel_name))
        return fn(q, k, v, offsets, tile_meta)
    if kernel_name == "varlen_causal_attention_tiled_compact" or kernel_name in MMA_KERNEL_NAMES:
        if tile_starts is None:
            tile_starts = make_attention_tile_starts(offsets, br=tile_br_for_kernel(kernel_name))
        return fn(q, k, v, offsets, tile_starts)
    return fn(q, k, v, offsets)


def reference_varlen(q, k, v, user_offsets):
    out = torch.empty_like(q)
    scale = 1.0 / math.sqrt(q.size(-1))
    offsets = user_offsets.detach().cpu().tolist()

    for start, end in zip(offsets[:-1], offsets[1:]):
        qi = q[:, :, start:end, :]
        ki = k[:, :, start:end, :]
        vi = v[:, :, start:end, :]
        scores = torch.matmul(qi, ki.transpose(-2, -1)) * scale
        local_len = end - start
        causal_mask = torch.tril(torch.ones((local_len, local_len), device=q.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask.view(1, 1, local_len, local_len), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out[:, :, start:end, :] = torch.matmul(attn, vi)

    return out


def reference_dense_current(q, k, v, user_offsets):
    scale = 1.0 / math.sqrt(q.size(-1))
    seq_len = q.size(2)
    token_user = torch.empty((seq_len,), device=q.device, dtype=torch.long)
    offsets = user_offsets.detach().cpu().tolist()
    for user, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        token_user[start:end] = user

    same_user = token_user.view(1, -1) == token_user.view(-1, 1)
    causal = torch.tril(torch.ones((seq_len, seq_len), device=q.device, dtype=torch.bool))
    mask = (same_user & causal).view(1, 1, seq_len, seq_len)

    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    scores = scores.masked_fill(~mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


def dtype_from_name(name):
    return {"float16": torch.float16}[name]


def tolerances(dtype):
    return (3e-3, 3e-3)


def make_qkv(lengths, heads=8, head_dim=64, seed=0, dtype=torch.float16):
    torch.manual_seed(seed)
    seq_len = sum(lengths)
    q = torch.randn((1, heads, seq_len, head_dim), device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    offsets = make_offsets(lengths, q.device)
    return q.contiguous(), k.contiguous(), v.contiguous(), offsets


def check_correctness(ext, kernel_name, lengths, dtype):
    q, k, v, offsets = make_qkv(lengths, dtype=dtype)
    tile_starts = make_attention_tile_starts(offsets, br=tile_br_for_kernel(kernel_name))
    tile_meta = make_attention_tile_meta(offsets, br=tile_br_for_kernel(kernel_name)) if kernel_name in META_KERNEL_NAMES else None
    expected = reference_varlen(q, k, v, offsets)
    actual = call_attention_kernel(ext, kernel_name, q, k, v, offsets, tile_starts, tile_meta)

    rtol, atol = tolerances(dtype)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    diff = (actual.float() - expected.float()).abs()
    print(
        f"[OK] correctness kernel={kernel_name}, dtype={dtype}, lengths={lengths}, seq_len={q.size(2)}, "
        f"max_abs_err={diff.max().item():.3e}, mean_abs_err={diff.mean().item():.3e}"
    )


def check_error_paths(ext):
    q, k, v, offsets = make_qkv([4, 3], dtype=torch.float32)

    for kernel_name in KERNEL_NAMES:
        tile_starts = make_attention_tile_starts(offsets, br=tile_br_for_kernel(kernel_name))
        tile_meta = make_attention_tile_meta(offsets, br=tile_br_for_kernel(kernel_name)) if kernel_name in META_KERNEL_NAMES else None
        try:
            call_attention_kernel(ext, kernel_name, q, k, v, offsets, tile_starts, tile_meta)
        except RuntimeError as exc:
            print(f"[OK] fp32 dtype check kernel={kernel_name} raised: {str(exc).splitlines()[0]}")
        else:
            raise AssertionError(f"{kernel_name} float32 input should raise RuntimeError")


def benchmark(ext, lengths, dtype, warmup=10, iters=50):
    q, k, v, offsets = make_qkv(lengths, seed=123, dtype=dtype)
    tile_starts_by_kernel = {
        "varlen_causal_attention_tiled_compact": make_attention_tile_starts(offsets),
        "varlen_causal_attention_mma_qk": make_attention_tile_starts(offsets, br=16),
        "varlen_causal_attention_mma_qk_pv": make_attention_tile_starts(offsets, br=16),
        "varlen_causal_attention_mma_qk_pv_padded": make_attention_tile_starts(offsets, br=16),
        "varlen_causal_attention_mma_qk_pv_padded_shared_kv": make_attention_tile_starts(offsets, br=16),
        "varlen_causal_attention_mma_qk_pv_padded_shared_kv_meta": make_attention_tile_starts(offsets, br=16),
        "varlen_causal_attention_mma_qk_pv_padded_shared_qkv_meta": make_attention_tile_starts(offsets, br=16),
        "varlen_causal_attention_mma_fa2_split_q_meta": make_attention_tile_starts(offsets, br=64),
        "varlen_causal_attention_mma_qk_pv_padded_shared_kv_meta_regpipe": make_attention_tile_starts(offsets, br=16),
    }
    tile_meta_by_kernel = {
        "varlen_causal_attention_mma_qk_pv_padded_shared_kv_meta": make_attention_tile_meta(offsets, br=16),
        "varlen_causal_attention_mma_qk_pv_padded_shared_qkv_meta": make_attention_tile_meta(offsets, br=16),
        "varlen_causal_attention_mma_fa2_split_q_meta": make_attention_tile_meta(offsets, br=64),
        "varlen_causal_attention_mma_qk_pv_padded_shared_kv_meta_regpipe": make_attention_tile_meta(offsets, br=16),
    }

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    custom_timings = {}
    for kernel_name in KERNEL_NAMES:
        tile_starts = tile_starts_by_kernel.get(kernel_name)
        tile_meta = tile_meta_by_kernel.get(kernel_name)
        for _ in range(warmup):
            call_attention_kernel(ext, kernel_name, q, k, v, offsets, tile_starts, tile_meta)
        torch.cuda.synchronize()

        start.record()
        for _ in range(iters):
            call_attention_kernel(ext, kernel_name, q, k, v, offsets, tile_starts, tile_meta)
        end.record()
        torch.cuda.synchronize()
        custom_timings[kernel_name] = start.elapsed_time(end) / iters

    for _ in range(3):
        reference_dense_current(q, k, v, offsets)
    torch.cuda.synchronize()

    start.record()
    for _ in range(max(3, iters // 10)):
        reference_dense_current(q, k, v, offsets)
    end.record()
    torch.cuda.synchronize()
    dense_iters = max(3, iters // 10)
    dense_ms = start.elapsed_time(end) / dense_iters

    seq_len = q.size(2)
    dense_scores = seq_len * seq_len
    causal_scores = sum(length * (length + 1) // 2 for length in lengths)
    custom_summary = ", ".join(f"{name}={ms:.6f} ms" for name, ms in custom_timings.items())
    print(
        f"[BENCH] users={len(lengths)}, seq_len={seq_len}, "
        f"dtype={dtype}, causal/dense={causal_scores / dense_scores:.6f}, "
        f"{custom_summary}, dense_torch={dense_ms:.6f} ms"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-users", type=int, default=20)
    parser.add_argument("--bench-len", type=int, default=128)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--dtype", choices=["float16"], default="float16")
    args = parser.parse_args()
    dtype = dtype_from_name(args.dtype)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run this test on a Linux/NVIDIA CUDA machine.")

    print(f"torch version: {torch.__version__}")
    print(f"torch cuda: {torch.version.cuda}")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"source: {CUDA_SRC}")

    ext = build_extension()

    for lengths in [
        [1],
        [2, 3, 1],
        [8, 5, 13],
        [32, 64, 17, 9],
        [100, 173, 401],
    ]:
        for kernel_name in KERNEL_NAMES:
            check_correctness(ext, kernel_name, lengths, dtype)

    check_error_paths(ext)
    benchmark(ext, [args.bench_len] * args.bench_users, dtype, iters=args.iters)


if __name__ == "__main__":
    main()
