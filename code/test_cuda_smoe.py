from pathlib import Path
import argparse
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


CUDA_SRC = REPO_ROOT / "CUDA" / "smoe_kernels.cu"
NUM_EXPERTS = 8
HIDDEN_DIM = 512
FF_DIM = 1024
TOPK = 2


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


def build_extension():
    configure_ninja()
    configure_cuda_arch()
    return load(
        name="routed_smoe_ext",
        sources=[str(CUDA_SRC)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )


def make_routing(n_tokens, pattern, device):
    idx = torch.empty((n_tokens, TOPK), device=device, dtype=torch.long)
    if n_tokens == 0:
        return idx

    token_ids = torch.arange(n_tokens, device=device, dtype=torch.long)
    if pattern == "balanced":
        idx[:, 0] = token_ids % NUM_EXPERTS
        idx[:, 1] = (token_ids + 3) % NUM_EXPERTS
    elif pattern == "front_skew":
        idx[:, 0] = 0
        idx[:, 1] = 1
    elif pattern == "back_skew":
        idx[:, 0] = 7
        idx[:, 1] = 6
    elif pattern == "duplicate":
        idx[:, 0] = 7
        idx[:, 1] = 7
    else:
        raise ValueError(f"unknown routing pattern: {pattern}")
    return idx


def make_case(n_tokens, pattern, score_dtype, seed):
    torch.manual_seed(seed)
    device = torch.device("cuda")
    x = torch.randn((n_tokens, HIDDEN_DIM), device=device, dtype=torch.float16)
    residual = torch.randn_like(x)
    w1 = torch.randn((NUM_EXPERTS, FF_DIM, HIDDEN_DIM), device=device, dtype=torch.float16) * 0.05
    b1 = torch.randn((NUM_EXPERTS, FF_DIM), device=device, dtype=torch.float16) * 0.05
    w2 = torch.randn((NUM_EXPERTS, HIDDEN_DIM, FF_DIM), device=device, dtype=torch.float16) * 0.05
    b2 = torch.randn((NUM_EXPERTS, HIDDEN_DIM), device=device, dtype=torch.float16) * 0.05
    topk_idx = make_routing(n_tokens, pattern, device)
    topk_score = torch.empty((n_tokens, TOPK), device=device, dtype=score_dtype)
    if n_tokens > 0:
        topk_score[:, 0] = 0.625
        topk_score[:, 1] = 0.375
    return x, residual, w1, b1, w2, b2, topk_idx, topk_score


def reference_smoe(x, residual, w1, b1, w2, b2, topk_idx, topk_score, with_residual):
    n_tokens = x.size(0)
    accum = torch.zeros((n_tokens, HIDDEN_DIM), device=x.device, dtype=torch.float32)
    for slot in range(TOPK):
        for expert in range(NUM_EXPERTS):
            mask = topk_idx[:, slot] == expert
            if not bool(mask.any()):
                continue
            x_e = x[mask]
            h = torch.matmul(x_e.float(), w1[expert].float().t())
            h = torch.relu(h + b1[expert].float()).to(torch.float16)
            y = torch.matmul(h.float(), w2[expert].float().t())
            y = (y + b2[expert].float()).to(torch.float16)
            score = topk_score[mask, slot].float().view(-1, 1)
            accum[mask] += score * y.float()

    if with_residual:
        accum += residual.float()
    return accum.to(torch.float16)


def max_abs_diff(actual, expected):
    if actual.numel() == 0:
        return 0.0
    return (actual.float() - expected.float()).abs().max().item()


def check_case(ext, n_tokens, pattern, score_dtype, seed, atol, rtol):
    case = make_case(n_tokens, pattern, score_dtype, seed)
    x, residual, w1, b1, w2, b2, topk_idx, topk_score = case

    actual = ext.smoe_forward(x, w1, b1, w2, b2, topk_idx, topk_score)
    expected = reference_smoe(x, residual, w1, b1, w2, b2, topk_idx, topk_score, False)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

    actual_residual = ext.smoe_forward_with_residual(
        x, residual, w1, b1, w2, b2, topk_idx, topk_score
    )
    expected_residual = reference_smoe(x, residual, w1, b1, w2, b2, topk_idx, topk_score, True)
    torch.testing.assert_close(actual_residual, expected_residual, atol=atol, rtol=rtol)

    print(
        f"[OK] n={n_tokens}, pattern={pattern}, score_dtype={score_dtype}, "
        f"max_abs={max_abs_diff(actual, expected):.4e}, "
        f"max_abs_residual={max_abs_diff(actual_residual, expected_residual):.4e}"
    )


def benchmark(ext, n_tokens, pattern, score_dtype, iters):
    case = make_case(n_tokens, pattern, score_dtype, seed=2026)
    x, residual, w1, b1, w2, b2, topk_idx, topk_score = case

    for _ in range(10):
        ext.smoe_forward_with_residual(x, residual, w1, b1, w2, b2, topk_idx, topk_score)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        ext.smoe_forward_with_residual(x, residual, w1, b1, w2, b2, topk_idx, topk_score)
    end.record()
    torch.cuda.synchronize()
    print(
        f"[BENCH] n={n_tokens}, pattern={pattern}, score_dtype={score_dtype}, "
        f"custom={start.elapsed_time(end) / iters:.6f} ms"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run this test on a Linux/NVIDIA CUDA machine.")

    print(f"torch version: {torch.__version__}")
    print(f"torch cuda: {torch.version.cuda}")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"source: {CUDA_SRC}")

    ext = build_extension()
    sizes = [0, 1, 2, 127, 128, 129, 255, 256, 257, 1024]
    patterns = ["balanced", "front_skew", "back_skew", "duplicate"]
    score_dtypes = [torch.float16, torch.float32]

    seed = 1234
    for n_tokens in sizes:
        for pattern in patterns:
            for score_dtype in score_dtypes:
                check_case(ext, n_tokens, pattern, score_dtype, seed, args.atol, args.rtol)
                seed += 1

    if args.bench:
        benchmark(ext, 4096, "balanced", torch.float16, args.iters)
        benchmark(ext, 4096, "back_skew", torch.float16, args.iters)
        benchmark(ext, 4096, "duplicate", torch.float16, args.iters)


if __name__ == "__main__":
    main()
