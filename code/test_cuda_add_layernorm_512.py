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
    import torch.nn.functional as F
    from torch.utils.cpp_extension import load
except ImportError as exc:
    raise SystemExit("Failed to import PyTorch from the active Python environment.") from exc

CUDA_SRC = REPO_ROOT / "CUDA" / "norm_kernels.cu"


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
        name="layernorm_512_ext",
        sources=[str(CUDA_SRC)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )


def dtype_from_name(name):
    return {"float16": torch.float16}[name]


def tolerances(dtype):
    return (2e-3, 2e-3)


def check_correctness(ext, shape, dtype, eps=1e-5):
    residual = torch.randn(shape, device="cuda", dtype=dtype)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    weight = torch.randn((512,), device="cuda", dtype=dtype)
    bias = torch.randn((512,), device="cuda", dtype=dtype)

    expected = F.layer_norm(residual + x, (512,), weight, bias, eps)
    actual = ext.add_layernorm_512(residual, x, weight, bias, eps)
    actual_residual, actual_norm = ext.add_layernorm_512_with_residual(residual, x, weight, bias, eps)

    rtol, atol = tolerances(dtype)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(actual_residual, residual + x, rtol=rtol, atol=atol)
    torch.testing.assert_close(actual_norm, expected, rtol=rtol, atol=atol)
    max_abs_err = (actual.float() - expected.float()).abs().max().item()
    residual_max_abs_err = (actual_residual.float() - (residual + x).float()).abs().max().item()
    with_residual_max_abs_err = (actual_norm.float() - expected.float()).abs().max().item()
    print(
        f"[OK] correctness dtype={dtype}, shape={shape}, "
        f"single_max_abs_err={max_abs_err:.3e}, "
        f"residual_max_abs_err={residual_max_abs_err:.3e}, "
        f"with_residual_max_abs_err={with_residual_max_abs_err:.3e}"
    )


def check_error_paths(ext):
    residual = torch.randn((2, 512), device="cuda", dtype=torch.float16)
    x = torch.randn((2, 256), device="cuda", dtype=torch.float16)
    weight = torch.randn((512,), device="cuda", dtype=torch.float16)
    bias = torch.randn((512,), device="cuda", dtype=torch.float16)

    try:
        ext.add_layernorm_512(residual, x, weight, bias, 1e-5)
    except RuntimeError as exc:
        print(f"[OK] shape check raised: {str(exc).splitlines()[0]}")
    else:
        raise AssertionError("shape mismatch should raise RuntimeError")

    residual = torch.randn((2, 512), device="cuda", dtype=torch.float32)
    x = torch.randn((2, 512), device="cuda", dtype=torch.float32)
    weight = torch.randn((512,), device="cuda", dtype=torch.float32)
    bias = torch.randn((512,), device="cuda", dtype=torch.float32)

    try:
        ext.add_layernorm_512(residual, x, weight, bias, 1e-5)
    except RuntimeError as exc:
        print(f"[OK] fp32 dtype check raised: {str(exc).splitlines()[0]}")
    else:
        raise AssertionError("float32 input should raise RuntimeError")

    try:
        ext.add_layernorm_512_with_residual(residual, x, weight, bias, 1e-5)
    except RuntimeError as exc:
        print(f"[OK] with_residual fp32 dtype check raised: {str(exc).splitlines()[0]}")
    else:
        raise AssertionError("float32 input should raise RuntimeError")

    try:
        ext.add_layernorm_512_with_residual(residual, x, weight, bias, 1e-5)
    except RuntimeError as exc:
        print(f"[OK] with_residual shape check raised: {str(exc).splitlines()[0]}")
    else:
        raise AssertionError("shape mismatch should raise RuntimeError")


def benchmark(ext, shape, dtype, eps=1e-5, warmup=50, iters=500):
    residual = torch.randn(shape, device="cuda", dtype=dtype)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    weight = torch.randn((512,), device="cuda", dtype=dtype)
    bias = torch.randn((512,), device="cuda", dtype=dtype)

    for _ in range(warmup):
        ext.add_layernorm_512_with_residual(residual, x, weight, bias, eps)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        ext.add_layernorm_512_with_residual(residual, x, weight, bias, eps)
    end.record()
    torch.cuda.synchronize()
    custom_ms = start.elapsed_time(end) / iters

    start.record()
    for _ in range(iters):
        attn_residual = residual + x
        F.layer_norm(attn_residual, (512,), weight, bias, eps)
    end.record()
    torch.cuda.synchronize()
    torch_ms = start.elapsed_time(end) / iters

    rows = x.numel() // 512
    print(
        f"[BENCH] dtype={dtype}, shape={shape}, rows={rows}, "
        f"custom={custom_ms:.6f} ms, torch_add_layernorm={torch_ms:.6f} ms"
    )


def main():
    parser = argparse.ArgumentParser()
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

    shapes = [
        (1, 512),
        (1, 128, 512),
        (1, 1024, 512),
        (4, 2048, 512),
    ]
    for shape in shapes:
        check_correctness(ext, shape, dtype)

    check_error_paths(ext)
    benchmark(ext, (4, 2048, 512), dtype)


if __name__ == "__main__":
    main()
