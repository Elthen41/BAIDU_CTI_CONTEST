from pathlib import Path
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
    local_torch = PROJECT_LIBRARIES / "torch"
    local_torch_c_files = []
    if local_torch.exists():
        local_torch_c_files = list(local_torch.glob("_C*.so")) + list(local_torch.glob("_C*.pyd"))

    if local_torch.exists() and not local_torch_c_files:
        detail = (
            f"Found {local_torch}, but it does not look like a complete wheel-installed "
            "PyTorch package because torch._C is missing.\n"
        )
    else:
        detail = ""

    raise SystemExit(
        "Failed to import PyTorch.\n"
        + detail
        + "Install a CUDA-enabled PyTorch wheel into ./libraries, or activate an environment "
        "where PyTorch is already installed."
    ) from exc

CUDA_SRC = REPO_ROOT / "CUDA" / "residual_add.cu"


def configure_ninja():
    if shutil.which("ninja") is not None:
        return

    candidates = []
    candidates.append(PROJECT_LIBRARIES / "bin" / "ninja")
    candidates.append(PROJECT_LIBRARIES / "ninja" / "data" / "bin" / "ninja")

    try:
        import ninja
        bin_dir = getattr(ninja, "BIN_DIR", None)
        if bin_dir is not None:
            candidates.append(Path(bin_dir) / "ninja")
    except ImportError:
        pass

    if PROJECT_LIBRARIES.exists():
        candidates.extend(
            path for path in PROJECT_LIBRARIES.rglob("ninja")
            if path.is_file()
        )

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

    raise RuntimeError(
        "Ninja is required to build this PyTorch CUDA extension. "
        "Install it with: python -m pip install --target ./libraries --upgrade ninja"
    )


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
        name="residual_add_ext",
        sources=[str(CUDA_SRC)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )


def check_correctness(ext, shape):
    a = torch.randn(shape, device="cuda", dtype=torch.float32)
    b = torch.randn(shape, device="cuda", dtype=torch.float32)

    expected = a + b
    actual = ext.residual_add(a, b)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    print(f"[OK] correctness shape={shape}, max_abs_err={(actual - expected).abs().max().item():.3e}")


def check_error_paths(ext):
    a = torch.randn((2, 3), device="cuda", dtype=torch.float32)
    b = torch.randn((2, 4), device="cuda", dtype=torch.float32)

    try:
        ext.residual_add(a, b)
    except RuntimeError as exc:
        print(f"[OK] shape check raised: {str(exc).splitlines()[0]}")
    else:
        raise AssertionError("shape mismatch should raise RuntimeError")

    try:
        ext.residual_add(a.double(), a.double())
    except RuntimeError as exc:
        print(f"[OK] dtype check raised: {str(exc).splitlines()[0]}")
    else:
        raise AssertionError("float64 input should raise RuntimeError")


def benchmark(ext, shape, warmup=50, iters=500):
    a = torch.randn(shape, device="cuda", dtype=torch.float32)
    b = torch.randn(shape, device="cuda", dtype=torch.float32)

    for _ in range(warmup):
        ext.residual_add(a, b)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        ext.residual_add(a, b)
    end.record()
    torch.cuda.synchronize()

    custom_ms = start.elapsed_time(end) / iters

    start.record()
    for _ in range(iters):
        a + b
    end.record()
    torch.cuda.synchronize()

    torch_ms = start.elapsed_time(end) / iters
    n = a.numel()
    print(
        f"[BENCH] shape={shape}, numel={n}, "
        f"custom={custom_ms:.6f} ms, torch={torch_ms:.6f} ms"
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run this test on a Linux/NVIDIA CUDA machine.")

    print(f"torch version: {torch.__version__}")
    print(f"torch cuda: {torch.version.cuda}")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"source: {CUDA_SRC}")

    ext = build_extension()

    shapes = [
        (1,),
        (1, 128, 512),
        (1, 1024, 512),
        (4, 2048, 512),
    ]
    for shape in shapes:
        check_correctness(ext, shape)

    check_error_paths(ext)
    benchmark(ext, (4, 2048, 512))


if __name__ == "__main__":
    main()
