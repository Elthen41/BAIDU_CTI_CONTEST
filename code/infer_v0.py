from pathlib import Path
import os
import shutil
import stat
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_LIBRARIES = (REPO_ROOT / "libraries").resolve()
CUDA_SRC = REPO_ROOT / "CUDA" / "norm_kernels.cu"

if PROJECT_LIBRARIES.exists():
    sys.path.insert(0, str(PROJECT_LIBRARIES))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load

import infer_init as _base
from infer_init import *  # noqa: F403


_ORIGINAL_LOAD_MODEL = _base.load_model
_ORIGINAL_TRANSFORMER_ENCODER = _base.TransformerEncoder
_LAYER_NORM_EXT = None


def _configure_ninja():
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
                print(f"[INFO] using local ninja: {candidate}")
                return

    raise RuntimeError(
        "Ninja is required to build the custom CUDA LayerNorm extension. "
        "Install it with: python -m pip install --target ./libraries --upgrade ninja"
    )


def _configure_cuda_arch():
    if "TORCH_CUDA_ARCH_LIST" in os.environ or not torch.cuda.is_available():
        return

    major, minor = torch.cuda.get_device_capability()
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    print(f"[INFO] using TORCH_CUDA_ARCH_LIST={os.environ['TORCH_CUDA_ARCH_LIST']}")


def _get_layernorm_ext():
    global _LAYER_NORM_EXT
    if _LAYER_NORM_EXT is not None:
        return _LAYER_NORM_EXT

    _configure_ninja()
    _configure_cuda_arch()
    _LAYER_NORM_EXT = load(
        name="layernorm_512_ext",
        sources=[str(CUDA_SRC)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=os.environ.get("CUDA_EXT_VERBOSE") == "1",
    )
    return _LAYER_NORM_EXT


class CustomLayerNorm512(nn.Module):
    """Drop-in LayerNorm(512) replacement backed by CUDA/norm_kernels.cu."""

    def __init__(self, normalized_shape=512, eps=1e-5, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        else:
            normalized_shape = tuple(normalized_shape)

        if normalized_shape != (512,):
            raise ValueError(f"CustomLayerNorm512 only supports normalized_shape=(512,), got {normalized_shape}")

        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(512))
            self.bias = nn.Parameter(torch.zeros(512))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        if (
            x.is_cuda
            and x.dtype == torch.float32
            and self.elementwise_affine
            and self.weight is not None
            and self.bias is not None
        ):
            return _get_layernorm_ext().layernorm_512(
                x.contiguous(),
                self.weight.contiguous(),
                self.bias.contiguous(),
                self.eps,
            )

        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

    def extra_repr(self):
        return (
            f"{self.normalized_shape}, eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine}, backend=custom_cuda"
        )


class TransformerEncoder(_ORIGINAL_TRANSFORMER_ENCODER):
    def __init__(self, d_model, n_heads, num_layers, dim_ff, act="relu",
                 attention_fn=_base.scaled_dot_product):
        super().__init__(
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dim_ff=dim_ff,
            act=act,
            attention_fn=attention_fn,
        )

        if d_model == 512:
            self.norm1 = nn.ModuleList([CustomLayerNorm512(d_model) for _ in range(num_layers)])
            self.norm2 = nn.ModuleList([CustomLayerNorm512(d_model) for _ in range(num_layers)])
        else:
            print(f"[WARNING] d_model={d_model}; keeping PyTorch LayerNorm because custom kernel only supports 512")


def _install_overrides():
    _base.TransformerEncoder = TransformerEncoder
    _base.load_model = load_model


def _warmup_custom_layernorm(device):
    dev = torch.device(device)
    if dev.type != "cuda":
        return

    ext = _get_layernorm_ext()
    with torch.no_grad():
        x = torch.randn((1, 512), device=dev, dtype=torch.float32)
        weight = torch.ones((512,), device=dev, dtype=torch.float32)
        bias = torch.zeros((512,), device=dev, dtype=torch.float32)
        ext.layernorm_512(x, weight, bias, 1e-5)
    torch.cuda.synchronize(dev)


def load_model(device='cuda:0', ckpt_path=None):
    _base.TransformerEncoder = TransformerEncoder
    model, dev = _ORIGINAL_LOAD_MODEL(device=device, ckpt_path=ckpt_path)
    _warmup_custom_layernorm(dev)
    print("[INFO] infer_v0 uses custom CUDA LayerNorm(512) for TransformerEncoder norm1/norm2")
    return model, dev


def main():
    _install_overrides()
    return _base.main()


_install_overrides()


if __name__ == '__main__':
    main()
