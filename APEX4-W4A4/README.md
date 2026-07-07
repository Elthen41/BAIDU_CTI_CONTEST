# APEX4: Efficient Pure W4A4 LLM Inference via Intra-SM Compute Rebalancing

📄 **Paper:** [arXiv:2606.08761](https://arxiv.org/abs/2606.08761)

This repository provides the APEX4 W4A4 (4-bit weight, 4-bit activation) CUDA kernels,
the quantized-inference layer, accuracy-evaluation code, and pre-quantized checkpoints.

This package covers **inference and evaluation**. The quantization that produced the
checkpoints uses the OmniQuant method (Shao et al., ICLR 2024); the quantization code is
not included — see [Producing the checkpoints](#producing-the-checkpoints).

## Overview

W4A4 quantization promises full utilization of INT4 Tensor Cores, but group-dequantization
overhead on CUDA Cores has driven existing systems to mixed-precision fallbacks. We show
that **intra-SM compute balance** — captured by the Tensor-Core : CUDA-Core throughput
ratio **ρ** — is the primary hardware indicator of whether W4A4 pays off: the W4A4-g128
kernel reaches 2.0–2.5× speedup on RTX 3090 (ρ=16) but degrades to 0.43–0.47× on A100
(ρ=64) in compute-bound settings. Guided by this, **APEX4** co-designs pure INT4 GEMM
kernels with **ρ-aware granularity adaptation** to remove the CUDA-Core dequantization
bottleneck.

**Highlights**
- **Accuracy** — perplexity within 0.63 of FP16 on Llama-2-70B; +4.0–4.4% zero-shot
  accuracy over W4Ax Atom-g128.
- **Speed** — drop-in for unmodified vLLM: up to 1.66× (L40S, ρ=8), 1.78× (RTX 3090, ρ=16),
  2.09× (A40, ρ=16) end-to-end; A100 (ρ=64) recovered to 1.20–1.40× via the mixed-granularity mode.
- **Finding** — W4A4 viability is platform-dependent (governed by ρ), not universally infeasible.

> This repository covers **accuracy** (inference kernels + evaluation + checkpoints).
> Throughput numbers are produced by a separate vLLM-based package.

---

## Model Zoo

Pre-quantized W4A4 checkpoints — download a directory and evaluate it directly
(each is self-contained: `config.json` + tokenizer + `*.safetensors`).

| Model | Config | Eval flags | Link |
|-------|--------|-----------|----|
| Llama-3-8B | W4A4 (uniform g128) | `--group_size 128 --act_group_size 128` | [APEX4-W4A4/Llama-3-8b-g128](https://huggingface.co/APEX4-W4A4/Llama-3-8b-g128) |
| Llama-3-8B | W4A4-mix (channel + v/down g32) | `--layer_mix --group_size -1 --mix_group_size 32 --act_group_size -1` | [APEX4-W4A4/Llama-3-8b-mix](https://huggingface.co/APEX4-W4A4/Llama-3-8b-mix) |
| Qwen2.5-7B | W4A4 (uniform g128) | `--group_size 128 --act_group_size 128` | [APEX4-W4A4/Qwen2.5-7b-g128](https://huggingface.co/APEX4-W4A4/Qwen2.5-7b-g128) |
| Qwen2.5-7B | W4A4-mix (channel + v/down g32) | `--layer_mix --group_size -1 --mix_group_size 32 --act_group_size -1` | [APEX4-W4A4/Qwen2.5-7b-mix](https://huggingface.co/APEX4-W4A4/Qwen2.5-7b-mix) |

Download a checkpoint to a local directory, e.g.:

```bash
huggingface-cli download APEX4-W4A4/Llama-3-8b-g128 --local-dir ./checkpoints/llama3-8b-w4a4-g128
```

## What's in this repo

| Component | Included | Location |
|-----------|----------|----------|
| W4A4 CUDA kernels | ✅ | `kernels/` |
| Quantized inference layer | ✅ | `quantize/quant_linear.py` |
| Accuracy evaluation | ✅ | `eval_model.py`, `scripts/` |
| Quantization / calibration ("compression") | ❌ — not included | OmniQuant method (Shao et al., ICLR 2024) |

---

## Installation

Requires **Python 3.11**, and an NVIDIA GPU (compute capability >= 8.0) with a
**CUDA 12.x toolkit** (nvcc; tested with 12.4) and a C++ compiler — needed to build
the kernels. The PyTorch CUDA build must match the toolkit (we use the cu124 build
of PyTorch 2.5.1).

```bash
conda create -n apex4 python=3.11 -y && conda activate apex4

# CUDA 12.4 toolkit (nvcc) for building the kernels.
# Skip if your system already provides nvcc 12.x on PATH.
conda install -y -c "nvidia/label/cuda-12.4.0" cuda-toolkit

# Install PyTorch first (CUDA 12.4 build; the kernels compile against it)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# Build the W4A4 CUDA kernels (PyTorch locates nvcc on PATH automatically;
# if needed, set CUDA_HOME to the toolkit root)
cd kernels
python setup.py build_ext --inplace      # produces kernels/APEX4/_CUDA*.so
cd ..
```

If you modify any kernel source (`kernels/csrc/*.cu` / `*.h`), rebuild with the same
command (`cd kernels && python setup.py build_ext --inplace`) to pick up the changes.

## Evaluate a checkpoint

`CKPT` is a downloaded checkpoint directory; it is self-contained, so it is used for
both `--model_name_or_path` and `--model_path`.

```bash
CKPT=./checkpoints/llama3-8b-w4a4-g128      # uniform g128
python eval_model.py \
    --model_name_or_path "$CKPT" --model_path "$CKPT" \
    --bits 4 --act_bits 4 --group_size 128 --act_group_size 128 \
    --symmetric --real_pack \
    --eval_ppl --ppl_datasets wikitext2 --seed 42 \
    --tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande
```

For a **W4A4-mix** checkpoint, swap the granularity flags:
`--layer_mix --group_size -1 --mix_group_size 32 --act_group_size -1`.

Ready-to-run scripts: [`scripts/eval_llama3_8b_w4a4.sh`](scripts/eval_llama3_8b_w4a4.sh),
[`scripts/eval_qwen2.5_7b_w4a4.sh`](scripts/eval_qwen2.5_7b_w4a4.sh).

## Reproducing the paper

| Model | Variant | Command | WikiText-2 PPL |
|-------|---------|---------|----------------|
| Llama-3-8B | uniform g128 | `bash scripts/eval_llama3_8b_w4a4.sh` | 7.70 |
| Llama-3-8B | mix | (use Variant 2 in the same script) | 7.85 |
| Qwen2.5-7B | uniform g128 | `bash scripts/eval_qwen2.5_7b_w4a4.sh` | 7.87 |
| Qwen2.5-7B | mix | (use Variant 2 in the same script) | 8.09 |

> Throughput numbers in the paper come from a **separate** vLLM-based package on
> uncalibrated weights. This repo is for accuracy only; do not cross-use them.

## Producing the checkpoints

The released checkpoints are produced with the **OmniQuant** method (Shao et al.,
ICLR 2024), followed by packing the calibrated weights into the W4A4 kernel format.
Quantization settings: 4-bit symmetric weights and activations — group size 128
(uniform), or per-channel weights with `v_proj`/`down_proj` at group 32 for the
*-mix* variants — with random Hadamard rotation, calibrated on WikiText-2. The
quantization code is not part of this package; the full setup is described in the paper.

## License & attribution

Apache-2.0 (see [LICENSE](LICENSE)). The kernels derive from **Marlin → QQQ**; the
quantization method is **OmniQuant** (Shao et al., ICLR 2024); evaluation uses
EleutherAI's lm-eval-harness. Full per-component provenance and our contributions are
in [NOTICE](NOTICE).

## Citation

```bibtex
@article{guo2026apex4,
  title={APEX4: Efficient Pure W4A4 LLM Inference via Intra-SM Compute Rebalancing},
  author={Guo, Hong and Guo, Nianhui and Wang, Weixing and Otholt, Jona and Meinel, Christoph and Yang, Haojin},
  journal={arXiv preprint arXiv:2606.08761},
  year={2026}
}
```
