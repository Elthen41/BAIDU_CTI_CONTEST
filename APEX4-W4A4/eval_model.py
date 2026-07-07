"""
W4A4 inference + accuracy evaluation entry point.

This script loads a *pre-quantized* checkpoint, reconstructs the W4A4 quantized
linear layers (which invoke our custom W4A4 CUDA kernel), and evaluates accuracy
(WikiText-2 perplexity and, optionally, zero-shot tasks via lm-eval-harness).

It does NOT quantize a model. The quantization (calibration) is done with the
OmniQuant method (Shao et al., ICLR 2024); see README.md for the quantization setup.
This repository ships only the inference + kernel + evaluation code plus the
resulting pre-quantized checkpoints.

Usage
-----
    # CKPT is the pre-quantized checkpoint dir (self-contained: config +
    # tokenizer + weights), used for both --model_name_or_path and --model_path.

    # Perplexity only
    python eval_model.py \
        --model_name_or_path $CKPT --model_path $CKPT \
        --bits 4 --act_bits 4 --group_size 128 --act_group_size 128 \
        --symmetric --real_pack --eval_ppl --ppl_datasets wikitext2

    # Perplexity + zero-shot tasks (lm-eval-harness)
    python eval_model.py \
        --model_name_or_path $CKPT --model_path $CKPT \
        --bits 4 --act_bits 4 --group_size 128 --act_group_size 128 \
        --symmetric --real_pack --eval_ppl --ppl_datasets wikitext2 \
        --tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande

    # W4A4-mix checkpoints: --layer_mix --group_size -1 --mix_group_size 32 --act_group_size -1
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import accelerate
from safetensors.torch import load_file as load_safetensors
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers.utils import logging as transformers_logging

# ── local imports (inference-only closure; no training/calibration code) ──
from quantize.quant_linear import QuantLinearTorch
from models.models_utils import find_layers, BaseLM
from datautils import get_loaders

torch.backends.cudnn.benchmark = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ======================================================================
#  Quantization-config helpers (lightweight; no dependency on the
#  calibration code — these only describe the packed weight layout)
# ======================================================================

class SimpleQParams:
    def __init__(self, group_size=128, bits=4):
        self.group_size = {bits: group_size}
        self.bits = [bits]


class SimpleAParams:
    def __init__(self, bits=4, group_size=128):
        self.bits = [bits]
        self.group_size = {bits: group_size}
        self.symmetric = True
        self.channel_wise = False
        self.scale_type = "tensor"


# ======================================================================
#  Model surgery: rebuild W4A4 quantized layers, then load packed weights
# ======================================================================

def replace_linear_with_quant(model, group_size=128, bits=4, symmetric=False,
                              real_pack=True, act_bits=4, act_group_size=128,
                              layer_mix=False, mix_group_size=32):
    """Replace every nn.Linear (except lm_head) with a QuantLinearTorch that
    runs the W4A4 kernel. Shapes only — weights are filled by load_state_dict.

    For the channel (-1) variant, the quantizer forces v_proj/down_proj to
    group 32 (see models/int_llama_layer_mix.py), so pass
    --layer_mix --group_size -1 --mix_group_size 32 to match it."""
    linear_layers = find_layers(model, layers=[nn.Linear])

    qparams_default = SimpleQParams(group_size=group_size, bits=bits)
    aparams_default = SimpleAParams(bits=act_bits, group_size=act_group_size)

    if layer_mix:
        qparams_mix = SimpleQParams(group_size=mix_group_size, bits=bits)
        aparams_mix = SimpleAParams(bits=act_bits, group_size=mix_group_size)
        print(f"Layer-mix mode: v_proj/down_proj -> g{mix_group_size}, "
              f"others -> g{group_size}")

    replaced = 0
    for name, linear in tqdm(linear_layers.items(),
                             desc="Replacing linear layers"):
        if "lm_head" in name:
            continue

        if layer_mix and (name.endswith("v_proj")
                          or name.endswith("down_proj")):
            qparams, aparams = qparams_mix, aparams_mix
        else:
            qparams, aparams = qparams_default, aparams_default

        bias = linear.bias is not None
        quant_linear = QuantLinearTorch(
            qparams=qparams, aparams=aparams,
            infeatures=linear.in_features,
            outfeatures=linear.out_features,
            bias=bias, symmetric=symmetric,
            real_pack=real_pack, cuda_optimized=True,
        )
        if hasattr(linear, "weight") and linear.weight is not None:
            quant_linear = quant_linear.to(linear.weight.device)

        parts = name.rsplit(".", 1)
        parent = model
        if len(parts) == 2:
            for p in parts[0].split("."):
                parent = getattr(parent, p)
            setattr(parent, parts[1], quant_linear)
        else:
            setattr(model, parts[0], quant_linear)
        replaced += 1

    print(f"Replaced {replaced}/{len(linear_layers)} linear layers [W4A4]")
    return model


def load_model(args, device="cuda", dtype=torch.float16):
    """Load a pre-quantized W4A4 checkpoint into a runnable model."""
    transformers_logging.set_verbosity_error()

    config = AutoConfig.from_pretrained(
        args.model_name_or_path, trust_remote_code=True)

    # Build an empty skeleton, then swap in W4A4 layers.
    with accelerate.init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, config=config,
            torch_dtype=dtype, trust_remote_code=True,
        )

    model = replace_linear_with_quant(
        model,
        group_size=args.group_size, bits=args.bits,
        symmetric=args.symmetric, real_pack=args.real_pack,
        act_bits=args.act_bits, act_group_size=args.act_group_size,
        layer_mix=args.layer_mix, mix_group_size=args.mix_group_size,
    )

    if config.tie_word_embeddings and hasattr(model, "lm_head"):
        if hasattr(model.lm_head, "weight"):
            delattr(model.lm_head, "weight")
        model.tie_weights()

    # Load the packed quantized weights from disk.
    model_path = Path(args.model_path)
    print(f"Loading quantized weights from {model_path} ...")
    if model_path.is_dir():
        model = accelerate.load_checkpoint_and_dispatch(
            model=model, checkpoint=str(model_path),
            device_map="auto",
            no_split_module_classes=getattr(model, "no_split_modules", None),
        )
    else:
        if model_path.suffix == ".safetensors":
            state_dict = load_safetensors(str(model_path))
        else:
            state_dict = torch.load(str(model_path), map_location="cpu")
            if "model" in state_dict:
                state_dict = state_dict["model"]
        # Shape-safe load: drop checkpoint keys whose shape doesn't match the
        # rebuilt model (e.g. the packing-only `qmax_list` buffer, which is not
        # used by forward_w4a4). Prevents a load_state_dict size-mismatch error.
        model_sd = model.state_dict()
        dropped = [k for k, v in state_dict.items()
                   if k in model_sd and v.shape != model_sd[k].shape]
        for k in dropped:
            state_dict.pop(k)
        if dropped:
            print(f"  Dropped {len(dropped)} shape-mismatched keys "
                  f"(e.g. {dropped[:3]}) — not used by W4A4 forward.")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
        if device != "auto":
            model = model.to(device)

    # NOTE: we deliberately do NOT call prepare_inference_buffers() here.
    # The authoritative evaluate.py runs forward_w4a4 with a freshly-zeroed
    # reduce buffer each call; pre-allocating + reusing a buffer could change
    # numerics. Matching evaluate.py keeps PPL bit-for-bit comparable.
    model.eval()
    if torch.cuda.is_available():
        print(f"VRAM used: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")
    return model


# ======================================================================
#  Perplexity — identical math/loop to the authoritative evaluate.py
#  (WikiText-2 / C4 / PTB; llama/qwen forward path)
# ======================================================================

@torch.no_grad()
def eval_ppl(model, args, device="cuda"):
    seqlen = args.eval_seq_length
    results = {}
    for dataset in args.ppl_datasets:
        _, testloader = get_loaders(
            dataset, seed=args.seed,
            model=args.model_name_or_path, seqlen=seqlen,
        )
        # C4's loader is already an indexable tensor; others expose .input_ids
        testenc = testloader if "c4" in dataset else testloader.input_ids
        nsamples = testenc.numel() // seqlen

        use_cache = model.config.use_cache
        model.config.use_cache = False
        model.eval()

        nlls = []
        for i in tqdm(range(nsamples), desc=f"{dataset} PPL"):
            batch = testenc[:, (i * seqlen):((i + 1) * seqlen)].to(device)
            # llama / qwen / mistral family: inner transformer then lm_head
            outputs = model.model(batch)
            hidden = outputs[0].to(model.lm_head.weight.dtype)
            logits = model.lm_head(hidden)
            shift_logits = logits[:, :-1, :]
            shift_labels = testenc[:, (i * seqlen):((i + 1) * seqlen)][:, 1:].to(
                model.lm_head.weight.device)
            loss = nn.CrossEntropyLoss()(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            nlls.append(loss.float() * seqlen)

        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
        model.config.use_cache = use_cache
        print(f"{dataset} perplexity: {ppl.item():.4f}")
        results[dataset] = ppl.item()
    return results


# ======================================================================
#  lm-eval-harness wrapper around an already-loaded quantized model
# ======================================================================

class QuantLM(BaseLM):
    """Minimal BaseLM adapter so a pre-loaded quantized model can be scored by
    lm-eval-harness. Mirrors models/LMClass.py but takes a ready model."""

    def __init__(self, model, tokenizer, batch_size=8, device="cuda"):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.batch_size_per_gpu = batch_size
        self.seqlen = model.config.max_position_embeddings
        self.vocab_size = tokenizer.vocab_size

    @property
    def eot_token(self):
        return self.tokenizer.eos_token

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self.model.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_encode_batch(self, strings):
        return self.tokenizer(strings, padding=True, add_special_tokens=False,
                              return_tensors="pt")

    def tok_decode(self, tokens):
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    def _model_call(self, inps):
        with torch.no_grad():
            return self.model(inps)["logits"]

    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(context, max_length=max_length,
                                   eos_token_id=eos_token_id, do_sample=False)


def eval_tasks(model, tokenizer, args, device="cuda"):
    from lm_eval import evaluator
    lm = QuantLM(model, tokenizer, batch_size=args.batch_size, device=device)
    # This vendored lm_eval's simple_evaluate splits `tasks` internally,
    # so pass the comma-separated string (matches authoritative evaluate.py).
    results = evaluator.simple_evaluate(
        lm, tasks=args.tasks,
        num_fewshot=args.num_fewshot, limit=None,
    )
    from pprint import pprint
    pprint(results["results"])
    return results


# ======================================================================

def main():
    p = argparse.ArgumentParser(
        description="W4A4 inference + accuracy evaluation (loads a pre-quantized checkpoint)")
    # paths
    p.add_argument("--model_name_or_path", required=True,
                   help="FP16 base model id/path — provides config, tokenizer, skeleton")
    p.add_argument("--model_path", required=True,
                   help="Directory (or file) with the pre-quantized W4A4 checkpoint")
    # quant config (must match how the checkpoint was produced)
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--act_bits", type=int, default=4)
    p.add_argument("--group_size", type=int, default=128)
    p.add_argument("--act_group_size", type=int, default=128)
    p.add_argument("--symmetric", action="store_true")
    p.add_argument("--real_pack", action="store_true", default=True)
    p.add_argument("--layer_mix", action="store_true",
                   help="W4A4-mix: v_proj/down_proj use mix_group_size")
    p.add_argument("--mix_group_size", type=int, default=32)
    # eval
    p.add_argument("--eval_ppl", action="store_true")
    p.add_argument("--ppl_datasets", nargs="+", default=["wikitext2"],
                   help="Perplexity datasets (paper reports WikiText-2; pass more to add c4/ptb)")
    p.add_argument("--eval_seq_length", type=int, default=2048)
    p.add_argument("--tasks", type=str, default="",
                   help="Comma-separated lm-eval tasks, e.g. piqa,arc_easy")
    p.add_argument("--num_fewshot", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)  # matches evaluate.py (affects C4 sampling)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args, device=device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True)

    if args.eval_ppl:
        eval_ppl(model, args, device=device)
    if args.tasks:
        eval_tasks(model, tokenizer, args, device=device)


if __name__ == "__main__":
    print(sys.argv)
    main()
