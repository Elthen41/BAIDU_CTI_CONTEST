#!/usr/bin/env bash
# Evaluate Qwen2.5-7B W4A4 accuracy (WikiText-2 PPL + zero-shot tasks).
# Set CKPT to your pre-quantized checkpoint directory. It is self-contained
# (config.json + tokenizer + *.safetensors), so it is used for BOTH
# --model_name_or_path and --model_path.
set -e

# Variant 1 — uniform group 128
CKPT=${CKPT:-./checkpoints/qwen2.5-7b-w4a4-g128}
python eval_model.py \
    --model_name_or_path "$CKPT" --model_path "$CKPT" \
    --bits 4 --act_bits 4 \
    --group_size 128 --act_group_size 128 \
    --symmetric --real_pack \
    --eval_ppl --ppl_datasets wikitext2 --seed 42 \
    --tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande \
    --num_fewshot 0 --batch_size 8

# Variant 2 — channel weights + v_proj/down_proj group 32 (uncomment to use)
# CKPT=${CKPT:-./checkpoints/qwen2.5-7b-w4a4-mix}
# python eval_model.py \
#     --model_name_or_path "$CKPT" --model_path "$CKPT" \
#     --bits 4 --act_bits 4 \
#     --layer_mix --group_size -1 --mix_group_size 32 --act_group_size -1 \
#     --symmetric --real_pack \
#     --eval_ppl --ppl_datasets wikitext2 --seed 42 \
#     --tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande \
#     --num_fewshot 0 --batch_size 8
