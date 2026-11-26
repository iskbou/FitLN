r"""Full fine-tuning baseline: all parameters trainable, weights loaded in bf16.

Usage:
    python -u decoder/fullft_decoder.py --model Qwen/Qwen2.5-1.5B --task sst2 \
        --lr 2e-5 --batch_size 8 --num_epoch 3 --seed 1
"""
import argparse

import torch

import common
import run_common

if __name__ == "__main__":
    parser = argparse.ArgumentParser("fullft_decoder")
    common.add_common_decoder_args(parser, default_lr=2e-5)
    args = parser.parse_args()

    run_common.set_seed(args.seed)
    print("Arguments parsed\n\n")

    tokenizer = run_common.load_tokenizer(args.model)
    # bf16 weights halve the checkpoint size.
    model = run_common.load_causal_lm(args.model, torch_dtype=torch.bfloat16)
    run_common.log_model_facts(model, args.model)
    model.requires_grad_(True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    print("Model loaded (full fine-tuning, no freezing)\n\n")

    def save_marker(model, output_dir: str) -> str:
        # No compact checkpoint for full fine-tuning.
        return None

    run_common.run_training(
        args,
        method="fullft",
        model=model,
        tokenizer=tokenizer,
        compact_save_fn=save_marker,
        # Smaller checkpoints; an interrupted run cannot resume.
        training_args_overrides={"save_only_model": True, "save_total_limit": 1},
    )
