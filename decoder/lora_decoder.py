r"""LoRA baseline on the attention projections (lora_target_modules in common.py), with
the LM head frozen. lora_alpha defaults to 2 * lora_r and lora_dropout to 0.05;
--include_mlp_targets (default False) also adapts the MLP projections.

Usage:
    python -u decoder/lora_decoder.py --model Qwen/Qwen2.5-1.5B --task sst2 \
        --lr 2e-4 --lora_r 8 --batch_size 8 --num_epoch 3 --seed 1
"""
import argparse
import os

from peft import LoraConfig, TaskType, get_peft_model

import common
import run_common

if __name__ == "__main__":
    parser = argparse.ArgumentParser("lora_decoder")
    common.add_common_decoder_args(parser, default_lr=1e-4)
    parser.add_argument("--lora_r", type=int, default=8, choices=[8, 1])
    parser.add_argument("--lora_alpha", type=int, default=None,
                         help="Defaults to 2*lora_r.")
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--include_mlp_targets", type=common.str2bool, default=False)
    args = parser.parse_args()

    run_common.set_seed(args.seed)
    print("Arguments parsed\n\n")

    tokenizer = run_common.load_tokenizer(args.model)
    model = run_common.load_causal_lm(args.model)
    run_common.log_model_facts(model, args.model)

    model_cfg = common.MODEL_CONFIGS[args.model]
    target_modules = model_cfg["lora_target_modules_with_mlp"] if args.include_mlp_targets else model_cfg["lora_target_modules"]
    assert target_modules is not None, (
        f"common.MODEL_CONFIGS[{args.model!r}] has no lora_target_modules."
    )
    lora_alpha = args.lora_alpha if args.lora_alpha is not None else 2 * args.lora_r
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    print("Model loaded (full precision) and LoRA-wrapped\n\n")

    def save_adapter(model, output_dir: str) -> str:
        adapter_dir = os.path.join(output_dir, "adapter")
        model.save_pretrained(adapter_dir)
        print(f"[lora_decoder] Saved LoRA adapter to {adapter_dir}")
        return adapter_dir

    method = f"lora_r{args.lora_r}" + ("" if args.include_mlp_targets else "_attnonly")
    run_common.run_training(
        args,
        method=method,
        model=peft_model,
        tokenizer=tokenizer,
        compact_save_fn=save_adapter,
    )
