"""Model configs and shared command-line arguments for the decoder scripts."""
import argparse

# Llama-3.2 and Qwen2.5 share module names (post_attention_layernorm, model.norm,
# q/k/v/o_proj, gate/up/down_proj), so norm_freeze.py needs no per-architecture code.
# Qwen3.5 alternates linear-attention and full-attention layers, so its LoRA targets
# list the projections of both layer types. lora_target_modules is attention-only;
# lora_target_modules_with_mlp (lora_decoder.py --include_mlp_targets) is not used for
# the reported results.
_ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]
_MLP = ["gate_proj", "up_proj", "down_proj"]
_QWEN35_ATTN = _ATTN + ["in_proj_a", "in_proj_b", "in_proj_qkv", "in_proj_z", "out_proj"]

MODEL_CONFIGS = {
    "Qwen/Qwen2.5-1.5B": dict(
        model_type="qwen2",
        lora_target_modules=_ATTN,
        lora_target_modules_with_mlp=_ATTN + _MLP,
        weight_decay=0.01,
    ),
    "meta-llama/Llama-3.2-1B": dict(
        model_type="llama",
        lora_target_modules=_ATTN,
        lora_target_modules_with_mlp=_ATTN + _MLP,
        weight_decay=0.01,
    ),
    "meta-llama/Llama-3.2-3B": dict(
        model_type="llama",
        lora_target_modules=_ATTN,
        lora_target_modules_with_mlp=_ATTN + _MLP,
        weight_decay=0.01,
    ),
    "Qwen/Qwen2.5-3B": dict(
        model_type="qwen2",
        lora_target_modules=_ATTN,
        lora_target_modules_with_mlp=_ATTN + _MLP,
        weight_decay=0.01,
    ),
    # 24 layers: 18 linear attention, 6 full attention.
    "Qwen/Qwen3.5-2B-Base": dict(
        model_type="qwen3_5_text",
        needs_text_extraction=True,  # loaded by qwen35_loader
        lora_target_modules=_QWEN35_ATTN,
        lora_target_modules_with_mlp=_QWEN35_ATTN + _MLP,
        weight_decay=0.01,
    ),
    # 32 layers: 24 linear attention, 8 full attention.
    "Qwen/Qwen3.5-4B-Base": dict(
        model_type="qwen3_5_text",
        needs_text_extraction=True,
        lora_target_modules=_QWEN35_ATTN,
        lora_target_modules_with_mlp=_QWEN35_ATTN + _MLP,
        weight_decay=0.01,
    ),
}

# Filesystem-safe model names used in output paths.
MODEL_SHORT = {
    "Qwen/Qwen2.5-1.5B": "qwen2_5_1_5b",
    "meta-llama/Llama-3.2-1B": "llama3_2_1b",
    "Qwen/Qwen3.5-2B-Base": "qwen3_5_2b_base",
    "meta-llama/Llama-3.2-3B": "llama3_2_3b",
    "Qwen/Qwen2.5-3B": "qwen2_5_3b",
    "Qwen/Qwen3.5-4B-Base": "qwen3_5_4b_base",
}

# GLUE tasks as generation, scored by label log-likelihood.
GLUE_GEN_TASKS = ["sst2", "rte", "mrpc", "cola"]

# Free-form generation tasks; each needs an entry in run_common.FREE_GEN_REGISTRY.
FREE_GEN_TASKS = ["alpaca_mini", "gsm8k", "samsum"]

ALL_TASKS = GLUE_GEN_TASKS + FREE_GEN_TASKS

# task -> (train_n argument, eval_n argument).
FREE_GEN_DATA_ARGS = {
    "alpaca_mini": ("alpaca_train_n", "alpaca_eval_n"),
    "gsm8k": ("gsm8k_train_n", "gsm8k_eval_n"),
    "samsum": ("samsum_train_n", "samsum_eval_n"),
}

# task -> (metric key in test_results.json, label for tables).
FREE_GEN_METRIC = {
    "alpaca_mini": ("rouge_l", "ROUGE-L"),
    "samsum": ("rouge_l", "ROUGE-L"),
    "gsm8k": ("exact_match", "Exact Match"),
}


def task_type(task: str) -> str:
    """Return 'glue_gen' or 'free_gen'."""
    if task in GLUE_GEN_TASKS:
        return "glue_gen"
    if task in FREE_GEN_TASKS:
        return "free_gen"
    raise ValueError(f"Unknown task {task!r}; expected one of {ALL_TASKS}")


def model_short_name(model_name: str) -> str:
    """Short model name from MODEL_SHORT, or a derived one."""
    return MODEL_SHORT.get(model_name, model_name.split("/")[-1].replace("-", "_").lower())


def str2bool(v):
    """Boolean parser for argparse (type=bool treats the string 'False' as True)."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def add_common_decoder_args(parser: argparse.ArgumentParser, default_lr: float) -> argparse.ArgumentParser:
    """Arguments shared by fitln_decoder.py, lora_decoder.py and fullft_decoder.py."""
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B", choices=sorted(MODEL_CONFIGS.keys()))
    parser.add_argument("--task", default="sst2", choices=ALL_TASKS)
    parser.add_argument("--lr", default=default_lr, type=float)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--num_epoch", default=3, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--max_seq_length", default=256, type=int)
    parser.add_argument("--alpaca_train_n", default=50000, type=int,
                         help="alpaca_mini: number of training examples (of 52,002).")
    parser.add_argument("--alpaca_eval_n", default=None, type=int,
                         help="alpaca_mini: eval examples; default is all the examples left (2,002).")
    parser.add_argument("--gsm8k_train_n", default=None, type=int,
                         help="gsm8k: cap on training examples; default is the full split (7,473).")
    parser.add_argument("--gsm8k_eval_n", default=None, type=int,
                         help="gsm8k: cap on eval examples; default is the full test split (1,319).")
    parser.add_argument("--samsum_train_n", default=None, type=int,
                         help="samsum: cap on training examples; default is the full split (14,731).")
    parser.add_argument("--samsum_eval_n", default=None, type=int,
                         help="samsum: cap on eval examples; default is the full test split (819).")
    parser.add_argument("--max_new_tokens", default=64, type=int,
                         help="Tokens generated at eval for free-form tasks; use 256 for gsm8k.")
    parser.add_argument("--gen_batch_size", default=16, type=int,
                         help="Batch size for generation at eval; lower it if eval runs out of memory.")
    parser.add_argument("--output_dir", default="out/decoder/", help="Root output dir; method/model/task/lr/seed are appended.")
    parser.add_argument(
        "--dry_run", type=str2bool, default=False,
        help="Unused.",
    )
    return parser


def build_output_dir(root: str, method: str, model_name: str, task: str, lr: float, seed: int) -> str:
    """{root}/{method}/{model_short}/{task}/{lr}/{seed}/"""
    import os
    import numpy as np

    return os.path.join(
        root,
        method,
        model_short_name(model_name),
        task,
        str(np.round(lr, 6)),
        str(seed),
    )
