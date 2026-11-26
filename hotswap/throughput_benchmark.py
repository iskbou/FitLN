"""Inference and training-step throughput per method on one decoder model.

Run it alone on a GPU: other jobs on the same device change the timings. Each timing
discards a warm-up phase and uses CUDA events. Per method it measures
  inference: forward pass only, samples/s;
  train_step: forward, backward and optimizer step, samples/s.
lora1 and lora8 get a second inference row with the adapter merged in place (merge_adapter),
to measure the forward-pass overhead of the unmerged adapter. The same dtype is used for all
methods. The reported results use --dtype fp16.

Usage:
    python hotswap/throughput_benchmark.py --model meta-llama/Llama-3.2-1B --dtype fp16 \
        --methods fitln fullft lora1 lora8 --batch_size 8 --seq_len 256 --n_warmup 10 --n_iters 50

Writes {results_dir}/throughput_{model}.json and .md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

DECODER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "decoder")
sys.path.insert(0, DECODER_DIR)

import common as decoder_common
import norm_freeze
import run_common


def _timed_run(fn, n_warmup: int, n_iters: int, device: torch.device) -> dict:
    """Mean, median and std in ms of n_iters calls of fn after n_warmup discarded calls."""
    for _ in range(n_warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()

    times_ms = []
    for _ in range(n_iters):
        if device.type == "cuda":
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times_ms.append(start.elapsed_time(end))
        else:
            t0 = time.perf_counter()
            fn()
            times_ms.append((time.perf_counter() - t0) * 1000)

    mean = sum(times_ms) / len(times_ms)
    sorted_t = sorted(times_ms)
    median = sorted_t[len(sorted_t) // 2]
    std = (sum((t - mean) ** 2 for t in times_ms) / len(times_ms)) ** 0.5
    return {"mean_ms": mean, "median_ms": median, "std_ms": std, "n": n_iters}


def make_batch(tokenizer, batch_size: int, seq_len: int, device: torch.device) -> dict:
    """Random token ids of a fixed shape; throughput does not depend on the content."""
    vocab_size = tokenizer.vocab_size
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def setup_method(model_name: str, method: str, torch_dtype):
    """Returns (model, number of trainable parameters, info). Freezing and LoRA settings
    follow fitln_decoder.py, lora_decoder.py (dropout 0, alpha = 2r) and fullft_decoder.py."""
    model = run_common.load_causal_lm(model_name, torch_dtype=torch_dtype)
    info = {}

    if method == "fitln":
        norm_freeze.freeze_for_fitln_decoder(model, include_final_norm=True)
    elif method == "fullft":
        for p in model.parameters():
            p.requires_grad_(True)
    elif method in ("lora1", "lora8"):
        from peft import LoraConfig, TaskType, get_peft_model
        r = 1 if method == "lora1" else 8
        target_modules = decoder_common.MODEL_CONFIGS[model_name]["lora_target_modules"]
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False,
            r=r, lora_alpha=2 * r, lora_dropout=0.0, target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)
        info["peft_model"] = True
    else:
        raise ValueError(f"unknown method {method!r}")

    # gradient checkpointing as in training; without enable_input_require_grads a frozen
    # backbone gives no gradients under checkpointing
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, trainable, info


def run_one(model_name: str, method: str, device: torch.device,
            batch_size: int, seq_len: int, n_warmup: int, n_iters: int,
            torch_dtype) -> list[dict]:
    """Result rows of one method: inference, train_step, and for LoRA a merged inference row."""
    tokenizer = run_common.load_tokenizer(model_name)
    model, trainable, info = setup_method(model_name, method, torch_dtype)
    model = model.to(device)
    model.train()

    rows = []
    batch = make_batch(tokenizer, batch_size, seq_len, device)

    model.eval()
    with torch.no_grad():
        infer = _timed_run(lambda: model(input_ids=batch["input_ids"],
                                         attention_mask=batch["attention_mask"]),
                           n_warmup, n_iters, device)
    rows.append(dict(method=method, regime="inference", merged=False,
                     trainable_params=trainable, **infer))

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    model.train()

    def train_step():
        optim.zero_grad(set_to_none=True)
        out = model(**batch)
        out.loss.backward()
        optim.step()

    train = _timed_run(train_step, n_warmup, n_iters, device)
    rows.append(dict(method=method, regime="train_step", merged=False,
                     trainable_params=trainable, **train))

    if info.get("peft_model"):
        model.eval()
        model.merge_adapter()
        with torch.no_grad():
            infer_merged = _timed_run(lambda: model(input_ids=batch["input_ids"],
                                                    attention_mask=batch["attention_mask"]),
                                      n_warmup, n_iters, device)
        rows.append(dict(method=method, regime="inference", merged=True,
                         trainable_params=trainable, **infer_merged))
        model.unmerge_adapter()

    # optimizer state holds GPU tensors of its own, so delete it before emptying the cache
    del model, optim, batch
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def to_markdown(model_name: str, rows: list[dict], batch_size: int, seq_len: int, dtype: str) -> str:
    lines = [
        f"# Throughput: {model_name}",
        "",
        f"Single process, batch_size={batch_size}, seq_len={seq_len}, dtype={dtype} for all methods.",
        "",
        "| Method | Regime | Merged | Trainable params | samples/sec | ms/batch (mean) | ms/batch (median) | std |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        sps = batch_size / (r["mean_ms"] / 1000)
        lines.append(
            f"| {r['method']} | {r['regime']} | {r['merged']} | {r['trainable_params']:,} | "
            f"{sps:.1f} | {r['mean_ms']:.3f} | {r['median_ms']:.3f} | {r['std_ms']:.3f} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser("throughput_benchmark")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--methods", nargs="*", default=["fitln", "fullft", "lora1", "lora8"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--n_warmup", type=int, default=10)
    parser.add_argument("--n_iters", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "bf16", "fp16"],
                        help="Applied to every method.")
    parser.add_argument("--results_dir", default="results/hotswap")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}, dtype={args.dtype}")
    else:
        print("cpu timings only check that the code runs")

    all_rows = []
    for method in args.methods:
        print(f"=== {method} ===")
        rows = run_one(args.model, method, device, args.batch_size, args.seq_len,
                       args.n_warmup, args.n_iters, torch_dtype)
        for r in rows:
            print(f"  {r['regime']:12} merged={r['merged']!s:5} "
                  f"mean={r['mean_ms']:.3f}ms  samples/sec={args.batch_size/(r['mean_ms']/1000):.1f}")
        all_rows.extend(rows)

    os.makedirs(args.results_dir, exist_ok=True)
    model_tag = args.model.split("/")[-1].lower().replace(".", "_").replace("-", "_")
    json_path = os.path.join(args.results_dir, f"throughput_{model_tag}.json")
    md_path = os.path.join(args.results_dir, f"throughput_{model_tag}.md")
    with open(json_path, "w") as f:
        json.dump(dict(model=args.model, batch_size=args.batch_size, seq_len=args.seq_len,
                       device=str(device), dtype=args.dtype, rows=all_rows), f, indent=2)
    with open(md_path, "w") as f:
        f.write(to_markdown(args.model, all_rows, args.batch_size, args.seq_len, args.dtype))
    print("wrote", json_path, md_path)
