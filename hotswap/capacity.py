"""Maximum number of task-specific parameter sets resident on one GPU next to one frozen
backbone, from the parameter counts in registry.py. No GPU or download needed.

Resident weights only: backbone plus N task deltas must fit in (1 - reserve_frac) of the GPU
memory. Activations, KV cache and CUDA context are not modelled; --reserve_frac stands in for
them. FullFT has no shared backbone, so each resident task is a full copy of the model.
For encoders every method also stores the classifier/pooler head (registry head_params); use
--include_head false to count the deltas alone. Decoders have no task head.

Usage:
    python hotswap/capacity.py --dtype bf16 --reserve_frac 0.10 --out_dir results/hotswap
    python hotswap/capacity.py --include_head false --out_dir results/hotswap
"""
from __future__ import annotations

import argparse
import json
import os

import registry

GIB = 1024 ** 3


def str2bool(v):
    return v.lower() in ("true", "1", "yes")


def max_resident_tasks(gpu_bytes: float, backbone_bytes: float, per_task_bytes: float,
                       reserve_frac: float = 0.10) -> int:
    usable = gpu_bytes * (1 - reserve_frac) - backbone_bytes
    if usable <= 0 or per_task_bytes <= 0:
        return 0
    return int(usable // per_task_bytes)


def build_rows(dtype: str = "bf16", reserve_frac: float = 0.10, include_head: bool = True) -> list[dict]:
    rows = []
    for arch, models in (("encoder", registry.ENCODER_MODELS), ("decoder", registry.DECODER_MODELS)):
        for model, cfg in models.items():
            backbone_bytes = registry.param_bytes(cfg["base_params"], dtype)
            head_bytes = registry.param_bytes(cfg.get("head_params", 0), dtype) if include_head else 0.0
            for method in registry.METHODS:
                param_key = registry.PARAM_KEY_BY_METHOD[method]
                if param_key not in cfg:
                    continue
                shares = registry.METHOD_SHARES_BACKBONE[method]
                per_task_bytes = registry.param_bytes(cfg[param_key], dtype) + head_bytes
                for gpu_label, gpu_gb in registry.GPU_MEMORY_PRESETS_GB.items():
                    gpu_bytes = gpu_gb * GIB
                    n_tasks = max_resident_tasks(gpu_bytes, backbone_bytes if shares else 0.0,
                                                 per_task_bytes, reserve_frac)
                    rows.append(dict(
                        arch=arch, model=model, method=method, gpu=gpu_label, gpu_gb=gpu_gb,
                        dtype=dtype, backbone_bytes=backbone_bytes, per_task_bytes=per_task_bytes,
                        head_bytes=head_bytes, n_tasks=n_tasks,
                    ))
    return rows


def to_markdown(rows: list[dict], dtype: str, reserve_frac: float, include_head: bool) -> str:
    gpu_labels = list(registry.GPU_MEMORY_PRESETS_GB)
    head = "included" if include_head else "not included"
    lines = [
        "# Capacity: maximum resident tasks per GPU memory budget",
        "",
        f"Weights in {dtype}, {100 * reserve_frac:.0f}% of GPU memory reserved. Encoder task "
        f"head: {head}. FullFT: one full model copy per task. Parameter counts are those "
        "of the paper tables.",
    ]
    for arch, models in (("encoder", registry.ENCODER_MODELS), ("decoder", registry.DECODER_MODELS)):
        lines += ["", f"## {arch.capitalize()}"]
        for model in models:
            lines += ["", f"### {model}", "", "| Method | " + " | ".join(gpu_labels) + " |",
                      "|---|" + "|".join(["---"] * len(gpu_labels)) + "|"]
            for method in registry.METHODS:
                cell_rows = [r for r in rows if r["arch"] == arch and r["model"] == model and r["method"] == method]
                if not cell_rows:
                    continue
                by_gpu = {r["gpu"]: r["n_tasks"] for r in cell_rows}
                cells = " | ".join(str(by_gpu.get(g, "-")) for g in gpu_labels)
                lines.append(f"| {method} (delta={registry.human_bytes(cell_rows[0]['per_task_bytes'])}) | {cells} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser("capacity")
    parser.add_argument("--dtype", choices=list(registry.BYTES_PER_PARAM), default="bf16")
    parser.add_argument("--reserve_frac", type=float, default=0.10)
    parser.add_argument("--include_head", type=str2bool, default=True)
    parser.add_argument("--out_dir", default="results/hotswap")
    args = parser.parse_args()

    rows = build_rows(dtype=args.dtype, reserve_frac=args.reserve_frac, include_head=args.include_head)
    suffix = "" if args.include_head else "_no_head"
    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, f"capacity_table{suffix}.json")
    md_path = os.path.join(args.out_dir, f"capacity_table{suffix}.md")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    with open(md_path, "w") as f:
        f.write(to_markdown(rows, args.dtype, args.reserve_frac, args.include_head))
    print(f"wrote {len(rows)} rows to {json_path} and {md_path}")
