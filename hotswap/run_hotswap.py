"""One (arch, model, method) run: resident-task capacity (as in capacity.py, without the task
head) and swap latencies (swap_benchmark.py), written to
{out_root}/{arch}/{model}/{method}/hotswap_result.json.

Usage:
    python hotswap/run_hotswap.py --arch decoder --model llama-3.2-1b --method fitln \
        --device cuda --n_resident 8 --n_switches 300 --real_model True
"""
from __future__ import annotations

import argparse
import json
import os

import torch

import capacity
import registry
import swap_benchmark


if __name__ == "__main__":
    parser = argparse.ArgumentParser("run_hotswap")
    parser.add_argument("--arch", choices=["encoder", "decoder"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", choices=list(registry.METHODS), required=True)
    parser.add_argument("--dtype", choices=list(registry.BYTES_PER_PARAM), default="bf16")
    parser.add_argument("--reserve_frac", type=float, default=0.10)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_resident", type=int, default=8)
    parser.add_argument("--n_switches", type=int, default=50)
    parser.add_argument("--real_model", type=lambda s: s.lower() == "true", default=False)
    parser.add_argument("--out_root", default="out/hotswap")
    args = parser.parse_args()

    models = registry.ENCODER_MODELS if args.arch == "encoder" else registry.DECODER_MODELS
    if args.model not in models:
        raise SystemExit(f"{args.model!r} is not in the {args.arch} registry: {sorted(models)}")

    cfg = models[args.model]
    param_key = registry.PARAM_KEY_BY_METHOD[args.method]
    if param_key not in cfg:
        raise SystemExit(f"{args.method!r} is not defined for {args.arch}/{args.model}")

    backbone_bytes = registry.param_bytes(cfg["base_params"], args.dtype)
    per_task_bytes = registry.param_bytes(cfg[param_key], args.dtype)
    shares = registry.METHOD_SHARES_BACKBONE[args.method]
    capacity_by_gpu = {}
    for gpu_label, gpu_gb in registry.GPU_MEMORY_PRESETS_GB.items():
        gpu_bytes = gpu_gb * (1024 ** 3)
        capacity_by_gpu[gpu_label] = capacity.max_resident_tasks(
            gpu_bytes, backbone_bytes if shares else 0.0, per_task_bytes, args.reserve_frac)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available")
    latency = swap_benchmark.run_benchmark(
        args.arch, args.model, args.method, args.dtype, device,
        args.n_resident, args.n_switches, args.real_model)

    result = dict(
        arch=args.arch, model=args.model, method=args.method, dtype=args.dtype,
        backbone_bytes=backbone_bytes, per_task_bytes=per_task_bytes,
        capacity_by_gpu=capacity_by_gpu,
        latency=latency,
    )
    out_path = os.path.join(args.out_root, args.arch, args.model, args.method, "hotswap_result.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print("wrote", out_path)
