"""Time to switch the active task on a resident model.

Two regimes are measured:
  resident: the new task's values are already in device memory. fitln, bitfit and fullft
    overwrite the model's own parameters with a tensor copy (resident_swap). lora1 and lora8
    flip the active adapter, which moves no data (resident_swap).
  cold: the new task's values are in host memory and are copied in (cold_swap), for every
    method.
Each method is also timed with the other mechanism. fitln, bitfit and fullft are timed with
Parameter.data reassignment (resident_swap_pointer). lora1 and lora8 are timed with a copy
into fixed A/B buffers (resident_swap_copy). lora1 and lora8 also get merge_swap, the time
to fuse the adapter into the base weights, W' = W + scale * B @ A.

Without --real_model, tensors are synthetic, with the count and size given by registry.py.
With --real_model True the Hugging Face model is loaded and the swap targets the parameters
the method trains (fitln, bitfit, fullft only; for fullft the pointer variant is skipped).
Use --device cuda for GPU numbers; CPU numbers only check that the code runs.

Usage:
    python hotswap/swap_benchmark.py --arch decoder --model llama-3.2-1b --method fitln \
        --device cuda --n_resident 8 --n_switches 300
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch

import registry

DECODER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "decoder")


def _registry(arch: str, model: str) -> dict:
    return (registry.ENCODER_MODELS if arch == "encoder" else registry.DECODER_MODELS)[model]


def build_synthetic_tensors(arch: str, model: str, method: str, dtype: str, device: torch.device,
                            n_copies: int = 1) -> list[list[torch.Tensor]]:
    """n_copies independent task deltas; each is a list of tensors with the tensor count
    of registry.tensor_layout and the parameter count of the registry."""
    reg = _registry(arch, model)
    param_key = registry.PARAM_KEY_BY_METHOD[method]
    if param_key not in reg:
        raise ValueError(f"{method} not defined for {arch}/{model}")
    total_params = reg[param_key]
    n_tensors = registry.tensor_layout(arch, model, method)
    torch_dtype = getattr(torch, registry.TORCH_DTYPE_NAME[dtype])

    copies = []
    for _ in range(n_copies):
        base = total_params // n_tensors
        remainder = total_params - base * n_tensors
        sizes = [base + (1 if i < remainder else 0) for i in range(n_tensors)]
        # torch.empty: the values do not matter, and random sampling is far slower
        copies.append([torch.empty(max(1, s), dtype=torch_dtype, device=device) for s in sizes])
    return copies


def build_synthetic_live_params(arch: str, model: str, method: str, dtype: str,
                                device: torch.device) -> list[torch.Tensor]:
    """Buffers standing in for the model's own parameters, overwritten by copy swaps."""
    return build_synthetic_tensors(arch, model, method, dtype, device, n_copies=1)[0]


def _load_real_backbone(arch: str, model: str, dtype: str, device: torch.device):
    reg = _registry(arch, model)
    torch_dtype = getattr(torch, registry.TORCH_DTYPE_NAME[dtype])

    if arch == "decoder":
        from transformers import AutoModelForCausalLM
        if reg["hf_id"].startswith("Qwen/Qwen3.5"):
            sys.path.append(DECODER_DIR)
            import qwen35_loader
            backbone = qwen35_loader.load_qwen35_text_only(reg["hf_id"], torch_dtype=torch_dtype)
        else:
            backbone = AutoModelForCausalLM.from_pretrained(reg["hf_id"], torch_dtype=torch_dtype)
    else:
        from transformers import AutoModelForSequenceClassification
        backbone = AutoModelForSequenceClassification.from_pretrained(
            reg["hf_id"], num_labels=2, torch_dtype=torch_dtype)
    return backbone.to(device).eval()


def _freeze_fitln_encoder(model) -> None:
    """Attention-output LayerNorm (attn_norm for ModernBERT), final_norm, classifier and
    pooler stay trainable."""
    model_type = model.config.model_type
    if "modernbert" in model_type:
        key = "attn_norm"
    elif "new" in model_type:
        key = "attn_ln"
    else:
        key = "attention.output.LayerNorm"
    model.requires_grad_(False)
    n_trainable = 0
    for name, param in model.named_parameters():
        if key in name or "final_norm" in name or "classifier" in name or "pooler" in name:
            param.requires_grad = True
            n_trainable += 1
    assert n_trainable, f"no parameter matches {key!r}"


def _freeze_bitfit(model) -> None:
    """Biases, classifier, pooler and head stay trainable."""
    model.requires_grad_(False)
    for name, param in model.named_parameters():
        if name.endswith(".bias") or any(k in name for k in ("classifier", "pooler", "head")):
            param.requires_grad = True


def _select_trainable_params(arch: str, method: str, backbone) -> list[torch.nn.Parameter]:
    """Live parameters that a task delta of fitln or bitfit overwrites."""
    if method == "fitln":
        if arch == "decoder":
            sys.path.append(DECODER_DIR)
            import norm_freeze
            norm_freeze.freeze_for_fitln_decoder(backbone)
        else:
            _freeze_fitln_encoder(backbone)
    elif method == "bitfit":
        _freeze_bitfit(backbone)
    else:
        raise ValueError(f"{method!r} does not overwrite parameters in place")
    return [p for p in backbone.parameters() if p.requires_grad]


def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


_warned_no_foreach = False


def timed_copy_swap(live_params: list[torch.Tensor], source_tensors: list[torch.Tensor],
                    device: torch.device) -> float:
    """Overwrite live_params in place with source_tensors. One fused torch._foreach_copy_
    call, so that methods with many small tensors are not dominated by launch overhead; falls
    back to a per-tensor loop if it is rejected."""
    global _warned_no_foreach
    _sync(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        try:
            torch._foreach_copy_(live_params, source_tensors)
        except (AttributeError, RuntimeError):
            if not _warned_no_foreach:
                print("torch._foreach_copy_ unavailable, using a per-tensor loop")
                _warned_no_foreach = True
            for live, src in zip(live_params, source_tensors):
                live.copy_(src, non_blocking=(device.type == "cuda"))
    _sync(device)
    return time.perf_counter() - t0


def timed_data_pointer_swap(live_params: list[torch.nn.Parameter],
                            source_tensors: list[torch.Tensor], device: torch.device) -> float:
    """Reassign each parameter's storage (Parameter.data = tensor); no bytes are copied.
    Inference only."""
    _sync(device)
    t0 = time.perf_counter()
    for p, t in zip(live_params, source_tensors):
        p.data = t
    _sync(device)
    return time.perf_counter() - t0


def build_lora_merge_tensors(arch: str, model: str, method: str, dtype: str, device: torch.device):
    """Weight, A and B matrices for the merge timing. Every targeted projection is taken as
    a hidden_size x hidden_size matrix (an approximation for grouped-query attention)."""
    reg = _registry(arch, model)
    hidden, n_layers, n_proj = reg["hidden_size"], reg["num_layers"], reg["lora_target_proj"]
    rank = 1 if method == "lora1" else 8
    torch_dtype = getattr(torch, registry.TORCH_DTYPE_NAME[dtype])
    n = n_layers * n_proj
    weights = [torch.empty(hidden, hidden, dtype=torch_dtype, device=device) for _ in range(n)]
    a_mats = [torch.empty(rank, hidden, dtype=torch_dtype, device=device) for _ in range(n)]
    b_mats = [torch.empty(hidden, rank, dtype=torch_dtype, device=device) for _ in range(n)]
    return weights, a_mats, b_mats


def timed_lora_merge_once(weights, a_mats, b_mats, device: torch.device, scale: float = 2.0) -> float:
    """Fuse the adapter into the base weights, W' = W + scale * (B @ A). The scale is
    alpha / r = 2, as in lora_decoder.py."""
    _sync(device)
    t0 = time.perf_counter()
    for w, a, b in zip(weights, a_mats, b_mats):
        w.add_(b @ a, alpha=scale)
    _sync(device)
    return time.perf_counter() - t0


def timed_pointer_swap() -> float:
    """Adapter switch as a pointer change (stands in for set_adapter); no data movement."""
    t0 = time.perf_counter()
    _active_adapter_flag = object()
    t1 = time.perf_counter()
    return t1 - t0


def run_benchmark(arch: str, model: str, method: str, dtype: str, device: torch.device,
                  n_resident: int, n_switches: int, real_model: bool) -> dict:
    swap_kind = registry.METHOD_SWAP_KIND[method]

    if device.type == "cpu" and method == "fullft" and not real_model and n_resident > 1:
        # a FullFT delta is backbone sized; n_resident copies in host RAM can exhaust it
        print(f"capping n_resident {n_resident} -> 1 for fullft on cpu")
        n_resident = 1

    if real_model:
        backbone = _load_real_backbone(arch, model, dtype, device)
        if method in ("fitln", "bitfit"):
            live_params = _select_trainable_params(arch, method, backbone)
            resident = [[torch.empty_like(p).normal_() for p in live_params] for _ in range(n_resident)]
            cold = [torch.empty_like(p).normal_().cpu() for p in live_params]
            if device.type == "cuda":
                cold = [t.pin_memory() for t in cold]
        elif method == "fullft":
            live_params = [p for p in backbone.parameters()]
            resident = [[torch.empty_like(p).normal_() for p in live_params] for _ in range(min(n_resident, 1))]
            cold = [torch.empty_like(p).normal_().cpu() for p in live_params]
            if device.type == "cuda":
                cold = [t.pin_memory() for t in cold]
        else:
            raise NotImplementedError("--real_model True is implemented for fitln, bitfit and "
                                      "fullft only; use synthetic tensors for LoRA")
    else:
        live_params = build_synthetic_live_params(arch, model, method, dtype, device)
        resident = build_synthetic_tensors(arch, model, method, dtype, device, n_copies=n_resident)
        cold = build_synthetic_tensors(arch, model, method, dtype, torch.device("cpu"), n_copies=1)[0]
        if device.type == "cuda":
            # pinned memory, so that host-to-device copies run at full bandwidth
            cold = [t.pin_memory() for t in cold]

    resident_times, cold_times, merge_times = [], [], []
    for i in range(n_switches):
        if swap_kind == "pointer" and n_resident > 0:
            resident_times.append(timed_pointer_swap())
        elif n_resident > 0:
            src = resident[i % len(resident)]
            resident_times.append(timed_copy_swap(live_params, src, device))
        cold_times.append(timed_copy_swap(live_params, cold, device))

    resident_pointer_times = []
    if swap_kind == "copy" and n_resident > 0 and not (real_model and method == "fullft"):
        pointer_live_params = build_synthetic_live_params(arch, model, method, dtype, device) \
            if not real_model else [p.clone() for p in live_params]
        for i in range(n_switches):
            src = resident[i % len(resident)]
            resident_pointer_times.append(timed_data_pointer_swap(pointer_live_params, src, device))

    resident_copy_times = []
    if swap_kind == "pointer" and n_resident > 0:
        copy_live_params = build_synthetic_live_params(arch, model, method, dtype, device)
        for i in range(n_switches):
            src = resident[i % len(resident)]
            resident_copy_times.append(timed_copy_swap(copy_live_params, src, device))

    if method in ("lora1", "lora8") and not real_model:
        # one merge touches every targeted weight matrix, so it is repeated at most 10 times
        merge_reps = min(n_switches, 10)
        weights, a_mats, b_mats = build_lora_merge_tensors(arch, model, method, dtype, device)
        for _ in range(merge_reps):
            merge_times.append(timed_lora_merge_once(weights, a_mats, b_mats, device))

    def summarize(xs):
        if not xs:
            return None
        ms = [1000 * x for x in xs]
        return dict(mean_ms=statistics.mean(ms),
                    median_ms=statistics.median(ms),
                    std_ms=statistics.pstdev(ms) if len(ms) > 1 else 0.0,
                    min_ms=min(ms), max_ms=max(ms), n=len(ms),
                    raw_ms=ms)

    return dict(
        arch=arch, model=model, method=method, dtype=dtype, device=str(device),
        real_model=real_model, n_resident=n_resident, n_switches=n_switches,
        swap_kind=swap_kind,
        resident_swap=summarize(resident_times),
        resident_swap_pointer=summarize(resident_pointer_times) if resident_pointer_times else None,
        resident_swap_copy=summarize(resident_copy_times) if resident_copy_times else None,
        cold_swap=summarize(cold_times),
        merge_swap=summarize(merge_times) if merge_times else None,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("swap_benchmark")
    parser.add_argument("--arch", choices=["encoder", "decoder"], required=True)
    parser.add_argument("--model", required=True, help="short name from registry.py")
    parser.add_argument("--method", choices=list(registry.METHODS), required=True)
    parser.add_argument("--dtype", choices=list(registry.BYTES_PER_PARAM), default="bf16")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_resident", type=int, default=8,
                        help="Number of task deltas held in memory for the resident swap.")
    parser.add_argument("--n_switches", type=int, default=50)
    parser.add_argument("--real_model", type=lambda s: s.lower() == "true", default=False)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available")
    device = torch.device(args.device)
    if device.type == "cpu":
        print("cpu timings only check that the code runs")

    result = run_benchmark(args.arch, args.model, args.method, args.dtype, device,
                           args.n_resident, args.n_switches, args.real_model)
    print(json.dumps(result, indent=2))
