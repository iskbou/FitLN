"""Task-vector analysis for the decoder FitLN checkpoints (scale-only, no beta).

Checkpoints are read from out/decoder/fitln/{model}/{task}/{lr}/{seed}/
fitln_decoder_weights.bin. TaskVector and cosine come from analysis.task_vectors, so the
repository root must be on sys.path (added below).
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.task_vectors import TaskVector, cosine  # noqa: E402

FITLN_WEIGHTS_FILE = "fitln_decoder_weights.bin"

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def _layer_index(key: str) -> int | None:
    m = _LAYER_RE.search(key)
    return int(m.group(1)) if m else None




@dataclass
class DecoderCheckpointRef:
    """One FitLN checkpoint on disk."""
    model_short: str
    task: str
    lr: float
    seed: int
    ckpt_path: Path
    val_loss: float | None
    test_metric: float | None
    metric_name: str | None

    def load(self, map_location: str = "cpu") -> dict[str, torch.Tensor]:
        return torch.load(self.ckpt_path, map_location=map_location, weights_only=True)


def _read_json(path: Path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def list_fitln_checkpoints(
    root: str | Path = "out/decoder/fitln",
    model_short: str | None = None,
    task: str | None = None,
) -> list[DecoderCheckpointRef]:
    """Return every complete FitLN checkpoint under root/{model}/{task}/{lr}/{seed}/.
    val_loss is eval_loss from val_results.json; test_metric is the first test_* scalar
    in test_results.json (informational only).
    """
    root = Path(root)
    refs: list[DecoderCheckpointRef] = []
    if not root.exists():
        return refs

    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir() or (model_short and model_dir.name != model_short):
            continue
        for task_dir in sorted(model_dir.iterdir()):
            if not task_dir.is_dir() or (task and task_dir.name != task):
                continue
            for lr_dir in sorted(task_dir.iterdir()):
                try:
                    lr = float(lr_dir.name)
                except ValueError:
                    continue
                for seed_dir in sorted(lr_dir.iterdir()):
                    try:
                        seed = int(seed_dir.name)
                    except ValueError:
                        continue
                    ckpt = seed_dir / FITLN_WEIGHTS_FILE
                    if not ckpt.exists():
                        continue
                    val = _read_json(seed_dir / "val_results.json") or {}
                    test = _read_json(seed_dir / "test_results.json") or {}
                    test_metric, metric_name = None, None
                    for k, v in test.items():
                        if k.startswith("test_") and isinstance(v, (int, float)):
                            test_metric, metric_name = v, k
                            break
                    refs.append(DecoderCheckpointRef(
                        model_short=model_dir.name, task=task_dir.name, lr=lr, seed=seed,
                        ckpt_path=ckpt, val_loss=val.get("eval_loss"),
                        test_metric=test_metric, metric_name=metric_name,
                    ))
    return refs


def is_valid_seed_run(ref: DecoderCheckpointRef) -> bool:
    """False if the validation loss or any checkpoint tensor is non-finite."""
    if ref.val_loss is None or not math.isfinite(ref.val_loss):
        return False
    sd = ref.load()
    return all(torch.isfinite(v).all() for v in sd.values())


def select_best_lr(
    refs: list[DecoderCheckpointRef],
    require_valid: bool = True,
) -> dict[tuple[str, str], list[DecoderCheckpointRef]]:
    """Same selection as aggregate_decoder_results.select_best_lr_across_seeds: per
    (model, task), the learning rate with the lowest mean validation loss across seeds;
    returns every valid seed run at that learning rate.
    """
    groups: dict[tuple[str, str], dict[float, list[DecoderCheckpointRef]]] = {}
    for r in refs:
        if require_valid and not is_valid_seed_run(r):
            continue
        groups.setdefault((r.model_short, r.task), {}).setdefault(r.lr, []).append(r)

    best: dict[tuple[str, str], list[DecoderCheckpointRef]] = {}
    for key, by_lr in groups.items():
        best_lr, best_mean = None, math.inf
        for lr, lr_refs in by_lr.items():
            losses = [r.val_loss for r in lr_refs if r.val_loss is not None]
            if not losses:
                continue
            mean_loss = sum(losses) / len(losses)
            if mean_loss < best_mean:
                best_mean, best_lr = mean_loss, lr
        if best_lr is not None:
            best[key] = sorted(by_lr[best_lr], key=lambda r: r.seed)
    return best




def extract_pretrained_norms(model) -> dict[str, torch.Tensor]:
    """Keep only the norm tensors needed for the task vector (per-layer
    post_attention_layernorm.weight and model.norm.weight), so the model can be freed.
    """
    sd = model.state_dict()
    return {
        k: v.detach().cpu().clone() for k, v in sd.items()
        if k.endswith("post_attention_layernorm.weight") or k == "model.norm.weight"
    }


def compute_task_vector_decoder(
    finetuned_sd: dict[str, torch.Tensor],
    pretrained_sd: dict[str, torch.Tensor],
) -> TaskVector:
    """Scale-only task vector. post_attention_layernorm deltas go in per-layer buckets and
    model.norm.weight in layer -1. There is no beta term.
    """
    tv = TaskVector(model_type="decoder", attn_only=True)
    for k, v in finetuned_sd.items():
        if k not in pretrained_sd:
            continue
        delta = (v - pretrained_sd[k]).detach().clone()
        tv.raw_deltas[k] = delta
        is_layer_norm = k.endswith("post_attention_layernorm.weight")
        layer = _layer_index(k) if is_layer_norm else -1
        slot = tv.deltas.setdefault(layer, {"gamma": None, "beta": None})
        slot["gamma"] = delta if slot["gamma"] is None else slot["gamma"] + delta
    return tv


# Seed stability: do independent seeds move the same channels?


def active_mask(delta_gamma: torch.Tensor, pretrained_gamma: torch.Tensor, eps: float = 0.01) -> torch.Tensor:
    """Channels where |delta| > eps * max|pretrained| (same threshold as sparsity_profile)."""
    thresh = eps * pretrained_gamma.abs().max()
    return delta_gamma.abs() > thresh


def jaccard(mask_a: torch.Tensor, mask_b: torch.Tensor) -> float:
    inter = (mask_a & mask_b).sum().item()
    union = (mask_a | mask_b).sum().item()
    return inter / union if union > 0 else math.nan


def seed_pair_cosine(tv_a: TaskVector, tv_b: TaskVector) -> float:
    """Cosine similarity between the task vectors of two seeds of the same (model, task)."""
    return cosine(tv_a, tv_b, include_beta=False)


def per_layer_seed_jaccard(
    tv_a: TaskVector,
    tv_b: TaskVector,
    pretrained_sd: dict[str, torch.Tensor],
    eps: float = 0.01,
) -> dict[int, float]:
    """Per-layer Jaccard similarity of the active channels of two seeds' task vectors."""
    layers = sorted(set(tv_a.layer_indices()) & set(tv_b.layer_indices()))
    out: dict[int, float] = {}
    for ell in layers:
        ga, gb = tv_a.gamma(ell), tv_b.gamma(ell)
        if ga is None or gb is None:
            continue
        pre_key = "model.norm.weight" if ell == -1 else next(
            (k for k in tv_a.raw_deltas if _layer_index(k) == ell
             and k.endswith("post_attention_layernorm.weight")),
            None,
        )
        if pre_key is None or pre_key not in pretrained_sd:
            continue
        pre = pretrained_sd[pre_key]
        out[ell] = jaccard(active_mask(ga, pre, eps), active_mask(gb, pre, eps))
    return out
