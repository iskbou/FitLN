"""Task vectors (delta gamma, delta beta) of the normalization layers: the fine-tuned
value minus the pretrained value, per layer.

TaskVector.deltas maps layer index -> {'gamma': Tensor, 'beta': Tensor or None}. The
LayerNorm parameter names are identified by an architecture-specific substring (see
_arch_family); for decoders (RMSNorm) there is no beta.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

import torch



# Substring identifying the LN parameters per architecture: (attn_only_key, all_ln_key).
_LN_KEY_DISPATCH = {
    "modernbert": ("attn_norm", "norm"),
    "new":        ("attn_ln",   "_ln"),
    "default":    ("attention.output.LayerNorm", "LayerNorm"),
}


def _arch_family(model_type: str) -> str:
    """Architecture family from the model type."""
    if "modernbert" in model_type:
        return "modernbert"
    if "new" in model_type:
        return "new"
    return "default"


def ln_keys(model_type: str, attn_only: bool) -> str:
    """Substring identifying the relevant LayerNorm parameters."""
    fam = _arch_family(model_type)
    attn_key, all_key = _LN_KEY_DISPATCH[fam]
    return attn_key if attn_only else all_key


# Regex for the layer index, e.g. layer.{N}.attention.output.LayerNorm.weight or layers.{N}.attn_norm.weight.
_LAYER_RE = re.compile(r"\.layer[s]?\.(\d+)\.")


def _layer_index(key: str) -> int | None:
    m = _LAYER_RE.search(key)
    return int(m.group(1)) if m else None


def _is_weight(key: str) -> bool:
    return key.endswith(".weight")


def _is_bias(key: str) -> bool:
    return key.endswith(".bias")




@dataclass
class TaskVector:
    """Per-layer (delta gamma, delta beta) of one fine-tuned model.

    deltas: {layer: {'gamma': Tensor, 'beta': Tensor or None}}; several LNs in one layer are
    summed. raw_deltas: unaggregated differences per parameter key. head_delta: optional
    classifier delta, kept apart from the task vector.
    """

    model_type: str
    attn_only: bool
    deltas: dict[int, dict[str, torch.Tensor | None]] = field(default_factory=dict)
    raw_deltas: dict[str, torch.Tensor] = field(default_factory=dict)
    head_delta: dict[str, torch.Tensor] = field(default_factory=dict)


    def layer_indices(self) -> list[int]:
        return sorted(self.deltas.keys())

    def gamma(self, layer: int) -> torch.Tensor | None:
        return self.deltas.get(layer, {}).get("gamma")

    def beta(self, layer: int) -> torch.Tensor | None:
        return self.deltas.get(layer, {}).get("beta")

    def flat(self, include_beta: bool = True) -> torch.Tensor:
        """Concatenate the per-layer gamma (and optionally beta) deltas in layer order."""
        parts: list[torch.Tensor] = []
        for ell in self.layer_indices():
            g = self.gamma(ell)
            if g is not None:
                parts.append(g.flatten())
            if include_beta:
                b = self.beta(ell)
                if b is not None:
                    parts.append(b.flatten())
        if not parts:
            return torch.zeros(0)
        return torch.cat(parts)




def compute_task_vector(
    finetuned_sd: dict[str, torch.Tensor],
    pretrained_sd: dict[str, torch.Tensor],
    model_type: str,
    attn_only: bool,
) -> TaskVector:
    """Fine-tuned minus pretrained LayerNorm parameters matching the architecture's
    substring. Classifier and pooler values go to head_delta.
    """
    key_sub = ln_keys(model_type, attn_only)
    tv = TaskVector(model_type=model_type, attn_only=attn_only)

    for k, v in finetuned_sd.items():
        if key_sub not in k:
            # classifier, pooler: no pretrained value, store the fine-tuned one
            if "classifier" in k or "pooler" in k:
                tv.head_delta[k] = v.detach().clone()
            continue

        if k not in pretrained_sd:
            # key missing from the pretrained model: skip
            continue

        delta = (v - pretrained_sd[k]).detach().clone()
        tv.raw_deltas[k] = delta

        layer = _layer_index(k)
        if layer is None:
            # not a per-layer LN (e.g. the final norm): synthetic layer -1
            layer = -1
        slot = tv.deltas.setdefault(layer, {"gamma": None, "beta": None})

        if _is_weight(k):
            slot["gamma"] = delta if slot["gamma"] is None else slot["gamma"] + delta
        elif _is_bias(k):
            slot["beta"] = delta if slot["beta"] is None else slot["beta"] + delta

    return tv




def magnitude_profile(tv: TaskVector) -> dict[str, list[float]]:
    """Per-layer L2 norms of delta gamma and delta beta:
    {'layers': [int], 'gamma_norm': [float], 'beta_norm': [float or nan]}.
    """
    import math
    layers = tv.layer_indices()
    gamma_norm = []
    beta_norm = []
    for ell in layers:
        g = tv.gamma(ell)
        b = tv.beta(ell)
        gamma_norm.append(float(g.norm().item()) if g is not None else math.nan)
        beta_norm.append(float(b.norm().item()) if b is not None else math.nan)
    return {"layers": layers, "gamma_norm": gamma_norm, "beta_norm": beta_norm}


def sparsity_profile(
    tv: TaskVector,
    pretrained_sd: dict[str, torch.Tensor],
    eps: float = 0.01,
) -> dict[str, list[float]]:
    """Per-layer fraction of channels with |delta gamma| > eps * max|pretrained gamma|."""
    import math
    layers = tv.layer_indices()
    out = {"layers": layers, "gamma_active_frac": [], "beta_active_frac": []}
    for ell in layers:
        # matching pretrained key for this layer's gamma
        gamma_pre_inf = None
        beta_pre_inf = None
        for k in tv.raw_deltas.keys():
            li = _layer_index(k)
            if li != ell:
                continue
            base = pretrained_sd.get(k)
            if base is None:
                continue
            if _is_weight(k) and gamma_pre_inf is None:
                gamma_pre_inf = float(base.abs().max().item())
            elif _is_bias(k) and beta_pre_inf is None:
                beta_pre_inf = float(base.abs().max().item())

        g = tv.gamma(ell)
        b = tv.beta(ell)
        if g is not None and gamma_pre_inf:
            out["gamma_active_frac"].append(float((g.abs() > eps * gamma_pre_inf).float().mean().item()))
        else:
            out["gamma_active_frac"].append(math.nan)
        if b is not None and beta_pre_inf:
            out["beta_active_frac"].append(float((b.abs() > eps * beta_pre_inf).float().mean().item()))
        else:
            out["beta_active_frac"].append(math.nan)
    return out


def cosine(tv_a: TaskVector, tv_b: TaskVector, include_beta: bool = True) -> float:
    """Cosine similarity between two task vectors."""
    a = tv_a.flat(include_beta=include_beta)
    b = tv_b.flat(include_beta=include_beta)
    if a.numel() == 0 or b.numel() == 0 or a.numel() != b.numel():
        raise ValueError(f"Incompatible task-vector shapes: {a.shape} vs {b.shape}. "
                         "Are these from the same model and attn_only setting?")
    return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def gamma_beta_ratio(tv: TaskVector) -> list[float]:
    """Per-layer norm(delta gamma) / norm(delta beta); NaN where beta is absent."""
    import math
    out = []
    for ell in tv.layer_indices():
        g = tv.gamma(ell)
        b = tv.beta(ell)
        if g is None or b is None or b.norm().item() == 0.0:
            out.append(math.nan)
        else:
            out.append(float((g.norm() / b.norm()).item()))
    return out


def hand_verify_norm(
    finetuned_sd: dict[str, torch.Tensor],
    pretrained_sd: dict[str, torch.Tensor],
    key: str,
) -> float:
    """L2 norm of fine-tuned[key] - pretrained[key], to cross-check raw_deltas."""
    return float((finetuned_sd[key] - pretrained_sd[key]).norm().item())
