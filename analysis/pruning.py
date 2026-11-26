"""Top-k channel pruning of a FitLN checkpoint.

Each LayerNorm keeps only its k% channels with the largest |delta gamma|; the other channels
(gamma and beta) are reset to the pretrained values.
"""

from __future__ import annotations

from typing import Literal

import torch

from .task_vectors import _layer_index, _is_weight, _is_bias, ln_keys


def _ln_param_keys(finetuned_sd: dict[str, torch.Tensor],
                   model_type: str, attn_only: bool) -> list[str]:
    key_sub = ln_keys(model_type, attn_only)
    return [k for k in finetuned_sd if key_sub in k]


def topk_channel_mask(
    finetuned_sd: dict[str, torch.Tensor],
    pretrained_sd: dict[str, torch.Tensor],
    model_type: str,
    attn_only: bool,
    k_frac: float,
    selector: Literal["gamma", "delta_norm"] = "gamma",
) -> dict[str, torch.Tensor]:
    """State dict where each LayerNorm keeps its top k_frac channels (0 < k_frac <= 1).
    selector "gamma" ranks channels by |delta gamma|, "delta_norm" by the norm of
    (delta gamma, delta beta). Other keys (classifier, pooler) are unchanged."""
    if not (0 < k_frac <= 1.0):
        raise ValueError(f"k_frac must be in (0, 1]; got {k_frac}")

    new_sd = {k: v.clone() for k, v in finetuned_sd.items()}
    ln_keys_list = _ln_param_keys(finetuned_sd, model_type, attn_only)

    # Group gamma and beta of each LayerNorm so they are masked together.
    by_ln: dict[tuple[int | None, str], dict[str, str]] = {}
    for k in ln_keys_list:
        layer = _layer_index(k)
        prefix = k.rsplit(".", 1)[0]
        slot = by_ln.setdefault((layer, prefix), {})
        if _is_weight(k):
            slot["weight"] = k
        elif _is_bias(k):
            slot["bias"] = k

    for (_, _), slot in by_ln.items():
        w_key = slot.get("weight")
        b_key = slot.get("bias")
        if w_key is None:
            continue
        gamma_ft = finetuned_sd[w_key]
        gamma_pre = pretrained_sd.get(w_key)
        if gamma_pre is None:
            continue
        d_gamma = gamma_ft - gamma_pre

        if selector == "gamma" or b_key is None or b_key not in pretrained_sd:
            score = d_gamma.abs()
        else:
            beta_ft = finetuned_sd[b_key]
            beta_pre = pretrained_sd[b_key]
            d_beta = beta_ft - beta_pre
            score = torch.sqrt(d_gamma.pow(2) + d_beta.pow(2))

        n = score.numel()
        k = max(1, int(round(k_frac * n)))
        if k >= n:
            continue
        top_idx = torch.topk(score, k).indices
        keep_mask = torch.zeros(n, dtype=torch.bool, device=score.device)
        keep_mask[top_idx] = True

        new_gamma = torch.where(keep_mask, gamma_ft, gamma_pre)
        new_sd[w_key] = new_gamma
        if b_key is not None and b_key in pretrained_sd:
            beta_ft = finetuned_sd[b_key]
            beta_pre = pretrained_sd[b_key]
            new_sd[b_key] = torch.where(keep_mask, beta_ft, beta_pre)

    return new_sd
