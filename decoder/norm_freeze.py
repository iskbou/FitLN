"""FitLN freezing for decoder-only LMs (Llama, Qwen2, Qwen2.5, Qwen3.5).

Trains each layer's post_attention_layernorm.weight and, optionally, the final
model.norm.weight. Everything else is frozen and no task head is added.

In these pre-norm models post_attention_layernorm is the norm before the MLP, the
closest counterpart of the encoder's attention-output LayerNorm but not equivalent.
RMSNorm has a scale but no bias, so the decoder variant is scale-only. All supported
models expose the same parameter names, so no per-architecture branching is needed;
the assertions fail if a model does not.
"""
import torch.nn as nn


def _is_final_norm(name: str) -> bool:
    """True for a top-level final norm, not for norms nested inside a layer."""
    return name.endswith(".norm.weight") and ".layers." not in name


def freeze_for_fitln_decoder(model: nn.Module, include_final_norm: bool = True) -> list:
    """Freeze all parameters, then unfreeze every post_attention_layernorm.weight and,
    if include_final_norm, model.norm.weight. Returns the unfrozen parameter names.
    """
    model.requires_grad_(False)
    unfrozen = []
    for n, p in model.named_parameters():
        if n.endswith("post_attention_layernorm.weight"):
            p.requires_grad = True
            unfrozen.append(n)
            print(n)
        elif include_final_norm and _is_final_norm(n):
            p.requires_grad = True
            unfrozen.append(n)
            print(n)

    assert len(unfrozen) > 0, (
        "No post_attention_layernorm.weight parameter found; the model's module naming "
        "does not match what this function expects."
    )
    if include_final_norm:
        assert any(_is_final_norm(n) for n in unfrozen), (
            "include_final_norm=True but no final norm parameter was found."
        )

    # Cross-check against the declared layer count (config.num_hidden_layers, or
    # config.text_config.num_hidden_layers for nested configs like Qwen3.5).
    n_post_attn_ln = sum(1 for n in unfrozen if n.endswith("post_attention_layernorm.weight"))
    declared_layers = getattr(model.config, "num_hidden_layers", None)
    if declared_layers is not None:
        assert n_post_attn_ln == declared_layers, (
            f"Found {n_post_attn_ln} post_attention_layernorm.weight tensors but "
            f"model.config.num_hidden_layers={declared_layers}."
        )
    else:
        print(
            "[norm_freeze] no top-level num_hidden_layers in the config, layer count not "
            f"checked ({n_post_attn_ln} post_attention_layernorm.weight tensors found)."
        )

    n_trainable = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_trainable} / {n_total} ({100 * n_trainable / n_total:.5f}%)")
    return unfrozen
