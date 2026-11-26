"""Text-only backbone extraction for the Qwen3.5 base checkpoints.

These models are only published as multimodal wrappers (Qwen3_5ForConditionalGeneration)
with the language model under the model.language_model.* prefix. This loader loads the
full checkpoint, strips that prefix, loads the weights into a plain Qwen3_5ForCausalLM,
and drops the vision tower and the mtp block. It needs a transformers release with
qwen3_5 support (5.x) and fails on any missing or unexpected key. The LM head is tied to
the embeddings.
"""
import torch


def load_qwen35_text_only(model_name: str, torch_dtype=None):
    """Return a Qwen3_5ForCausalLM holding the text backbone of model_name.

    Qwen3_5ForCausalLM is built in fp32, so with torch_dtype=None the result is fp32;
    otherwise the model is cast to torch_dtype. The full multimodal checkpoint is loaded
    first, so peak memory is higher than the final model's size.
    """
    from transformers import AutoConfig
    from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration

    full_config = AutoConfig.from_pretrained(model_name)
    text_config = full_config.text_config

    load_kwargs = {}
    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype
    full_model = Qwen3_5ForConditionalGeneration.from_pretrained(model_name, **load_kwargs)

    causal_model = Qwen3_5ForCausalLM(text_config)
    lang_prefix = "model.language_model."
    full_state = full_model.state_dict()
    lang_state = {k[len(lang_prefix):]: v for k, v in full_state.items() if k.startswith(lang_prefix)}

    missing, unexpected = causal_model.model.load_state_dict(lang_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Qwen3.5 text backbone extraction failed: missing={missing}, "
            f"unexpected={unexpected}. Check the transformers version (checked with 5.13.0)."
        )

    if text_config.tie_word_embeddings:
        causal_model.tie_weights()
    else:
        lm_head_key = "lm_head.weight"
        if lm_head_key in full_state:
            with torch.no_grad():
                causal_model.lm_head.weight.copy_(full_state[lm_head_key])
        else:
            raise RuntimeError(
                "tie_word_embeddings is False but the checkpoint has no lm_head.weight."
            )

    del full_model
    if torch_dtype is not None:
        causal_model = causal_model.to(torch_dtype)
    return causal_model
