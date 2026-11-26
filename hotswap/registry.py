"""Model registry and memory arithmetic shared by the hot-swap scripts.

Parameter counts are the ones of the paper tables (encoders: params table; decoders:
decoder params table). hidden_size and num_layers come from the Hugging Face configs; they
only decide how many tensors a task delta is split into. The check at the bottom of the
file verifies that hidden_size * num_layers reproduces every FitLN parameter count.
"""
from __future__ import annotations

BYTES_PER_PARAM = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "int8": 1,
    "int4": 0.5,
}

TORCH_DTYPE_NAME = {
    "fp32": "float32",
    "fp16": "float16",
    "bf16": "bfloat16",
}

# head_params is the classifier/pooler head stored per task in addition to the method's own
# delta (1,051,650 = pooler 1024x1024+1024 plus classifier 1024x2+2). ModernBERT's pooling
# head is named "head", so only its classifier (2,050) counts. RoBERTa uses the same figure
# as BERT.
ENCODER_MODELS = {
    "bert-large": dict(
        hf_id="bert-large-uncased",
        hidden_size=1024, num_layers=24,
        base_params=340_000_000,
        fitln_params=49_152, fitln_has_beta=True,
        bitfit_params=270_000,
        lora1_params=147_000, lora8_params=1_200_000, lora_target_proj=3,
        head_params=1_051_650,
    ),
    "roberta-large": dict(
        hf_id="roberta-large",
        hidden_size=1024, num_layers=24,
        base_params=355_000_000,
        fitln_params=49_152, fitln_has_beta=True,
        bitfit_params=270_000,
        lora1_params=147_000, lora8_params=1_200_000, lora_target_proj=3,
        head_params=1_051_650,
    ),
    "deberta-v3-large": dict(
        hf_id="microsoft/deberta-v3-large",
        hidden_size=1024, num_layers=24,
        base_params=435_000_000,
        fitln_params=49_152, fitln_has_beta=True,
        bitfit_params=270_000,
        lora1_params=147_000, lora8_params=1_200_000, lora_target_proj=3,
        head_params=1_051_650,
    ),
    "modernbert-large": dict(
        hf_id="answerdotai/ModernBERT-large",
        hidden_size=1024, num_layers=28,
        base_params=395_000_000,
        fitln_params=28_672, fitln_has_beta=False,
        bitfit_params=291_000,
        lora1_params=344_000, lora8_params=2_800_000, lora_target_proj=1,
        head_params=2_050,
    ),
}

# No BitFit entry for decoders: BitFit was not run on them.
DECODER_MODELS = {
    "llama-3.2-1b": dict(
        hf_id="meta-llama/Llama-3.2-1B",
        hidden_size=2048, num_layers=16,
        base_params=1_240_000_000,
        fitln_params=34_816,
        lora1_params=213_000, lora8_params=1_700_000, lora_target_proj=4,
    ),
    "qwen2.5-1.5b": dict(
        hf_id="Qwen/Qwen2.5-1.5B",
        hidden_size=1536, num_layers=28,
        base_params=1_540_000_000,
        fitln_params=44_544,
        lora1_params=272_000, lora8_params=2_180_000, lora_target_proj=4,
    ),
    "qwen3.5-2b-base": dict(
        hf_id="Qwen/Qwen3.5-2B-Base",
        hidden_size=2048, num_layers=24,
        base_params=1_880_000_000,
        fitln_params=51_200,
        lora1_params=461_000, lora8_params=3_690_000, lora_target_proj=4,
    ),
    "llama-3.2-3b": dict(
        hf_id="meta-llama/Llama-3.2-3B",
        hidden_size=3072, num_layers=28,
        base_params=3_210_000_000,
        fitln_params=89_088,
        lora1_params=573_000, lora8_params=4_590_000, lora_target_proj=4,
    ),
    "qwen2.5-3b": dict(
        hf_id="Qwen/Qwen2.5-3B",
        hidden_size=2048, num_layers=36,
        base_params=3_090_000_000,
        fitln_params=75_776,
        lora1_params=461_000, lora8_params=3_690_000, lora_target_proj=4,
    ),
    "qwen3.5-4b-base": dict(
        hf_id="Qwen/Qwen3.5-4B-Base",
        hidden_size=2560, num_layers=32,
        base_params=4_210_000_000,
        fitln_params=84_480,
        lora1_params=899_000, lora8_params=7_190_000, lora_target_proj=4,
    ),
}

METHODS = ("fitln", "bitfit", "lora1", "lora8", "fullft")

# Whether the task parameters sit on top of one shared backbone (FullFT: one backbone each).
METHOD_SHARES_BACKBONE = {
    "fitln": True, "bitfit": True, "lora1": True, "lora8": True, "fullft": False,
}

# copy: the task values are the model's own parameters, so switching overwrites them in place.
# pointer: the task values are a separate adapter, so switching is a pointer change.
METHOD_SWAP_KIND = {
    "fitln": "copy", "bitfit": "copy", "lora1": "pointer", "lora8": "pointer", "fullft": "copy",
}

GPU_MEMORY_PRESETS_GB = {
    "24GB (e.g. RTX 4090 / L4)": 24,
    "40GB (e.g. A100-40GB)": 40,
    "80GB (e.g. A100/H100-80GB)": 80,
}

PARAM_KEY_BY_METHOD = {
    "fitln": "fitln_params",
    "bitfit": "bitfit_params",
    "lora1": "lora1_params",
    "lora8": "lora8_params",
    "fullft": "base_params",
}


def param_bytes(n_params: float, dtype: str = "bf16") -> float:
    return n_params * BYTES_PER_PARAM[dtype]


def human_bytes(n: float) -> str:
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}PB"


def tensor_layout(arch: str, model: str, method: str) -> int:
    """Number of separate tensors a task delta is split into. Exact for fitln (one tensor
    per unfrozen norm); an approximation for the other methods."""
    reg = ENCODER_MODELS[model] if arch == "encoder" else DECODER_MODELS[model]
    n_layers = reg["num_layers"]
    if method == "fitln":
        if arch == "decoder":
            return n_layers + 1
        return 2 * n_layers if reg.get("fitln_has_beta", True) else n_layers
    if method == "bitfit":
        # about 6 bias tensors per layer
        return max(1, 6 * n_layers)
    if method in ("lora1", "lora8"):
        # an A and a B matrix per targeted projection per layer
        return max(1, 2 * reg["lora_target_proj"] * n_layers)
    if method == "fullft":
        # about 6 weight matrices per layer plus embeddings and head
        return max(1, 6 * n_layers + 2)
    raise ValueError(method)


def _check_fitln_counts():
    for model, reg in DECODER_MODELS.items():
        expected = reg["hidden_size"] * (reg["num_layers"] + 1)
        assert expected == reg["fitln_params"], f"{model}: {expected} != {reg['fitln_params']}"
    for model, reg in ENCODER_MODELS.items():
        mult = 2 if reg.get("fitln_has_beta", True) else 1
        expected = mult * reg["hidden_size"] * reg["num_layers"]
        assert expected == reg["fitln_params"], f"{model}: {expected} != {reg['fitln_params']}"


_check_fitln_counts()
