"""GLUE tasks (sst2, rte, mrpc, cola) as generation: prompt templates, label tokens and
tokenization.

Label ids: sst2 0=negative 1=positive; rte 0=entailment 1=not_entailment;
mrpc 0=not_equivalent 1=equivalent; cola 0=unacceptable 1=acceptable.

Each task has a prompt function and, per label, candidate verbalizer strings.
resolve_label_tokens picks the first candidate that is a single token for the
tokenizer and raises if there is none. Check this when adding a model.
"""
import torch
from datasets import load_dataset

TASK_TEXT_FIELDS = {
    "sst2": ["sentence"],
    "rte": ["sentence1", "sentence2"],
    "mrpc": ["sentence1", "sentence2"],
    "cola": ["sentence"],
}


def _sst2_prompt(ex):
    return f"Review: {ex['sentence'].strip()}\nSentiment:"


def _rte_prompt(ex):
    return (
        f"Premise: {ex['sentence1'].strip()}\n"
        f"Hypothesis: {ex['sentence2'].strip()}\n"
        f"Question: Does the premise entail the hypothesis? Answer:"
    )


def _mrpc_prompt(ex):
    return (
        f"Sentence 1: {ex['sentence1'].strip()}\n"
        f"Sentence 2: {ex['sentence2'].strip()}\n"
        f"Question: Are these two sentences paraphrases of each other? Answer:"
    )


def _cola_prompt(ex):
    return f"Sentence: {ex['sentence'].strip()}\nQuestion: Is this sentence grammatically acceptable? Answer:"


# label id -> candidate verbalizer strings (the leading space matters for BPE).
PROMPT_TEMPLATES = {
    "sst2": (_sst2_prompt, {0: [" negative", " Negative", "negative"], 1: [" positive", " Positive", "positive"]}),
    "rte": (_rte_prompt, {0: [" yes", " Yes", "yes"], 1: [" no", " No", "no"]}),
    "mrpc": (_mrpc_prompt, {0: [" no", " No", "no"], 1: [" yes", " Yes", "yes"]}),
    "cola": (_cola_prompt, {0: [" no", " No", "no"], 1: [" yes", " Yes", "yes"]}),
}


def resolve_label_tokens(tokenizer, task: str) -> dict:
    """Return {label_id: token_id}; raises unless every label is a single token."""
    _, label_candidates = PROMPT_TEMPLATES[task]
    resolved = {}
    for label_id, candidates in label_candidates.items():
        chosen = None
        for cand in candidates:
            ids = tokenizer.encode(cand, add_special_tokens=False)
            if len(ids) == 1:
                chosen = ids[0]
                print(f"[data_glue_gen] task={task} label={label_id} -> {cand!r} -> token id {chosen}")
                break
        assert chosen is not None, (
            f"No single-token candidate in {candidates!r} for task={task} label={label_id}; "
            f"add one to PROMPT_TEMPLATES."
        )
        resolved[label_id] = chosen
    assert len(set(resolved.values())) == len(resolved), (
        f"Two labels for task={task} resolved to the same token id ({resolved})."
    )
    return resolved


def build_dataset(task: str, tokenizer, split: str, max_seq_length: int = 256, include_answer: bool = True):
    """Load a GLUE split and tokenize the verbalizer prompt. Returns (dataset, label_token_ids).

    include_answer=True (training): the gold answer token is appended and labels are -100
    everywhere except that token.

    include_answer=False (evaluation): the input is the prompt only. scoring.score_labels
    reads the prediction after the prompt, so the answer must not be in the input.

    The raw GLUE label is kept as gold_label.
    """
    template_fn, _ = PROMPT_TEMPLATES[task]
    label_token_ids = resolve_label_tokens(tokenizer, task)
    raw = load_dataset("glue", task, split=split)

    def _map(example):
        prompt = template_fn(example)
        max_len = max_seq_length - 1 if include_answer else max_seq_length
        enc = tokenizer(prompt, truncation=True, max_length=max_len)
        if include_answer:
            answer_token_id = label_token_ids[example["label"]]
            input_ids = enc["input_ids"] + [answer_token_id]
            attention_mask = enc["attention_mask"] + [1]
            labels = [-100] * len(enc["input_ids"]) + [answer_token_id]
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "gold_label": example["label"],
            }
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "gold_label": example["label"],
        }

    keep_cols = {"input_ids", "attention_mask", "labels", "gold_label"}
    remove_cols = [c for c in raw.column_names if c not in keep_cols]
    ds = raw.map(_map, remove_columns=remove_cols)
    ds.set_format("torch")
    return ds, label_token_ids
