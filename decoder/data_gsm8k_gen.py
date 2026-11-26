"""GSM8K (openai/gsm8k, config main) math reasoning task, scored by exact match of the
final number.

Uses the real train/test splits (7,473 / 1,319 examples) unless train_n/eval_n are
given, in which case the split is shuffled deterministically and truncated. The gold
answer contains the chain of thought followed by '#### <number>'; it is the training
target and the source of the reference number.
"""
import re

from datasets import load_dataset

PROMPT_TEMPLATE = "Question: {question}\nAnswer: Let's think step by step."

_ANSWER_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
_LAST_NUMBER_RE = re.compile(r"-?[\d,]+(?:\.\d+)?")


def build_prompt(example: dict) -> str:
    return PROMPT_TEMPLATE.format(question=example["question"].strip())


def extract_final_answer(text: str) -> str:
    """Number after '#### ', else the last number in the text. Commas are removed."""
    m = _ANSWER_RE.search(text)
    if m is None:
        candidates = _LAST_NUMBER_RE.findall(text)
        if not candidates:
            return ""
        m_str = candidates[-1]
    else:
        m_str = m.group(1)
    return m_str.replace(",", "").rstrip(".")


def load_gsm8k(train_n: int = None, eval_n: int = None, seed: int = 0):
    train_raw = load_dataset("openai/gsm8k", "main", split="train")
    test_raw = load_dataset("openai/gsm8k", "main", split="test")
    train = train_raw if train_n is None else train_raw.shuffle(seed=seed).select(range(min(train_n, len(train_raw))))
    eval_ = test_raw if eval_n is None else test_raw.shuffle(seed=seed).select(range(min(eval_n, len(test_raw))))
    return train, eval_


def _build_teacher_forced(dataset_slice, tokenizer, max_seq_length: int):
    eos = tokenizer.eos_token or ""

    def _map(example):
        prompt = build_prompt(example)
        response = " " + example["answer"].strip() + eos
        prompt_ids = tokenizer(prompt, truncation=True, max_length=max_seq_length // 2)["input_ids"]
        response_ids = tokenizer(response, truncation=True, max_length=max_seq_length - len(prompt_ids),
                                  add_special_tokens=False)["input_ids"]
        input_ids = prompt_ids + response_ids
        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(prompt_ids) + response_ids
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    ds = dataset_slice.map(_map, remove_columns=dataset_slice.column_names)
    ds.set_format("torch")
    return ds


def build_train_dataset(tokenizer, train_n: int = None, eval_n: int = None, seed: int = 0, max_seq_length: int = 512):
    train, _ = load_gsm8k(train_n=train_n, eval_n=eval_n, seed=seed)
    return _build_teacher_forced(train, tokenizer, max_seq_length)


def build_eval_examples(tokenizer, train_n: int = None, eval_n: int = None, seed: int = 0, max_seq_length: int = 512):
    """Prompt-only examples; reference is the extracted gold final number."""
    _, eval_ = load_gsm8k(train_n=train_n, eval_n=eval_n, seed=seed)
    examples = []
    for ex in eval_:
        prompt = build_prompt(ex)
        enc = tokenizer(prompt, truncation=True, max_length=max_seq_length // 2)
        examples.append({
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "reference": extract_final_answer(ex["answer"]),
        })
    return examples
