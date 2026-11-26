"""SAMSum dialogue summarization task, scored with ROUGE-L like alpaca_mini.

Dataset: knkarthick/samsum (a mirror of SAMSum), falling back to Samsung/samsum. Uses
the real splits (14,731 train / 819 test) unless train_n/eval_n are given.
"""
from datasets import load_dataset

PROMPT_TEMPLATE = (
    "Summarize the following conversation.\n\n### Conversation:\n{dialogue}\n\n### Summary:"
)


def build_prompt(example: dict) -> str:
    return PROMPT_TEMPLATE.format(dialogue=example["dialogue"].strip())


def load_samsum(train_n: int = None, eval_n: int = None, seed: int = 0):
    try:
        raw_train = load_dataset("knkarthick/samsum", split="train")
        raw_eval = load_dataset("knkarthick/samsum", split="test")
    except Exception:
        raw_train = load_dataset("Samsung/samsum", split="train")
        raw_eval = load_dataset("Samsung/samsum", split="test")
    train = raw_train if train_n is None else raw_train.shuffle(seed=seed).select(range(min(train_n, len(raw_train))))
    eval_ = raw_eval if eval_n is None else raw_eval.shuffle(seed=seed).select(range(min(eval_n, len(raw_eval))))
    return train, eval_


def _build_teacher_forced(dataset_slice, tokenizer, max_seq_length: int):
    eos = tokenizer.eos_token or ""

    def _map(example):
        prompt = build_prompt(example)
        response = " " + example["summary"].strip() + eos
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
    train, _ = load_samsum(train_n=train_n, eval_n=eval_n, seed=seed)
    return _build_teacher_forced(train, tokenizer, max_seq_length)


def build_eval_examples(tokenizer, train_n: int = None, eval_n: int = None, seed: int = 0, max_seq_length: int = 512):
    _, eval_ = load_samsum(train_n=train_n, eval_n=eval_n, seed=seed)
    examples = []
    for ex in eval_:
        prompt = build_prompt(ex)
        enc = tokenizer(prompt, truncation=True, max_length=max_seq_length // 2)
        examples.append({
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "reference": ex["summary"].strip(),
        })
    return examples
