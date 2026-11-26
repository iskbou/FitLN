"""Free-form instruction-following task (alpaca_mini) on tatsu-lab/alpaca (Stanford
Alpaca, 52,002 examples, single split).

The split is shuffled with a fixed seed: by default 50,000 examples are used for
training and the remaining 2,002 for evaluation (ROUGE-L against the reference
responses). This is a held-out split of the Alpaca instruction set, not AlpacaEval.

The prompt template is the one from the Stanford Alpaca repository.
"""
from datasets import load_dataset

PROMPT_INPUT = (
    "Below is an instruction that describes a task, paired with an input that provides "
    "further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
)
PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that appropriately "
    "completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:"
)


def build_prompt(example: dict) -> str:
    if example.get("input", ""):
        return PROMPT_INPUT.format(instruction=example["instruction"], input=example["input"])
    return PROMPT_NO_INPUT.format(instruction=example["instruction"])


def load_alpaca_mini(train_n: int = 50000, eval_n: int = None, seed: int = 0):
    """Deterministic train/eval partition: shuffle with a fixed seed and take disjoint
    slices. eval_n=None uses all examples left after the train_n slice.
    """
    raw = load_dataset("tatsu-lab/alpaca", split="train")
    shuffled = raw.shuffle(seed=seed)
    train = shuffled.select(range(train_n))
    eval_end = len(raw) if eval_n is None else min(train_n + eval_n, len(raw))
    eval_ = shuffled.select(range(train_n, eval_end))
    return train, eval_


def _build_teacher_forced(dataset_slice, tokenizer, max_seq_length: int):
    """Teacher-forced examples: only response tokens carry a loss (prompt masked to -100)."""
    eos = tokenizer.eos_token or ""

    def _map(example):
        prompt = build_prompt(example)
        response = " " + example["output"].strip() + eos
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


def build_train_dataset(tokenizer, train_n: int = 50000, eval_n: int = None, seed: int = 0, max_seq_length: int = 512):
    train, _ = load_alpaca_mini(train_n=train_n, eval_n=eval_n, seed=seed)
    return _build_teacher_forced(train, tokenizer, max_seq_length)


def build_eval_examples(tokenizer, train_n: int = 50000, eval_n: int = None, seed: int = 0, max_seq_length: int = 512):
    """Prompt-only examples for generation, each with its reference response."""
    _, eval_ = load_alpaca_mini(train_n=train_n, eval_n=eval_n, seed=seed)
    examples = []
    for ex in eval_:
        prompt = build_prompt(ex)
        enc = tokenizer(prompt, truncation=True, max_length=max_seq_length // 2)
        examples.append({
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "reference": ex["output"].strip(),
        })
    return examples
