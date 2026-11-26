"""Free-form generation scoring for alpaca_mini, gsm8k and samsum (uses generate()).
ROUGE-L needs the rouge_score package; if it is missing the result carries an error
message instead of a score. Generation is batched with left padding, which is set
temporarily on the tokenizer.
"""
import torch


def compute_eval_perplexity(model, tokenizer, dataset, batch_size: int = 4) -> float:
    """Perplexity (exp of mean loss over response tokens) on a teacher-forced dataset.
    Not called by run_common; aggregation derives perplexity from val_results eval_loss.
    """
    from torch.utils.data import DataLoader
    from transformers import DataCollatorForSeq2Seq

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=-100)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator, shuffle=False)
    device = next(model.parameters()).device
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            n_tokens = (batch["labels"] != -100).sum().item()
            total_loss += out.loss.item() * n_tokens
            total_tokens += n_tokens
    mean_loss = total_loss / total_tokens if total_tokens > 0 else float("nan")
    return float(torch.exp(torch.tensor(mean_loss)))


def _generate_completions(model, tokenizer, eval_examples: list, max_new_tokens: int, batch_size: int = 16) -> list:
    """Greedy decoding in batches of batch_size, with left padding restored afterwards."""
    device = next(model.parameters()).device
    model.eval()
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    completions = []
    try:
        with torch.no_grad():
            for i in range(0, len(eval_examples), batch_size):
                batch = eval_examples[i:i + batch_size]
                encoded = tokenizer.pad(
                    {"input_ids": [ex["input_ids"] for ex in batch],
                     "attention_mask": [ex["attention_mask"] for ex in batch]},
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)
                out_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=pad_token_id,
                )
                gen_only = out_ids[:, input_ids.shape[1]:]
                for row in gen_only:
                    completions.append(tokenizer.decode(row, skip_special_tokens=True).strip())
    finally:
        tokenizer.padding_side = original_padding_side
    return completions


def generate_and_score_rouge(model, tokenizer, eval_examples: list, max_new_tokens: int = 64,
                              batch_size: int = 16) -> dict:
    """ROUGE-L of greedy completions against the references. Returns {'rouge_l': None,
    'error': ...} if rouge_score is unavailable.
    """
    try:
        import evaluate as hf_evaluate
        rouge = hf_evaluate.load("rouge")
    except Exception as e:  # e.g. rouge_score not installed
        return {
            "rouge_l": None,
            "error": (
                f"ROUGE-L scoring unavailable: {e}. Install rouge_score to enable it."
            ),
        }

    completions = _generate_completions(model, tokenizer, eval_examples, max_new_tokens, batch_size)
    references = [ex["reference"] for ex in eval_examples]
    result = rouge.compute(predictions=completions, references=references, rouge_types=["rougeL"])
    return {"rouge_l": float(result["rougeL"])}


def generate_and_score_exact_match(model, tokenizer, eval_examples: list, max_new_tokens: int = 256,
                                    batch_size: int = 16) -> dict:
    """gsm8k scorer: exact match of the final '#### <number>' answer."""
    import data_gsm8k_gen

    completions = _generate_completions(model, tokenizer, eval_examples, max_new_tokens, batch_size)
    correct = 0
    for completion, ex in zip(completions, eval_examples):
        predicted = data_gsm8k_gen.extract_final_answer(completion)
        if predicted != "" and predicted == ex["reference"]:
            correct += 1
    n = len(eval_examples)
    return {"exact_match": correct / n if n > 0 else float("nan"), "n": n}
