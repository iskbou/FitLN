"""Scoring for the GLUE-as-generation tasks. Each label is a single token (enforced in
data_glue_gen.py), so a prediction is the argmax over the label tokens' log-probabilities
at the answer position of one forward pass. No generate() call.
"""
import torch


def score_labels(model, input_ids, attention_mask, label_token_ids: list) -> torch.Tensor:
    """Log-probs of each label token at the last real token of each example, shape
    [batch, num_labels]. The last real token is found from the attention mask, since
    batches are right-padded.
    """
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    batch_size, seq_len, vocab_size = out.logits.shape
    last_real_idx = attention_mask.sum(dim=-1) - 1            # [batch]
    gather_idx = last_real_idx.view(-1, 1, 1).expand(-1, 1, vocab_size)
    last_logits = out.logits.gather(dim=1, index=gather_idx).squeeze(1)  # [batch, vocab]
    log_probs = torch.log_softmax(last_logits.float(), dim=-1)  # [batch, vocab]
    label_ids_t = torch.tensor(label_token_ids, device=log_probs.device)
    return log_probs[:, label_ids_t]                         # [batch, num_labels]


def predict_labels(model, input_ids, attention_mask, label_token_ids: list) -> torch.Tensor:
    scores = score_labels(model, input_ids, attention_mask, label_token_ids)
    return scores.argmax(dim=-1)


def evaluate_accuracy(model, tokenizer, dataset, label_token_ids: dict, batch_size: int = 16,
                       show_progress: bool = False, progress_desc: str = "eval") -> dict:
    """Accuracy of the model on a prompt-only dataset (data_glue_gen.build_dataset with
    include_answer=False). label_token_ids maps label id to token id.
    """
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding

    ordered_label_ids = sorted(label_token_ids.keys())
    ordered_token_ids = [label_token_ids[lid] for lid in ordered_label_ids]

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    gold_labels = dataset["gold_label"]
    eval_cols = dataset.remove_columns(["gold_label"])
    loader = DataLoader(eval_cols, batch_size=batch_size, collate_fn=collator, shuffle=False)
    if show_progress:
        from tqdm import tqdm
        loader = tqdm(loader, desc=progress_desc, total=len(loader), leave=False)

    device = next(model.parameters()).device
    model.eval()
    correct = 0
    total = 0
    idx = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pred_positions = predict_labels(model, input_ids, attention_mask, ordered_token_ids)
            pred_label_ids = torch.tensor(ordered_label_ids, device=pred_positions.device)[pred_positions]
            batch_n = input_ids.shape[0]
            gold_batch = torch.as_tensor(gold_labels[idx:idx + batch_n])
            correct += (pred_label_ids.cpu() == gold_batch).sum().item()
            total += batch_n
            idx += batch_n

    return {"accuracy": correct / total if total > 0 else 0.0, "n": total}
