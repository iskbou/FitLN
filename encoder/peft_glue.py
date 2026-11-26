"""LoRA and BitFit baselines on a GLUE task, with the Hugging Face Trainer.

Writes val_results.json, test_results.json and the trained weights (adapter/ for LoRA,
bitfit_weights.bin for BitFit) to {output_dir}/{model}/{task}/{lr}/{seed}/. The default
output_dir is out/bitfit/ or out/lora_r{lora_r}/. The training set is split 90/10 for
validation and the GLUE validation set is the test set.

Usage:
    python encoder/peft_glue.py --method lora --lora_r 1 --model FacebookAI/roberta-large \
        --task rte --lr 1e-4 --batch_size 16 --num_epoch 10
    python encoder/peft_glue.py --method bitfit --model FacebookAI/roberta-large \
        --task rte --lr 5e-4 --batch_size 16 --num_epoch 10

For mrpc, stsb and rte, --mnli_init takes the output of the MNLI run (the adapter directory
for LoRA, bitfit_weights.bin for BitFit); classifier and head weights are not loaded.
"""
import os
import time
import argparse
import random
import json
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from transformers.trainer_callback import EarlyStoppingCallback
from torch.optim import AdamW
from peft import LoraConfig, TaskType, PeftModelForSequenceClassification
import evaluate

import common

BITFIT_WEIGHTS_FILE = "bitfit_weights.bin"

# LoRA target modules and fully trained modules per model. The rank is --lora_r (three
# times that for ModernBERT's fused Wqkv). Other models use the BERT/RoBERTa entry.
LORA_MODULES = {
    "FacebookAI/roberta-large": (["query", "key", "value"], ["classifier"]),
    "google-bert/bert-large-uncased": (["query", "key", "value"], ["classifier"]),
    "bert-large-uncased": (["query", "key", "value"], ["classifier"]),
    "microsoft/deberta-v3-large": (["query_proj", "key_proj", "value_proj"], ["classifier", "pooler"]),
    "answerdotai/ModernBERT-large": (["Wqkv"], ["head", "classifier"]),
}
LORA_DEFAULT_MODULES = (["query", "key", "value"], ["classifier"])

# Modules trained in full by BitFit next to the bias terms.
HEAD_MODULE_KEYS = ("classifier", "pooler", "head")


def lora_kwargs(model_name, lora_r):
    target_modules, modules_to_save = LORA_MODULES.get(model_name, LORA_DEFAULT_MODULES)
    r = lora_r * 3 if "ModernBERT" in model_name else lora_r
    return dict(r=r, target_modules=target_modules, modules_to_save=modules_to_save)


def inject_biases(model: nn.Module) -> int:
    # ModernBERT's Linear modules have no bias, which would make BitFit a no-op there.
    n_injected = 0
    for module in model.modules():
        if isinstance(module, nn.Linear) and module.bias is None:
            module.bias = nn.Parameter(torch.zeros(module.out_features, device=module.weight.device, dtype=module.weight.dtype))
            n_injected += 1
    return n_injected


def freeze_for_bitfit(model: nn.Module) -> int:
    model.requires_grad_(False)
    for name, param in model.named_parameters():
        if name.endswith(".bias") or any(key in name for key in HEAD_MODULE_KEYS):
            param.requires_grad = True
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class BitFitTrainer(Trainer):
    # Checkpoints contain only the trainable parameters.
    def _trainable_state_dict(self, full_state_dict=None):
        trainable = {n for n, p in self.model.named_parameters() if p.requires_grad}
        sd = full_state_dict if full_state_dict is not None else self.model.state_dict()
        return {k: v for k, v in sd.items() if k in trainable}

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self._trainable_state_dict(state_dict), os.path.join(output_dir, BITFIT_WEIGHTS_FILE))
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)

    def _load_best_model(self):
        if self.state.best_model_checkpoint is None:
            return
        weights_path = os.path.join(self.state.best_model_checkpoint, BITFIT_WEIGHTS_FILE)
        if not os.path.exists(weights_path):
            return
        partial = torch.load(weights_path, map_location="cpu")
        missing, unexpected = self.model.load_state_dict(partial, strict=False)
        if unexpected:
            print(f"[BitFitTrainer] unexpected keys when loading best model: {unexpected}")


def load_adapter_weights(adapter_dir):
    safetensors_file = os.path.join(adapter_dir, "adapter_model.safetensors")
    if os.path.exists(safetensors_file):
        from safetensors.torch import load_file
        return load_file(safetensors_file)
    return torch.load(os.path.join(adapter_dir, "adapter_model.bin"), map_location="cpu", weights_only=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("peft_glue")
    parser.add_argument("--method", required=True, choices=["lora", "bitfit"])
    parser.add_argument("--task", default="mnli", required=False)
    parser.add_argument("--output_dir", default=None, required=False)
    parser.add_argument("--seed", default=0, required=False, type=int)
    parser.add_argument("--brainfloat", action="store_true")
    parser.add_argument("--batch_size", default=32, required=False, type=int)
    parser.add_argument("--num_epoch", default=10, required=False, type=int)
    parser.add_argument("--lr", default=None, required=False, type=float,
                        help="Default 5e-5 for LoRA and 5e-4 for BitFit.")
    parser.add_argument("--model", default="FacebookAI/roberta-large", required=False)
    parser.add_argument("--lora_r", default=1, required=False, type=int)
    parser.add_argument("--lora_alpha", default=8, required=False, type=int)
    parser.add_argument("--mnli_init", default=None, required=False)
    args = parser.parse_args()
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    method = args.method
    lr = args.lr if args.lr is not None else (5e-5 if method == "lora" else 5e-4)
    num_epoch = args.num_epoch
    task = args.task
    brainfloat = args.brainfloat
    batch_size = args.batch_size
    model_name = args.model
    default_dir = f"out/lora_r{args.lora_r}/" if method == "lora" else "out/bitfit/"
    output_dir = os.path.join(
        os.getcwd(),
        args.output_dir if args.output_dir is not None else default_dir,
        model_name,
        str(args.task),
        str(np.round(lr, 6)),
        str(args.seed),
    )
    os.makedirs(output_dir, exist_ok=True)
    num_label = common.TASK_NUM_LABELS[task]
    text_fields = common.TASK_TEXT_FIELDS[task]
    print("Arguments parsed\n\n")

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_label)

    if method == "lora":
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.1,
            **lora_kwargs(model_name, args.lora_r),
        )
        model.add_adapter(lora_config, adapter_name="lora_default")
        model = PeftModelForSequenceClassification(model=model, peft_config=lora_config)
        if args.mnli_init:
            mnli_weights = load_adapter_weights(args.mnli_init)
            filtered = {k: v for k, v in mnli_weights.items() if "classifier" not in k and "head" not in k}
            missing, _ = model.load_state_dict(filtered, strict=False)
            print(f"Loaded MNLI LoRA adapter: {len(filtered)} tensors, {len(missing)} missing")
    else:
        if "answerdotai" in model_name:
            n_injected = inject_biases(model)
            print(f"Injected biases into {n_injected} Linear modules (ModernBERT)")
        n_trainable = freeze_for_bitfit(model)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"Trainable params: {n_trainable} / {n_total} ({100 * n_trainable / n_total:.4f}%)")
        if args.mnli_init:
            mnli_weights = torch.load(args.mnli_init, map_location="cpu", weights_only=True)
            filtered = {k: v for k, v in mnli_weights.items() if "classifier" not in k and "head" not in k}
            missing, _ = model.load_state_dict(filtered, strict=False)
            print(f"Loaded MNLI BitFit checkpoint: {len(filtered)} tensors, {len(missing)} missing")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    if method == "bitfit" and hasattr(model, "enable_input_require_grads"):
        # Needed for gradient flow when the whole backbone is frozen.
        model.enable_input_require_grads()
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Model Loaded and initialized\n\n")
    if method == "lora":
        print(model.print_trainable_parameters())

    raw_datasets = load_dataset("glue", task)

    def tokenize_function(example):
        return tokenizer(*[example[field] for field in text_fields], truncation=True)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

    tokenized_datasets = tokenized_datasets.remove_columns([field for field in text_fields] + ["idx"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    train_ds = tokenized_datasets["train"]
    split_tokenized_datasets = train_ds.train_test_split(test_size=0.1, seed=seed)
    eval_dataset = split_tokenized_datasets["test"]
    test_dataset = (
        tokenized_datasets["validation"]
        if task != "mnli"
        else {
            "matched": tokenized_datasets["validation_matched"],
            "mismatched": tokenized_datasets["validation_mismatched"],
        }
    )

    early_stopping = EarlyStoppingCallback(early_stopping_patience=5)
    eval_module = evaluate.load("glue", task, experiment_id=str(time.time()) + str(lr) + task)
    print(f"{task} Data Loaded and Preprocessed\n\n")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if task != 'stsb':
            predictions = np.argmax(logits, axis=-1)
        else:
            predictions = 5 * logits
        return eval_module.compute(predictions=predictions, references=labels)

    # The weight decay of the optimizer is the AdamW default (0.01).
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epoch,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        gradient_checkpointing=(method == "lora"),
        bf16=brainfloat,
        fp16=not brainfloat,
        learning_rate=lr,
        seed=seed,
        data_seed=seed,
        logging_steps=200,
        logging_first_step=True,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        save_total_limit=2,
        lr_scheduler_type="constant",
        eval_steps=2000,
        save_steps=2000,
        eval_on_start=True,
        report_to="tensorboard",
        metric_for_best_model="loss",
        **({"save_safetensors": False} if method == "bitfit" else {}),
    )

    trainer_cls = Trainer if method == "lora" else BitFitTrainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=split_tokenized_datasets["train"],
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        optimizers=(optimizer, None),
        compute_metrics=compute_metrics,
        callbacks=[early_stopping],
    )

    print("Begin Training\n\n")
    trainer.train()

    print("Training finished, evaluation begins\n\n")
    result = trainer.evaluate()
    with open(os.path.join(output_dir, "val_results.json"), "w", encoding="utf-8") as js:
        json.dump({k: float(v) for k, v in result.items()}, js)
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    with open(os.path.join(output_dir, "test_results.json"), "w", encoding="utf-8") as js:
        json.dump({k: float(v) for k, v in test_results.items()}, js)

    if method == "lora":
        model.save_pretrained(os.path.join(output_dir, "adapter"))
        print(f"Saved LoRA adapter to {os.path.join(output_dir, 'adapter')}")
    else:
        final_weights = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
        torch.save(final_weights, os.path.join(output_dir, BITFIT_WEIGHTS_FILE))
        print(f"Saved {len(final_weights)} trainable tensors to {BITFIT_WEIGHTS_FILE}")
