"""Shared training, evaluation and saving code for fitln_decoder.py, lora_decoder.py
and fullft_decoder.py.
"""
import json
import os
import random

import numpy as np
import torch
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments
from transformers.trainer_callback import EarlyStoppingCallback

import common
import data_alpaca_gen
import data_glue_gen
import data_gsm8k_gen
import data_samsum_gen
import gen_eval
import mem_utils
import scoring

# task -> (data module, scorer). Data modules provide build_train_dataset and
# build_eval_examples; scorers follow gen_eval.generate_and_score_rouge.
FREE_GEN_REGISTRY = {
    "alpaca_mini": (data_alpaca_gen, gen_eval.generate_and_score_rouge),
    "gsm8k": (data_gsm8k_gen, gen_eval.generate_and_score_exact_match),
    "samsum": (data_samsum_gen, gen_eval.generate_and_score_rouge),
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        # Llama and Qwen tokenizers often have no pad token; reuse eos.
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm(model_name: str, torch_dtype=None):
    """Load a decoder backbone. Models flagged needs_text_extraction (Qwen3.5) go through
    qwen35_loader; all others use AutoModelForCausalLM.

    torch_dtype=None leaves the weight dtype to the installed transformers version
    (fp32 for Qwen3.5). fullft_decoder.py passes bfloat16 to halve its checkpoint size.
    """
    from transformers import AutoModelForCausalLM

    cfg = common.MODEL_CONFIGS.get(model_name, {})
    if cfg.get("needs_text_extraction"):
        import qwen35_loader
        return qwen35_loader.load_qwen35_text_only(model_name, torch_dtype=torch_dtype)

    load_kwargs = {}
    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype
    return AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)


def log_model_facts(model, model_name: str):
    """Print the loaded class and whether the LM head is tied."""
    print(f"[run_common] model_name={model_name} -> loaded class {type(model).__name__}")
    print(f"[run_common] tie_word_embeddings={getattr(model.config, 'tie_word_embeddings', None)}")


def build_task_datasets(args, tokenizer):
    """Return the datasets for args.task (glue_gen or free_gen)."""
    kind = common.task_type(args.task)
    if kind == "glue_gen":
        train_answer, label_token_ids = data_glue_gen.build_dataset(
            args.task, tokenizer, split="train", max_seq_length=args.max_seq_length, include_answer=True,
        )
        # Periodic eval uses a held-out slice of train capped at 300 examples. The full
        # validation split is only used for the final accuracy.
        eval_holdout_size = min(300, max(1, len(train_answer) // 10))
        split_answer = train_answer.train_test_split(test_size=eval_holdout_size, seed=args.seed)

        train_noanswer, _ = data_glue_gen.build_dataset(
            args.task, tokenizer, split="train", max_seq_length=args.max_seq_length, include_answer=False,
        )
        # Same split arguments as above, so the same held-out indices, without the answer token.
        split_noanswer = train_noanswer.train_test_split(test_size=eval_holdout_size, seed=args.seed)

        test_noanswer, _ = data_glue_gen.build_dataset(
            args.task, tokenizer, split="validation", max_seq_length=args.max_seq_length, include_answer=False,
        )
        return dict(
            kind="glue_gen",
            train=split_answer["train"],
            eval_loss=split_answer["test"],
            eval_acc=split_noanswer["test"],
            test_acc=test_noanswer,
            label_token_ids=label_token_ids,
        )

    # free_gen: use the data module and train_n/eval_n arguments registered for the task.
    data_module, _ = FREE_GEN_REGISTRY[args.task]
    train_n_attr, eval_n_attr = common.FREE_GEN_DATA_ARGS[args.task]
    train_n, eval_n = getattr(args, train_n_attr), getattr(args, eval_n_attr)

    train_full = data_module.build_train_dataset(
        tokenizer, train_n=train_n, eval_n=eval_n, seed=args.seed, max_seq_length=args.max_seq_length,
    )
    # Hold out up to 300 train examples for periodic loss eval and early stopping. The
    # real test split is only used for the final generation metric.
    eval_loss_size = min(300, max(1, len(train_full) // 10))
    split = train_full.train_test_split(test_size=eval_loss_size, seed=args.seed)
    train = split["train"]
    eval_loss = split["test"]

    eval_gen_examples = data_module.build_eval_examples(
        tokenizer, train_n=train_n, eval_n=eval_n, seed=args.seed, max_seq_length=args.max_seq_length,
    )
    return dict(kind="free_gen", train=train, eval_loss=eval_loss, eval_gen_examples=eval_gen_examples)


def make_training_args(output_dir: str, args, weight_decay: float, gradient_checkpointing: bool,
                        training_args_overrides: dict = None) -> TrainingArguments:
    """Training arguments shared by all methods: constant LR, no warm-up, best checkpoint
    by held-out loss, eval and save every 200 steps, bf16 autocast on GPU.
    training_args_overrides replaces any default (fullft_decoder.py uses it to shrink
    its checkpoints).
    """
    kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=args.num_epoch,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if gradient_checkpointing else None,
        bf16=torch.cuda.is_available(),
        fp16=False,
        learning_rate=args.lr,
        seed=args.seed,
        data_seed=args.seed,
        logging_steps=50,
        logging_first_step=True,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        save_total_limit=1,
        lr_scheduler_type="constant",
        eval_steps=200,
        save_steps=200,
        eval_on_start=True,
        weight_decay=weight_decay,
        report_to="tensorboard",
        metric_for_best_model="loss",
    )
    if training_args_overrides:
        kwargs.update(training_args_overrides)
    return TrainingArguments(**kwargs)


def run_training(args, method: str, model, tokenizer, trainer_cls=Trainer, extra_trainer_kwargs=None,
                  compact_save_fn=None, training_args_overrides=None):
    """Train, evaluate and save one run. trainer_cls and extra_trainer_kwargs allow a custom
    Trainer (used by fitln_decoder.py). compact_save_fn(model, output_dir) writes the final
    compact checkpoint and returns its path, or None.
    """
    output_dir = common.build_output_dir(args.output_dir, method, args.model, args.task, args.lr, args.seed)
    os.makedirs(output_dir, exist_ok=True)

    weight_decay = common.MODEL_CONFIGS.get(args.model, {}).get("weight_decay", 0.01)
    datasets = build_task_datasets(args, tokenizer)
    trainable_param_names = [n for n, p in model.named_parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())

    if hasattr(model, "enable_input_require_grads"):
        # Needed when the backbone is frozen with gradient checkpointing, otherwise backward
        # produces no gradients.
        model.enable_input_require_grads()

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=-100)
    training_args = make_training_args(output_dir, args, weight_decay, gradient_checkpointing=True,
                                        training_args_overrides=training_args_overrides)

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["eval_loss"],
        processing_class=tokenizer,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=8)],
    )
    if extra_trainer_kwargs:
        trainer_kwargs.update(extra_trainer_kwargs)

    trainer = trainer_cls(**trainer_kwargs)

    mem_utils.reset_peak_memory()
    print("Begin Training\n\n")
    trainer.train()
    # Peak CUDA memory over the whole train() call; None without a GPU.
    peak_mem_mb = mem_utils.get_peak_memory_mb()

    print("Training finished, evaluation begins\n\n")
    eval_loss_result = trainer.evaluate()
    with open(os.path.join(output_dir, "val_results.json"), "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in eval_loss_result.items()}, f)

    if datasets["kind"] == "glue_gen":
        eval_acc = scoring.evaluate_accuracy(
            trainer.model, tokenizer, datasets["eval_acc"], datasets["label_token_ids"],
            batch_size=args.batch_size,
        )
        test_acc = scoring.evaluate_accuracy(
            trainer.model, tokenizer, datasets["test_acc"], datasets["label_token_ids"],
            batch_size=args.batch_size,
        )
        with open(os.path.join(output_dir, "test_results.json"), "w", encoding="utf-8") as f:
            json.dump({"eval_accuracy": eval_acc["accuracy"], "test_accuracy": test_acc["accuracy"],
                       "eval_n": eval_acc["n"], "test_n": test_acc["n"]}, f)
    else:
        _, scorer = FREE_GEN_REGISTRY[args.task]
        gen_result = scorer(
            trainer.model, tokenizer, datasets["eval_gen_examples"], max_new_tokens=args.max_new_tokens,
            batch_size=args.gen_batch_size,
        )
        with open(os.path.join(output_dir, "test_results.json"), "w", encoding="utf-8") as f:
            json.dump(gen_result, f)

    compact_path = None
    if compact_save_fn is not None:
        compact_path = compact_save_fn(trainer.model, output_dir)

    checkpoint_bytes = mem_utils.checkpoint_size_on_disk(compact_path) if compact_path else None
    train_meta = {
        "trainable_params": n_trainable,
        "total_params": n_total,
        "trainable_pct": 100 * n_trainable / n_total,
        "trainable_param_names": trainable_param_names,
        "peak_mem_mb": peak_mem_mb,
        "checkpoint_bytes": checkpoint_bytes,
    }
    with open(os.path.join(output_dir, "train_meta.json"), "w", encoding="utf-8") as f:
        json.dump(train_meta, f)

    print(f"[run_common] wrote results + train_meta to {output_dir}")
    return output_dir, train_meta
