"""Train FitLN on MS MARCO for retrieval with sentence-transformers.

Trains the attention-output LayerNorms (and the final norm, classifier and pooler if the
model has them) with a cached in-batch negatives loss on 1.25M MS MARCO triplets with mined
hard negatives, for one epoch. The model is saved to
out_ri/msmarco/{model}/{lr}/1/{model_short}-DPR-{lr}/final.

Usage:
    python encoder/beir_train.py --model FacebookAI/roberta-large --lr 1.2e-3
"""
import os
import argparse

import numpy as np
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers


def main():
    parser = argparse.ArgumentParser("beir_train")
    parser.add_argument("--lr", default=5e-4, required=False, type=float)
    parser.add_argument("--model", default="bert-base-uncased", required=False)
    parser.add_argument("--mini_batch_size", default=16, required=False, type=int,
                        help="Mini-batch size of the cached loss; the training batch size is 512.")
    args = parser.parse_args()
    lr = args.lr
    model_name = args.model
    model_shortname = model_name.split("/")[-1]
    savedir = os.getcwd() + "/out_ri/" + "msmarco" + "/" + model_name + "/" + str(lr) + "/" + str(1) + "/"

    model = SentenceTransformer(model_name)

    train_param = []
    train_names = []
    model.requires_grad_(False)
    if "ModernBERT" in model_name:
        layernorm_key = "attn_norm"
    else:
        layernorm_key = "attention.output.LayerNorm"
    for n, p in model.named_parameters():
        for key in (layernorm_key, "final_norm", "classifier", "pooler"):
            if key in n:
                p.requires_grad = True
                print(n)
                train_param.append(p)
                train_names.append(n)

    num_param = 0
    for p in train_param:
        num_param += p.numel()
    print(f"Trainable params: {num_param}, Ratio: {num_param/np.sum([p.numel() for p in model.parameters()])}")

    dataset = load_dataset(
        "sentence-transformers/msmarco-co-condenser-margin-mse-sym-mnrl-mean-v1",
        "triplet-hard",
        split="train",
    )
    dataset_dict = dataset.train_test_split(test_size=1_000, seed=12)
    train_dataset = dataset_dict["train"].select(range(1_250_000))
    eval_dataset = dataset_dict["test"]

    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=args.mini_batch_size)

    run_name = f"{model_shortname}-DPR-{lr}"
    training_args = SentenceTransformerTrainingArguments(
        output_dir=f"output/{model_shortname}/{run_name}",
        num_train_epochs=1,
        per_device_train_batch_size=512,
        per_device_eval_batch_size=512,
        warmup_ratio=0.05,
        fp16=True,
        bf16=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        learning_rate=lr,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        logging_steps=100,
        run_name=run_name,
    )

    dev_evaluator = TripletEvaluator(
        anchors=eval_dataset["query"],
        positives=eval_dataset["positive"],
        negatives=eval_dataset["negative"],
        name="msmarco-co-condenser-dev",
    )
    dev_evaluator(model)

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=dev_evaluator,
    )
    trainer.train()

    dev_evaluator(model)

    model.save_pretrained(f"{savedir}{run_name}/final")


if __name__ == "__main__":
    main()
