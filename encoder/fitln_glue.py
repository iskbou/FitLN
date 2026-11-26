"""FitLN on a GLUE task.

Trains the attention-output LayerNorms (and the classifier) of an encoder and saves the
trained parameters and the scores of the best epoch to
out/{task}/{model}/{attn_only|full}/{lr}/{num_epoch}/.

Usage:
    python encoder/fitln_glue.py --model roberta-large --task rte --lr 1e-3 --batch_size 16 \
        --num_epoch 10

For mrpc, stsb and rte, --mnli_init takes the checkpoint of the MNLI run (the classifier
is not loaded).
"""
import argparse
import json
import os
from datetime import datetime

import evaluate
import numpy as np
import torch
import transformers

import common
from glue_data import SimpleDataModule
from glue_trainer import Trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser("fitln_glue")
    parser.add_argument("--task", default="stsb", required=False)
    parser.add_argument("--batch_size", default=2, required=False, type=int)
    parser.add_argument("--attn_only", default=True, required=False, type=common.str2bool)
    parser.add_argument("--num_epoch", default=15, required=False, type=int)
    parser.add_argument("--lr", default=5e-4, required=False, type=float)
    parser.add_argument("--model", default="bert-base-uncased", required=False)
    parser.add_argument("--mnli_init", default=None, required=False,
                        help="Checkpoint of the MNLI run used to initialize the LayerNorms.")
    parser.add_argument("--train_pooler", default=False, required=False, type=common.str2bool)
    parser.add_argument("--lr_schedule", default="constant", choices=["constant", "cosine"],
                        help="constant: fixed learning rate, no warm-up. cosine: cosine with warm-up.")
    args = parser.parse_args()

    model_name = args.model
    task = args.task
    batch_size = args.batch_size
    num_epoch = args.num_epoch
    lr = args.lr
    attn_only = args.attn_only
    attn_dir = "attn_only" if attn_only else "full"
    savedir = os.getcwd() + "/out/" + task + "/" + model_name + "/" + attn_dir + "/" + str(lr) + "/" + str(num_epoch) + "/"

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    dm = SimpleDataModule(model_name_or_path=model_name, train_batch_size=batch_size, eval_batch_size=256,
                          task_name=task, seed=0)
    dm.setup()
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=dm.num_labels).to(dm.device)
    if args.mnli_init:
        mnli_ckpt = torch.load(args.mnli_init, weights_only=False, map_location=dm.device)
        model.load_state_dict({k: v for k, v in mnli_ckpt.items() if "classifier" not in k}, strict=False)
        print("MNLI checkpoint successfully loaded")

    train_dataloader = dm.train_dataloader()
    val_dataloader = dm.val_dataloader()
    test_dataloader = dm.test_dataloader()
    metric = evaluate.load("glue", task, experiment_id=str(datetime.now().strftime("%d-%m-%Y_%H-%M-%S")) + "_task=" + args.task + "_lr=" + str(args.lr))
    trainer = Trainer(model=model, train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                      test_dataloader=test_dataloader, metric=metric)

    loss_ln, acc_ln, eloss_ln, eacc_ln, best_model, train_names = trainer.train_model(
        model, train_dataloader=train_dataloader, val_dataloader=val_dataloader, metric=metric,
        epochs=num_epoch, attn_only=attn_only, lr=lr, train_pooler=args.train_pooler,
        lr_schedule=args.lr_schedule)
    model_sd = best_model.state_dict()
    light_sd = {k: model_sd[k] for k in model_sd.keys() if k in train_names}
    metric_name = list(metric.compute(predictions=[0, 0.5], references=[0, 1]).keys())[0]
    torch.save(light_sd, savedir + "fitln_weights.bin")

    if task != "mnli":
        eloss, eacc = trainer.eval_model(best_model, val_dataloader=val_dataloader, metric=metric)
        tloss, tacc = trainer.test_model(best_model, test_dataloader=test_dataloader, metric=metric)
        results = {"val_loss": float(np.mean(eloss)), "val_" + metric_name: float(np.mean(eacc)),
                   "test_loss": float(np.mean(tloss)), "test_" + metric_name: float(np.mean(tacc))}
        with open(savedir + "results.json", mode="w", encoding="utf-8") as res:
            json.dump(results, res)
    else:
        eloss, eacc = trainer.eval_model(best_model, val_dataloader=val_dataloader[-1], metric=metric)
        for name, split in (("results_m.json", 0), ("results_mm.json", 1)):
            tloss, tacc = trainer.test_model(best_model, test_dataloader=test_dataloader[split], metric=metric)
            results = {"val_loss": float(np.mean(eloss)), "val_" + metric_name: float(np.mean(eacc)),
                       "test_loss": float(np.mean(tloss)), "test_" + metric_name: float(np.mean(tacc))}
            with open(savedir + name, mode="w", encoding="utf-8") as res:
                json.dump(results, res)
