"""Training loop for FitLN on GLUE.

Trains the attention-output LayerNorm of every layer (the final norm for ModernBERT) and
the classifier; the pooler is trained only with train_pooler=True.
"""
import copy
import tqdm
import numpy as np
import torch
from transformers import get_cosine_schedule_with_warmup
from torch.optim import AdamW


class Trainer():
    def __init__(self, model, train_dataloader, val_dataloader, test_dataloader, metric=None) -> None:
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.metric = metric

    def train_model(self, model=None, train_dataloader=None, val_dataloader=None, metric=None,
                    epochs=10, lr=10e-4, attn_only=False, train_pooler=False, lr_schedule="constant"):
        """Returns train losses, train metric, eval losses, eval metric per epoch, the best
        model (by validation metric, later epochs win ties) and the names of trained parameters.

        lr_schedule "constant": fixed learning rate, no warm-up. "cosine": cosine schedule with
        warm-up."""
        model = self.model if model is None else model
        train_dataloader = self.train_dataloader if train_dataloader is None else train_dataloader
        val_dataloader = self.val_dataloader if val_dataloader is None else val_dataloader
        metric = self.metric if metric is None else metric
        train_param = []
        train_names = []
        model.requires_grad_(False)
        if "modernbert" in model.config.model_type:
            layernorm_key = "attn_norm" if attn_only else "norm"
        elif "new" in model.config.model_type:
            layernorm_key = "attn_ln" if attn_only else "_ln"
        else:
            layernorm_key = "attention.output.LayerNorm" if attn_only else "LayerNorm"
        keys = [layernorm_key, "final_norm", "classifier"] + (["pooler"] if train_pooler else [])
        for n, p in model.named_parameters():
            for key in keys:
                if key in n:
                    p.requires_grad = True
                    print(n)
                    train_param.append(p)
                    train_names.append(n)

        num_param = 0
        for p in train_param:
            num_param += p.numel()
        print(f"Trainable params: {num_param}, Ratio: {num_param/np.sum([p.numel() for p in model.parameters()])}")
        # weight_decay_rate is not an AdamW option, so the default weight decay (0.01) applies.
        optimizer = AdamW([{'params': train_param, 'weight_decay_rate': 0.0}], lr=lr)
        scheduler = None
        if lr_schedule == "cosine":
            # num_steps counts examples, not batches, so the schedule is mostly warm-up.
            num_steps = int(len(train_dataloader.dataset) * epochs) + 1
            scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=int(num_steps / 100),
                                                        num_training_steps=num_steps)
        losses = []
        accuracies = []
        elosses = []
        eacc = []
        ep_prog = tqdm.tqdm(range(epochs))
        best_model = model
        best_epoch = 0
        best_acc = 0
        for epoch in ep_prog:
            full_preds = []
            full_labels = []
            train_prog = tqdm.tqdm(train_dataloader)
            for b_idx, batch in enumerate(train_prog):
                model.train()
                optimizer.zero_grad()
                outputs = model(**{kx: vx.to(model.device, non_blocking=True) for kx, vx in batch.items()})
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                if outputs.logits.shape[-1] <= 1 or len(outputs.logits.shape) <= 1:
                    pred = outputs.logits.squeeze().detach().cpu().numpy()
                else:
                    pred = outputs.logits.argmax(dim=-1).detach().cpu().numpy()
                losses.append(loss.item())
                train_prog.set_postfix({"Train Loss": loss.item()})
                full_preds.extend(list(pred))
                full_labels.extend(batch["labels"].detach().cpu().numpy())

            acc = metric.compute(predictions=full_preds, references=full_labels)
            metric_name = list(acc.keys())[0]
            metric_value = list(acc.values())[0]
            accuracies.append(metric_value)

            # MNLI has several validation sets; the last one is the held-out part of train.
            if type(val_dataloader) == list:
                eval_losses, eval_acc = self.eval_model(model, val_dataloader=val_dataloader[-1], metric=metric)
            else:
                eval_losses, eval_acc = self.eval_model(model, val_dataloader=val_dataloader, metric=metric)
            elosses.append(np.mean(eval_losses))
            if eval_acc >= best_acc:
                best_model = copy.deepcopy(model)
                best_epoch = copy.copy(epoch)
                best_acc = eval_acc
            eacc.append(eval_acc)
            ep_prog.set_postfix({metric_name: metric_value, "Eval Loss": np.mean(eval_losses), "eval_" + metric_name: eval_acc})
        if type(val_dataloader) == list:
            _, best_model_run = self.eval_model(best_model, val_dataloader=val_dataloader[-1], metric=metric)
        else:
            _, best_model_run = self.eval_model(best_model, val_dataloader=val_dataloader, metric=metric)

        print(f"Best Eval Accuracy: {best_model_run} at epoch {best_epoch}")
        return losses, accuracies, elosses, eacc, best_model, train_names

    def eval_model(self, model, val_dataloader, metric):
        model = self.model if model is None else model
        val_dataloader = self.val_dataloader if val_dataloader is None else val_dataloader
        metric = self.metric if metric is None else metric
        losses = []
        accuracies = []
        eval_prog = tqdm.tqdm(val_dataloader)
        full_preds = []
        full_labels = []
        for batch in eval_prog:
            model.eval()
            with torch.no_grad():
                outputs = model(**{kx: vx.to(model.device, non_blocking=True) for kx, vx in batch.items()})
                loss = outputs.loss
                if outputs.logits.shape[-1] <= 1 or len(outputs.logits.shape) <= 1:
                    pred = outputs.logits.squeeze().detach().cpu().numpy()
                else:
                    pred = outputs.logits.argmax(dim=-1).detach().cpu().numpy()

                losses.append(loss.item())
                eval_prog.set_postfix({"Eval Loss": loss.item()})
                full_preds.extend(pred)
                full_labels.extend(batch["labels"].detach().cpu().numpy())
        acc = metric.compute(predictions=full_preds, references=full_labels)
        accuracies.append(list(acc.values())[0])
        print(f"Mean loss: {np.mean(losses)}, {list(acc.keys())[0]}:{list(acc.values())[0]}")
        return losses, accuracies[-1]

    def test_model(self, model, test_dataloader, metric):
        model = self.model if model is None else model
        test_dataloader = self.test_dataloader if test_dataloader is None else test_dataloader
        metric = self.metric if metric is None else metric
        return self.eval_model(model, val_dataloader=test_dataloader, metric=metric)
