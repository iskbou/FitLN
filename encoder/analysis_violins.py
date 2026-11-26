"""Violin plots of the attention-output LayerNorm weights and biases of RoBERTa-large per layer.

One plot per GLUE task (checkpoint with the best validation metric among learning rates 6e-4,
8e-4 and 1e-3 at 10 epochs), one for the pretrained model, and optionally one for a model
trained with beir_train.py.

cache: save the LayerNorm parameters of the selected checkpoints and of the pretrained model.
plot:  draw the plots from that file.

Usage (from the repository root):
    python encoder/analysis_violins.py cache --root out --out results/encoder/roberta_violin_params.pt
    python encoder/analysis_violins.py plot --cache results/encoder/roberta_violin_params.pt --figdir figures/encoder
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

from analysis.loader import list_checkpoints

MODEL = "FacebookAI/roberta-large"
TASKS = ["mrpc", "stsb", "rte", "cola", "qnli", "sst2", "qqp"]
LRS = [6e-4, 8e-4, 10e-4]
EPOCHS = 10

plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 17,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.titlesize": 18,
})


def is_ln_weight_or_bias(name):
    return "orm" in name


def build_cache(root, out, msmarco_ckpt=None):
    import transformers
    pre = transformers.AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2).state_dict()
    cache = {"pretrained": {k: v.clone() for k, v in pre.items() if "attention.output.LayerNorm" in k}, "checkpoints": {}}
    for task in TASKS:
        refs = [r for r in list_checkpoints(root, task=task, model=MODEL, mode="attn_only")
                if r.val_metric is not None and r.lr in LRS and r.epochs == EPOCHS]
        if not refs:
            continue
        best = max(refs, key=lambda r: r.val_metric)
        cache["checkpoints"][task] = {"lr": best.lr, "state": {k: v for k, v in best.load().items() if is_ln_weight_or_bias(k)}}
        print(task, best.lr)
    if msmarco_ckpt:
        state = torch.load(msmarco_ckpt, map_location="cpu", weights_only=False)
        state = {k.replace("auto_model.encoder.layer.", "roberta.encoder.layer."): v for k, v in state.items()}
        cache["checkpoints"]["MsMARCO"] = {"lr": None, "state": {k: v for k, v in state.items() if is_ln_weight_or_bias(k)}}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, out)
    print("saved", out)


def violin(state, figpath):
    w_diff, b_diff = [], []
    for name in [k for k in state if "orm" in k and "weight" in k]:
        b_diff.append(state[name.replace("weight", "bias")].tolist())
        w_diff.append(state[name].tolist())
    df = pd.DataFrame({
        "layer": list(range(len(w_diff))) + list(range(len(b_diff))),
        "module": ["weight"] * len(w_diff) + ["bias"] * len(b_diff),
        "diff": w_diff + b_diff,
    })
    df = df.explode(column="diff", ignore_index=True)
    df["diff"] = df["diff"].astype(float)
    fig = plt.figure(figsize=(20, 8))
    sns.violinplot(data=df, x="layer", y="diff", hue="module", split=True)
    plt.ylabel("")
    plt.xlabel("")
    plt.ylim(-1.2, 1.2)
    plt.savefig(figpath, bbox_inches="tight")
    plt.close(fig)
    print("saved", figpath)


def plot(cache_path, figdir):
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    for task, entry in cache["checkpoints"].items():
        violin(entry["state"], Path(figdir) / f"violin_component_{task}.pdf")
    violin(cache["pretrained"], Path(figdir) / "violin_component_pretrained.pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("cache")
    p.add_argument("--root", default="out")
    p.add_argument("--out", default="results/encoder/roberta_violin_params.pt")
    p.add_argument("--msmarco_ckpt", default=None, help="Checkpoint of a model trained with beir_train.py (optional).")
    p = sub.add_parser("plot")
    p.add_argument("--cache", default="results/encoder/roberta_violin_params.pt")
    p.add_argument("--figdir", default="figures/encoder")
    args = parser.parse_args()
    if args.command == "cache":
        build_cache(args.root, args.out, args.msmarco_ckpt)
    else:
        os.makedirs(args.figdir, exist_ok=True)
        plot(args.cache, args.figdir)


if __name__ == "__main__":
    main()
