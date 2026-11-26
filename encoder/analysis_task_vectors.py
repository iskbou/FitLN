"""Task-vector analysis of the encoder FitLN checkpoints.

cache:        read the best-learning-rate checkpoint of every (model, task) in out/ and the
              pretrained models, and save only their attention-output LayerNorm parameters to
              one file (a few MB). MNLI has no best checkpoint (see analysis/loader.py).
figures:      cosine matrices, sparsity profiles and cross-model magnitude profiles from that file.
pruning:      accuracy of the checkpoints with only the top-k% channels per LayerNorm kept
              (needs out/, the models, the GLUE data and a GPU); writes a csv.
pruning_plot: plot the pruning csv.

Usage (from the repository root):
    python encoder/analysis_task_vectors.py cache --root out --out results/encoder/ln_params.pt
    python encoder/analysis_task_vectors.py figures --cache results/encoder/ln_params.pt --figdir figures/encoder
    python encoder/analysis_task_vectors.py pruning --root out --out results/encoder/pruning_curves.csv
    python encoder/analysis_task_vectors.py pruning_plot --csv results/encoder/pruning_curves.csv --figdir figures/encoder
"""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from analysis.loader import find_best_checkpoint
from analysis.pruning import topk_channel_mask
from analysis.task_vectors import (TaskVector, compute_task_vector, cosine, ln_keys,
                                   magnitude_profile, sparsity_profile)

MODELS = {
    'BERT-L':       'bert-large-uncased',
    'RoBERTa-L':    'FacebookAI/roberta-large',
    'DeBERTa-L':    'microsoft/deberta-v3-large',
    'ModernBERT-L': 'answerdotai/ModernBERT-large',
}
TASKS = ['cola', 'sst2', 'mrpc', 'stsb', 'qqp', 'mnli', 'qnli', 'rte']
K_GRID = [0.01, 0.05, 0.10, 0.25, 0.50, 1.0]
PRUNING_TASKS = ['mrpc', 'sst2', 'rte', 'cola']

plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 17,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.titlesize": 18,
})


def model_tag(model_short):
    return model_short.lower().replace("-", "")


def build_cache(root, out):
    import transformers
    cache = {"models": {}, "checkpoints": {}}
    for short, hf in MODELS.items():
        model = transformers.AutoModelForSequenceClassification.from_pretrained(hf, num_labels=2)
        model_type = model.config.model_type
        key_sub = ln_keys(model_type, True)
        cache["models"][short] = {
            "hf": hf, "model_type": model_type,
            "pretrained": {k: v.clone() for k, v in model.state_dict().items() if key_sub in k},
        }
        del model
        for task in TASKS:
            ref = find_best_checkpoint(task=task, model=hf, mode="attn_only", root=root)
            if ref is None:
                ref = find_best_checkpoint(task=task, model=hf.split("/")[-1], mode="attn_only", root=root)
            if ref is None:
                continue
            state = {k: v for k, v in ref.load().items() if key_sub in k}
            cache["checkpoints"][f"{short}/{task}"] = {"lr": ref.lr, "epochs": ref.epochs, "state": state}
            print(short, task, ref.lr, ref.epochs)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, out)
    print("saved", out)


class Vectors:
    """Task vectors computed from the cache file."""

    def __init__(self, path):
        self.cache = torch.load(path, map_location="cpu", weights_only=False)
        self.tv = {}

    def pretrained(self, model_short):
        return self.cache["models"][model_short]["pretrained"]

    def get(self, model_short, task):
        key = (model_short, task)
        if key not in self.tv:
            entry = self.cache["checkpoints"].get(f"{model_short}/{task}")
            if entry is None:
                self.tv[key] = None
            else:
                self.tv[key] = compute_task_vector(
                    entry["state"], self.pretrained(model_short),
                    model_type=self.cache["models"][model_short]["model_type"], attn_only=True)
        return self.tv[key]


def gamma_only(tv: TaskVector) -> TaskVector:
    out = TaskVector(model_type=tv.model_type, attn_only=tv.attn_only)
    for ell, slot in tv.deltas.items():
        out.deltas[ell] = {'gamma': slot['gamma'], 'beta': None}
    out.raw_deltas = {k: v for k, v in tv.raw_deltas.items() if k.endswith('.weight')}
    return out


def beta_only(tv: TaskVector) -> TaskVector:
    out = TaskVector(model_type=tv.model_type, attn_only=tv.attn_only)
    for ell, slot in tv.deltas.items():
        out.deltas[ell] = {'gamma': None, 'beta': slot['beta']}
    out.raw_deltas = {k: v for k, v in tv.raw_deltas.items() if k.endswith('.bias')}
    return out


def cosine_matrix(vectors, model_short, decomposition='full'):
    tvs = {}
    for task in TASKS:
        tv = vectors.get(model_short, task)
        if tv is None:
            continue
        if decomposition == 'gamma':
            tv = gamma_only(tv)
        elif decomposition == 'beta':
            tv = beta_only(tv)
        tvs[task] = tv
    keys = list(tvs.keys())
    n = len(keys)
    if n == 0:
        return None, None
    M = np.zeros((n, n))
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            try:
                M[i, j] = cosine(tvs[a], tvs[b])
            except ValueError:
                M[i, j] = np.nan
    return M, keys


def plot_cosine(vectors, figdir):
    for model_short in MODELS:
        decomps = ['full', 'gamma'] if model_short == 'ModernBERT-L' else ['full', 'gamma', 'beta']
        fig, axes = plt.subplots(1, len(decomps), figsize=(5 * len(decomps), 5.5))
        for ax, decomp in zip(axes, decomps):
            M, keys = cosine_matrix(vectors, model_short, decomp)
            if M is None:
                ax.set_visible(False)
                continue
            sns.heatmap(M, annot=True, fmt='.2f', xticklabels=keys, yticklabels=keys,
                        cmap='RdBu_r', center=0, vmin=-1, vmax=1, ax=ax, cbar=(decomp == decomps[-1]))
            ax.set_title(f'{model_short} — {decomp}')
        plt.tight_layout()
        out = Path(figdir) / f'fig_cosine_matrix_{model_tag(model_short)}.pdf'
        plt.savefig(out, bbox_inches='tight')
        plt.close(fig)
        print('saved', out)


def plot_sparsity(vectors, figdir):
    for model_short in MODELS:
        pre_sd = vectors.pretrained(model_short)
        rows = []
        for task in TASKS:
            tv = vectors.get(model_short, task)
            if tv is None:
                continue
            sp = sparsity_profile(tv, pre_sd, eps=0.01)
            for layer, gf, bf in zip(sp['layers'], sp['gamma_active_frac'], sp['beta_active_frac']):
                rows.append({'task': task, 'layer': layer, 'gamma_active': gf, 'beta_active': bf})
        if not rows:
            continue
        df = pd.DataFrame(rows)
        if model_short == 'ModernBERT-L':
            fig, a1 = plt.subplots(1, 1, figsize=(5, 3.5))
            sns.lineplot(data=df, x='layer', y='gamma_active', errorbar='sd', ax=a1)
            a1.set_title(f'{model_short}: γ active fraction')
        else:
            fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.5))
            sns.lineplot(data=df, x='layer', y='gamma_active', errorbar='sd', ax=a1)
            sns.lineplot(data=df, x='layer', y='beta_active', errorbar='sd', ax=a2)
            a1.set_title(f'{model_short}: γ active fraction')
            a2.set_title(f'{model_short}: β active fraction')
        out = Path(figdir) / f'fig_sparsity_{model_tag(model_short)}.pdf'
        plt.tight_layout()
        plt.savefig(out, bbox_inches='tight')
        plt.close(fig)
        print('saved', out)


def plot_crossmodel(vectors, figdir):
    for task in TASKS:
        if all(vectors.get(m, task) is None for m in MODELS):
            continue
        fig = plt.figure(figsize=(7, 3.5))
        for model_short in MODELS:
            tv = vectors.get(model_short, task)
            if tv is None:
                continue
            prof = magnitude_profile(tv)
            L = max(prof['layers']) + 1
            plt.plot([l / max(1, L - 1) for l in prof['layers']], prof['gamma_norm'], label=model_short)
        plt.xlabel('Relative layer depth')
        plt.ylabel(r'$\|\Delta\gamma\|_2$')
        plt.title(f'Cross-arch ‖Δγ‖ on {task}')
        plt.legend()
        out = Path(figdir) / f'fig_crossmodel_gamma_{task}.pdf'
        plt.savefig(out, bbox_inches='tight')
        plt.close(fig)
        print('saved', out)


def evaluate_state_dict(model_name, task, state_dict, split='test', batch_size=256):
    """Metric of the pretrained model with state_dict loaded on a GLUE split."""
    import evaluate as hf_evaluate
    import transformers
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from glue_data import SimpleDataModule
    from glue_trainer import Trainer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dm = SimpleDataModule(model_name_or_path=model_name, task_name=task, train_batch_size=batch_size,
                          eval_batch_size=batch_size, max_seq_length=512, seed=0)
    dm.setup()
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=dm.num_labels).to(dev)
    model.load_state_dict({k: v.to(dev) for k, v in state_dict.items()}, strict=False)
    metric = hf_evaluate.load("glue", task, experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S_%f") + f"_{task}")
    loader = dm.val_dataloader() if split == "val" else dm.test_dataloader()
    if isinstance(loader, list):
        loader = loader[-1]
    trainer = Trainer(model=model, train_dataloader=None, val_dataloader=loader, test_dataloader=loader, metric=metric)
    _, acc = trainer.eval_model(model, val_dataloader=loader, metric=metric)
    return float(acc)


def run_pruning(root, out):
    import transformers
    results = []
    for short, hf in MODELS.items():
        pre = transformers.AutoModelForSequenceClassification.from_pretrained(hf, num_labels=2)
        pre_sd, model_type = pre.state_dict(), pre.config.model_type
        for task in PRUNING_TASKS:
            ref = find_best_checkpoint(task=task, model=hf, mode="attn_only", root=root)
            if ref is None:
                ref = find_best_checkpoint(task=task, model=hf.split("/")[-1], mode="attn_only", root=root)
            if ref is None:
                continue
            ft = ref.load(map_location="cpu")
            for k in K_GRID:
                sd = topk_channel_mask(ft, pre_sd, model_type=model_type, attn_only=True, k_frac=k)
                value = evaluate_state_dict(hf, task, sd)
                results.append({'model': short, 'task': task, 'k': k, 'metric': value})
                print(f'{short:14s} {task:5s} k={k:>4} -> {value:.4f}')
    pd.DataFrame(results).to_csv(out, index=False)


def plot_pruning(csv_path, figdir):
    df = pd.read_csv(csv_path)
    for model_short in df['model'].unique():
        sub = df[df['model'] == model_short]
        fig = plt.figure(figsize=(6, 3.5))
        sns.lineplot(data=sub, x='k', y='metric', hue='task', marker='o')
        plt.xscale('log')
        plt.title(f'{model_short}: perf vs top-k% channels retained')
        plt.xlabel('k (fraction of channels retained per LN)')
        plt.xticks(K_GRID, [str(int(k * 100)) + "%" for k in K_GRID])
        plt.ylabel('Task metric')
        out = Path(figdir) / f'fig_pruning_curve_{model_tag(model_short)}_logscale.pdf'
        plt.savefig(out, bbox_inches='tight')
        plt.close(fig)
        print('saved', out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("cache")
    p.add_argument("--root", default="out")
    p.add_argument("--out", default="results/encoder/ln_params.pt")
    p = sub.add_parser("figures")
    p.add_argument("--cache", default="results/encoder/ln_params.pt")
    p.add_argument("--figdir", default="figures/encoder")
    p = sub.add_parser("pruning")
    p.add_argument("--root", default="out")
    p.add_argument("--out", default="results/encoder/pruning_curves.csv")
    p = sub.add_parser("pruning_plot")
    p.add_argument("--csv", default="results/encoder/pruning_curves.csv")
    p.add_argument("--figdir", default="figures/encoder")
    args = parser.parse_args()

    if args.command == "cache":
        build_cache(args.root, args.out)
    elif args.command == "figures":
        os.makedirs(args.figdir, exist_ok=True)
        vectors = Vectors(args.cache)
        plot_cosine(vectors, args.figdir)
        plot_sparsity(vectors, args.figdir)
        plot_crossmodel(vectors, args.figdir)
    elif args.command == "pruning":
        run_pruning(args.root, args.out)
    else:
        os.makedirs(args.figdir, exist_ok=True)
        plot_pruning(args.csv, args.figdir)


if __name__ == "__main__":
    main()
