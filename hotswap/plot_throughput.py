"""Throughput figure and cross-model summary table from throughput_benchmark.py.

Reads {results_dir}/throughput_*.json and writes {figdir}/fig_throughput.pdf and
{results_dir}/throughput_summary.md. Panel A is inference throughput, with LoRA also shown
merged (hatched); panel B is training-step throughput.

Usage:
    python hotswap/plot_throughput.py --results_dir results/hotswap --figdir figures/hotswap
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHOD_COLOR = {
    "fitln":  "#0072B2",
    "bitfit": "#E69F00",
    "lora1":  "#009E73",
    "lora8":  "#CC79A7",
    "fullft": "#D55E00",
}
METHOD_LABEL = {
    "fitln": "FitLN", "bitfit": "BitFit", "lora1": "LoRA (r=1)",
    "lora8": "LoRA (r=8)", "fullft": "FullFT",
}
METHOD_ORDER = ["fitln", "bitfit", "lora1", "lora8", "fullft"]

plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 17,
    "axes.labelsize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "figure.titlesize": 18,
})


def load_all_models(results_dir: str) -> dict:
    """{model: contents of throughput_*.json}"""
    out = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "throughput_*.json"))):
        with open(path) as f:
            d = json.load(f)
        out[d["model"]] = d
    return out


def short_model_label(hf_id: str) -> str:
    return hf_id.split("/")[-1]


def samples_per_sec(row: dict, batch_size: int) -> float:
    return batch_size / (row["mean_ms"] / 1000)


def _bar_positions(n_models: int, n_methods: int, bar_width: float = 0.14):
    group_width = n_methods * bar_width
    centers = np.arange(n_models)
    offsets = (np.arange(n_methods) - (n_methods - 1) / 2) * bar_width
    return centers, offsets, group_width


def _present_methods(data: dict, regime: str) -> list:
    return [m for m in METHOD_ORDER
            if any(any(r["method"] == m and r["regime"] == regime for r in data[model]["rows"])
                   for model in data)]


def _row(data, model, method, regime, merged=False):
    for r in data[model]["rows"]:
        if r["method"] == method and r["regime"] == regime and r["merged"] == merged:
            return r
    return None


def plot_inference_panel(ax, data: dict, yscale: str):
    models = list(data.keys())
    labels = [short_model_label(m) for m in models]
    methods = _present_methods(data, "inference")
    centers, offsets, _ = _bar_positions(len(models), len(methods))

    for mi, method in enumerate(methods):
        for model_i, model in enumerate(models):
            rows = [r for r in data[model]["rows"]
                    if r["method"] == method and r["regime"] == "inference"]
            if not rows:
                continue
            bs = data[model]["batch_size"]
            unmerged = next((r for r in rows if not r["merged"]), None)
            merged = next((r for r in rows if r["merged"]), None)
            x = centers[model_i] + offsets[mi]
            if unmerged:
                ax.bar(x, samples_per_sec(unmerged, bs), width=0.13,
                       color=METHOD_COLOR[method], edgecolor="white", linewidth=0.6, zorder=3)
            if merged:
                ax.bar(x, samples_per_sec(merged, bs), width=0.13,
                       facecolor="none", edgecolor=METHOD_COLOR[method], hatch="///",
                       linewidth=1.4, zorder=4)

    ax.set_xticks(centers)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Inference throughput (samples/sec)")
    ax.set_title("(A) Inference throughput\n(solid = unmerged, hatched = LoRA merged)", pad=10)
    ax.set_yscale(yscale)
    ax.grid(True, which="major", axis="y", alpha=0.25, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_training_panel(ax, data: dict, yscale: str):
    models = list(data.keys())
    labels = [short_model_label(m) for m in models]
    methods = _present_methods(data, "train_step")
    centers, offsets, _ = _bar_positions(len(models), len(methods))

    for mi, method in enumerate(methods):
        for model_i, model in enumerate(models):
            rows = [r for r in data[model]["rows"]
                    if r["method"] == method and r["regime"] == "train_step"]
            if not rows:
                continue
            bs = data[model]["batch_size"]
            x = centers[model_i] + offsets[mi]
            ax.bar(x, samples_per_sec(rows[0], bs), width=0.13,
                   color=METHOD_COLOR[method], edgecolor="white", linewidth=0.6, zorder=3)

    ax.set_xticks(centers)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Training-step throughput (samples/sec)")
    ax.set_title("(B) Training-step throughput\n(forward + backward + optimizer.step())", pad=10)
    ax.set_yscale(yscale)
    ax.grid(True, which="major", axis="y", alpha=0.25, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _ratio(num, den):
    return num / den if (num is not None and den is not None) else None


def _fmt(r):
    return f"{r:.2f}x" if r is not None else "-"


def _range(vals, nd=3):
    return f"{min(vals):.{nd}f}-{max(vals):.{nd}f}x" if vals else "-"


def write_summary_markdown(results_dir: str, md_path: str) -> None:
    data = load_all_models(results_dir)
    if not data:
        raise SystemExit(f"No throughput_*.json found in {results_dir}")
    models = list(data.keys())

    lines = ["# Throughput summary", "",
             f"{len(models)} model(s): {', '.join(short_model_label(m) for m in models)}. "
             "Per-row detail is in throughput_{model}.md.", ""]

    lines += ["## 1. Inference, FitLN vs. FullFT", "",
              "| Model | FitLN (samples/s) | FullFT (samples/s) | ratio |",
              "|---|---|---|---|"]
    for model in models:
        bs = data[model]["batch_size"]
        fitln, fullft = _row(data, model, "fitln", "inference"), _row(data, model, "fullft", "inference")
        if fitln and fullft:
            f_sps, ff_sps = samples_per_sec(fitln, bs), samples_per_sec(fullft, bs)
            lines.append(f"| {short_model_label(model)} | {f_sps:.1f} | {ff_sps:.1f} | {f_sps/ff_sps:.3f}x |")
    lines.append("")

    lines += ["## 2. LoRA inference, unmerged and merged", "",
              "| Model | LoRA-r1 unmerged | LoRA-r1 merged | overhead removed | "
              "merged vs FitLN | LoRA-r8 unmerged | LoRA-r8 merged | overhead removed | merged vs FitLN |",
              "|---|---|---|---|---|---|---|---|---|"]
    for model in models:
        bs = data[model]["batch_size"]
        fitln = _row(data, model, "fitln", "inference")
        row_cells = [short_model_label(model)]
        for method in ("lora1", "lora8"):
            um, m = _row(data, model, method, "inference", merged=False), _row(data, model, method, "inference", merged=True)
            if um and m and fitln:
                um_sps, m_sps, f_sps = samples_per_sec(um, bs), samples_per_sec(m, bs), samples_per_sec(fitln, bs)
                row_cells += [f"{um_sps:.1f}", f"{m_sps:.1f}", f"+{(m_sps/um_sps-1)*100:.1f}%", f"{m_sps/f_sps:.3f}x"]
            else:
                row_cells += ["-", "-", "-", "-"]
        lines.append("| " + " | ".join(row_cells) + " |")
    lines.append("")

    lines += ["## 3. Training step (forward, backward, optimizer step)", "",
              "| Model | FitLN (samples/s) | FullFT (samples/s) | FitLN speedup | "
              "LoRA-r1 (samples/s) | LoRA-r8 (samples/s) | "
              "FitLN / LoRA-r1 | FitLN / LoRA-r8 | LoRA-r1 / FullFT | LoRA-r8 / FullFT |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    ratio_cols = {"fit_l1": [], "fit_l8": [], "l1_full": [], "l8_full": []}
    ratio_fullft = []
    for model in models:
        bs = data[model]["batch_size"]
        fitln, fullft = _row(data, model, "fitln", "train_step"), _row(data, model, "fullft", "train_step")
        l1, l8 = _row(data, model, "lora1", "train_step"), _row(data, model, "lora8", "train_step")
        if fitln and fullft:
            f_sps, ff_sps = samples_per_sec(fitln, bs), samples_per_sec(fullft, bs)
            l1_v = samples_per_sec(l1, bs) if l1 else None
            l8_v = samples_per_sec(l8, bs) if l8 else None
            l1_sps = f"{l1_v:.1f}" if l1_v is not None else "-"
            l8_sps = f"{l8_v:.1f}" if l8_v is not None else "-"
            r = {"fit_l1": _ratio(f_sps, l1_v), "fit_l8": _ratio(f_sps, l8_v),
                 "l1_full": _ratio(l1_v, ff_sps), "l8_full": _ratio(l8_v, ff_sps)}
            for k, v in r.items():
                if v is not None:
                    ratio_cols[k].append(v)
            ratio_fullft.append(f_sps / ff_sps)
            lines.append(f"| {short_model_label(model)} | {f_sps:.1f} | {ff_sps:.1f} | "
                         f"{f_sps/ff_sps:.2f}x | {l1_sps} | {l8_sps} | "
                         f"{_fmt(r['fit_l1'])} | {_fmt(r['fit_l8'])} | "
                         f"{_fmt(r['l1_full'])} | {_fmt(r['l8_full'])} |")
    lines.append("")

    lines += ["Range across models (min-max):", "",
              "| Comparison | Range |", "|---|---|",
              f"| FitLN vs. FullFT | {_range(ratio_fullft)} |",
              f"| FitLN vs. LoRA-r8 | {_range(ratio_cols['fit_l8'])} |",
              f"| FitLN vs. LoRA-r1 | {_range(ratio_cols['fit_l1'])} |",
              f"| LoRA-r1 vs. FullFT | {_range(ratio_cols['l1_full'])} |",
              f"| LoRA-r8 vs. FullFT | {_range(ratio_cols['l8_full'])} |",
              ""]

    d0 = next(iter(data.values()))
    lines += [f"Setting: one GPU, single process, batch size {d0['batch_size']}, sequence length "
              f"{d0['seq_len']}, {d0['dtype']} for every method. Training-step throughput, not "
              "end-to-end training time.", ""]

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", md_path)


def make_figure(results_dir: str, out_path_pdf: str, yscale: str = "log"):
    data = load_all_models(results_dir)
    if not data:
        raise SystemExit(f"No throughput_*.json found in {results_dir}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
    plot_inference_panel(ax1, data, yscale)
    plot_training_panel(ax2, data, yscale)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=METHOD_COLOR[m], edgecolor="white",
                             label=METHOD_LABEL[m])
               for m in METHOD_ORDER if any(
                   any(r["method"] == m for r in data[model]["rows"]) for model in data)]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="black",
                                 hatch="///", label="LoRA, merged"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, -0.08), frameon=False, columnspacing=1.4, handletextpad=0.6)

    n_models = len(data)
    fig.suptitle(f"Single-process throughput, {n_models} model(s)", y=1.04)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_path_pdf, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote", out_path_pdf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("plot_throughput")
    parser.add_argument("--results_dir", default="results/hotswap")
    parser.add_argument("--figdir", default="figures/hotswap")
    parser.add_argument("--yscale", default="log", choices=["log", "linear"])
    args = parser.parse_args()
    os.makedirs(args.figdir, exist_ok=True)
    make_figure(args.results_dir, os.path.join(args.figdir, "fig_throughput.pdf"), yscale=args.yscale)
    write_summary_markdown(args.results_dir, os.path.join(args.results_dir, "throughput_summary.md"))
