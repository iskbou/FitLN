r"""Spearman correlation between full fine-tuning's GSM8K score (a proxy for backbone
reasoning capacity) and FitLN/FullFT retention across the six models, for GSM8K and for
SAMSum and Alpaca as controls. With n=6 the coefficients are suggestive only.
Alpaca is the 2,002-example held-out split of the Stanford Alpaca set (not AlpacaEval).
Reads the free-form table of global_summary.md.

Usage:
    python decoder/reasoning_gap_correlation.py \
        --summary_md results/decoder/global_summary.md --out_dir results/decoder
"""
from __future__ import annotations

import argparse
import os
import re

import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Deliberately generous font sizes: figure text must stay legible at column width.
plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 17,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.titlesize": 18,
})

MODEL_DISPLAY = {
    "llama3_2_1b": "Llama-3.2-1B",
    "qwen2_5_1_5b": "Qwen2.5-1.5B",
    "qwen3_5_2b_base": "Qwen3.5-2B-Base",
    "llama3_2_3b": "Llama-3.2-3B",
    "qwen2_5_3b": "Qwen2.5-3B",
    "qwen3_5_4b_base": "Qwen3.5-4B-Base",
}

TASK_LABEL = {
    "gsm8k": "GSM8K (Exact Match)",
    "samsum": "SAMSum (ROUGE-L) -- control",
    "alpaca_mini": "Alpaca (ROUGE-L) -- control",
}

_MEAN_RE = re.compile(r"^\s*([\d.]+)\s*(?:±.*)?\s*$")


def _parse_mean(cell: str) -> float | None:
    m = _MEAN_RE.match(cell)
    return float(m.group(1)) if m else None


def parse_freeform_table(md_path):
    """Returns {(model_short, method): {task: mean_score}}."""
    with open(md_path, encoding="utf-8") as f:
        lines = f.readlines()

    start = next((i for i, l in enumerate(lines)
                  if l.strip().startswith("## Free-form generation results")), None)
    if start is None:
        raise ValueError(f"{md_path} has no '## Free-form generation results' section")

    table_lines = []
    for line in lines[start + 1:]:
        if line.strip().startswith("|"):
            table_lines.append(line.strip())
        elif table_lines:
            break

    header = [c.strip() for c in table_lines[0].strip("|").split("|")]
    task_cols = {}
    for task in TASK_LABEL:
        for i, h in enumerate(header):
            if h.startswith(task):
                task_cols[task] = i
                break
    model_idx, method_idx = header.index("Model"), header.index("Method")

    out = {}
    for line in table_lines[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        model_short, method = cells[model_idx], cells[method_idx]
        scores = {task: _parse_mean(cells[idx]) for task, idx in task_cols.items()}
        out[(model_short, method)] = scores
    return out


def build_dataset(scores_by_pair):
    """Return (model_shorts, fft_gsm8k, retention); retention[task] is aligned with model_shorts."""
    model_shorts = sorted({m for m, method in scores_by_pair if method == "fullft"})
    fft_gsm8k = [scores_by_pair[(m, "fullft")]["gsm8k"] for m in model_shorts]
    retention = {}
    for task in TASK_LABEL:
        vals = []
        for m in model_shorts:
            fitln = scores_by_pair.get((m, "fitln"), {}).get(task)
            fullft = scores_by_pair.get((m, "fullft"), {}).get(task)
            vals.append(fitln / fullft if fitln is not None and fullft else None)
        retention[task] = vals
    return model_shorts, fft_gsm8k, retention


def plot_correlations(model_shorts, fft_gsm8k, retention, out_path):
    # Spearman is computed for every task; only the GSM8K panel is plotted.
    results = {t: spearmanr(fft_gsm8k, retention[t]) for t in TASK_LABEL}

    fig = plt.figure(figsize=(6, 8))
    task = "gsm8k"
    y = retention[task]
    rho, pval = results[task]
    plt.scatter(fft_gsm8k, y, s=60)
    for x_i, y_i, m in zip(fft_gsm8k, y, model_shorts):
        plt.annotate(MODEL_DISPLAY.get(m, m), (x_i, y_i),
                    textcoords="offset points", xytext=(2, -14), fontsize=12)
    plt.xlabel("FFT GSM8K score\n(backbone reasoning capacity proxy)")
    plt.ylabel("FitLN / FFT retention")
    plt.title(f"{TASK_LABEL[task]}\nSpearman ρ = {rho:.2f} (p = {pval:.2f}, n={len(fft_gsm8k)})")
    plt.ylim(0.0, 1.05)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return results


def leave_one_out(model_shorts, fft_gsm8k, retention, drop_model):
    """Spearman rho for every task with drop_model excluded."""
    keep = [i for i, m in enumerate(model_shorts) if m != drop_model]
    x = [fft_gsm8k[i] for i in keep]
    out = {}
    for task in TASK_LABEL:
        y = [retention[task][i] for i in keep]
        out[task] = spearmanr(x, y)
    return out


def to_markdown(model_shorts, fft_gsm8k, retention, results):
    lines = ["## Reasoning-gap correlation", ""]
    lines.append("| Model | FFT GSM8K | GSM8K retention | SAMSum retention | Alpaca retention |")
    lines.append("| --- | --- | --- | --- | --- |")
    for i, m in enumerate(model_shorts):
        lines.append("| " + " | ".join([
            MODEL_DISPLAY.get(m, m),
            f"{fft_gsm8k[i]:.4f}",
            f"{retention['gsm8k'][i]:.4f}",
            f"{retention['samsum'][i]:.4f}",
            f"{retention['alpaca_mini'][i]:.4f}",
        ]) + " |")
    lines.append("")
    lines.append("| Task | Spearman ρ (n=6) | p-value (informal -- see script docstring) |")
    lines.append("| --- | --- | --- |")
    for task in ["gsm8k", "samsum", "alpaca_mini"]:
        rho, pval = results[task]
        lines.append(f"| {task} | {rho:.3f} | {pval:.3f} |")

    weakest = min(range(len(model_shorts)), key=lambda i: fft_gsm8k[i])
    loo = leave_one_out(model_shorts, fft_gsm8k, retention, model_shorts[weakest])
    lines.append("")
    lines.append(f"### Sensitivity check: leave-{MODEL_DISPLAY.get(model_shorts[weakest], model_shorts[weakest])}-out (n=5)")
    lines.append("")
    lines.append(
        f"{MODEL_DISPLAY.get(model_shorts[weakest], model_shorts[weakest])} is the weakest "
        f"backbone by FFT GSM8K score, so it sits at the low extreme of every panel "
        f"simultaneously -- removing it checks whether it alone is driving the n=6 correlations."
    )
    lines.append("")
    lines.append("| Task | Spearman ρ (n=5, leave-one-out) |")
    lines.append("| --- | --- |")
    for task in ["gsm8k", "samsum", "alpaca_mini"]:
        rho, _ = loo[task]
        lines.append(f"| {task} | {rho:.3f} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_md", default="results/decoder/global_summary.md")
    parser.add_argument("--out_dir", default="results/decoder")
    args = parser.parse_args()

    if not os.path.isfile(args.summary_md):
        print(f"[reasoning_gap_correlation] {args.summary_md} does not exist -- "
              f"run summarize_results.py first.")
        return

    scores_by_pair = parse_freeform_table(args.summary_md)
    model_shorts, fft_gsm8k, retention = build_dataset(scores_by_pair)

    os.makedirs(args.out_dir, exist_ok=True)
    fig_path = os.path.join(args.out_dir, "fig_reasoning_gap_correlation.pdf")
    results = plot_correlations(model_shorts, fft_gsm8k, retention, fig_path)
    print(f"[reasoning_gap_correlation] wrote {fig_path}")

    md_path = os.path.join(args.out_dir, "reasoning_gap_correlation.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(to_markdown(model_shorts, fft_gsm8k, retention, results))
        f.write("\n")
    print(f"[reasoning_gap_correlation] wrote {md_path}")

    print("\n=== Spearman correlations (FFT GSM8K score vs. FitLN/FFT retention) ===")
    for task in ["gsm8k", "samsum", "alpaca_mini"]:
        rho, pval = results[task]
        print(f"{task:12s} rho={rho:+.3f}  p={pval:.3f}  (n={len(fft_gsm8k)})")


if __name__ == "__main__":
    main()
