"""Whole-sweep summary in one file, global_summary.md: GLUE-as-generation results,
free-form results, and model/method characteristics, one row per (model, method).
Uses the same run discovery, learning-rate selection, statistics and --seeds filter
as aggregate_decoder_results.py.

Usage:
    python decoder/summarize_results.py --root out/decoder --out results/decoder/global_summary.md
"""
import argparse
import os
import statistics

import common
from aggregate_decoder_results import (REPORTED_SEEDS, iter_runs, parse_seeds_arg,
                                       select_best_lr_across_seeds, _mean_std, _fmt_mean_std)


def _to_markdown(rows) -> str:
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def build_glue_table(runs):
    """One row per (model, method): mean and std per task, and the mean of the four task means."""
    by_pair = {}
    for r in runs:
        if r["task"] not in common.GLUE_GEN_TASKS:
            continue
        by_pair.setdefault((r["model_short"], r["method"]), {}).setdefault(r["task"], []).append(r)

    header = ["Model", "Method", "SST2", "RTE", "MRPC", "CoLA", "avg."]
    rows = [header, ["---"] * len(header)]
    for (model_short, method), task_runs in sorted(by_pair.items()):
        cells, means = [], []
        for task in ["sst2", "rte", "mrpc", "cola"]:
            group = task_runs.get(task)
            if not group:
                cells.append("--")
                continue
            accs = [r["test_results"].get("test_accuracy") for r in group]
            cells.append(_fmt_mean_std(accs, prec=4))
            mean, _ = _mean_std(accs)
            if mean is not None:
                means.append(mean)
        avg_str = f"{sum(means) / len(means):.4f}" if means else "--"
        rows.append([model_short, method, *cells, avg_str])
    return rows


def build_freegen_table(runs):
    """One row per (model, method): mean and std per free-form task. No average column,
    since the tasks use different metrics.
    """
    by_pair = {}
    for r in runs:
        if r["task"] not in common.FREE_GEN_TASKS:
            continue
        by_pair.setdefault((r["model_short"], r["method"]), {}).setdefault(r["task"], []).append(r)

    header = ["Model", "Method"] + [f"{t} ({common.FREE_GEN_METRIC[t][1]})" for t in common.FREE_GEN_TASKS]
    rows = [header, ["---"] * len(header)]
    for (model_short, method), task_runs in sorted(by_pair.items()):
        cells = []
        for task in common.FREE_GEN_TASKS:
            group = task_runs.get(task)
            if not group:
                cells.append("--")
                continue
            metric_key, _ = common.FREE_GEN_METRIC[task]
            vals = [r["test_results"].get(metric_key) for r in group]
            if all(v is None for v in vals):
                cells.append("N/A (rouge_score not installed)" if any(
                    r["test_results"].get("error") for r in group) else "--")
            else:
                cells.append(_fmt_mean_std(vals, prec=4))
        rows.append([model_short, method, *cells])
    return rows


def _summarize(values, fmt="{:.1f}"):
    """One value if all entries agree, else mean and std."""
    values = [v for v in values if v is not None]
    if not values:
        return "N/A (cluster only)"
    if len(values) == 1:
        return fmt.format(values[0])
    scale = max(abs(v) for v in values) or 1.0
    if (max(values) - min(values)) < 1e-6 * scale:
        return fmt.format(values[0])
    return f"{fmt.format(statistics.mean(values))} ± {fmt.format(statistics.stdev(values))}"


def build_characteristics_table(runs):
    """One row per (model, method) over all task and seed runs: seeds, trainable parameters,
    checkpoint size, peak memory.
    """
    by_pair = {}
    for r in runs:
        by_pair.setdefault((r["model_short"], r["method"]), []).append(r)

    header = ["Model", "Method", "Seeds", "Trainable Params", "Checkpoint Size (KB)", "Peak GPU Mem (MB)"]
    rows = [header, ["---"] * len(header)]
    for (model_short, method), method_runs in sorted(by_pair.items()):
        n_seeds = len({r["seed"] for r in method_runs})
        n_trainable_str = _summarize(
            [r["train_meta"].get("trainable_params") for r in method_runs], fmt="{:.0f}",
        )
        ckpt_kb = [r["train_meta"].get("checkpoint_bytes") / 1024
                   for r in method_runs if r["train_meta"].get("checkpoint_bytes") is not None]
        ckpt_str = _summarize(ckpt_kb) if ckpt_kb else "N/A (cluster only)"
        mem_mb = [r["train_meta"].get("peak_mem_mb") for r in method_runs]
        mem_str = _summarize(mem_mb)
        rows.append([model_short, method, str(n_seeds), n_trainable_str, ckpt_str, mem_str])
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="out/decoder")
    parser.add_argument("--out", default="results/decoder/global_summary.md")
    parser.add_argument("--seeds", nargs="+", default=list(REPORTED_SEEDS),
                         help="Seeds to aggregate (default 1 2 42); 'all' uses every seed found.")
    args = parser.parse_args()

    if not os.path.isdir(args.root):
        print(f"[summarize_results] {args.root} does not exist -- no runs to summarize yet.")
        return

    runs = select_best_lr_across_seeds(list(iter_runs(args.root, parse_seeds_arg(args.seeds))))
    if not runs:
        print(f"[summarize_results] no completed runs (train_meta.json) found under {args.root}.")
        return

    glue = build_glue_table(runs)
    freegen = build_freegen_table(runs)
    chars = build_characteristics_table(runs)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"## GLUE-as-generation results\n\n{_to_markdown(glue)}\n\n")
        f.write(f"## Free-form generation results\n\n{_to_markdown(freegen)}\n\n")
        f.write(f"## Model / method characteristics\n\n{_to_markdown(chars)}\n")
    print(f"[summarize_results] wrote {args.out}")


if __name__ == "__main__":
    main()
