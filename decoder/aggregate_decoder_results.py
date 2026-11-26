"""Collect completed runs under out/decoder/{method}/{model}/{task}/{lr}/{seed}/ and write
one markdown table per task, one file per model.

The learning rate is chosen once per (method, model, task) as the one with the lowest
mean validation eval_loss across seeds. Reported values are the mean and sample std
over the seed runs at that learning rate. Only seeds 1, 2 and 42 are used by default;
--seeds all uses every seed found. Peak memory and checkpoint size print as
'N/A (cluster only)' when unmeasured, never as 0.

Usage:
    python decoder/aggregate_decoder_results.py --root out/decoder --out_dir results/decoder
"""
import argparse
import math
import os
import json
from collections import defaultdict

import common


def _fmt_mem(mb):
    return "N/A (cluster only)" if mb is None else f"{mb:.1f}"


def _fmt_bytes_kb(b):
    return "N/A (cluster only)" if b is None else f"{b / 1024:.1f}"


def _read_json(path):
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _mean_std(values):
    """Sample mean and std over non-None values; (None, None) if empty, std 0 for one value."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    n = len(vals)
    mean = sum(vals) / n
    if n == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    return mean, math.sqrt(var)


def _fmt_mean_std(values, prec=4):
    mean, std = _mean_std(values)
    if mean is None:
        return "--"
    return f"{mean:.{prec}f} ± {std:.{prec}f}"


REPORTED_SEEDS = ("1", "2", "42")


def parse_seeds_arg(values):
    """None or ['all'] means no filter; otherwise a set of seed strings."""
    if not values or [v.lower() for v in values] == ["all"]:
        return None
    return {str(v) for v in values}


def iter_runs(root: str, seeds=None):
    """Yield every completed run (one with train_meta.json). seeds=None keeps all seeds;
    otherwise runs with other seeds are skipped and reported.
    """
    skipped = set()
    yield from _iter_runs(root, seeds, skipped)
    if skipped:
        print(f"[aggregate] skipped seeds not in the selected set: {sorted(skipped)}")


def _iter_runs(root: str, seeds, skipped):
    for dirpath, _, filenames in os.walk(root):
        if "train_meta.json" not in filenames:
            continue
        rel = os.path.relpath(dirpath, root)
        parts = rel.split(os.sep)
        if len(parts) != 5:
            print(f"[aggregate_decoder_results] skipping unexpected path shape: {dirpath}")
            continue
        method, model_short, task, lr, seed = parts
        if seeds is not None and seed not in seeds:
            skipped.add(seed)
            continue
        train_meta = _read_json(os.path.join(dirpath, "train_meta.json"))
        val_results = _read_json(os.path.join(dirpath, "val_results.json"))
        test_results = _read_json(os.path.join(dirpath, "test_results.json"))
        yield dict(
            path=dirpath, method=method, model_short=model_short, task=task, lr=lr, seed=seed,
            train_meta=train_meta, val_results=val_results, test_results=test_results,
        )


def select_best_lr_across_seeds(runs, group_keys=("method", "model_short", "task")):
    """Per (method, model, task), keep the runs at the learning rate with the lowest mean
    validation eval_loss across seeds.
    """
    # group -> lr -> [runs]
    by_group_lr = defaultdict(lambda: defaultdict(list))
    for r in runs:
        gk = tuple(r[k] for k in group_keys)
        by_group_lr[gk][r["lr"]].append(r)

    surviving = []
    for gk, by_lr in by_group_lr.items():
        best_lr, best_mean_loss = None, math.inf
        for lr, lr_runs in by_lr.items():
            losses = [r["val_results"].get("eval_loss", math.inf) for r in lr_runs]
            mean_loss = sum(losses) / len(losses) if losses else math.inf
            if mean_loss < best_mean_loss:
                best_mean_loss, best_lr = mean_loss, lr
        surviving.extend(by_lr[best_lr])
    return surviving


def build_task_table(runs, model_short: str, task: str):
    """One row per method for task: best LR, seeds, trainable parameters, task metric,
    checkpoint size and peak memory (metric, perplexity and memory as mean and std over seeds).
    Perplexity is exp(eval_loss) from val_results.json and only shown for free-form tasks.
    """
    is_glue = task in common.GLUE_GEN_TASKS
    metric_key, metric_label = ("test_accuracy", "Test Accuracy") if is_glue else common.FREE_GEN_METRIC[task]

    task_runs = [r for r in runs if r["task"] == task and r["model_short"] == model_short]
    header = ["Model", "Method", "Best LR", "Seeds", "Trainable Params", "%"]
    if not is_glue:
        header.append("Eval Perplexity")
    header += [metric_label, "Checkpoint Size (KB)", "Peak GPU Mem (MB)"]
    rows = [header, ["---"] * len(header)]

    by_method = defaultdict(list)
    for r in task_runs:
        by_method[r["method"]].append(r)

    for method in sorted(by_method):
        method_runs = by_method[method]
        rep = method_runs[0]  # seed-invariant fields come from any run in the group
        n_trainable = rep["train_meta"].get("trainable_params", "N/A")
        pct = rep["train_meta"].get("trainable_pct")
        pct_str = f"{pct:.5f}" if pct is not None else "N/A"
        n_seeds = len({r["seed"] for r in method_runs})
        row = [model_short, method, str(rep.get("lr", "N/A")), str(n_seeds), str(n_trainable), pct_str]

        if not is_glue:
            perps = []
            for r in method_runs:
                eval_loss = r["val_results"].get("eval_loss")
                perps.append(math.exp(eval_loss) if eval_loss is not None else None)
            row.append(_fmt_mean_std(perps, prec=3))

        metric_vals = [r["test_results"].get(metric_key) for r in method_runs]
        if all(v is None for v in metric_vals):
            metric_str = "N/A (rouge_score not installed)" if any(
                r["test_results"].get("error") for r in method_runs) else "--"
        else:
            metric_str = _fmt_mean_std(metric_vals, prec=4)
        row.append(metric_str)

        # Checkpoint size is deterministic given the param count -> take a representative run.
        row.append(_fmt_bytes_kb(rep["train_meta"].get("checkpoint_bytes")))

        # Peak memory varies slightly run-to-run -> report mean +/- std (None if unmeasured).
        mems = [r["train_meta"].get("peak_mem_mb") for r in method_runs]
        if all(m is None for m in mems):
            row.append("N/A (cluster only)")
        else:
            mean_m, std_m = _mean_std(mems)
            row.append(f"{mean_m:.1f} ± {std_m:.1f}")
        rows.append(row)
    return rows


def _to_markdown(rows) -> str:
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="out/decoder")
    parser.add_argument("--out_dir", default="results/decoder")
    parser.add_argument("--seeds", nargs="+", default=list(REPORTED_SEEDS),
                         help="Seeds to aggregate (default 1 2 42); 'all' uses every seed found.")
    args = parser.parse_args()

    if not os.path.isdir(args.root):
        print(f"[aggregate_decoder_results] {args.root} does not exist -- no runs to aggregate.")
        return

    runs = select_best_lr_across_seeds(list(iter_runs(args.root, parse_seeds_arg(args.seeds))))
    if not runs:
        print(f"[aggregate_decoder_results] no completed runs (train_meta.json) found under {args.root}.")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    model_shorts = sorted({r["model_short"] for r in runs})
    for model_short in model_shorts:
        out_path = os.path.join(args.out_dir, f"{model_short}.md")
        with open(out_path, "w", encoding="utf-8") as f:
            for task in common.ALL_TASKS:
                if not any(r["task"] == task and r["model_short"] == model_short for r in runs):
                    continue
                t = build_task_table(runs, model_short, task)
                f.write(f"## {task} -- {model_short}\n\n{_to_markdown(t)}\n\n")
        print(f"[aggregate_decoder_results] wrote {out_path}")


if __name__ == "__main__":
    main()
