"""Aggregate the GLUE results of FitLN, BitFit and LoRA into one table.

Layouts read:
  FitLN : {root}/{task}/{model}/attn_only/{lr}/{epochs}/results.json (MNLI: results_m.json,
          results_mm.json)
  PEFT  : {peft_root}/{method}/{model}/{task}/{lr}/{seed}/val_results.json, test_results.json
          where method is bitfit, lora_r1, lora_r8 (or lora), ...

For every (method, model, task) the learning rate with the highest mean validation score
is selected and the mean test score at that learning rate is reported. For FitLN the
{epochs} directories are averaged like seeds.

Metrics: Matthews correlation (cola), Pearson (stsb), accuracy for the other tasks
(including mrpc and qqp). MNLI: the test score of the baselines is the mean of the matched
and mismatched accuracy, the test score of FitLN is the mismatched accuracy only.

Usage:
    python encoder/aggregate_glue.py --root out --peft_root out --out results/encoder/glue_detail.csv --table
    python encoder/aggregate_glue.py --csv_in results/encoder/glue_detail.csv --table

The second form prints the table from an existing CSV, without the raw results.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

TASK_METRIC = {
    "cola": "matthews_correlation",
    "sst2": "accuracy",
    "mrpc": "accuracy",
    "stsb": "pearson",
    "qqp": "accuracy",
    "mnli": "accuracy",
    "qnli": "accuracy",
    "rte": "accuracy",
}
TABLE_TASKS = ["cola", "qqp", "mnli", "sst2", "stsb", "qnli", "rte", "mrpc"]

MODEL_SHORT = {
    "FacebookAI/roberta-large": "RoBERTa-L",
    "google-bert/bert-large-uncased": "BERT-L",
    "bert-large-uncased": "BERT-L",
    "microsoft/deberta-v3-large": "DeBERTa-L",
    "answerdotai/ModernBERT-large": "ModernBERT-L",
}


def load_json(path: Path) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def iter_models(parent: Path):
    """Yield (model name, directory) for the known models below parent."""
    for child in sorted(parent.iterdir()):
        if not child.is_dir():
            continue
        if child.name in MODEL_SHORT:
            yield child.name, child
        else:
            for sub in sorted(child.iterdir()):
                if sub.is_dir() and f"{child.name}/{sub.name}" in MODEL_SHORT:
                    yield f"{child.name}/{sub.name}", sub


def iter_numeric_dirs(parent: Path):
    for child in sorted(parent.iterdir()):
        if child.is_dir():
            try:
                yield float(child.name), child
            except ValueError:
                continue


def is_peft_method(name: str) -> bool:
    return name == "bitfit" or name.startswith("lora")


def collect_peft(root: Path) -> list[dict]:
    rows = []
    for method_dir in sorted(root.iterdir()):
        if not method_dir.is_dir() or not is_peft_method(method_dir.name):
            continue
        for model, model_dir in iter_models(method_dir):
            for task_dir in sorted(model_dir.iterdir()):
                task = task_dir.name
                if not task_dir.is_dir() or task not in TASK_METRIC:
                    continue
                metric = TASK_METRIC[task]
                for lr, lr_dir in iter_numeric_dirs(task_dir):
                    for seed_dir in sorted(lr_dir.iterdir()):
                        if not seed_dir.is_dir() or not seed_dir.name.isdigit():
                            continue
                        val_data = load_json(seed_dir / "val_results.json")
                        test_data = load_json(seed_dir / "test_results.json")
                        val_metric = val_data.get("eval_" + metric) if val_data else None
                        if task == "mnli":
                            test_metric = None
                            if test_data and "eval_matched_accuracy" in test_data and "eval_mismatched_accuracy" in test_data:
                                test_metric = (test_data["eval_matched_accuracy"] + test_data["eval_mismatched_accuracy"]) / 2
                        else:
                            test_metric = test_data.get("eval_" + metric) if test_data else None
                        if val_metric is None and test_metric is None:
                            continue
                        rows.append({"method": method_dir.name, "model": model, "model_short": MODEL_SHORT[model],
                                     "task": task, "lr": lr, "seed": int(seed_dir.name),
                                     "val_metric": val_metric, "test_metric": test_metric, "metric_name": metric})
    return rows


def collect_fitln(root: Path) -> list[dict]:
    rows = []
    for task_dir in sorted(root.iterdir()):
        task = task_dir.name
        if not task_dir.is_dir() or task not in TASK_METRIC:
            continue
        metric = TASK_METRIC[task]
        for model, model_dir in iter_models(task_dir):
            mode_dir = model_dir / "attn_only"
            if not mode_dir.is_dir():
                continue
            for lr, lr_dir in iter_numeric_dirs(mode_dir):
                for epoch_dir in sorted(lr_dir.iterdir()):
                    if not epoch_dir.is_dir():
                        continue
                    data = load_json(epoch_dir / "results.json")
                    test_key = "test_" + metric
                    if data is None:
                        data = load_json(epoch_dir / "results_m.json")
                        if data is None:
                            continue
                        mismatched = load_json(epoch_dir / "results_mm.json")
                        test_metric = mismatched.get(test_key) if mismatched else None
                    else:
                        test_metric = data.get(test_key)
                    val_metric = data.get("val_" + metric)
                    if val_metric is None and test_metric is None:
                        continue
                    rows.append({"method": "fitln_attn_only", "model": model, "model_short": MODEL_SHORT[model],
                                 "task": task, "lr": lr, "seed": int(epoch_dir.name),
                                 "val_metric": val_metric, "test_metric": test_metric, "metric_name": metric})
    return rows


def best_per_group(rows: list[dict]) -> list[dict]:
    """For each (method, model, task), the learning rate with the highest mean validation
    score and the mean test score at that learning rate."""
    groups: dict[tuple, list] = defaultdict(list)
    for r in rows:
        groups[(r["method"], r["model"], r["task"], r["lr"])].append(r)

    lr_stats: dict[tuple, dict] = {}
    for (method, model, task, lr), rlist in groups.items():
        vals = [r["val_metric"] for r in rlist if r["val_metric"] is not None]
        tests = [r["test_metric"] for r in rlist if r["test_metric"] is not None]
        if not vals:
            continue
        lr_stats[(method, model, task, lr)] = {
            "avg_val": sum(vals) / len(vals),
            "avg_test": sum(tests) / len(tests) if tests else None,
            "n_val": len(vals),
            "n_test": len(tests),
            "metric_name": rlist[0]["metric_name"],
            "model_short": rlist[0]["model_short"],
        }

    best: dict[tuple, dict] = {}
    for (method, model, task, lr), stats in lr_stats.items():
        key = (method, model, task)
        if key not in best or stats["avg_val"] > best[key]["avg_val"]:
            best[key] = {**stats, "best_lr": lr, "method": method, "model": model, "task": task}
    return sorted(best.values(), key=lambda r: (r["method"], r["model"], r["task"]))


def write_csv(best_rows: list[dict], path: str) -> None:
    fields = ["method", "model", "model_short", "task", "best_lr",
              "avg_val", "avg_test", "n_val", "n_test", "metric_name"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(best_rows)
    print(f"Saved {len(best_rows)} rows to {path}", file=sys.stderr)


def print_table(best_rows: list[dict]) -> None:
    """One line per (method, model): test scores in percent in the order of the paper table,
    then the mean over the tasks."""
    cell = {}
    for r in best_rows:
        cell.setdefault((r["method"], r["model_short"], r["task"]), r["avg_test"])
    for method, model_short in sorted({(r["method"], r["model_short"]) for r in best_rows}):
        scores = [cell.get((method, model_short, t)) for t in TABLE_TASKS]
        done = [s for s in scores if s is not None]
        avg = f"{100 * sum(done) / len(done):.1f}" if len(done) == len(TABLE_TASKS) else "-"
        cells = " & ".join("-" if s is None else f"{100 * s:.1f}" for s in scores)
        print(f"{method:16s} {model_short:13s} {cells} & {avg}")


def read_csv(path: str) -> list[dict]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["model_short"] = MODEL_SHORT.get(r["model"], r["model_short"].replace("ModernBERT-large", "ModernBERT-L"))
        r["avg_test"] = float(r["avg_test"]) if r["avg_test"] not in ("", None) else None
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="out", help="Directory with the FitLN results (out/{task}/...).")
    parser.add_argument("--peft_root", default=None, help="Directory with the BitFit and LoRA results (default: --root).")
    parser.add_argument("--csv_in", default=None, help="Read an aggregated CSV instead of the raw results.")
    parser.add_argument("--out", default=None, help="Write the CSV to this path.")
    parser.add_argument("--table", action="store_true", help="Print the test scores in the order of the paper table.")
    args = parser.parse_args()

    if args.csv_in:
        best_rows = read_csv(args.csv_in)
    else:
        root = Path(args.root)
        peft_root = Path(args.peft_root) if args.peft_root else root
        rows = collect_peft(peft_root) + collect_fitln(root)
        if not rows:
            print("No results found.", file=sys.stderr)
            sys.exit(0)
        best_rows = best_per_group(rows)
    if args.table:
        print_table(best_rows)
    if args.out:
        write_csv(best_rows, args.out)


if __name__ == "__main__":
    main()
