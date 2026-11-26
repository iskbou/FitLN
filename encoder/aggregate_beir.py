"""Learning-rate selection and result tables for the BEIR experiments.

Reads the mteb results written by beir_eval.py:
  {dir}/{model_short}-DPR-{lr}/results/no_model_name__available/no_revision_available/{task}.json

select: for each model, mean NDCG@10 over NFCorpus, SciFact, TRECCOVID and FiQA2018 for every
        learning rate in the grid (3e-4 to 12e-4); the best one is the selected learning rate.
table:  per-task NDCG@10 of FitLN (selected learning rate), FitLN* (fixed learning rate) and the
        ratio to full fine-tuning, with the meta-task averages. The learning rates are read from
        a json file {model: {"run": model_short, "selected": lr, "fixed": lr}}; the full
        fine-tuning scores are a csv with one row per model.

Usage:
    python encoder/aggregate_beir.py select --dir result_mteb_for_selection/mteb/model_name
    python encoder/aggregate_beir.py table --dir result_mteb/mteb --dir_star result_mteb_zs/mteb \
        --lrs results/encoder/beir_lrs.json --fullft results/encoder/beir_fullft.csv \
        --out results/encoder/beir_final_results.csv
"""
import argparse
import csv
import json

import numpy as np

SELECTION_TASKS = ["SciFact", "TRECCOVID", "FiQA2018", "NFCorpus"]
SELECTION_MODELS = ["bert-large-uncased", "roberta-large", "ModernBERT-large", "deberta-v3-large"]
LR_GRID = list(np.round(np.arange(3e-4, 13e-4, 1e-4), decimals=6))

# mteb task name -> name in the tables, in the order of the per-task table
TASKS = {
    "ArguAna": "ArguAna", "CQADupstackEnglishRetrieval": "CQADup.", "ClimateFEVER": "Clim.-FEVER",
    "DBPedia": "DBPedia", "FEVER": "FEVER", "FiQA2018": "FiQA", "HotpotQA": "HotpotQA",
    "MSMARCO": "MSMARCO", "NFCorpus": "NFCorpus", "NQ": "NQ", "QuoraRetrieval": "Quora",
    "SCIDOCS": "SciDocs", "SciFact": "SciFact", "TRECCOVID": "TREC-Covid", "Touche2020": "Touche2020",
}
# column names of the full fine-tuning csv
FULLFT_COLUMN = {
    "ArguAna": "ArguAna", "CQADupstackEnglishRetrieval": "CQADupstack", "ClimateFEVER": "Climate-FEVER",
    "DBPedia": "DBPedia", "FEVER": "FEVER", "FiQA2018": "FiQA", "HotpotQA": "HotpotQA",
    "MSMARCO": "MSMARCO", "NFCorpus": "NFCorpus", "NQ": "NQ", "QuoraRetrieval": "Quora",
    "SCIDOCS": "SciDocs", "SciFact": "SciFact", "TRECCOVID": "TREC-Covid", "Touche2020": "Touche2020",
}
META_TASKS = {
    "Passage Retrieval": ["MSMARCO"],
    "BioMed Retrieval": ["TRECCOVID", "NFCorpus"],
    "Question Answering": ["NQ", "HotpotQA", "FiQA2018"],
    "Argument Retrieval": ["ArguAna", "Touche2020"],
    "Dup.-Question Retrieval": ["CQADupstackEnglishRetrieval", "QuoraRetrieval"],
    "Entity Retrieval": ["DBPedia"],
    "Citation Prediction": ["SCIDOCS"],
    "Fact Checking": ["FEVER", "ClimateFEVER", "SciFact"],
}
MODELS = ["BERT", "Roberta", "Deberta", "ModernBERT"]


def read_ndcg(results_dir, run_name, task):
    path = f"{results_dir}/{run_name}/results/no_model_name__available/no_revision_available/{task}.json"
    with open(path, encoding="utf-8") as f:
        return float(json.load(f)["scores"]["test"][0]["ndcg_at_10"])


def select(args):
    for model in SELECTION_MODELS:
        scores = {}
        for lr in LR_GRID:
            run_name = f"{model}-DPR-{lr}"
            scores[lr] = np.mean([read_ndcg(args.dir, run_name, t) for t in SELECTION_TASKS])
        ranking = sorted(scores.items(), key=lambda kv: -kv[1])
        print(model, "best lr:", ranking[0][0])
        for lr, score in ranking:
            print(f"  {lr:g}  {score:.4f}")


def table(args):
    with open(args.lrs) as f:
        lrs = json.load(f)
    with open(args.fullft, newline="") as f:
        fullft = {row["model"]: row for row in csv.DictReader(f)}
    columns = list(TASKS.values())
    out_rows = []
    per_model = {}
    for model in MODELS:
        run, lr_sel, lr_fix = lrs[model]["run"], lrs[model]["selected"], lrs[model]["fixed"]
        full = {t: float(fullft[model][FULLFT_COLUMN[t]]) for t in TASKS}
        star = {t: round(read_ndcg(args.dir_star, f"{run}-DPR-{lr_fix}", t), 3) for t in TASKS}
        sel = {t: round(read_ndcg(args.dir, f"{run}-DPR-{lr_sel}", t), 3) for t in TASKS}
        ratio_star = {t: 100 * np.round(star[t] / full[t], decimals=4) for t in TASKS}
        ratio_sel = {t: 100 * np.round(sel[t] / full[t], decimals=4) for t in TASKS}
        per_model[model] = dict(full=full, star=star, sel=sel, ratio_star=ratio_star, ratio_sel=ratio_sel)
        for name, values in ((model, full), (model + "_fitln_zs", star), (model + "_ratio_zs", ratio_star),
                             (model + "_fitln", sel), (model + "_ratio", ratio_sel)):
            out_rows.append([name] + [values[t] for t in TASKS] + [round(float(np.mean(list(values.values()))), 3)])
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model"] + columns + ["avg"])
        w.writerows(out_rows)
    print(f"Saved {len(out_rows)} rows to {args.out}")
    meta_table(per_model)


def meta_table(per_model):
    """Meta-task averages of the per-task scores rounded to two decimals, and their ratio."""
    print("\nmeta-task          " + "".join(f"{m:>26s}" for m in MODELS))
    for meta, tasks in META_TASKS.items():
        cells = []
        for model in MODELS:
            d = per_model[model]
            full = np.mean([round(d["full"][t], 2) for t in tasks])
            fit = np.mean([round(d["sel"][t], 2) for t in tasks])
            cells.append(f"{full:.2f} {fit:.2f} {100 * fit / full:6.1f}")
        print(f"{meta:24s}" + "".join(f"{c:>22s}" for c in cells))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_sel = sub.add_parser("select")
    p_sel.add_argument("--dir", required=True)
    p_sel.set_defaults(func=select)
    p_tab = sub.add_parser("table")
    p_tab.add_argument("--dir", required=True, help="mteb results of FitLN (selected learning rate).")
    p_tab.add_argument("--dir_star", required=True, help="mteb results of FitLN* (fixed learning rate).")
    p_tab.add_argument("--lrs", required=True)
    p_tab.add_argument("--fullft", required=True)
    p_tab.add_argument("--out", required=True)
    p_tab.set_defaults(func=table)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
