"""Evaluate a model trained with beir_train.py on BEIR retrieval tasks with mteb (NDCG@10).

Scores are written by mteb to {cache_dir}/{model_short}-DPR-{lr}/results/. Without --task, the
15 BEIR tasks are evaluated in one run; with --task a single task is evaluated on its test split.

Usage:
    python encoder/beir_eval.py --model FacebookAI/roberta-large --lr 1.2e-3 --task SciFact
"""
import os
import argparse

import mteb
from sentence_transformers import SentenceTransformer

TASKS = ["NFCorpus", "SciFact", "TRECCOVID", "FiQA2018", "ArguAna", "ClimateFEVER", "DBPedia", "FEVER",
         "HotpotQA", "MSMARCO", "NQ", "QuoraRetrieval", "SCIDOCS", "Touche2020", "CQADupstackEnglishRetrieval"]


def main():
    print("Begining evaluation")
    parser = argparse.ArgumentParser("beir_eval")
    parser.add_argument("--task", default="", required=False)
    parser.add_argument("--lr", default=5e-4, required=False, type=float)
    parser.add_argument("--model", default="bert-base-uncased", required=False)
    parser.add_argument("--cache_dir", default="result_mteb/mteb", required=False,
                        help="Directory for the mteb results; use another one for the learning-rate selection runs.")
    args = parser.parse_args()
    lr = args.lr
    model_name = args.model
    model_shortname = model_name.split("/")[-1]
    savedir = os.getcwd() + "/out_ri/" + "msmarco" + "/" + model_name + "/" + str(lr) + "/" + str(1) + "/"
    run_name = f"{model_shortname}-DPR-{lr}"
    model = SentenceTransformer(f"{savedir}{run_name}/final").to("cuda")
    print("Model loaded")
    if args.task == "":
        tasks = mteb.get_tasks(tasks=TASKS)
    else:
        tasks = mteb.get_tasks(tasks=[args.task], eval_splits=["test"])
    print("Evaluation start")
    os.makedirs(f"{args.cache_dir}/{run_name}", exist_ok=True)
    cache = mteb.ResultCache(cache_path=f"{args.cache_dir}/{run_name}")
    results = mteb.evaluate(model, tasks, cache=cache, num_proc=4, overwrite_strategy="only-missing", show_progress_bar=True)
    print(results)


if __name__ == "__main__":
    main()
