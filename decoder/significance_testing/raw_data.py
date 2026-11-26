"""Data loading and statistics for the decoder significance tests.

Reads test_results.json for seeds 42, 1 and 2 under out/decoder/{method}/{model}/{task}/
{lr}/{seed}/, at the learning rate selected in results/decoder/{model}.md. Paths can be
overridden with FITLN_RESULTS_DIR and FITLN_OUT_DECODER.
"""
import json
import os
import re
import warnings

import numpy as np
from scipy import optimize, stats

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.environ.get("FITLN_RESULTS_DIR", os.path.join(REPO_ROOT, "results", "decoder"))
OUT_DECODER = os.environ.get("FITLN_OUT_DECODER", os.path.join(REPO_ROOT, "out", "decoder"))

SEEDS = [42, 1, 2]

MODELS = ["llama3_2_1b", "llama3_2_3b", "qwen2_5_1_5b", "qwen2_5_3b",
          "qwen3_5_2b_base", "qwen3_5_4b_base"]
MODEL_TITLES = {
    "llama3_2_1b": "Llama-3.2-1B", "llama3_2_3b": "Llama-3.2-3B",
    "qwen2_5_1_5b": "Qwen2.5-1.5B", "qwen2_5_3b": "Qwen2.5-3B",
    "qwen3_5_2b_base": "Qwen3.5-2B-Base", "qwen3_5_4b_base": "Qwen3.5-4B-Base",
}
TASK_ORDER = ["sst2", "rte", "mrpc", "cola", "alpaca_mini", "gsm8k", "samsum"]
METRIC_KEY = {
    "sst2": "test_accuracy", "rte": "test_accuracy", "mrpc": "test_accuracy", "cola": "test_accuracy",
    "gsm8k": "exact_match", "alpaca_mini": "rouge_l", "samsum": "rouge_l",
}
METHODS = ["fitln", "fullft", "lora_r1_attnonly", "lora_r8_attnonly"]
OTHER_METHODS = [("fullft", "Full-FT"), ("lora_r1_attnonly", "LoRA r1"), ("lora_r8_attnonly", "LoRA r8")]

_ROW_RE = re.compile(r"\|\s*(\S+)\s*\|\s*(\S+)\s*\|\s*(\S+)\s*\|\s*(\d+)\s*\|")


def parse_best_lr(model):
    """Parse results/decoder/{model}.md into {(task, method): (best_lr, n_seeds)}."""
    path = os.path.join(RESULTS_DIR, f"{model}.md")
    with open(path) as f:
        text = f.read()
    result = {}
    current_task = None
    for line in text.splitlines():
        m = re.match(r"## (\S+) -- ", line)
        if m:
            current_task = m.group(1)
            continue
        if line.startswith("|") and current_task is not None:
            m = _ROW_RE.match(line)
            if m:
                row_model, method, best_lr, n_seeds = m.groups()
                if row_model == model and method in METHODS:
                    result[(current_task, method)] = (best_lr, int(n_seeds))
    return result


def read_metric(method, model, task, lr, seed):
    path = os.path.join(OUT_DECODER, method, model, task, lr, str(seed), "test_results.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        d = json.load(f)
    return d.get(METRIC_KEY[task])


def build_raw_dataset():
    """Returns {(model, task, method): {seed: value}} for every cell found on disk."""
    raw = {}
    for model in MODELS:
        best_lr = parse_best_lr(model)
        for task in TASK_ORDER:
            for method in METHODS:
                key = (task, method)
                if key not in best_lr:
                    continue
                lr, _ = best_lr[key]
                vals = {}
                for seed in SEEDS:
                    v = read_metric(method, model, task, lr, seed)
                    if v is not None:
                        vals[seed] = v
                raw[(model, task, method)] = vals
    return raw


def mean_std_n(seed_value_dict):
    """Sample mean/std (ddof=1)/n over a {seed: value} dict. std=0.0 for n=1, (None,None,0) for n=0."""
    vals = list(seed_value_dict.values())
    n = len(vals)
    if n == 0:
        return None, None, 0
    m = float(np.mean(vals))
    s = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    return m, s, n


def welch(x, y):
    """Welch's t-test on two samples. Returns (t, df, p), all nan if either sample is empty or
    the combined variance is zero. df is the Welch-Satterthwaite value.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan, np.nan
    s1 = float(np.std(x, ddof=1)) if n1 > 1 else 0.0
    s2 = float(np.std(y, ddof=1)) if n2 > 1 else 0.0
    se2 = (s1 ** 2) / n1 + (s2 ** 2) / n2
    if se2 == 0:
        return np.nan, np.nan, np.nan
    num = se2 ** 2
    den = 0.0
    if n1 > 1:
        den += ((s1 ** 2 / n1) ** 2) / (n1 - 1)
    if n2 > 1:
        den += ((s2 ** 2 / n2) ** 2) / (n2 - 1)
    df = num / den if den > 0 else np.nan
    t, p = stats.ttest_ind(x, y, equal_var=False)
    return float(t), df, float(p)


def paired_ttest(fitln_vals, other_vals):
    """scipy.stats.ttest_rel on matched seeds. Returns dict(n, diff, t, df, p) or None if <2 common seeds."""
    common_seeds = sorted(set(fitln_vals) & set(other_vals))
    if len(common_seeds) < 2:
        return None
    x = np.array([fitln_vals[s] for s in common_seeds])
    y = np.array([other_vals[s] for s in common_seeds])
    diffs = x - y
    n = len(diffs)
    if np.std(diffs, ddof=1) == 0:
        t, p = np.nan, np.nan
    else:
        t, p = stats.ttest_rel(x, y)
    return dict(n=n, diff=float(np.mean(diffs)), t=t, df=n - 1, p=p)


def paired_correlation(fitln_vals, other_vals):
    """Pearson r between FitLN's and the other method's scores, matched by seed. None if degenerate."""
    common_seeds = sorted(set(fitln_vals) & set(other_vals))
    if len(common_seeds) < 3:
        return None
    x = np.array([fitln_vals[s] for s in common_seeds])
    y = np.array([other_vals[s] for s in common_seeds])
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    r, _ = stats.pearsonr(x, y)
    return r


def min_detectable_d(n, paired, power=0.8, alpha=0.05):
    """Smallest effect size (Cohen's d, or d_z if paired) detected with the given power by a
    two-sided t-test with n observations per group (n pairs if paired)."""
    df = n - 1 if paired else 2 * n - 2
    scale = np.sqrt(n) if paired else np.sqrt(n / 2)
    t_crit = stats.t.ppf(1 - alpha / 2, df)

    def gap(d):
        nc = d * scale
        return stats.nct.sf(t_crit, df, nc) + stats.nct.cdf(-t_crit, df, nc) - power

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(optimize.brentq(gap, 0.1, 20))


def bh_correct(rows, p_key="p", q_key="q_bh"):
    """In-place Benjamini-Hochberg FDR correction. Adds q_key to each row dict. Rows with nan p get nan q."""
    pvals = np.array([r[p_key] for r in rows])
    valid = ~np.isnan(pvals)
    m = int(valid.sum())
    q_full = np.full_like(pvals, np.nan)
    if m > 0:
        order = np.argsort(pvals[valid])
        ranked_p = pvals[valid][order]
        q = ranked_p * m / (np.arange(m) + 1)
        q = np.minimum.accumulate(q[::-1])[::-1]
        idx_valid = np.where(valid)[0]
        q_full[idx_valid[order]] = q
    for r, qv in zip(rows, q_full):
        r[q_key] = qv
    return rows


def stars(q, as_code=True):
    """Significance stars for a BH-adjusted q-value, wrapped in backticks so Markdown does
    not read them as emphasis.
    """
    if np.isnan(q):
        return ""
    if q < 0.001:
        sym = "***"
    elif q < 0.01:
        sym = "**"
    elif q < 0.05:
        sym = "*"
    else:
        return ""
    return f"`{sym}`" if as_code else sym
