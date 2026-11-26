"""Write results/decoder/significance_tests.md (Welch) and significance_tests_paired.md
(paired t-test) from the per-seed results in out/decoder/ (see raw_data.py).

FitLN is compared with full fine-tuning, LoRA r1 and LoRA r8: 3 comparisons x 6 models x
7 tasks = 126 tests per file, with Benjamini-Hochberg correction over the 126.

Usage:
    python decoder/significance_testing/generate_significance_tests.py
"""
import os

import numpy as np

import raw_data as rd

GLUE_TASKS = ["sst2", "rte", "mrpc", "cola"]

INTRO = """\
# Significance tests: FitLN vs. Full-FT, LoRA r1 and LoRA r8

{method}, on the test scores of seeds 42, 1 and 2 at the learning rate selected for
each method (n=3 per cell). {n} tests (3 comparisons x 6 models x 7 tasks), with
Benjamini-Hochberg correction over all of them (q).

diff = {diff}. Stars refer to q: `*` < 0.05, `**` < 0.01, `***` < 0.001.
{note}
"""

UNPAIRED_METHOD = "Welch's t-test (unpaired)"
UNPAIRED_DIFF = "FitLN mean - other mean"
UNPAIRED_NOTE = """\
With n=3 the degrees of freedom are 2 to 4, so p-values are a rough signal. A cell without
stars is not evidence of equal performance. Paired version: significance_tests_paired.md."""

PAIRED_METHOD = "Paired t-test (df=2)"
PAIRED_DIFF = "mean over seeds of (FitLN - other)"
PAIRED_NOTE = "Unpaired version: significance_tests.md."

POWER = """\
## Power with n=3

Minimum effect size detectable with 80% power (two-sided, alpha=0.05), from the noncentral
t distribution.

{table}

With 3 seeds per method the minimum is d = {d3:.2f}. With the median per-cell seed std of
{std_glue:.4f} (GLUE tasks) and {std_free:.4f} (free-form tasks), this is a gap of about
{gap_glue:.1f} and {gap_free:.1f} points. Smaller gaps cannot be confirmed. For a claim of
equivalence use a TOST with a stated margin.
"""

PAIRED_CORR = """\
Mean per-seed correlation between FitLN and the other method: r = {mean_r:.3f} (positive in
{pos:.0f}% of the {n_corr} comparisons). With df=2 pairing gains little unless r is high:
{n_sig}/{n} paired tests reach q < 0.05, against {n_ref}/{n} unpaired.
"""


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def fmt(x, spec):
    return "n/a" if np.isnan(x) else format(x, spec)


def build_rows(raw, paired):
    rows, correlations = [], []
    for model in rd.MODELS:
        for task in rd.TASK_ORDER:
            fitln = raw.get((model, task, "fitln"), {})
            fitln_mean, _, n1 = rd.mean_std_n(fitln)
            if n1 == 0:
                continue
            for key, label in rd.OTHER_METHODS:
                other = raw.get((model, task, key), {})
                other_mean, _, n2 = rd.mean_std_n(other)
                if n2 == 0:
                    continue
                row = dict(model=model, task=task, comparison=label)
                if paired:
                    res = rd.paired_ttest(fitln, other)
                    if res is None:
                        continue
                    row.update(res)
                    r = rd.paired_correlation(fitln, other)
                    if r is not None:
                        correlations.append(r)
                else:
                    t, df, p = rd.welch(list(fitln.values()), list(other.values()))
                    row.update(fitln_mean=fitln_mean, other_mean=other_mean,
                               diff=fitln_mean - other_mean, t=t, df=df, p=p)
                rows.append(row)
    return rd.bh_correct(rows), correlations


def model_tables(rows, paired):
    header = ["Task", "Comparison"] + ([] if paired else ["FitLN", "Other"]) + \
             ["Diff", "t", "df", "p", "q (BH)", "Sig"]
    parts = []
    for model in rd.MODELS:
        model_rows = sorted((r for r in rows if r["model"] == model),
                            key=lambda r: (rd.TASK_ORDER.index(r["task"]), r["comparison"]))
        body = []
        for r in model_rows:
            means = [] if paired else [f"{r['fitln_mean']:.4f}", f"{r['other_mean']:.4f}"]
            body.append([r["task"], "vs. " + r["comparison"]] + means +
                        [f"{r['diff']:+.4f}", fmt(r["t"], ".2f"), fmt(r["df"], ".1f" if not paired else "d"),
                         fmt(r["p"], ".4f"), fmt(r["q_bh"], ".4f"), rd.stars(r["q_bh"])])
        parts.append(f"## {rd.MODEL_TITLES[model]}\n\n{md_table(header, body)}\n")
    return "\n".join(parts)


def count_line(rows):
    n = len(rows)
    n_q = sum(r["q_bh"] < 0.05 for r in rows)
    n_p = sum(r["p"] < 0.05 for r in rows)
    return f"**{n_q}/{n}** tests have q < 0.05 ({n_p}/{n} have uncorrected p < 0.05).\n", n_q


def median_seed_std(raw, tasks):
    stds = [rd.mean_std_n(v)[1] for (m, t, k), v in raw.items() if t in tasks and len(v) > 1]
    return float(np.median(stds))


def power_section(raw):
    ns = [3, 6, 10, 20]
    table = md_table(["n", "Unpaired d", "Paired d_z"],
                     [[n, f"{rd.min_detectable_d(n, False):.2f}", f"{rd.min_detectable_d(n, True):.2f}"]
                      for n in ns])
    free_tasks = [t for t in rd.TASK_ORDER if t not in GLUE_TASKS]
    std_glue, std_free = median_seed_std(raw, GLUE_TASKS), median_seed_std(raw, free_tasks)
    d3 = rd.min_detectable_d(3, False)
    return POWER.format(table=table, d3=d3, std_glue=std_glue, std_free=std_free,
                        gap_glue=100 * d3 * std_glue, gap_free=100 * d3 * std_free)


def write(path, text):
    with open(path, "w") as f:
        f.write(text)
    print("wrote", path)


def main():
    raw = rd.build_raw_dataset()
    print(f"{len(raw)} (model, task, method) cells.")

    rows, _ = build_rows(raw, paired=False)
    counts, n_unpaired = count_line(rows)
    text = INTRO.format(method=UNPAIRED_METHOD, n=len(rows), diff=UNPAIRED_DIFF, note=UNPAIRED_NOTE)
    text += "\n" + model_tables(rows, paired=False) + "\n" + counts + "\n" + power_section(raw)
    write(os.path.join(rd.RESULTS_DIR, "significance_tests.md"), text)

    rows, correlations = build_rows(raw, paired=True)
    counts, n_paired = count_line(rows)
    text = INTRO.format(method=PAIRED_METHOD, n=len(rows), diff=PAIRED_DIFF, note=PAIRED_NOTE)
    text = text.replace("Significance tests:", "Significance tests (paired):", 1)
    text += "\n" + PAIRED_CORR.format(
        mean_r=np.mean(correlations), pos=100 * np.mean([r > 0 for r in correlations]),
        n_corr=len(correlations), n_sig=n_paired, n=len(rows), n_ref=n_unpaired)
    text += "\n" + model_tables(rows, paired=True) + "\n" + counts
    write(os.path.join(rd.RESULTS_DIR, "significance_tests_paired.md"), text)


if __name__ == "__main__":
    main()
