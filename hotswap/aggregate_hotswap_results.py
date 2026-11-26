"""Collect the run_hotswap.py outputs into a latency table and pairwise significance tests.

Reads {out_root}/{arch}/{model}/{method}/hotswap_result.json and writes
{results_dir}/swap_latency_table.md and {results_dir}/swap_latency_significance.json.

Significance: Welch's t-test on the raw cold_swap timings of each pair of methods (cold swap
is the only regime that all methods have), per (arch, model), with Benjamini-Hochberg
correction over the pairs of that (arch, model). The p-value uses a normal approximation
(no scipy), which is accurate for the sample sizes used here (300 switches per cell).

Usage:
    python hotswap/aggregate_hotswap_results.py --out_root results/hotswap/runs --results_dir results/hotswap
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import math
import os

METHOD_ORDER = {"fitln": 0, "bitfit": 1, "lora1": 2, "lora8": 3, "fullft": 4}

HEADER = """\
# Swap latency

Resident swap: the new task is already in device memory. Copy overwrites the model's own
parameters, pointer only changes which tensors are used, and costs no data movement. Cold swap:
the new task is copied in from host memory. Merge (lora1 and lora8 only): fusing the adapter
into the base weights. The comparable regimes are resident copy and cold swap; resident
pointer is near zero for every method. Times are means in ms; n is the number of switches.
"""

MERGE_NOTE = """\
Merge (lora1 and lora8 only) is not a hot-swap number: merging several tasks into the same
base weights needs one backbone per task.
"""

INTERPRETATION = """\
## Summary

- Provenance: device {devices}, {n_switches} switches per cell, {n_real} of {n_cells} cells
  with real pretrained weights and the others with synthetic tensors of the registry sizes.
- Resident pointer swap is {pointer} ms for every method and separates nothing.
- Cold swap, decoders: fitln {dec_fitln} ms, lora1 {dec_lora1} ms, lora8 {dec_lora8} ms,
  fullft {dec_fullft} ms. Encoders: fitln {enc_fitln} ms, fullft {enc_fullft} ms.
- Resident copy, decoders: fitln {copy_fitln} ms, fullft {copy_fullft} ms.
- LoRA copy into fixed buffers (decoders, lora1): {lora_copy} ms, against {lora_pointer} ms for
  the adapter switch.
- LoRA merge, rank 1: {merge_enc} ms (encoders), {merge_dec} ms (decoders).
"""


def _load_all(out_root: str) -> list[dict]:
    rows = []
    for path in glob.glob(os.path.join(out_root, "*", "*", "*", "hotswap_result.json")):
        with open(path) as f:
            rows.append(json.load(f))
    return rows


def _erf(x: float) -> float:
    # Abramowitz and Stegun 7.1.26
    sign = 1 if x >= 0 else -1
    x = abs(x)
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
    return sign * y


def _normal_sf(z: float) -> float:
    """Two-sided tail probability P(|Z| >= z) of the standard normal."""
    return 1.0 - _erf(z / math.sqrt(2))


def welch_t_test(a: list[float], b: list[float]) -> dict:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return dict(t=None, p=None, note="insufficient samples")
    mean_a, mean_b = sum(a) / na, sum(b) / nb
    var_a = sum((x - mean_a) ** 2 for x in a) / (na - 1)
    var_b = sum((x - mean_b) ** 2 for x in b) / (nb - 1)
    se = math.sqrt(var_a / na + var_b / nb)
    if se == 0:
        return dict(t=None, p=None, note="zero pooled variance")
    t = (mean_a - mean_b) / se
    df = (var_a / na + var_b / nb) ** 2 / (
        (var_a / na) ** 2 / (na - 1) + (var_b / nb) ** 2 / (nb - 1)
    ) if (var_a > 0 or var_b > 0) else na + nb - 2
    p = _normal_sf(abs(t))
    return dict(t=t, df=df, p=p, mean_a=mean_a, mean_b=mean_b, n_a=na, n_b=nb,
                note="normal-approximation p-value, not exact Student-t (no scipy dependency)")


def benjamini_hochberg(pvals: list[float]) -> list[float]:
    """BH-adjusted q-values in the order of pvals."""
    n = len(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])
    q = [0.0] * n
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        k = n - rank + 1
        val = min(prev, pvals[i] * n / k)
        q[i] = val
        prev = val
    return q


def pairwise_significance(rows: list[dict]) -> list[dict]:
    by_cell = {}
    for r in rows:
        key = (r["arch"], r["model"])
        chosen = r["latency"].get("cold_swap")
        if not chosen or not chosen.get("raw_ms"):
            continue
        by_cell.setdefault(key, {})[r["method"]] = chosen["raw_ms"]

    out = []
    for (arch, model), by_method in by_cell.items():
        methods = sorted(by_method)
        pair_results = []
        for m1, m2 in itertools.combinations(methods, 2):
            test = welch_t_test(by_method[m1], by_method[m2])
            pair_results.append(dict(arch=arch, model=model, method_a=m1, method_b=m2, **test))
        pvals = [pr["p"] for pr in pair_results if pr["p"] is not None]
        qvals = benjamini_hochberg(pvals) if pvals else []
        qi = iter(qvals)
        for pr in pair_results:
            pr["q_bh"] = next(qi) if pr["p"] is not None else None
        out.extend(pair_results)
    return out


def _regime_range(rows, arch, method, regime):
    """(min, max) of the mean ms over models for one (arch, method, regime), or None."""
    vals = []
    for r in rows:
        if r["arch"] != arch or r["method"] != method:
            continue
        lat = r.get("latency") or {}
        if regime == "pointer":
            blk = lat.get("resident_swap_pointer")
            if not (isinstance(blk, dict) and blk.get("mean_ms") is not None):
                blk = lat.get("resident_swap") if lat.get("swap_kind") == "pointer" else None
        elif regime == "copy":
            blk = lat.get("resident_swap") if lat.get("swap_kind") == "copy" else None
            if blk is None:
                blk = lat.get("resident_swap_copy")
        else:
            blk = lat.get({"cold": "cold_swap", "merge": "merge_swap"}[regime])
        if isinstance(blk, dict) and blk.get("mean_ms") is not None:
            vals.append(blk["mean_ms"])
    return (min(vals), max(vals)) if vals else None


def _span(rows, arch, method, regime, prec=3):
    r = _regime_range(rows, arch, method, regime)
    if r is None:
        return "not measured"
    lo, hi = r
    fmt = "%%.%df" % prec
    return (fmt % lo) if lo == hi else ((fmt + "-" + fmt) % (lo, hi))


def _global_pointer_span(rows):
    vals = []
    for arch in ("encoder", "decoder"):
        for method in METHOD_ORDER:
            r = _regime_range(rows, arch, method, "pointer")
            if r:
                vals.extend(r)
    return "not measured" if not vals else "%.4f-%.4f" % (min(vals), max(vals))


def latency_table(rows: list[dict]) -> str:
    lines = []
    for arch in ("encoder", "decoder"):
        arch_rows = [r for r in rows if r["arch"] == arch]
        if not arch_rows:
            continue
        lines += ["", f"## {arch.capitalize()}"]
        for model in sorted({r["model"] for r in arch_rows}):
            lines += ["", f"### {model}", "",
                      "| Method | device | n | resident copy (ms) | cold swap (ms) | merge (ms) | resident pointer (ms) | swap kind |",
                      "|---|---|---|---|---|---|---|---|"]
            for r in sorted((x for x in arch_rows if x["model"] == model),
                            key=lambda x: METHOD_ORDER.get(x["method"], 99)):
                lat = r["latency"]
                cold = lat.get("cold_swap") or {}
                resident = lat.get("resident_swap") or {}
                resident_ptr = lat.get("resident_swap_pointer")
                merge = lat.get("merge_swap") or {}
                merge_cell = f"{merge['mean_ms']:.4f}" if merge else "-"
                if lat["swap_kind"] == "pointer":
                    pointer_cell = f"{resident.get('mean_ms', float('nan')):.4f}"
                    resident_copy = lat.get("resident_swap_copy")
                    copy_cell = f"**{resident_copy['mean_ms']:.4f}**" if resident_copy else "-"
                else:
                    pointer_cell = f"{resident_ptr['mean_ms']:.4f}" if resident_ptr else "-"
                    copy_cell = f"**{resident.get('mean_ms', float('nan')):.4f}**"
                lines.append(
                    f"| {r['method']} | {lat['device']} | {resident.get('n', cold.get('n', '-'))} | "
                    f"{copy_cell} | **{cold.get('mean_ms', float('nan')):.4f}** | "
                    f"{merge_cell} | {pointer_cell} | {lat['swap_kind']} |"
                )
        lines += ["", MERGE_NOTE.strip()]
    return "\n".join(lines)


def significance_table(sig_rows: list[dict]) -> str:
    lines = ["", "## Pairwise significance of cold swap (Welch's t-test, BH-corrected per model)", "",
             "| Arch | Model | A vs B | mean A (ms) | mean B (ms) | t | p (normal approx.) | q (BH) | sig. |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in sig_rows:
        if r["p"] is None:
            continue
        sig = "*" if r["q_bh"] is not None and r["q_bh"] < 0.05 else ""
        lines.append(
            f"| {r['arch']} | {r['model']} | {r['method_a']} vs {r['method_b']} | "
            f"{r['mean_a']:.4f} | {r['mean_b']:.4f} | {r['t']:.2f} | {r['p']:.4g} | "
            f"{r['q_bh']:.4g} | {sig} |"
        )
    return "\n".join(lines)


def interpretation(rows: list[dict]) -> str:
    devices = sorted({(r.get("latency") or {}).get("device") for r in rows} - {None})
    n_switches = sorted({(r.get("latency") or {}).get("n_switches") for r in rows} - {None})
    n_real = sum(1 for r in rows if (r.get("latency") or {}).get("real_model"))
    return INTERPRETATION.format(
        devices="/".join(devices), n_switches="/".join(str(n) for n in n_switches),
        n_real=n_real, n_cells=len(rows), pointer=_global_pointer_span(rows),
        dec_fitln=_span(rows, "decoder", "fitln", "cold", 2),
        dec_lora1=_span(rows, "decoder", "lora1", "cold", 2),
        dec_lora8=_span(rows, "decoder", "lora8", "cold", 2),
        dec_fullft=_span(rows, "decoder", "fullft", "cold", 1),
        enc_fitln=_span(rows, "encoder", "fitln", "cold", 2),
        enc_fullft=_span(rows, "encoder", "fullft", "cold", 1),
        copy_fitln=_span(rows, "decoder", "fitln", "copy", 3),
        copy_fullft=_span(rows, "decoder", "fullft", "copy", 2),
        lora_copy=_span(rows, "decoder", "lora1", "copy", 4),
        lora_pointer=_span(rows, "decoder", "lora1", "pointer", 4),
        merge_enc=_span(rows, "encoder", "lora1", "merge", 1),
        merge_dec=_span(rows, "decoder", "lora1", "merge", 1),
    )


def to_markdown(rows: list[dict], sig_rows: list[dict]) -> str:
    return "\n".join([HEADER, latency_table(rows), significance_table(sig_rows), "",
                      interpretation(rows)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser("aggregate_hotswap_results")
    parser.add_argument("--out_root", default="out/hotswap")
    parser.add_argument("--results_dir", default="results/hotswap")
    args = parser.parse_args()

    rows = _load_all(args.out_root)
    if not rows:
        raise SystemExit(f"No hotswap_result.json found under {args.out_root}")
    sig_rows = pairwise_significance(rows)

    os.makedirs(args.results_dir, exist_ok=True)
    md_path = os.path.join(args.results_dir, "swap_latency_table.md")
    json_path = os.path.join(args.results_dir, "swap_latency_significance.json")
    with open(md_path, "w") as f:
        f.write(to_markdown(rows, sig_rows))
    with open(json_path, "w") as f:
        json.dump(sig_rows, f, indent=2)
    print(f"{len(rows)} result files, {len(sig_rows)} pairwise comparisons -> {md_path}, {json_path}")
