"""Two-panel figure. (A) per-task weight size against the maximum number of resident tasks on
an 80GB GPU, for encoders without and with the task head and for decoders. (B) bytes written
by a swap against the swap time: for fitln, bitfit and fullft the delta size of the resident
copy swap; for lora1 and lora8 the full target weight matrices that a merge writes.

Reads capacity_table.json and capacity_table_no_head.json (capacity.py) from --results_dir,
and the hotswap_result.json files (run_hotswap.py) from --runs.

Usage:
    python hotswap/plot_capacity_swap.py --results_dir results/hotswap --runs results/hotswap/runs \
        --figdir figures/hotswap
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import registry

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
ARCH_MARKER = {"encoder": "o", "decoder": "s"}
SCENARIO_MARKER = {"encoder": "o", "encoder_head": "^", "decoder": "s"}
SCENARIO_LABEL = {
    "encoder": "Encoder (excl. head)",
    "encoder_head": "Encoder (+ classifier/pooler head)",
    "decoder": "Decoder (no task head)",
}
GPU_LABEL_FOR_CAPACITY = "80GB (e.g. A100/H100-80GB)"

plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 17,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.titlesize": 18,
})


def _load(results_dir, name):
    with open(os.path.join(results_dir, name)) as f:
        return json.load(f)


def build_panel_a_points(results_dir):
    with_head = _load(results_dir, "capacity_table.json")
    no_head = _load(results_dir, "capacity_table_no_head.json")
    points = []
    for r in with_head:
        if r["gpu"] != GPU_LABEL_FOR_CAPACITY:
            continue
        scenario = "encoder_head" if r["arch"] == "encoder" else "decoder"
        points.append(dict(scenario=scenario, model=r["model"], method=r["method"],
                           x=r["per_task_bytes"], y=r["n_tasks"]))
    for r in no_head:
        if r["gpu"] != GPU_LABEL_FOR_CAPACITY or r["arch"] != "encoder":
            continue
        points.append(dict(scenario="encoder", model=r["model"], method=r["method"],
                           x=r["per_task_bytes"], y=r["n_tasks"]))
    return points


def build_panel_b_points(runs):
    """(arch, model, method) -> (bytes written, swap time in ms)."""
    points = []
    for path in glob.glob(os.path.join(runs, "*", "*", "*", "hotswap_result.json")):
        with open(path) as f:
            r = json.load(f)
        lat = r["latency"]
        method, arch, model = r["method"], r["arch"], r["model"]
        device = lat.get("device")
        if lat["swap_kind"] == "copy":
            resident = lat.get("resident_swap")
            if not resident:
                continue
            points.append(dict(arch=arch, model=model, method=method, device=device,
                               x=r["per_task_bytes"], y=resident["mean_ms"]))
        elif method in ("lora1", "lora8"):
            merge = lat.get("merge_swap")
            if not merge:
                continue
            reg = (registry.ENCODER_MODELS if arch == "encoder" else registry.DECODER_MODELS)[model]
            dtype_bytes = registry.BYTES_PER_PARAM["bf16"]
            bytes_touched = reg["hidden_size"] ** 2 * reg["num_layers"] * reg["lora_target_proj"] * dtype_bytes
            points.append(dict(arch=arch, model=model, method=method, device=device,
                               x=bytes_touched, y=merge["mean_ms"]))
    return points


def _panel_b_device_label(panel_b):
    """(axis label suffix, title note) from the devices in the data. The result files only
    record cuda or cpu; the GPU of the reported runs is an H100."""
    devices = {p.get("device") for p in panel_b}
    devices.discard(None)
    if devices == {"cuda"}:
        return "H100", "(H100 measurements)"
    if devices == {"cpu"}:
        return "CPU", "(CPU baseline)"
    print(f"panel B mixes devices {devices}")
    return "mixed devices", "(mixed CPU/GPU data)"


def _scatter_by_method(ax, points, ylabel, title, category_key="arch", marker_map=None,
                       yscale="log", xscale="log"):
    marker_map = marker_map or ARCH_MARKER
    seen_methods = set()
    seen_categories = set()
    for p in points:
        color = METHOD_COLOR[p["method"]]
        marker = marker_map[p[category_key]]
        ax.scatter(p["x"], p["y"], c=color, marker=marker, s=90,
                   edgecolors="white", linewidths=1.0, zorder=3, alpha=0.55)
        seen_methods.add(p["method"])
        seen_categories.add(p[category_key])
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)
    ax.grid(True, which="major", alpha=0.25, zorder=0)
    ax.grid(True, which="minor", alpha=0.1, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return seen_methods, seen_categories


def _bytes_formatter():
    def fmt(x, pos):
        if x >= 1e9:
            return f"{x/1e9:.0f}GB"
        if x >= 1e6:
            return f"{x/1e6:.0f}MB"
        if x >= 1e3:
            return f"{x/1e3:.0f}KB"
        return f"{x:.0f}B"
    return mticker.FuncFormatter(fmt)


def make_figure(panel_a, panel_b, out_path_pdf, panel_a_yscale="log", panel_a_xscale="log"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))

    methods_a, scenarios_a = _scatter_by_method(
        ax1, panel_a, ylabel="Max resident tasks (80GB GPU)",
        title="(A) Capacity vs. per-task weight size\n(every scenario: with and without the head)",
        category_key="scenario", marker_map=SCENARIO_MARKER, yscale=panel_a_yscale, xscale=panel_a_xscale)
    ax1.set_xlabel("Per-task checkpoint size")
    ax1.xaxis.set_major_formatter(_bytes_formatter())

    device_label, suptitle_note = _panel_b_device_label(panel_b)
    methods_b, _ = _scatter_by_method(ax2, panel_b,
                                      ylabel=f"Write-swap latency (ms, {device_label})",
                                      title="(B) Write-swap time vs. bytes touched")
    ax2.set_xlabel("Bytes touched by the write\n(copy: delta size; merge: full target matrices)")
    ax2.xaxis.set_major_formatter(_bytes_formatter())

    method_order = ["fitln", "bitfit", "lora1", "lora8", "fullft"]
    color_handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=METHOD_COLOR[m],
                                markersize=11, label=METHOD_LABEL[m])
                     for m in method_order if m in (methods_a | methods_b)]
    scenario_order = ["encoder", "encoder_head", "decoder"]
    scenario_handles = [plt.Line2D([0], [0], marker=SCENARIO_MARKER[s], color="w", markerfacecolor="#555555",
                                   markersize=11, label=SCENARIO_LABEL[s])
                        for s in scenario_order if s in scenarios_a]
    fig.legend(handles=color_handles + scenario_handles, loc="lower center",
               ncol=len(color_handles), bbox_to_anchor=(0.5, -0.10),
               frameon=False, columnspacing=1.4, handletextpad=0.5)

    fig.suptitle("Hot-swap capacity and write-swap cost scale with weight size\n"
                 + suptitle_note,
                 y=1.06)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(out_path_pdf, bbox_inches="tight", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("plot_capacity_swap")
    parser.add_argument("--results_dir", default="results/hotswap")
    parser.add_argument("--runs", default="results/hotswap/runs")
    parser.add_argument("--figdir", default="figures/hotswap")
    args = parser.parse_args()
    os.makedirs(args.figdir, exist_ok=True)
    pdf_path = os.path.join(args.figdir, "fig_capacity_swap.pdf")
    make_figure(build_panel_a_points(args.results_dir), build_panel_b_points(args.runs), pdf_path)
    print("wrote", pdf_path)
