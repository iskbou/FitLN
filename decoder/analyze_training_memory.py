r"""Peak training memory of FitLN vs full fine-tuning and vs LoRA, from the peak_mem_mb
values in global_summary.md (Model / method characteristics table).

Usage:
    python decoder/analyze_training_memory.py \
        --summary_md results/decoder/global_summary.md --out_dir results/decoder
"""
import argparse
import os
import re
import statistics


METHOD_ORDER = ["fitln", "lora_r1_attnonly", "lora_r8_attnonly", "fullft"]

# Row labels / tier grouping; keys match common.MODEL_SHORT.
MODEL_DISPLAY = {
    "llama3_2_1b": "Llama-3.2-1B",
    "qwen2_5_1_5b": "Qwen2.5-1.5B",
    "qwen3_5_2b_base": "Qwen3.5-2B-Base",
    "llama3_2_3b": "Llama-3.2-3B",
    "qwen2_5_3b": "Qwen2.5-3B",
    "qwen3_5_4b_base": "Qwen3.5-4B-Base",
}
TIER_1_2B = ["llama3_2_1b", "qwen2_5_1_5b", "qwen3_5_2b_base"]
TIER_3_4B = ["llama3_2_3b", "qwen2_5_3b", "qwen3_5_4b_base"]

_MEM_RE = re.compile(r"^\s*([\d.]+)\s*(?:±\s*([\d.]+))?\s*$")


def parse_characteristics_table(md_path):
    """Return {(model_short, method): (mean_mb, std_mb, n_seeds)}; unmeasured cells are skipped."""
    with open(md_path, encoding="utf-8") as f:
        lines = f.readlines()

    start = next((i for i, l in enumerate(lines) if l.strip().startswith("## Model / method characteristics")), None)
    if start is None:
        raise ValueError(f"{md_path} has no '## Model / method characteristics' section")

    table_lines = []
    for line in lines[start + 1:]:
        if line.strip().startswith("|"):
            table_lines.append(line.strip())
        elif table_lines:
            break

    header = [c.strip() for c in table_lines[0].strip("|").split("|")]
    mem_idx = header.index("Peak GPU Mem (MB)")
    seeds_idx = header.index("Seeds")
    model_idx = header.index("Model")
    method_idx = header.index("Method")

    out = {}
    for line in table_lines[2:]:  # skip header + '---' separator row
        cells = [c.strip() for c in line.strip("|").split("|")]
        model_short, method = cells[model_idx], cells[method_idx]
        n_seeds = int(cells[seeds_idx])
        mem_cell = cells[mem_idx]
        if mem_cell.startswith("N/A"):
            continue
        m = _MEM_RE.match(mem_cell)
        if not m:
            raise ValueError(f"Could not parse Peak GPU Mem cell {mem_cell!r} in {md_path}")
        mean = float(m.group(1))
        std = float(m.group(2)) if m.group(2) is not None else 0.0
        out[(model_short, method)] = (mean, std, n_seeds)
    return out


def build_ratio_table(mem_by_pair, model_shorts):
    rows = []
    for model_short in model_shorts:
        row = {"model": model_short}
        for method in METHOD_ORDER:
            row[method] = mem_by_pair.get((model_short, method), (None, None, 0))
        fitln_mean = row["fitln"][0]
        row["ratio_fft_over_fitln"] = (
            row["fullft"][0] / fitln_mean if fitln_mean and row["fullft"][0] else None
        )
        row["ratio_lora8_over_fitln"] = (
            row["lora_r8_attnonly"][0] / fitln_mean if fitln_mean and row["lora_r8_attnonly"][0] else None
        )
        row["ratio_lora1_over_fitln"] = (
            row["lora_r1_attnonly"][0] / fitln_mean if fitln_mean and row["lora_r1_attnonly"][0] else None
        )
        rows.append(row)
    return rows


def _fmt_mem(mean, std, n):
    if mean is None:
        return "N/A (cluster only)"
    return f"{mean:.0f} ± {std:.0f}" if n > 1 else f"{mean:.0f}"


def _fmt_ratio(r):
    return f"{r:.2f}×" if r is not None else "N/A"


def to_markdown(rows):
    header = ["Model", "FitLN (MB)", "LoRA (1) (MB)", "LoRA (8) (MB)", "FFT (MB)",
              "FFT / FitLN", "LoRA(8) / FitLN"]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join([
            MODEL_DISPLAY.get(row["model"], row["model"]),
            _fmt_mem(*row["fitln"]),
            _fmt_mem(*row["lora_r1_attnonly"]),
            _fmt_mem(*row["lora_r8_attnonly"]),
            _fmt_mem(*row["fullft"]),
            _fmt_ratio(row["ratio_fft_over_fitln"]),
            _fmt_ratio(row["ratio_lora8_over_fitln"]),
        ]) + " |")
    return "\n".join(lines)


def to_latex_table(rows):
    by_model = {row["model"]: row for row in rows}

    def cell(row, method):
        mean, std, n = row[method]
        if mean is None:
            return "--"
        return f"${mean:.0f}$" if n <= 1 else f"${mean:.0f}_{{\\pm{std:.0f}}}$"

    def row_line(model_short):
        row = by_model[model_short]
        ratio_fft = row["ratio_fft_over_fitln"]
        ratio_lora8 = row["ratio_lora8_over_fitln"]
        return (
            f"{MODEL_DISPLAY.get(model_short, model_short)} & {cell(row, 'fitln')} & "
            f"{cell(row, 'lora_r1_attnonly')} & {cell(row, 'lora_r8_attnonly')} & "
            f"{cell(row, 'fullft')} & "
            f"{f'{ratio_fft:.2f}' if ratio_fft else '--'}$\\times$ & "
            f"{f'{ratio_lora8:.2f}' if ratio_lora8 else '--'}$\\times$ \\\\"
        )

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\footnotesize")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{l | r r r r | r r}")
    lines.append(r"Model & FitLN & LoRA (1) & LoRA (8) & FFT & FFT/FitLN & LoRA(8)/FitLN \\")
    lines.append(r"\toprule")
    lines.append(r"\toprule")
    lines.append(r"\multicolumn{7}{l}{\textit{1--2B tier}}\\")
    lines.append(r"\midrule")
    for model_short in TIER_1_2B:
        if model_short in by_model:
            lines.append(row_line(model_short))
    lines.append(r"\midrule")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{7}{l}{\textit{3--4B tier}}\\")
    lines.append(r"\midrule")
    for model_short in TIER_3_4B:
        if model_short in by_model:
            lines.append(row_line(model_short))
    lines.append(r"\toprule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Peak GPU training memory (MB, mean $\pm$ std across seeds at each "
        r"method's selected best LR; same runs as Table~\ref{tab:decoder_params}). Ratio "
        r"columns are FFT and LoRA (8) peak memory divided by FitLN's, per model.}"
        r"\label{tab:decoder_memory}"
    )
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_md", default="results/decoder/global_summary.md")
    parser.add_argument("--out_dir", default="results/decoder")
    args = parser.parse_args()

    if not os.path.isfile(args.summary_md):
        print(f"[analyze_training_memory] {args.summary_md} does not exist -- "
              f"run summarize_results.py first.")
        return

    mem_by_pair = parse_characteristics_table(args.summary_md)
    model_shorts = sorted({model for model, _ in mem_by_pair})
    rows = build_ratio_table(mem_by_pair, model_shorts)

    os.makedirs(args.out_dir, exist_ok=True)

    md_path = os.path.join(args.out_dir, "peak_memory_table.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("## Peak GPU training memory by method and model\n\n")
        f.write("Source: `global_summary.md`'s \"Model / method characteristics\" table "
                 "(seed set as selected when summarize_results.py was run; default 1, 2, 42).\n\n")
        f.write(to_markdown(rows))
        f.write("\n")
    print(f"[analyze_training_memory] wrote {md_path}")

    tex_path = os.path.join(args.out_dir, "peak_memory_tex_block.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Ready-to-paste LaTeX table block (peak training memory by method and model)\n\n")
        f.write(to_latex_table(rows))
        f.write("\n")
    print(f"[analyze_training_memory] wrote {tex_path}")

    fft_ratios = [r["ratio_fft_over_fitln"] for r in rows if r["ratio_fft_over_fitln"] is not None]
    lora8_ratios = [r["ratio_lora8_over_fitln"] for r in rows if r["ratio_lora8_over_fitln"] is not None]
    lora1_ratios = [r["ratio_lora1_over_fitln"] for r in rows if r["ratio_lora1_over_fitln"] is not None]

    print("\n=== Headline numbers ===")
    if fft_ratios:
        print(f"FFT / FitLN peak memory ratio: mean {statistics.mean(fft_ratios):.2f}x "
              f"(range {min(fft_ratios):.2f}x - {max(fft_ratios):.2f}x, n={len(fft_ratios)} models)")
    if lora8_ratios:
        print(f"LoRA(8) / FitLN peak memory ratio: mean {statistics.mean(lora8_ratios):.2f}x "
              f"(range {min(lora8_ratios):.2f}x - {max(lora8_ratios):.2f}x, n={len(lora8_ratios)} models)")
    if lora1_ratios:
        print(f"LoRA(1) / FitLN peak memory ratio: mean {statistics.mean(lora1_ratios):.2f}x "
              f"(range {min(lora1_ratios):.2f}x - {max(lora1_ratios):.2f}x, n={len(lora1_ratios)} models)")


if __name__ == "__main__":
    main()
