"""Find the best-learning-rate FitLN checkpoint of a (task, model) in out/.

Layout: {root}/{task}/{model}/{mode}/{lr}/{epochs}/{fitln_weights.bin|lay.bin} with
results.json next to it. The validation metric is the first metric of the GLUE metric
(accuracy for mrpc and qqp). MNLI runs write results_m.json instead of results.json, so
they have no validation metric here and are skipped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

CHECKPOINT_NAMES = ("fitln_weights.bin", "lay.bin")

FITLN_VAL_METRIC = {
    "cola":  "val_matthews_correlation",
    "sst2":  "val_accuracy",
    "mrpc":  "val_accuracy",
    "stsb":  "val_pearson",
    "qqp":   "val_accuracy",
    "mnli":  "val_accuracy",
    "qnli":  "val_accuracy",
    "rte":   "val_accuracy",
    "wnli":  "val_accuracy",
}

# Tried in order when the primary metric is missing.
FITLN_FALLBACK_METRICS = {
    "mrpc":  ["val_f1"],
    "qqp":   ["val_f1"],
    "stsb":  ["val_spearmanr"],
}


@dataclass
class CheckpointRef:
    task: str
    model: str
    mode: str
    lr: float
    epochs: int
    ckpt_path: Path
    results_path: Path
    val_metric: float | None
    metric_name: str

    def load(self, map_location: str = "cpu") -> dict[str, torch.Tensor]:
        return torch.load(self.ckpt_path, map_location=map_location, weights_only=False)


def _safe_float(s: str) -> float | None:
    try:
        return float(s)
    except ValueError:
        return None


def _safe_int(s: str) -> int | None:
    try:
        return int(s)
    except ValueError:
        return None


def list_checkpoints(
    root: Path | str = "out",
    task: str | None = None,
    model: str | None = None,
    mode: str | None = None,
) -> list[CheckpointRef]:
    """Every checkpoint below root matching the filters. A model is either
    {task}/{model}/ or {task}/{org}/{model}/."""
    root = Path(root)
    results: list[CheckpointRef] = []
    if not root.exists():
        return results

    for task_dir in sorted(root.iterdir()):
        if not task_dir.is_dir():
            continue
        if task and task_dir.name != task:
            continue
        t = task_dir.name
        if t not in FITLN_VAL_METRIC:
            continue

        for first in sorted(task_dir.iterdir()):
            if not first.is_dir():
                continue

            # `first` is the model directory if it contains a mode directory, else an org directory.
            sub_names = {p.name for p in first.iterdir() if p.is_dir()}
            mode_like = {"attn_only", "full", "gamma", "beta", "gamma_beta_zero"}
            if sub_names & mode_like:
                model_dirs = [(first.name, first)]
            else:
                model_dirs = [(f"{first.name}/{m.name}", m)
                              for m in sorted(first.iterdir()) if m.is_dir()]

            for full_model_name, model_dir in model_dirs:
                if model and full_model_name != model and model_dir.name != model:
                    continue
                for mode_dir in sorted(model_dir.iterdir()):
                    if not mode_dir.is_dir():
                        continue
                    if mode and mode_dir.name != mode:
                        continue
                    for lr_dir in sorted(mode_dir.iterdir()):
                        lr = _safe_float(lr_dir.name)
                        if lr is None:
                            continue
                        for ep_dir in sorted(lr_dir.iterdir()):
                            ep = _safe_int(ep_dir.name)
                            if ep is None:
                                continue
                            ckpt = next((ep_dir / n for n in CHECKPOINT_NAMES if (ep_dir / n).exists()), None)
                            res = ep_dir / "results.json"
                            if ckpt is None:
                                continue
                            val = None
                            metric_name = FITLN_VAL_METRIC[t]
                            if res.exists():
                                try:
                                    with open(res) as f:
                                        data = json.load(f)
                                    val = data.get(metric_name)
                                    if val is None:
                                        for alt in FITLN_FALLBACK_METRICS.get(t, []):
                                            if alt in data:
                                                val = data[alt]
                                                metric_name = alt
                                                break
                                except (OSError, json.JSONDecodeError):
                                    pass
                            results.append(CheckpointRef(
                                task=t,
                                model=full_model_name,
                                mode=mode_dir.name,
                                lr=lr,
                                epochs=ep,
                                ckpt_path=ckpt,
                                results_path=res,
                                val_metric=val,
                                metric_name=metric_name,
                            ))
    return results


def find_best_checkpoint(
    task: str,
    model: str,
    mode: str = "attn_only",
    root: Path | str = "out",
) -> CheckpointRef | None:
    """The checkpoint with the highest validation metric, or None."""
    refs = [r for r in list_checkpoints(root, task=task, model=model, mode=mode)
            if r.val_metric is not None]
    if not refs:
        return None
    return max(refs, key=lambda r: r.val_metric)
