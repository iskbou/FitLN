"""Shared definitions for the encoder GLUE scripts."""
import argparse

TASK_TEXT_FIELDS = {
    "cola": ["sentence"],
    "sst2": ["sentence"],
    "mrpc": ["sentence1", "sentence2"],
    "qqp": ["question1", "question2"],
    "stsb": ["sentence1", "sentence2"],
    "mnli": ["premise", "hypothesis"],
    "qnli": ["question", "sentence"],
    "rte": ["sentence1", "sentence2"],
    "wnli": ["sentence1", "sentence2"],
}

TASK_NUM_LABELS = {
    "cola": 2,
    "sst2": 2,
    "mrpc": 2,
    "qqp": 2,
    "stsb": 1,
    "mnli": 3,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
}


def str2bool(v):
    """Boolean parser for argparse (type=bool treats the string 'False' as True)."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes", "y", "t"):
        return True
    if v.lower() in ("false", "0", "no", "n", "f", ""):
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean, got {v!r}")
