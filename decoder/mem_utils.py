"""Peak GPU memory and checkpoint size measurement.

get_peak_memory_mb returns None (not 0) when CUDA is unavailable, so an unmeasured
value is never read as zero.
"""
import os

import torch


def reset_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    else:
        print(
            "[mem_utils] CUDA unavailable, peak memory is not tracked."
        )


def get_peak_memory_mb():
    """Peak allocated CUDA memory in MB since the last reset, or None without CUDA."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return None


def checkpoint_size_bytes(state_dict: dict) -> int:
    """Size in bytes of a state dict."""
    return sum(v.numel() * v.element_size() for v in state_dict.values())


def checkpoint_size_on_disk(path: str) -> int:
    """Size in bytes of a checkpoint file, or of a directory (summed recursively)."""
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            total += os.path.getsize(os.path.join(dirpath, f))
    return total
