from __future__ import annotations

from math import gcd
from typing import Any


def compute_dynamic_global_batch_size(
    num_samples: int,
    *,
    dp_size: int,
    target_steps: int | None = None,
) -> tuple[int, int, int, int]:
    """Return dynamic GBS plus desired steps, realized steps, and wasted samples."""
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    if dp_size <= 0:
        raise ValueError(f"dp_size must be positive, got {dp_size}")

    desired_steps = int(target_steps) if target_steps is not None and target_steps > 0 else 1
    per_step_target = max(1, num_samples // desired_steps)
    dynamic_gbs = (per_step_target // dp_size) * dp_size

    if dynamic_gbs == 0:
        dynamic_gbs = dp_size

    realized_steps = max(1, num_samples // dynamic_gbs)
    wasted = num_samples % dynamic_gbs
    return dynamic_gbs, desired_steps, realized_steps, wasted


def get_aligned_train_data_trim_len(num_samples: int, *, global_batch_size: int, dp_size: int) -> int:
    """Choose a trim length that keeps DP partitions even after sample filtering."""
    if num_samples <= 0:
        return 0
    if global_batch_size <= 0:
        raise ValueError(f"global_batch_size must be positive, got {global_batch_size}")
    if dp_size <= 0:
        raise ValueError(f"dp_size must be positive, got {dp_size}")

    trim_multiple = global_batch_size if num_samples >= global_batch_size else dp_size
    if trim_multiple % dp_size != 0:
        trim_multiple = trim_multiple * dp_size // gcd(trim_multiple, dp_size)
    return (num_samples // trim_multiple) * trim_multiple


def trim_sample_aligned_data(data: dict[str, Any], trim_len: int, original_len: int) -> None:
    """Trim per-sample list fields in ``data`` in place."""
    for key, val in list(data.items()):
        if isinstance(val, list) and len(val) == original_len:
            data[key] = val[:trim_len]
