from __future__ import annotations

import logging
from typing import Any, Dict, List

import wandb
from slime.utils import logging_utils
from slime.utils.types import Sample
from slime.ray.rollout import compute_rollout_step

logger = logging.getLogger(__name__)


def _ensure_terminal_step_metric(args) -> None:
    if not getattr(args, "use_wandb", False):
        return
    try:
        wandb.define_metric("terminal/*", step_metric="rollout/step")
    except Exception as e:
        logger.warning("Failed to define wandb step metric for terminal/*: %s", e)


def _percentile(sorted_xs: List[float], q: float) -> float:
    n = len(sorted_xs)
    if n == 0:
        return float("nan")
    if n == 1:
        return sorted_xs[0]
    k = (n - 1) * q
    f = int(k)
    c = min(f + 1, n - 1)
    return sorted_xs[f] + (sorted_xs[c] - sorted_xs[f]) * (k - f)


def _aggregate_timings(samples) -> Dict[str, float]:
    """Aggregate per-sample `_perf` dicts attached by generate.py into mean /
    p50 / p95 / max distributions. Dedups by (group_index, index) because
    build_samples_from_outcome emits one Sample per turn, all sharing the
    same per-episode timing dict.
    """
    seen = set()
    perfs: List[Dict[str, float]] = []
    for s in samples:
        key = (getattr(s, "group_index", None), getattr(s, "index", None))
        if key in seen:
            continue
        seen.add(key)
        md = getattr(s, "metadata", None) or {}
        perf = md.get("_perf")
        if isinstance(perf, dict):
            perfs.append(perf)
    if not perfs:
        return {}

    keys = set()
    for p in perfs:
        keys.update(p.keys())

    out: Dict[str, float] = {}
    out["terminal/timing/n_samples"] = len(perfs)
    for k in sorted(keys):
        vals = [float(p[k]) for p in perfs if isinstance(p.get(k), (int, float))]
        if not vals:
            continue
        vals_sorted = sorted(vals)
        prefix = f"terminal/timing/{k}"
        out[f"{prefix}/mean"] = sum(vals) / len(vals)
        out[f"{prefix}/p50"] = _percentile(vals_sorted, 0.50)
        out[f"{prefix}/p95"] = _percentile(vals_sorted, 0.95)
        out[f"{prefix}/max"] = vals_sorted[-1]

    # Per-sample sglang share of total (avoids ratio-of-means bias).
    shares = []
    for p in perfs:
        total = p.get("total")
        sg = p.get("sglang_generate")
        if isinstance(total, (int, float)) and total > 0 and isinstance(sg, (int, float)):
            shares.append(sg / total)
    if shares:
        out["terminal/timing/sglang_share/mean"] = sum(shares) / len(shares)
        out["terminal/timing/sglang_share/p50"] = _percentile(sorted(shares), 0.50)
    return out


def rollout_log(rollout_id, args, samples, rollout_extra_metrics, rollout_time):

    trainable = [s for s in samples if not getattr(s, "remove_sample", False)]
    non_trainable = [s for s in samples if getattr(s, "remove_sample", False)]

    log_dict: Dict[str, Any] = {}

    total = len(samples)
    n_failed = sum(1 for s in samples if s.status == Sample.Status.FAILED)
    n_aborted = sum(1 for s in samples if s.status == Sample.Status.ABORTED)
    n_truncated = sum(1 for s in samples if s.status == Sample.Status.TRUNCATED)
    n_completed = sum(1 for s in samples if s.status == Sample.Status.COMPLETED)

    log_dict["terminal/total_samples"] = total
    log_dict["terminal/completed"] = n_completed
    log_dict["terminal/truncated"] = n_truncated
    log_dict["terminal/failed"] = n_failed
    log_dict["terminal/aborted"] = n_aborted
    log_dict["terminal/failed_ratio"] = n_failed / total if total else 0.0
    log_dict["terminal/non_trainable_ratio"] = (
        len(non_trainable) / total if total else 0.0
    )

    if trainable:
        trainable_rewards = [s.reward["score"] for s in trainable]
        log_dict["terminal/reward_mean"] = sum(trainable_rewards) / len(
            trainable_rewards
        )
        log_dict["terminal/reward_min"] = min(trainable_rewards)
        log_dict["terminal/reward_max"] = max(trainable_rewards)

        trainable_accs = []
        for s in trainable:
            if isinstance(s.reward, dict) and "accuracy" in s.reward:
                trainable_accs.append(float(s.reward["accuracy"]))
        if trainable_accs:
            log_dict["terminal/accuracy"] = sum(trainable_accs) / len(trainable_accs)

        trainable_prm = []
        for s in trainable:
            if isinstance(s.reward, dict) and "prm_turn_score" in s.reward:
                trainable_prm.append(float(s.reward["prm_turn_score"]))
        if trainable_prm:
            log_dict["terminal/prm_turn_score"] = sum(trainable_prm) / len(
                trainable_prm
            )

    log_dict["terminal/rollout_time"] = rollout_time

    log_dict.update(_aggregate_timings(samples))

    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step
    _ensure_terminal_step_metric(args)
    logging_utils.log(args, log_dict, step_key="rollout/step")
    logger.info("rollout %s metrics: %s", rollout_id, log_dict)

    return False
