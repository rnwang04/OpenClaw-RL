from __future__ import annotations

import logging

from slime.rollout.data_source import RolloutDataSourceWithBuffer

from task_filter import BLOCKED_TASKS_ENV, get_blocked_task_names, is_task_blocked


logger = logging.getLogger(__name__)


def _sample_is_blocked(sample, blocked_task_names: frozenset[str]) -> bool:
    metadata = sample.metadata or {}
    return is_task_blocked(metadata.get("task_name", ""), blocked_task_names)


class FilteredRolloutDataSourceWithBuffer(RolloutDataSourceWithBuffer):
    """Rollout data source that removes explicitly blocked terminal tasks."""

    def __init__(self, args):
        super().__init__(args)
        self.blocked_task_names = get_blocked_task_names()
        if self.dataset is None or not self.blocked_task_names:
            return

        original_count = len(self.dataset.origin_samples)
        allowed_origin_samples = [
            sample
            for sample in self.dataset.origin_samples
            if not _sample_is_blocked(sample, self.blocked_task_names)
        ]
        allowed_samples = [
            sample
            for sample in self.dataset.samples
            if not _sample_is_blocked(sample, self.blocked_task_names)
        ]
        removed_count = original_count - len(allowed_origin_samples)
        if removed_count == 0:
            return
        if not allowed_origin_samples:
            raise ValueError(
                f"All rollout prompts were blocked by {BLOCKED_TASKS_ENV}="
                f"{','.join(sorted(self.blocked_task_names))}"
            )

        self.dataset.origin_samples = allowed_origin_samples
        self.dataset.samples = allowed_samples
        logger.warning(
            "Filtered %d blocked terminal task prompt(s); blocked tasks=%s",
            removed_count,
            ",".join(sorted(self.blocked_task_names)),
        )

    def add_samples(self, samples):
        if not self.blocked_task_names:
            return super().add_samples(samples)

        allowed_groups = [
            group
            for group in samples
            if group
            and not _sample_is_blocked(group[0], self.blocked_task_names)
        ]
        if len(allowed_groups) != len(samples):
            logger.warning(
                "Dropped %d blocked sample group(s) from rollout buffer",
                len(samples) - len(allowed_groups),
            )
        if allowed_groups:
            super().add_samples(allowed_groups)
