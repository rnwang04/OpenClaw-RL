from __future__ import annotations

import os
import re
from collections.abc import Iterable


BLOCKED_TASKS_ENV = "TERMINAL_RL_BLOCKED_TASKS"


def parse_task_names(raw: str | None) -> frozenset[str]:
    if not raw:
        return frozenset()
    return frozenset(part for part in re.split(r"[\s,]+", raw.strip()) if part)


def get_blocked_task_names() -> frozenset[str]:
    return parse_task_names(os.getenv(BLOCKED_TASKS_ENV))


def is_task_blocked(
    task_name: object,
    blocked_task_names: Iterable[str] | None = None,
) -> bool:
    blocked = (
        frozenset(blocked_task_names)
        if blocked_task_names is not None
        else get_blocked_task_names()
    )
    return str(task_name).strip() in blocked
