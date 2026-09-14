"""Opt-in public conversation limits, independent of signed member sessions."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AnonymousLimits:
    """Bound one anonymous conversation's resource usage."""

    max_active: int
    max_queries: int
    timeout_s: int


def anonymous_enabled() -> bool:
    """Return the explicitly configured public-entry state.

    Returns:
        Whether anonymous public conversations are enabled.

    Raises:
        ValueError: The enable flag is invalid.
    """
    value = os.environ.get("GROUP_AGENT_ANONYMOUS_ENABLED", "0").strip().lower()
    if value not in {"0", "1", "false", "true", "off", "on"}:
        msg = "GROUP_AGENT_ANONYMOUS_ENABLED must be 0/1, false/true, or off/on"
        raise ValueError(msg)
    return value in {"1", "true", "on"}


def anonymous_limits() -> AnonymousLimits:
    """Validate the independent public-entry concurrency and query budgets.

    Returns:
        Validated limits for a public conversation.

    Raises:
        ValueError: A limit is not an integer in its allowed range.
    """
    values: list[int] = []
    for key, default, maximum in (
        ("GROUP_AGENT_ANONYMOUS_MAX_ACTIVE", "2", 10),
        ("GROUP_AGENT_ANONYMOUS_MAX_QUERIES", "3", 5),
        ("GROUP_AGENT_ANONYMOUS_TIMEOUT_S", "45", 120),
    ):
        value = int(os.environ.get(key, default))
        if not 1 <= value <= maximum:
            msg = f"{key} must be between 1 and {maximum}"
            raise ValueError(msg)
        values.append(value)
    return AnonymousLimits(*values)
