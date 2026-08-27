"""YAML-gated invented-candidate scrub (chk.invented_candidate)."""

from __future__ import annotations

import logging
import time

from apps.group_agent_api.agent_factory.checks.invented_candidate.ids import CHECK_ID

_logger = logging.getLogger("uvicorn.error")

__all__ = [
    "invented_candidate_enabled",
    "scrub_invented_candidate_if_enabled",
]


def invented_candidate_enabled(*, enabled: bool | None = None) -> bool:
    if enabled is not None:
        return bool(enabled)
    from apps.group_agent_api.agent_factory.module_config import load_modules_config

    return load_modules_config().is_check_enabled(CHECK_ID)


def scrub_invented_candidate_if_enabled(
    text: str | None,
    *,
    enabled: bool | None = None,
) -> str:
    """When enabled, drop invented-candidate narrative paragraphs.

    Off = return stripped original (no scrub).
    """
    started = time.perf_counter()
    raw = text if text is not None else ""
    if not invented_candidate_enabled(enabled=enabled):
        _log_span(started, skipped=True)
        return raw.strip()

    from apps.group_agent_api.agent_factory.content_quality import (
        scrub_invented_candidate_narrative as _raw_scrub,
    )

    out = _raw_scrub(raw)
    _log_span(started, skipped=False)
    return out


def _log_span(started: float, *, skipped: bool) -> None:
    _logger.info(
        "action=module_span check_id=%s skipped=%s elapsed_ms=%s",
        CHECK_ID,
        skipped,
        int((time.perf_counter() - started) * 1000),
    )
