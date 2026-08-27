"""YAML-gated action-claim guard (chk.action_claim)."""

from __future__ import annotations

import logging
import time

from apps.group_agent_api.agent_factory.checks.action_claim.ids import CHECK_ID

_logger = logging.getLogger("uvicorn.error")

__all__ = [
    "action_claim_enabled",
    "apply_action_claim_guard",
]


def action_claim_enabled(*, enabled: bool | None = None) -> bool:
    """Resolve YAML switch; explicit ``enabled`` wins (tests / callers)."""
    if enabled is not None:
        return bool(enabled)
    from apps.group_agent_api.agent_factory.module_config import load_modules_config

    return load_modules_config().is_check_enabled(CHECK_ID)


def apply_action_claim_guard(
    text: str | None,
    *,
    enabled: bool | None = None,
) -> tuple[str, bool]:
    """When enabled, replace unauthorized action-completion claims.

    Returns ``(reply, blocked)``. When the check is off, returns the stripped
    original text and ``blocked=False`` (skipped).
    """
    started = time.perf_counter()
    raw = (text or "").strip()
    if not action_claim_enabled(enabled=enabled):
        _log_span(started, skipped=True, blocked=False)
        return raw, False

    from apps.group_agent_api.agent_factory.content_quality import (
        guard_action_claims as _raw_guard,
    )

    out, blocked = _raw_guard(raw)
    _log_span(started, skipped=False, blocked=blocked)
    return out, blocked


def _log_span(started: float, *, skipped: bool, blocked: bool) -> None:
    _logger.info(
        "action=module_span check_id=%s skipped=%s blocked=%s elapsed_ms=%s",
        CHECK_ID,
        skipped,
        blocked,
        int((time.perf_counter() - started) * 1000),
    )
