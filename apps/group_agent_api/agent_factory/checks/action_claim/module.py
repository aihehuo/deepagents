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

    De-dupe vs ``mod.brain.reply_grounding`` L0 ``unverified_action`` (same
    detector): when RG is on and ``enabled`` was not forced True, skip this
    silent replace so RG owns deny→rewrite. Force ``enabled=True`` in tests
    that need the legacy silent path with RG still on.
    """
    started = time.perf_counter()
    raw = (text or "").strip()
    if not action_claim_enabled(enabled=enabled):
        _log_span(started, skipped=True, blocked=False, reason="yaml_off")
        return raw, False

    # Prefer RG L0 when both would run (async/chat hang before finalize/RG).
    if enabled is None:
        from apps.group_agent_api.agent_factory.module_config import (
            reply_grounding_enabled,
        )

        if reply_grounding_enabled():
            _log_span(
                started,
                skipped=True,
                blocked=False,
                reason="deferred_to_reply_grounding_l0",
            )
            return raw, False

    from apps.group_agent_api.agent_factory.content_quality import (
        guard_action_claims as _raw_guard,
    )

    out, blocked = _raw_guard(raw)
    _log_span(started, skipped=False, blocked=blocked, reason="applied")
    return out, blocked


def _log_span(
    started: float,
    *,
    skipped: bool,
    blocked: bool,
    reason: str = "",
) -> None:
    _logger.info(
        "action=module_span check_id=%s skipped=%s blocked=%s reason=%s "
        "elapsed_ms=%s",
        CHECK_ID,
        skipped,
        blocked,
        reason or "n/a",
        int((time.perf_counter() - started) * 1000),
    )
