"""YAML-gated profile quality Layer-2 LLM (chk.profile_quality_llm).

Soft check under ``mod.brain.check``: master off or check off → skip the
second LLM and keep Layer-1 rules only (TSD-14 §4.3: 只走规则闸).
"""

from __future__ import annotations

import logging
import time

from apps.group_agent_api.agent_factory.checks.profile_quality.ids import CHECK_ID

_logger = logging.getLogger("uvicorn.error")

__all__ = [
    "profile_quality_llm_enabled",
]


def profile_quality_llm_enabled(*, enabled: bool | None = None) -> bool:
    """Resolve YAML switch; explicit ``enabled`` wins (tests / callers)."""
    if enabled is not None:
        return bool(enabled)
    from apps.group_agent_api.agent_factory.module_config import load_modules_config

    return load_modules_config().is_check_enabled(CHECK_ID)


def log_profile_quality_span(*, skipped: bool, reason: str = "") -> None:
    _logger.info(
        "action=module_span check_id=%s skipped=%s reason=%s elapsed_ms=0",
        CHECK_ID,
        skipped,
        reason or ("yaml_off" if skipped else "applied"),
    )


def timed_enabled(*, enabled: bool | None = None) -> bool:
    """Like ``profile_quality_llm_enabled`` but emits a module_span."""
    started = time.perf_counter()
    on = profile_quality_llm_enabled(enabled=enabled)
    _logger.info(
        "action=module_span check_id=%s skipped=%s reason=%s elapsed_ms=%s",
        CHECK_ID,
        not on,
        "yaml_off" if not on else "applied",
        int((time.perf_counter() - started) * 1000),
    )
    return on
