"""YAML-gated FORCE_SAVE retry loop (chk.force_save_retry).

Hard persistence rescue: missing YAML key → **on** (fail-closed).
Not soft under ``mod.brain.check`` — master off does not disable this.
Off = skip FORCE_SAVE ``ainvoke`` loop (model must call save itself).
"""

from __future__ import annotations

import logging
import time

from apps.group_agent_api.agent_factory.checks.force_save_retry.ids import CHECK_ID

_logger = logging.getLogger("uvicorn.error")

__all__ = ["force_save_retry_enabled"]


def force_save_retry_enabled(*, enabled: bool | None = None) -> bool:
    """Resolve YAML; explicit ``enabled`` wins. Missing key → True."""
    if enabled is not None:
        return bool(enabled)
    from apps.group_agent_api.agent_factory.module_config import load_modules_config

    cfg = load_modules_config()
    if CHECK_ID not in cfg.checks:
        return True
    return cfg.is_check_enabled(CHECK_ID)


def log_force_save_span(*, skipped: bool) -> None:
    _logger.info(
        "action=module_span check_id=%s skipped=%s reason=%s elapsed_ms=0",
        CHECK_ID,
        skipped,
        "yaml_off" if skipped else "applied",
    )


def timed_enabled(*, enabled: bool | None = None) -> bool:
    started = time.perf_counter()
    on = force_save_retry_enabled(enabled=enabled)
    _logger.info(
        "action=module_span check_id=%s skipped=%s reason=%s elapsed_ms=%s",
        CHECK_ID,
        not on,
        "yaml_off" if not on else "applied",
        int((time.perf_counter() - started) * 1000),
    )
    return on
