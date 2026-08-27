"""YAML-gated deterministic profile save fallback.

Hard persistence rescue: missing YAML key → **on** (fail-closed).
Not soft under ``mod.brain.check``. Off = no code-path ``save_group_profile``
after FORCE_SAVE retries fail.
"""

from __future__ import annotations

import logging
import time

from apps.group_agent_api.agent_factory.checks.deterministic_profile_save.ids import (
    CHECK_ID,
)

_logger = logging.getLogger("uvicorn.error")

__all__ = ["deterministic_profile_save_enabled"]


def deterministic_profile_save_enabled(*, enabled: bool | None = None) -> bool:
    """Resolve YAML; explicit ``enabled`` wins. Missing key → True."""
    if enabled is not None:
        return bool(enabled)
    from apps.group_agent_api.agent_factory.module_config import load_modules_config

    cfg = load_modules_config()
    if CHECK_ID not in cfg.checks:
        return True
    return cfg.is_check_enabled(CHECK_ID)


def timed_enabled(*, enabled: bool | None = None) -> bool:
    started = time.perf_counter()
    on = deterministic_profile_save_enabled(enabled=enabled)
    _logger.info(
        "action=module_span check_id=%s skipped=%s reason=%s elapsed_ms=%s",
        CHECK_ID,
        not on,
        "yaml_off" if not on else "applied",
        int((time.perf_counter() - started) * 1000),
    )
    return on
