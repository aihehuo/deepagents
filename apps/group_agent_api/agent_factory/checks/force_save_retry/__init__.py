"""chk.force_save_retry — FORCE_SAVE prompt retry when profile not persisted.

Hang: ``chat.py`` / ``async_manager.py`` after first-turn agent reply when
``_should_force_profile_save`` is true.

Default **on** (fail-closed). Explicit YAML ``false`` skips the retry loop
(no second ``ainvoke`` with FORCE_SAVE_PROMPT). Deterministic fallback is a
separate check (``chk.deterministic_profile_save``).

Not soft under ``mod.brain.check``.
"""

from __future__ import annotations

from apps.group_agent_api.agent_factory.checks.force_save_retry.ids import (
    CHECK_ID,
    PROTOCOL_NAME,
)
from apps.group_agent_api.agent_factory.checks.force_save_retry.module import (
    force_save_retry_enabled,
)

__all__ = [
    "CHECK_ID",
    "PROTOCOL_NAME",
    "force_save_retry_enabled",
]
