"""chk.deterministic_profile_save — regex/natural extract → tool invoke fallback.

Hang: ``async_manager._attempt_deterministic_profile_save`` (also called from
``chat.py``) after FORCE_SAVE retries leave profile unpersisted.

Default **on**. Explicit YAML ``false`` skips the harness save.
Not soft under ``mod.brain.check``.
"""

from __future__ import annotations

from apps.group_agent_api.agent_factory.checks.deterministic_profile_save.ids import (
    CHECK_ID,
    PROTOCOL_NAME,
)
from apps.group_agent_api.agent_factory.checks.deterministic_profile_save.module import (
    deterministic_profile_save_enabled,
)

__all__ = [
    "CHECK_ID",
    "PROTOCOL_NAME",
    "deterministic_profile_save_enabled",
]
