"""chk.finalize_templates — confirmation + next_step template assembly.

When **on** (default / ``current`` preset): ``finalize_user_visible_reply``
may stack profile confirmation + authoritative match/invite next-step copy
(keep-DA already prefers substantive model wording when not stub/pending).

When **off** (``model_voice`` / keep-DA-reply): return the hard-scrubbed DA
reply only — **no** confirmation / next_step templates. Do **not** reintroduce
Micro Renderer.

Scrub / capability / reply_grounding remain separately switchable.
"""

from __future__ import annotations

from apps.group_agent_api.agent_factory.checks.finalize_templates.ids import (
    CHECK_ID,
    PROTOCOL_NAME,
)
from apps.group_agent_api.agent_factory.checks.finalize_templates.module import (
    finalize_templates_enabled,
)

__all__ = [
    "CHECK_ID",
    "PROTOCOL_NAME",
    "finalize_templates_enabled",
]
