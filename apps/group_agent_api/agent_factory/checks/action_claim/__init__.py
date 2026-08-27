"""chk.action_claim — block unauthorized group-send / @ / admin-notify claims.

Thin YAML-gated facade over ``content_quality.guard_action_claims``. Detector
helpers stay in ``content_quality`` so reply_grounding L0
(``unverified_action``) can keep sharing them without duplicating regex.

Overlap with ``mod.brain.reply_grounding`` L0:
- This check (when on): early silent replace with capability-boundary copy.
- RG L0: fail → ``ctx.check_deny`` → same-turn rewrite.
Both may run when both YAML bits are on; do **not** relocate the RG gate.
Off = identity (orchestrator keeps model text).
"""

from __future__ import annotations

from apps.group_agent_api.agent_factory.checks.action_claim.ids import (
    CHECK_ID,
    PROTOCOL_NAME,
)
from apps.group_agent_api.agent_factory.checks.action_claim.module import (
    apply_action_claim_guard,
    action_claim_enabled,
)

__all__ = [
    "CHECK_ID",
    "PROTOCOL_NAME",
    "action_claim_enabled",
    "apply_action_claim_guard",
]
