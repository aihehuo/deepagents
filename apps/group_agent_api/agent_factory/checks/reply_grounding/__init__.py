"""mod.brain.reply_grounding — semantic reply-vs-ground check (TSD-14 §4.6).

Switchable Module. When disabled, the orchestrator skips this package entirely.
Does not rewrite user-facing sentences; fail returns check_deny for repair.
"""

from apps.group_agent_api.agent_factory.checks.reply_grounding.deny import (
    format_check_deny,
)
from apps.group_agent_api.agent_factory.checks.reply_grounding.gate import (
    ReplyGroundingGateResult,
    apply_reply_grounding_gate,
    default_repair_fn,
)
from apps.group_agent_api.agent_factory.checks.reply_grounding.ground import (
    build_ground_from_turn,
)
from apps.group_agent_api.agent_factory.checks.reply_grounding.ids import (
    CHECK_ID,
    MODULE_ID,
    PROTOCOL_NAME,
)
from apps.group_agent_api.agent_factory.checks.reply_grounding.module import (
    check_reply_grounding,
)
from apps.group_agent_api.agent_factory.checks.reply_grounding.protocol import (
    CandidateGround,
    FactItem,
    GroundBlock,
    MatchEvidenceItem,
    ReplyGroundingInput,
    ReplyGroundingOutput,
    RepairableBy,
    Verdict,
)

__all__ = [
    "MODULE_ID",
    "CHECK_ID",
    "PROTOCOL_NAME",
    "CandidateGround",
    "FactItem",
    "GroundBlock",
    "MatchEvidenceItem",
    "ReplyGroundingInput",
    "ReplyGroundingOutput",
    "RepairableBy",
    "Verdict",
    "ReplyGroundingGateResult",
    "apply_reply_grounding_gate",
    "build_ground_from_turn",
    "check_reply_grounding",
    "default_repair_fn",
    "format_check_deny",
]
