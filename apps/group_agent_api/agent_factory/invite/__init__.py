"""mod.brain.invite_copy — YAML Module facade over invite scaffold / polish / enrich.

Default on ≡ today's behavior. Off → empty invite_text and no per-candidate
copy fields. Sub-checks: ``chk.invite_scaffold``, ``chk.invite_llm_polish``.
"""

from __future__ import annotations

from apps.group_agent_api.agent_factory.invite.ids import (
    CHECK_LLM_POLISH,
    CHECK_SCAFFOLD,
    MODULE_ID,
    PROTOCOL_NAME,
)
from apps.group_agent_api.agent_factory.invite.module import (
    enrich_candidate_with_single_copy,
    enrich_candidates_with_single_copy,
    generate_invite_with_optional_llm,
    invite_copy_enabled,
    invite_scaffold_enabled,
    should_emit_invite_artifact,
)

__all__ = [
    "CHECK_LLM_POLISH",
    "CHECK_SCAFFOLD",
    "MODULE_ID",
    "PROTOCOL_NAME",
    "enrich_candidate_with_single_copy",
    "enrich_candidates_with_single_copy",
    "generate_invite_with_optional_llm",
    "invite_copy_enabled",
    "invite_scaffold_enabled",
    "should_emit_invite_artifact",
]
