"""YAML-gated invite scaffold + polish Module (mod.brain.invite_copy).

Default on ≡ today's invite scaffold + optional LLM polish + per-candidate
enrich. Off → empty / withheld invite_text and no per-candidate copy fields.

Sub-flags (already in modules.yaml checks):
- ``chk.invite_scaffold`` — scaffold emission (this Module's hard gate)
- ``chk.invite_llm_polish`` — optional polish via ``llm_polish_enabled``
  (ENV ``GROUP_AGENT_LLM_POLISH`` still overrides when explicitly set)
"""

from __future__ import annotations

from typing import Any

from apps.group_agent_api.agent_factory.invite.ids import (
    CHECK_LLM_POLISH,
    CHECK_SCAFFOLD,
    MODULE_ID,
    PROTOCOL_NAME,
)
from apps.group_agent_api.agent_factory.invite_copy import InviteResult
from apps.group_agent_api.agent_factory.profile_schema import GroupProfile

__all__ = [
    "MODULE_ID",
    "CHECK_SCAFFOLD",
    "CHECK_LLM_POLISH",
    "PROTOCOL_NAME",
    "invite_copy_enabled",
    "invite_scaffold_enabled",
    "should_emit_invite_artifact",
    "generate_invite_with_optional_llm",
    "enrich_candidate_with_single_copy",
    "enrich_candidates_with_single_copy",
]


def invite_copy_enabled(*, enabled: bool | None = None) -> bool:
    """Master Module switch. Missing YAML key → on (≡ today's always-on)."""
    if enabled is not None:
        return bool(enabled)
    from apps.group_agent_api.agent_factory.module_config import load_modules_config

    cfg = load_modules_config()
    if MODULE_ID not in cfg.modules:
        return True
    return cfg.is_module_enabled(MODULE_ID)


def invite_scaffold_enabled(*, enabled: bool | None = None) -> bool:
    """Scaffold emission: Module on AND ``chk.invite_scaffold``.

    Missing check key → on when Module is on (preserve prior hard-on path).
    """
    if enabled is not None:
        return bool(enabled)
    if not invite_copy_enabled():
        return False
    from apps.group_agent_api.agent_factory.module_config import load_modules_config

    cfg = load_modules_config()
    if CHECK_SCAFFOLD not in cfg.checks:
        return True
    return cfg.is_check_enabled(CHECK_SCAFFOLD)


def should_emit_invite_artifact(
    *,
    match_status: str,
    match_reason: str | None,
    candidate_count: int,
) -> bool:
    """Chat/async gate: Module+scaffold YAML plus match/candidate rules."""
    if not invite_scaffold_enabled():
        return False
    from apps.group_agent_api.agent_factory.invite_copy import (
        should_emit_invite_artifact as _base,
    )

    return _base(
        match_status=match_status,
        match_reason=match_reason,
        candidate_count=candidate_count,
    )


def generate_invite_with_optional_llm(
    *,
    profile: GroupProfile,
    candidates: list[dict[str, Any]],
    match_status: str,
    willing_to_at: bool,
    user_id: str = "",
    group_id: str = "",
    model: Any | None = None,
    use_llm: bool | None = None,
    _broken_first_draft: bool = False,
) -> InviteResult:
    """Facade: off → empty invite_text; on → existing scaffold (+ optional polish)."""
    if not invite_scaffold_enabled():
        return InviteResult(
            kind="undirected",
            text="",
            topic="",
            match_status=match_status,
            willing_to_at=willing_to_at,
            mentioned_user_ids=[],
            elements=None,
            honest_note=None,
            ok=False,
            violations=["invite_scaffold_off"],
            assert_attempts=0,
            candidates=[],
        )

    from apps.group_agent_api.agent_factory.invite_llm import (
        generate_invite_with_optional_llm as _base,
    )

    return _base(
        profile=profile,
        candidates=candidates,
        match_status=match_status,
        willing_to_at=willing_to_at,
        user_id=user_id,
        group_id=group_id,
        model=model,
        use_llm=use_llm,
        _broken_first_draft=_broken_first_draft,
    )


def enrich_candidate_with_single_copy(
    candidate: dict[str, Any],
    profile: GroupProfile,
) -> dict[str, Any]:
    """Per-candidate four-field enrich; Module off → identity (no copy fields)."""
    if not invite_copy_enabled():
        return dict(candidate)
    from apps.group_agent_api.agent_factory.per_candidate_copy import (
        enrich_candidate_with_single_copy as _base,
    )

    return _base(candidate, profile)


def enrich_candidates_with_single_copy(
    candidates: list[dict[str, Any]],
    profile: GroupProfile,
) -> list[dict[str, Any]]:
    if not invite_copy_enabled():
        return [dict(c) for c in (candidates or [])]
    from apps.group_agent_api.agent_factory.per_candidate_copy import (
        enrich_candidates_with_single_copy as _base,
    )

    return _base(candidates, profile)
