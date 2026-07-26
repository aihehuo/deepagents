"""Bind plaintext group_id to membership.event_id (REQ-007 FIX)."""

from __future__ import annotations

import logging

from fastapi import HTTPException

from apps.group_agent_api.agent_factory.capability import unlocks_network
from apps.group_agent_api.agent_factory.disclosure import (
    public_match_basis,
    stable_candidate_user_id,
)
from apps.group_agent_api.agent_factory.integrations.config import integration_mode
from apps.group_agent_api.agent_factory.integrations.membership_client import (
    MembershipResult,
)
from apps.group_agent_api.agent_factory.match_stub import MatchResult

_logger = logging.getLogger("uvicorn.error")


def resolve_trusted_group_id(
    *,
    plaintext_group_id: str,
    membership: MembershipResult,
    force_mode: str | None = None,
) -> str:
    """HTTP: membership.event_id is the sole trusted group id.

    Plaintext group_id must match event_id when both present; mismatch → 400.
    Stub: plaintext group_id wins.
    """
    mode = (force_mode or integration_mode()).strip().lower()
    plain = (plaintext_group_id or "").strip()
    if mode != "http":
        if not plain:
            raise HTTPException(
                status_code=400,
                detail={"error": "missing_group_id"},
            )
        return plain

    event_id = (membership.event_id or "").strip() or None
    if event_id:
        if plain and plain != event_id:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "group_id_mismatch",
                    "plaintext_group_id": plain,
                    "event_id": event_id,
                },
            )
        return event_id

    # No event_id from micro: never unlock network on this path.
    if unlocks_network(membership.tier):
        raise HTTPException(
            status_code=502,
            detail={
                "error": "membership_missing_event_id",
                "tier": membership.tier.value,
            },
        )
    if not plain:
        raise HTTPException(
            status_code=400,
            detail={"error": "missing_group_id", "message": "no event_id and no plaintext"},
        )
    return plain


def align_match_to_trusted_group(
    result: MatchResult,
    *,
    trusted_group_id: str,
) -> MatchResult:
    """Drop / empty match if new_api group_id disagrees with trusted event_id."""
    reported = (result.group_id or "").strip()
    if reported and reported != trusted_group_id:
        return MatchResult(
            status="empty",
            candidates=[],
            query=result.query,
            group_id=trusted_group_id,
            reason=f"group_id_mismatch:match={reported}:trusted={trusted_group_id}",
        )
    # Filter candidates whose source/group disagrees
    safe = []
    rejected_for_missing_basis = False
    rejected_for_missing_id = False
    seen_ids: set[str] = set()
    accepted_ids: set[str] = set()
    identity_violations: list[str] = []
    for c in result.candidates or []:
        user_id = stable_candidate_user_id(c)
        if user_id is None:
            rejected_for_missing_id = True
            identity_violations.append("missing_candidate_id")
            continue
        if user_id in seen_ids:
            identity_violations.append(f"duplicate_candidate_id:{user_id}")
        seen_ids.add(user_id)
        src = str(c.get("source_group_id") or c.get("group_id") or "")
        if src and src != trusted_group_id:
            continue
        # Force source to trusted id for downstream guard
        item = dict(c)
        item["group_id"] = trusted_group_id
        item["source_group_id"] = trusted_group_id
        if not public_match_basis(item):
            rejected_for_missing_basis = True
            continue
        if user_id in accepted_ids:
            continue
        accepted_ids.add(user_id)
        safe.append(item)
    if identity_violations:
        _logger.warning(
            "action=candidate_identity_gate violations=%s",
            ",".join(identity_violations),
        )
    status = result.status
    reason = result.reason
    if status == "matched" and not safe:
        status = "empty"
        if rejected_for_missing_basis:
            reason = "no_auditable_public_match_basis"
        elif rejected_for_missing_id:
            reason = "no_stable_candidate_id"
    return MatchResult(
        status=status,  # type: ignore[arg-type]
        candidates=safe,
        query=result.query,
        group_id=trusted_group_id,
        reason=reason,
    )
