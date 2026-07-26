"""POST /invite — FR-04/05/05B copy (+ optional LLM polish) on gated candidates."""

from __future__ import annotations

import asyncio

from fastapi import HTTPException, Request

from apps.group_agent_api.agent_factory.capability import unlocks_network
from apps.group_agent_api.agent_factory.guard import enforce_capability_guard
from apps.group_agent_api.agent_factory.integrations.config import integration_mode
from apps.group_agent_api.agent_factory.integrations.group_bind import (
    align_match_to_trusted_group,
)
from apps.group_agent_api.agent_factory.integrations.match_backend import run_match
from apps.group_agent_api.agent_factory.invite_llm import generate_invite_with_optional_llm
from apps.group_agent_api.agent_factory.match_stub import build_query_from_profile
from apps.group_agent_api.agent_factory.profile_store import load_profile
from apps.group_agent_api.app.models import InviteRequest, InviteResponse
from apps.group_agent_api.app.session import resolve_trusted_session
from apps.group_agent_api.app.state import AppState


def _empty_request() -> Request:
    return Request(
        {"type": "http", "method": "POST", "path": "/invite", "headers": []}
    )


async def invite(
    req: InviteRequest, state: AppState, request: Request | None = None
) -> InviteResponse:
    session = await resolve_trusted_session(
        request or _empty_request(),
        body_user_id=req.user_id,
        body_group_id=req.group_id,
        body_membership=req.membership,
        body_unionid=req.unionid,
        body_group_token=req.group_token,
        body_user_token=req.user_token,
    )
    user_id = session.principal.user_id
    group_id = session.group_id
    tier = session.membership.tier

    profile = await asyncio.to_thread(load_profile, state.base_dir, user_id, group_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="profile_missing")

    match_status = "skipped"
    candidates: list = []
    match_reason: str | None = None
    mode = integration_mode()

    if unlocks_network(tier):
        # HTTP: forbid caller-supplied candidates (bypass reachable pool).
        if req.candidates is not None and mode == "http":
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "candidates_injection_forbidden",
                    "message": "HTTP mode only accepts server run_match candidates",
                },
            )
        if req.candidates is not None and mode != "http":
            candidates = list(req.candidates)
            match_status = req.match_status or (
                "matched" if candidates else "empty"
            )
        else:
            query = (req.query or "").strip() or build_query_from_profile(profile)
            result = await asyncio.to_thread(
                run_match,
                query=query,
                group_id=group_id,
                excluded_ids=[user_id],
                group_token=session.group_token,
                user_bearer=session.principal.user_token,
            )
            aligned = align_match_to_trusted_group(result, trusted_group_id=group_id)
            match_status = aligned.status
            match_reason = aligned.reason
            candidates = aligned.candidates
    else:
        match_status = "skipped"
        match_reason = f"capability_{tier.value}_no_network"

    guarded = enforce_capability_guard(
        tier=tier,
        reply="",
        candidates=candidates,
        caller_group_id=group_id,
        user_id=user_id,
    )
    candidates = guarded.candidates
    if not unlocks_network(tier):
        match_status = "skipped"
    elif match_status == "matched" and not candidates:
        match_status = "empty"
        match_reason = "no_auditable_public_match_basis"

    invite_res = await asyncio.to_thread(
        generate_invite_with_optional_llm,
        profile=profile,
        candidates=candidates,
        match_status="empty" if not unlocks_network(tier) else match_status,
        willing_to_at=req.willing_to_at if unlocks_network(tier) else False,
        user_id=user_id,
        group_id=group_id,
        model=state.polish_model,
        use_llm=req.use_llm_polish,
    )

    return InviteResponse(
        user_id=user_id,
        group_id=group_id,
        capability=tier.value,  # type: ignore[arg-type]
        capability_source=session.membership.source,
        match_status=match_status,  # type: ignore[arg-type]
        match_reason=match_reason,
        candidates=candidates,
        delivery_kind=invite_res.kind,
        invite_text=invite_res.text,
        topic=invite_res.topic,
        mentioned_user_ids=invite_res.mentioned_user_ids,
        elements=invite_res.elements,
        honest_note=invite_res.honest_note,
        invite_ok=invite_res.ok,
        invite_violations=invite_res.violations,
        invite_assert_attempts=invite_res.assert_attempts,
        guard_blocked=guarded.blocked,
        guard_violations=guarded.violations,
        willing_to_at=req.willing_to_at,
    )
