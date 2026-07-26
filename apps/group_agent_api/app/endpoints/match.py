"""POST /match — FR-03 match under capability + disclosure guards (stub or HTTP)."""

from __future__ import annotations

import asyncio

from fastapi import HTTPException, Request

from apps.group_agent_api.agent_factory.capability import unlocks_network
from apps.group_agent_api.agent_factory.guard import enforce_capability_guard
from apps.group_agent_api.agent_factory.integrations.group_bind import (
    align_match_to_trusted_group,
)
from apps.group_agent_api.agent_factory.integrations.match_backend import run_match
from apps.group_agent_api.agent_factory.match_stub import build_query_from_profile
from apps.group_agent_api.agent_factory.profile_store import load_profile
from apps.group_agent_api.app.models import MatchRequest, MatchResponse
from apps.group_agent_api.app.session import resolve_trusted_session
from apps.group_agent_api.app.state import AppState


def _empty_request() -> Request:
    return Request(
        {"type": "http", "method": "POST", "path": "/match", "headers": []}
    )


async def match(
    req: MatchRequest, state: AppState, request: Request | None = None
) -> MatchResponse:
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

    if not unlocks_network(tier):
        guarded = enforce_capability_guard(
            tier=tier,
            reply="",
            candidates=[],
            caller_group_id=group_id,
            user_id=user_id,
        )
        return MatchResponse(
            user_id=user_id,
            group_id=group_id,
            capability=tier.value,  # type: ignore[arg-type]
            capability_source=session.membership.source,
            match_status="skipped",
            candidates=[],
            match_reason=f"capability_{tier.value}_no_network",
            query="",
            guard_blocked=guarded.blocked,
            guard_violations=guarded.violations,
        )

    profile = await asyncio.to_thread(load_profile, state.base_dir, user_id, group_id)
    query = (req.query or "").strip()
    if not query:
        if profile is None:
            raise HTTPException(
                status_code=404,
                detail="profile_missing: save profile first or pass query",
            )
        query = build_query_from_profile(profile)

    excluded = list(req.excluded_ids or [])
    if user_id not in excluded:
        excluded.append(user_id)

    result = await asyncio.to_thread(
        run_match,
        query=query,
        group_id=group_id,
        excluded_ids=excluded,
        group_token=session.group_token,
        user_bearer=session.principal.user_token,
    )
    aligned = align_match_to_trusted_group(result, trusted_group_id=group_id)
    guarded = enforce_capability_guard(
        tier=tier,
        reply="",
        candidates=aligned.candidates,
        caller_group_id=group_id,
        user_id=user_id,
    )
    return MatchResponse(
        user_id=user_id,
        group_id=group_id,
        capability=tier.value,  # type: ignore[arg-type]
        capability_source=session.membership.source,
        match_status=(
            aligned.status
            if guarded.candidates or aligned.status == "empty"
            else "empty"
        ),
        candidates=guarded.candidates,
        match_reason=aligned.reason,
        query=query,
        guard_blocked=guarded.blocked,
        guard_violations=guarded.violations,
    )
