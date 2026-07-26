"""Shared HTTP-mode session resolution for chat/match/invite (REQ-007 FIX)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from fastapi import HTTPException, Request

from apps.group_agent_api.agent_factory.integrations.config import integration_mode
from apps.group_agent_api.agent_factory.integrations.group_bind import (
    resolve_trusted_group_id,
)
from apps.group_agent_api.agent_factory.integrations.membership_backend import (
    resolve_session_capability,
)
from apps.group_agent_api.agent_factory.integrations.membership_client import (
    MembershipResult,
)
from apps.group_agent_api.agent_factory.integrations.principal import (
    SessionPrincipal,
    resolve_session_principal,
)
from apps.group_agent_api.agent_factory.profile_store import (
    ProfileStoreError,
    validate_id,
)


@dataclass(frozen=True)
class TrustedSession:
    principal: SessionPrincipal
    membership: MembershipResult
    group_id: str
    group_token: str | None


async def resolve_trusted_session(
    request: Request,
    *,
    body_user_id: str | None,
    body_group_id: str | None,
    body_membership: str | None,
    body_unionid: str | None,
    body_group_token: str | None,
    body_user_token: str | None,
) -> TrustedSession:
    """Resolve principal + membership + trusted group_id off the event loop."""

    def _sync() -> TrustedSession:
        principal = resolve_session_principal(
            request,
            body_user_id=body_user_id,
            body_unionid=body_unionid,
            body_user_token=body_user_token,
        )
        try:
            user_id = validate_id(principal.user_id, field="user_id")
        except ProfileStoreError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        # Rebuild principal with validated id
        principal = SessionPrincipal(
            user_id=user_id,
            unionid=principal.unionid,
            user_token=principal.user_token,
            group_token=principal.group_token,
            source=principal.source,
        )

        mode = integration_mode()
        unionid = principal.unionid if mode == "http" else (body_unionid or principal.unionid)
        # HTTP: prefer signed header group token; body OK for POST JSON (not URL).
        if mode == "http":
            hdr_gt = (principal.group_token or "").strip() or None
            body_gt = (body_group_token or "").strip() or None
            if hdr_gt and body_gt and hdr_gt != body_gt:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "group_token_conflict",
                        "message": "body group_token must match X-GA-Group-Token",
                    },
                )
            group_token = hdr_gt or body_gt
        else:
            group_token = body_group_token
        try:
            plain_gid = validate_id(body_group_id or "", field="group_id") if (
                body_group_id or ""
            ).strip() else ""
        except ProfileStoreError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        membership = resolve_session_capability(
            membership_override=body_membership,
            unionid=unionid,
            group_token=group_token,
            group_id=plain_gid,
            user_id=user_id,
        )

        if mode != "http":
            if not plain_gid:
                raise HTTPException(status_code=400, detail="missing_group_id")
            group_id = plain_gid
        else:
            group_id = resolve_trusted_group_id(
                plaintext_group_id=plain_gid,
                membership=membership,
            )
            try:
                group_id = validate_id(group_id, field="group_id")
            except ProfileStoreError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        return TrustedSession(
            principal=principal,
            membership=membership,
            group_id=group_id,
            group_token=group_token,
        )

    return await asyncio.to_thread(_sync)
