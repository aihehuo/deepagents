"""GET profile — owner-guarded; tokens never in URL (REQ-007 FIX3)."""

from __future__ import annotations

from fastapi import HTTPException, Request

from apps.group_agent_api.agent_factory.integrations.principal import (
    HEADER_GROUP_TOKEN,
    HEADER_USER_TOKEN,
)
from apps.group_agent_api.agent_factory.profile_store import (
    load_profile,
    virtual_profile_path,
)
from apps.group_agent_api.app.models import ProfileQueryResponse
from apps.group_agent_api.app.session import resolve_trusted_session
from apps.group_agent_api.app.state import AppState


def _empty_request() -> Request:
    return Request(
        {"type": "http", "method": "GET", "path": "/profile", "headers": []}
    )


async def get_profile(
    *,
    user_id: str,
    group_id: str,
    state: AppState,
    request: Request | None = None,
    membership: str | None = "unknown",
    unionid: str | None = None,
) -> ProfileQueryResponse:
    """Read profile for the trusted owner.

    HTTP: `user_token` / `group_token` come only from signed headers
    (`X-GA-User-Token`, `X-GA-Group-Token`) — never from query string.
    Stub: membership/unionid kwargs for local tests only.
    """
    req = request or _empty_request()
    # Header-only credentials (FIX3). Do not accept query/body token kwargs.
    group_token = (req.headers.get(HEADER_GROUP_TOKEN) or "").strip() or None
    user_token = (req.headers.get(HEADER_USER_TOKEN) or "").strip() or None

    session = await resolve_trusted_session(
        req,
        body_user_id=user_id,
        body_group_id=group_id,
        body_membership=membership,
        body_unionid=unionid,
        body_group_token=group_token,
        body_user_token=user_token,
    )
    uid = session.principal.user_id
    gid = session.group_id

    # Cross-user / cross-group: plaintext request must not target another owner.
    req_uid = (user_id or "").strip()
    req_gid = (group_id or "").strip()
    if req_uid and req_uid != uid:
        raise HTTPException(
            status_code=403,
            detail={"error": "profile_owner_forbidden", "field": "user_id"},
        )
    if req_gid and req_gid != gid:
        raise HTTPException(
            status_code=403,
            detail={"error": "profile_owner_forbidden", "field": "group_id"},
        )

    path = virtual_profile_path(uid, gid)
    profile = load_profile(state.base_dir, uid, gid)
    return ProfileQueryResponse(
        user_id=uid,
        group_id=gid,
        exists=profile is not None,
        profile=profile.model_dump(mode="json") if profile else None,
        path=path,
    )
