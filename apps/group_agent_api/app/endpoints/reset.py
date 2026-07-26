"""Reset conversation (and optionally clear profile) — owner-guarded (REQ-007 FIX2)."""

from __future__ import annotations

import logging

from fastapi import HTTPException, Request

from apps.group_agent_api.agent_factory.profile_store import disk_profile_path
from apps.group_agent_api.app.models import ResetRequest, ResetResponse
from apps.group_agent_api.app.session import resolve_trusted_session
from apps.group_agent_api.app.state import AppState
from apps.group_agent_api.app.utils import get_agent_checkpointer, thread_id

_logger = logging.getLogger("uvicorn.error")


def _empty_request() -> Request:
    return Request(
        {"type": "http", "method": "POST", "path": "/reset", "headers": []}
    )


async def reset(
    req: ResetRequest, state: AppState, request: Request | None = None
) -> ResetResponse:
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

    req_uid = (req.user_id or "").strip()
    req_gid = (req.group_id or "").strip()
    if req_uid and req_uid != user_id:
        raise HTTPException(
            status_code=403,
            detail={"error": "reset_owner_forbidden", "field": "user_id"},
        )
    if req_gid and req_gid != group_id:
        raise HTTPException(
            status_code=403,
            detail={"error": "reset_owner_forbidden", "field": "group_id"},
        )

    tid = thread_id(
        user_id=user_id, group_id=group_id, conversation_id=req.conversation_id
    )
    checkpointer = get_agent_checkpointer(state.agent)
    if checkpointer is not None and hasattr(checkpointer, "delete_thread"):
        checkpointer.delete_thread(tid)
    state.thread_locks.pop(tid, None)

    profile_cleared = False
    if req.clear_profile:
        path = disk_profile_path(state.base_dir, user_id, group_id)
        if path.exists():
            path.unlink()
            profile_cleared = True
            _logger.info("Cleared profile %s", path)

    return ResetResponse(
        user_id=user_id,
        group_id=group_id,
        conversation_id=req.conversation_id,
        thread_id=tid,
        ok=True,
        profile_cleared=profile_cleared,
    )
