"""Async endpoint POST /call_async (REQ-009 / RESP-009-FIX5)."""

from __future__ import annotations

import asyncio
from fastapi import HTTPException, Request

from apps.group_agent_api.agent_factory.integrations.callback_client import (
    validate_and_normalize_callback_url,
)
from apps.group_agent_api.app.async_manager import (
    complete_idempotency_reservation,
    execute_async_run,
    reserve_idempotency_slot,
    rollback_idempotency_reservation,
)
from apps.group_agent_api.app.models import AsyncCallRequest, AsyncCallResponse
from apps.group_agent_api.app.session import resolve_trusted_session
from apps.group_agent_api.app.state import AppState
from apps.group_agent_api.app.utils import thread_id


async def call_async(
    req: AsyncCallRequest,
    state: AppState,
    request: Request,
) -> AsyncCallResponse:
    """Async endpoint: validates principal & SSRF, reserves idempotency slot, checks active locks, returns 202 ACK on successful commit."""
    # 1. Validate and normalize callback_url against SSRF allowlist
    canonical_url = validate_and_normalize_callback_url(req.callback_url)
    req.callback_url = canonical_url

    # 2. Resolve trusted session (validates GA-PRINCIPAL-V1 headers vs body fields)
    session = await resolve_trusted_session(
        request,
        body_user_id=req.user_id,
        body_group_id=req.group_id,
        body_membership=req.membership,
        body_unionid=req.unionid,
        body_group_token=req.group_token,
        body_user_token=req.user_token,
    )

    # 3. Reserve idempotency slot atomically with TrustedSession fingerprint binding
    status, cached, slot = await reserve_idempotency_slot(req, session)

    if status == "HIT" and cached is not None:
        return cached
    if status == "CONFLICT":
        raise HTTPException(
            status_code=409,
            detail={
                "error": "idempotency_conflict",
                "message": "Idempotency key, run_id, or request fingerprint conflicts with an existing distinct request",
            },
        )
    if status == "INITIALIZING":
        raise HTTPException(
            status_code=425,
            detail={
                "error": "request_initializing",
                "message": "Request initialization in progress, please retry",
            },
        )

    assert slot is not None

    tid = thread_id(
        user_id=session.principal.user_id,
        group_id=session.group_id,
        conversation_id=req.conversation_id,
    )

    # 4. Check global active limit & per-conversation lock
    if not state.try_start_agent_run(tid, "call_async"):
        await rollback_idempotency_reservation(slot)
        raise HTTPException(
            status_code=409,
            detail={
                "error": "run_in_progress",
                "message": "Agent run already in progress for this conversation or global max active limit reached",
                "thread_id": tid,
            },
        )

    # 5. Build response
    resp = AsyncCallResponse(
        success=True,
        run_id=req.run_id,
        session_id=tid,
        accepted=True,
        message="accepted",
    )

    # 6. Task creation, registration, commit, and error/cancellation compensation
    task: asyncio.Task[None] | None = None
    committed = False
    try:
        task = asyncio.create_task(
            execute_async_run(
                req=req,
                session=session,
                state=state,
                tid=tid,
                slot=slot,
            )
        )
        state.register_task(tid, task)
        committed = await complete_idempotency_reservation(slot, resp)
        if not committed:
            raise RuntimeError("Idempotency reservation commit returned False")
    except (asyncio.CancelledError, Exception) as exc:
        await rollback_idempotency_reservation(slot)
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        with state.active_agent_runs_lock:
            state.active_tasks.pop(tid, None)
        state.finish_agent_run(tid, "call_async")
        if isinstance(exc, asyncio.CancelledError):
            raise
        raise HTTPException(
            status_code=500,
            detail={"error": "idempotency_commit_failed", "message": "Failed to commit idempotency reservation"},
        ) from exc

    return resp
