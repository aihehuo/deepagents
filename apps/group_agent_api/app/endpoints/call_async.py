"""Async endpoint POST /call_async (REQ-009 / RESP-009-FIX5 / REQ-032)."""

from __future__ import annotations

import asyncio

from fastapi import HTTPException, Request

from apps.group_agent_api.agent_factory.integrations.callback_client import (
    validate_and_normalize_callback_url,
)
from apps.group_agent_api.app.async_manager import (
    calculate_request_fingerprint,
    complete_idempotency_reservation,
    execute_async_run,
    reserve_idempotency_slot,
    rollback_idempotency_reservation,
)
from apps.group_agent_api.app.models import AsyncCallRequest, AsyncCallResponse
from apps.group_agent_api.app.session import resolve_trusted_session
from apps.group_agent_api.app.state import AppState
from apps.group_agent_api.app.utils import thread_id
from apps.group_agent_api.execution.admission import admit_durable_async_call
from apps.group_agent_api.execution.broker import enqueue_delivery
from apps.group_agent_api.execution.config import durable_queue_enabled


async def call_async(
    req: AsyncCallRequest,
    state: AppState,
    request: Request,
) -> AsyncCallResponse:
    """Async endpoint: validates principal & SSRF, then durable or legacy admission."""
    if req.request_schema_version == 5:
        from apps.group_agent_api.agent_factory.integrations.config import integration_mode
        if integration_mode() != "http":
            raise HTTPException(503, detail={"error":"assistant_requires_signed_transport"})
    # 1. Validate and normalize callback_url against SSRF allowlist
    canonical_url = validate_and_normalize_callback_url(req.callback_url)
    if req.request_schema_version in {2, 3, 4, 5} and canonical_url != req.callback_url:
        raise HTTPException(status_code=400, detail={"error": "schema_invalid", "message": "typed callback_url must already be canonical"})
    req.callback_url = canonical_url
    raw_body = await request.body() if req.request_schema_version in {2, 3, 4, 5} else None
    if raw_body is not None and len(raw_body) > 128 * 1024:
        raise HTTPException(status_code=400, detail={"error": "schema_invalid", "message": "request budget exceeded"})
    if req.request_schema_version == 3:
        from apps.group_agent_api.app.execution_modes import validate_v3_wire

        try:
            validate_v3_wire(raw_body, req)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail={"error": "schema_invalid", "message": "invalid v3 wire body"}) from exc
    if req.request_schema_version == 4:
        from apps.group_agent_api.app.execution_modes import validate_v4_wire

        try:
            validate_v4_wire(raw_body, req)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail={"error": "schema_invalid", "message": "invalid v4 wire body"}) from exc

    # 2. Resolve trusted session (validates GA-PRINCIPAL-V1 headers vs body fields)
    session = await resolve_trusted_session(
        request,
        body_user_id=req.user_id,
        body_group_id=req.group_id,
        body_membership=req.membership,
        body_unionid=req.unionid,
        body_group_token=req.group_token,
        body_user_token=req.user_token,
        request_schema_version=req.request_schema_version or 1,
        request_body=raw_body,
    )

    # Typed execution modes are admitted only through the encrypted durable ledger. Never fall
    # back to the process-local legacy adapter when durable execution is off.
    if req.request_schema_version in {3, 4, 5}:
        try:
            calculate_request_fingerprint(req, session)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail={"error": "schema_invalid", "message": "frozen identity differs from principal"}) from exc
        if not durable_queue_enabled():
            raise HTTPException(status_code=503, detail={"error": "execution_mode_unavailable", "message": "typed execution requires durable execution"})

    tid = thread_id(
        user_id=session.principal.user_id,
        group_id=session.group_id,
        conversation_id=req.conversation_id,
        execution_mode=req.execution_mode if req.request_schema_version in {3, 4} else None,
        episode_id=req.episode_id if req.request_schema_version in {2, 3, 4, 5} else (
            str((req.metadata or {}).get("episode_id") or (req.metadata or {}).get("episodeId") or "").strip()
            or None
        ),
    )

    # REQ-032 durable path: no process-local idempotency / create_task.
    if durable_queue_enabled():
        if state.durable_store is None or state.durable_config is None or state.backpressure is None:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "queue_unavailable",
                    "message": "durable queue not initialized",
                },
                headers={"Retry-After": "5"},
            )

        def _enqueue(delivery):  # type: ignore[no-untyped-def]
            enqueue_delivery(state.durable_config, delivery)

        result = admit_durable_async_call(
            req=req,
            session=session,
            thread_id=tid,
            store=state.durable_store,
            config=state.durable_config,
            backpressure=state.backpressure,
            enqueue=_enqueue,
        )
        return result.response

    # --- Legacy process-local path (GROUP_AGENT_DURABLE_QUEUE_ENABLED=0) ---
    if req.request_schema_version in {2, 3, 4, 5}:
        raise HTTPException(status_code=503, detail={"error": "queue_unavailable", "message": "v2 requires durable execution"})
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
