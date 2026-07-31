"""Durable admission for POST /call_async (REQ-032-FIX1)."""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from fastapi import HTTPException
from fastapi.responses import JSONResponse

from apps.group_agent_api.app.models import AsyncCallRequest, AsyncCallResponse
from apps.group_agent_api.app.session import TrustedSession
from apps.group_agent_api.execution.backpressure import BackpressureController
from apps.group_agent_api.execution.config import (
    DurableQueueConfig,
    validate_request_fingerprint,
)
from apps.group_agent_api.execution.crypto import digest_id, encrypt_envelope
from apps.group_agent_api.execution.models import (
    QUEUE_SCHEMA_VERSION,
    REQUEST_SCHEMA_VERSION,
    BrokerDeliveryRef,
    EncryptedPayload,
    ExecutionRecord,
    ExecutionStatus,
)
from apps.group_agent_api.execution.redis_store import ExecutionStore, ExecutionStoreError

_logger = logging.getLogger("uvicorn.error")

EnqueueFn = Callable[[BrokerDeliveryRef], None]


@dataclass
class AdmissionResult:
    response: AsyncCallResponse
    created: bool
    execution_status: str


def build_trusted_envelope(
    *,
    req: AsyncCallRequest,
    session: TrustedSession,
    thread_id: str,
) -> dict[str, Any]:
    """Serialize trusted request + session for encrypted ledger storage."""
    return {
        "thread_id": thread_id,
        "request": req.model_dump(),
        "session": {
            "user_id": session.principal.user_id,
            "unionid": session.principal.unionid,
            "user_token": session.principal.user_token,
            "group_token": session.group_token,
            "group_id": session.group_id,
            "membership_tier": session.membership.tier.value,
            "membership_source": session.membership.source,
            "principal_source": session.principal.source,
        },
    }


def _error_response(
    status_code: int,
    error: str,
    message: str,
    *,
    retry_after: int | None = None,
) -> None:
    headers = {}
    if retry_after is not None:
        headers["Retry-After"] = str(retry_after)
    raise HTTPException(
        status_code=status_code,
        detail={"error": error, "message": message},
        headers=headers or None,
    )


def validate_durable_request_fields(req: AsyncCallRequest) -> tuple[int, str]:
    """Validate Micro fingerprint fields required in durable mode."""
    schema = getattr(req, "request_schema_version", None)
    fingerprint = getattr(req, "request_fingerprint", None)
    queue_schema = getattr(req, "queue_schema_version", None)

    if schema is None or fingerprint is None or queue_schema is None:
        _error_response(
            400,
            "durable_fields_required",
            "request_schema_version, request_fingerprint, queue_schema_version required",
        )
    if int(schema) != REQUEST_SCHEMA_VERSION:
        _error_response(400, "schema_invalid", "request_schema_version must be 1")
    if int(queue_schema) != QUEUE_SCHEMA_VERSION:
        _error_response(400, "schema_invalid", "queue_schema_version must be 1")
    try:
        fp = validate_request_fingerprint(str(fingerprint))
    except ValueError:
        _error_response(400, "schema_invalid", "request_fingerprint must be 64-char lowercase sha256 hex")
    return int(schema), fp


def admit_durable_async_call(
    *,
    req: AsyncCallRequest,
    session: TrustedSession,
    thread_id: str,
    store: ExecutionStore,
    config: DurableQueueConfig,
    backpressure: BackpressureController,
    enqueue: EnqueueFn,
) -> AdmissionResult:
    """Durable admission: hit-first, then quota, publish, mark queued.

    Never returns 202 unless execution record exists and broker accepted.
    Idempotent hits are not blocked by admission quota.
    """
    schema, fingerprint = validate_durable_request_fields(req)
    provider = (os.environ.get("GROUP_AGENT_PROVIDER") or "default").strip() or "default"

    # FIX1 BLOCKER-4: binding hit before quota — saturated system still returns original ACK.
    existing = store.get_by_idempotency(req.idempotency_key)
    if existing is not None:
        if (
            existing.request_schema_version != schema
            or existing.request_fingerprint != fingerprint
            or existing.run_id != req.run_id
        ):
            _error_response(409, "idempotency_conflict", "idempotency or run binding conflict")
        if existing.status in {ExecutionStatus.ENQUEUE_FAILED, ExecutionStatus.ACCEPTED}:
            return _enqueue_and_ack(
                req=req,
                thread_id=thread_id,
                store=store,
                loaded=existing,
                enqueue=enqueue,
                created=False,
                release_on_fail=False,
                backpressure=backpressure,
                session=session,
                reserved=False,
            )
        return AdmissionResult(
            response=_ack(req, thread_id, existing.status.value),
            created=False,
            execution_status=existing.status.value,
        )

    by_run = store.get(req.run_id)
    if by_run is not None and by_run.idempotency_key != req.idempotency_key:
        _error_response(409, "run_binding_conflict", "run_id bound to different idempotency_key")

    bp = backpressure.check_and_reserve(
        user_id=session.principal.user_id,
        group_id=session.group_id,
        conversation_id=req.conversation_id,
        provider=provider,
    )
    if not bp.allowed:
        _error_response(
            int(bp.http_status or 503),
            bp.error_code or "queue_saturated",
            bp.reason or "backpressure",
            retry_after=bp.retry_after_s,
        )

    reserved = True
    try:
        envelope = build_trusted_envelope(req=req, session=session, thread_id=thread_id)
        ciphertext: EncryptedPayload = encrypt_envelope(
            envelope,
            key=config.current_payload_key,
            key_version=config.payload_current_version,
            run_id=req.run_id,
            idempotency_key=req.idempotency_key,
            request_fingerprint=fingerprint,
            schema_version=schema,
        )
        now = store.redis_time()
        record = ExecutionRecord(
            run_id=req.run_id,
            idempotency_key=req.idempotency_key,
            request_schema_version=schema,
            request_fingerprint=fingerprint,
            queue_schema_version=QUEUE_SCHEMA_VERSION,
            status=ExecutionStatus.ACCEPTED,
            created_at=now,
            payload_key_version=ciphertext.key_version,
            payload_ciphertext=ciphertext,
            conversation_id=req.conversation_id,
            user_id_digest=digest_id(session.principal.user_id),
            group_id_digest=digest_id(session.group_id),
            provider=provider,
        )
        try:
            kind, loaded = store.create_or_get(record=record)
        except ExecutionStoreError as exc:
            backpressure.release_queued_reservation(
                user_id=session.principal.user_id,
                group_id=session.group_id,
            )
            reserved = False
            if exc.code in {"idempotency_conflict", "run_binding_conflict"}:
                _error_response(409, exc.code, "idempotency or run binding conflict")
            _error_response(503, "queue_unavailable", "execution ledger unavailable", retry_after=5)

        if kind == "hit":
            backpressure.release_queued_reservation(
                user_id=session.principal.user_id,
                group_id=session.group_id,
            )
            reserved = False
            if loaded.status in {ExecutionStatus.ENQUEUE_FAILED, ExecutionStatus.ACCEPTED}:
                return _enqueue_and_ack(
                    req=req,
                    thread_id=thread_id,
                    store=store,
                    loaded=loaded,
                    enqueue=enqueue,
                    created=False,
                    release_on_fail=False,
                    backpressure=backpressure,
                    session=session,
                    reserved=False,
                )
            return AdmissionResult(
                response=_ack(req, thread_id, loaded.status.value),
                created=False,
                execution_status=loaded.status.value,
            )

        return _enqueue_and_ack(
            req=req,
            thread_id=thread_id,
            store=store,
            loaded=loaded,
            enqueue=enqueue,
            created=True,
            release_on_fail=True,
            backpressure=backpressure,
            session=session,
            reserved=True,
        )
    except HTTPException:
        if reserved:
            backpressure.release_queued_reservation(
                user_id=session.principal.user_id,
                group_id=session.group_id,
            )
        raise
    except Exception:
        if reserved:
            backpressure.release_queued_reservation(
                user_id=session.principal.user_id,
                group_id=session.group_id,
            )
        _logger.error("durable_admission_error run_id=%s", req.run_id)
        _error_response(503, "queue_unavailable", "admission failed", retry_after=5)


def _ack(req: AsyncCallRequest, thread_id: str, status: str) -> AsyncCallResponse:
    return AsyncCallResponse(
        success=True,
        run_id=req.run_id,
        session_id=thread_id,
        accepted=True,
        message="accepted",
        idempotency_key=req.idempotency_key,
        execution_status=status,
        queue_schema_version=QUEUE_SCHEMA_VERSION,
    )


def _enqueue_and_ack(
    *,
    req: AsyncCallRequest,
    thread_id: str,
    store: ExecutionStore,
    loaded: ExecutionRecord,
    enqueue: EnqueueFn,
    created: bool,
    release_on_fail: bool,
    backpressure: BackpressureController,
    session: TrustedSession,
    reserved: bool,
) -> AdmissionResult:
    """Publish first; mark queued only after broker acceptance."""
    expected = loaded.status.value
    if expected not in {"accepted", "enqueue_failed"}:
        expected = "accepted"
    delivery = BrokerDeliveryRef(
        queue_schema_version=QUEUE_SCHEMA_VERSION,
        run_id=loaded.run_id,
        idempotency_key=loaded.idempotency_key,
        request_fingerprint=loaded.request_fingerprint,
        delivery_id=str(uuid.uuid4()),
    )
    # FIX3: admission shares publish ownership with recovery.
    ownership = store.claim_publish_delivery(
        loaded.run_id,
        expected_status=expected,
        delivery_id=delivery.delivery_id,
    )
    if ownership == "busy":
        # Another publisher owns this cycle — treat as accepted-in-flight.
        return AdmissionResult(
            response=_ack(req, thread_id, loaded.status.value),
            created=created,
            execution_status=loaded.status.value,
        )
    if ownership != "ok":
        _error_response(503, "enqueue_failed", "publish ownership unavailable", retry_after=5)

    try:
        enqueue(delivery)
    except Exception as exc:  # noqa: BLE001
        _logger.error(
            "enqueue_failed run_id=%s error_type=%s",
            loaded.run_id,
            type(exc).__name__,
        )
        store.release_recovery_claim(loaded.run_id, delivery_id=delivery.delivery_id)
        try:
            store.mark_accepted_enqueue_failed(loaded.run_id, "enqueue_failed")
        except ExecutionStoreError:
            pass
        _error_response(503, "enqueue_failed", "broker did not accept message", retry_after=5)

    try:
        queued = store.mark_queued(
            loaded.run_id,
            expected_status=expected,
            delivery_id=delivery.delivery_id,
        )
        store.release_recovery_claim(loaded.run_id, delivery_id=delivery.delivery_id)
    except ExecutionStoreError as exc:
        _logger.error("mark_queued_failed run_id=%s code=%s", loaded.run_id, exc.code)
        try:
            if loaded.status != ExecutionStatus.ENQUEUE_FAILED:
                store.mark_accepted_enqueue_failed(loaded.run_id, "queued_transition_failed")
        except ExecutionStoreError:
            pass
        _error_response(503, "enqueue_failed", "failed to persist queued evidence", retry_after=5)

    return AdmissionResult(
        response=_ack(req, thread_id, queued.status.value),
        created=created,
        execution_status=queued.status.value,
    )


def http_exception_to_json(exc: HTTPException) -> JSONResponse:
    """Helper for tests — mirrors FastAPI error body + Retry-After."""
    headers = dict(exc.headers or {})
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail}, headers=headers)
