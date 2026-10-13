"""Durable admission for POST /call_async (REQ-032-FIX1)."""

from __future__ import annotations

import hashlib
import logging
import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException
from fastapi.responses import JSONResponse

from apps.group_agent_api.app.execution_modes import (
    execution_binding,
    v4_queue_binding,
    validate_v3_request,
    validate_v4_request,
)
from apps.group_agent_api.app.models import AsyncCallRequest, AsyncCallResponse
from apps.group_agent_api.app.session import TrustedSession
from apps.group_agent_api.app.venue_protocol import lease_scope, validate_v2_request
from apps.group_agent_api.execution.backpressure import BackpressureController
from apps.group_agent_api.execution.config import (
    DurableQueueConfig,
    validate_request_fingerprint,
)
from apps.group_agent_api.execution.crypto import (
    PayloadCryptoError,
    decrypt_envelope,
    digest_id,
    encrypt_envelope,
)
from apps.group_agent_api.execution.models import (
    SUPPORTED_SCHEMA_PAIRS,
    BrokerDeliveryRef,
    EncryptedPayload,
    ExecutionRecord,
    ExecutionStatus,
)
from apps.group_agent_api.execution.redis_store import (
    ExecutionStore,
    ExecutionStoreError,
)

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
    from apps.group_agent_api.app.cafe_host_model_tools import cafe_host_link_for_admission

    envelope: dict[str, Any] = {
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
            "capabilities": {
                "can_search": session.capabilities.can_search,
                "can_write_profile": session.capabilities.can_write_profile,
            },
        },
    }
    link = cafe_host_link_for_admission(req)
    if link is not None:
        envelope["cafe_host_link"] = link
    return envelope


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
    if (
        type(schema) is not int
        or type(queue_schema) is not int
        or (schema, queue_schema) not in SUPPORTED_SCHEMA_PAIRS
    ):
        _error_response(400, "schema_invalid", "unsupported request/queue schema pair")
    if schema == 5:
        from apps.group_agent_api.app.assistant_protocol import validate
        try:
            validate(req.model_dump())
        except (ValueError, TypeError):
            _error_response(400, "schema_invalid", "invalid assistant request")
    if schema == 2:
        try:
            validate_v2_request(req.model_dump())
        except ValueError:
            _error_response(
                400, "schema_invalid", "v2 request fingerprint or structure invalid"
            )
    if schema == 3:
        try:
            validate_v3_request(req.model_dump())
        except ValueError:
            _error_response(400, "schema_invalid", "invalid v3 frozen request")
    if schema == 4:
        try:
            validate_v4_request(req.model_dump())
        except ValueError:
            _error_response(400, "schema_invalid", "invalid v4 frozen request")
    try:
        fp = validate_request_fingerprint(str(fingerprint))
    except ValueError:
        _error_response(
            400,
            "schema_invalid",
            "request_fingerprint must be 64-char lowercase sha256 hex",
        )
    return int(schema), fp


def _v4_admission_enabled(user_id: str) -> bool:
    """Return whether server configuration allows a new v4 run.

    Args:
        user_id: Authenticated principal identifier.
    """
    enabled = os.environ.get("CAFE_HOST_V4_ADMISSION_ENABLED", "0").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return False
    allowed = {
        item.strip()
        for item in os.environ.get("CAFE_HOST_V4_USER_ALLOWLIST", "").split(",")
        if item.strip()
    }
    return "*" in allowed or user_id in allowed


def _token_digest(token: str | None) -> str:
    """Return the protocol digest sentinel for one trusted header token.

    Args:
        token: Authenticated header credential, if present.
    """
    value = (token or "").strip()
    return hashlib.sha256(value.encode("utf-8")).hexdigest() if value else "-"


def _validate_v4_principal(req: AsyncCallRequest, session: TrustedSession) -> None:
    """Bind signed header credentials to the frozen v4 body digests.

    Args:
        req: Strict v4 request with no bearer fields.
        session: Principal resolved from signed headers.
    """
    binding = req.credential_binding or {}
    user_digest = _token_digest(session.principal.user_token)
    group_digest = _token_digest(session.group_token)
    if binding != {
        "user_token_sha256": user_digest,
        "group_token_sha256": group_digest,
    }:
        _error_response(
            400, "binding_mismatch", "credential digest differs from trusted principal"
        )
    mode = (req.chat_access or {}).get("mode")
    if mode == "ordinary" and (session.principal.user_token or session.group_token):
        _error_response(
            400, "binding_mismatch", "ordinary run cannot carry search credentials"
        )
    if mode == "active" and not session.principal.user_token:
        _error_response(
            400, "binding_mismatch", "active run requires authenticated user credential"
        )


def _validate_v4_idempotent_envelope(
    *,
    req: AsyncCallRequest,
    session: TrustedSession,
    thread_id: str,
    record: ExecutionRecord,
    config: DurableQueueConfig,
) -> None:
    """Require an idempotent v4 hit to match the first encrypted request.

    Args:
        req: Canonical request being replayed.
        session: Newly authenticated transport principal.
        thread_id: Canonical checkpoint binding for this replay.
        record: Existing immutable durable run.
        config: Queue keys used to authenticate and decrypt the first envelope.

    This comparison happens before returning the original ACK. It prevents a
    partial ledger projection from becoming the authority for a replay: the
    canonical request and authenticated credential identity must still equal
    the first accepted encrypted envelope byte-for-byte in meaning.
    """
    try:
        envelope = decrypt_envelope(
            record.payload_ciphertext,
            keys=config.payload_keys,
            run_id=record.run_id,
            idempotency_key=record.idempotency_key,
            request_fingerprint=record.request_fingerprint,
            schema_version=record.request_schema_version,
            mode_binding=record.mode_binding,
        )
    except PayloadCryptoError:
        _error_response(
            409, "run_binding_conflict", "stored v4 envelope binding is invalid"
        )
    stored_session = envelope.get("session")
    expected_identity = {
        "user_id": session.principal.user_id,
        "unionid": session.principal.unionid,
        "user_token": session.principal.user_token,
        "group_token": session.group_token,
        "group_id": session.group_id,
    }
    if (
        envelope.get("thread_id") != thread_id
        or envelope.get("request") != req.model_dump()
        or not isinstance(stored_session, dict)
        or any(
            stored_session.get(key) != value for key, value in expected_identity.items()
        )
    ):
        _error_response(
            409,
            "idempotency_conflict",
            "v4 replay differs from first accepted envelope",
        )


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
    if schema == 4:
        if not _v4_admission_enabled(session.principal.user_id):
            _error_response(
                503, "execution_mode_unavailable", "Cafe Host v4 admission is disabled"
            )
        _validate_v4_principal(req, session)
    if schema in {3, 4}:
        from apps.group_agent_api.app.utils import thread_id as canonical_thread_id

        expected_thread = canonical_thread_id(
            user_id=req.user_id,
            group_id=req.group_id,
            conversation_id=req.conversation_id,
            episode_id=req.episode_id,
            execution_mode=req.execution_mode,
        )
        if thread_id != expected_thread:
            _error_response(400, "schema_invalid", "invalid frozen thread binding")
    if schema in {2, 3, 4, 5} and (
        req.user_id != session.principal.user_id
        or req.unionid != session.principal.unionid
        or req.group_id != session.group_id
        or (
            schema != 4
            and (
                (req.user_token or "").strip()
                != (session.principal.user_token or "").strip()
                or (req.group_token or "").strip()
                != (session.group_token or "").strip()
            )
        )
    ):
        _error_response(
            400, "binding_mismatch", "frozen request must match trusted session"
        )
    provider = (
        os.environ.get("GROUP_AGENT_PROVIDER") or "default"
    ).strip() or "default"

    # FIX1 BLOCKER-4: binding hit before quota — saturated system still returns original ACK.
    existing = store.get_by_idempotency(req.idempotency_key)
    if existing is not None:
        if (
            existing.request_schema_version != schema
            or existing.queue_schema_version != req.queue_schema_version
            or existing.request_fingerprint != fingerprint
            or existing.run_id != req.run_id
            or (
                schema in {3, 4}
                and (
                    existing.mode_binding
                    != (
                        v4_queue_binding(req)
                        if schema == 4
                        else {
                            key: getattr(req, key)
                            for key in (
                                "user_id",
                                "execution_mode",
                                "conversation_id",
                                "episode_id",
                            )
                        }
                    )
                    or existing.conversation_id != lease_scope(req)
                )
            )
        ):
            _error_response(
                409, "idempotency_conflict", "idempotency or run binding conflict"
            )
        if schema == 4:
            _validate_v4_idempotent_envelope(
                req=req,
                session=session,
                thread_id=thread_id,
                record=existing,
                config=config,
            )
        if existing.status in {
            ExecutionStatus.ENQUEUE_FAILED,
            ExecutionStatus.ACCEPTED,
        }:
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
        _error_response(
            409, "run_binding_conflict", "run_id bound to different idempotency_key"
        )

    bp = backpressure.check_and_reserve(
        user_id=session.principal.user_id,
        group_id=session.group_id,
        conversation_id=lease_scope(req),
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
            mode_binding=(
                v4_queue_binding(req)
                if schema == 4
                else {
                    key: getattr(req, key)
                    for key in (
                        "user_id",
                        "execution_mode",
                        "conversation_id",
                        "episode_id",
                    )
                }
            )
            if schema in {3, 4}
            else None,
        )
        now = store.redis_time()
        record = ExecutionRecord(
            run_id=req.run_id,
            idempotency_key=req.idempotency_key,
            request_schema_version=schema,
            request_fingerprint=fingerprint,
            queue_schema_version=req.queue_schema_version,
            status=ExecutionStatus.ACCEPTED,
            created_at=now,
            payload_key_version=ciphertext.key_version,
            payload_ciphertext=ciphertext,
            conversation_id=lease_scope(req),
            user_id_digest=digest_id(session.principal.user_id),
            group_id_digest=digest_id(session.group_id),
            provider=provider,
            input_kind=req.input_kind,
            mode_binding=(
                v4_queue_binding(req)
                if schema == 4
                else {
                    key: getattr(req, key)
                    for key in (
                        "user_id",
                        "execution_mode",
                        "conversation_id",
                        "episode_id",
                    )
                }
            )
            if schema in {3, 4}
            else None,
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
            _error_response(
                503, "queue_unavailable", "execution ledger unavailable", retry_after=5
            )

        if kind == "hit":
            backpressure.release_queued_reservation(
                user_id=session.principal.user_id,
                group_id=session.group_id,
            )
            reserved = False
            if schema == 4:
                _validate_v4_idempotent_envelope(
                    req=req,
                    session=session,
                    thread_id=thread_id,
                    record=loaded,
                    config=config,
                )
            if loaded.status in {
                ExecutionStatus.ENQUEUE_FAILED,
                ExecutionStatus.ACCEPTED,
            }:
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
        queue_schema_version=req.queue_schema_version,
        execution_binding=execution_binding(req)
        if req.request_schema_version in {3, 4}
        else None,
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
        queue_schema_version=loaded.queue_schema_version,
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
        _error_response(
            503, "enqueue_failed", "publish ownership unavailable", retry_after=5
        )

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
        _error_response(
            503, "enqueue_failed", "broker did not accept message", retry_after=5
        )

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
                store.mark_accepted_enqueue_failed(
                    loaded.run_id, "queued_transition_failed"
                )
        except ExecutionStoreError:
            pass
        _error_response(
            503, "enqueue_failed", "failed to persist queued evidence", retry_after=5
        )

    return AdmissionResult(
        response=_ack(req, thread_id, queued.status.value),
        created=created,
        execution_status=queued.status.value,
    )


def http_exception_to_json(exc: HTTPException) -> JSONResponse:
    """Helper for tests — mirrors FastAPI error body + Retry-After."""
    headers = dict(exc.headers or {})
    return JSONResponse(
        status_code=exc.status_code, content={"detail": exc.detail}, headers=headers
    )
