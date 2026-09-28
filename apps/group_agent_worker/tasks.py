"""Celery tasks for Group Agent durable execution (REQ-032-FIX2)."""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import logging
import os
import threading
from typing import Any

from billiard.exceptions import SoftTimeLimitExceeded
from celery import shared_task
from celery.exceptions import Reject
from fastapi import HTTPException

from apps.group_agent_api.agent_factory.capability import (
    CapabilityTier,
    SessionCapabilities,
    resolve_session_capabilities,
)
from apps.group_agent_api.agent_factory.integrations.callback_client import (
    CallbackProtocolError,
    CallbackRunTerminal,
    send_callback_event,
)
from apps.group_agent_api.agent_factory.integrations.config import async_run_timeout_s
from apps.group_agent_api.agent_factory.integrations.membership_client import (
    MembershipResult,
)
from apps.group_agent_api.agent_factory.integrations.principal import SessionPrincipal
from apps.group_agent_api.agent_factory.intent_routing.execution import QueryPlanError
from apps.group_agent_api.agent_factory.venue_greeting import VenueGreetingError
from apps.group_agent_api.agent_factory.write_skills import profile_write_mode
from apps.group_agent_api.app.async_manager import (
    VenueEventExecutorUnavailable,
    execute_async_run_core,
)
from apps.group_agent_api.app.models import AsyncCallRequest, CallbackEnvelope
from apps.group_agent_api.app.session import TrustedSession
from apps.group_agent_api.app.utils import thread_id
from apps.group_agent_api.app.venue_protocol import callback_bindings, lease_scope
from apps.group_agent_api.execution.active_fence import (
    ActiveAttemptFence,
    FenceRejectedError,
    clear_active_fence,
    set_active_fence,
)
from apps.group_agent_api.execution.crypto import PayloadCryptoError, decrypt_envelope
from apps.group_agent_api.execution.dlq import push_dlq_index
from apps.group_agent_api.execution.fence import LeaseLostError, SideEffectFence
from apps.group_agent_api.execution.models import (
    SUPPORTED_SCHEMA_PAIRS,
    BrokerDeliveryRef,
    ExecutionStatus,
    LeaseClaim,
)
from apps.group_agent_api.execution.redis_store import ExecutionStoreError
from apps.group_agent_api.execution.retry import decide_retry
from apps.group_agent_worker.event_loop import run_on_worker_event_loop
from apps.group_agent_worker.runtime import get_worker_runtime

_logger = logging.getLogger(__name__)


def _strict_callback_enabled(req: AsyncCallRequest) -> bool:
    return (
        req.input_kind != "venue_event"
        and os.environ.get("GROUP_AGENT_CALLBACK_ACK_SCHEMA_VERSION", "0").strip()
        == "1"
        and os.environ.get("GROUP_AGENT_INTENT_ROUTING_MODE", "off").strip().lower()
        == "enforce"
    )


def _progress_payload(
    phase: str,
    *,
    intent: str | None = None,
    routing_snapshot_hash: str | None = None,
    display_code: str | None = None,
    plan_id: str | None = None,
    plan_step_id: str | None = None,
    request_index: int | None = None,
    elapsed_ms: int = 0,
    step_result_count: int | None = None,
    aggregate_result_count: int | None = None,
) -> dict[str, Any]:
    codes = {
        "started": "starting",
        "intent_resolving": "resolving_intent",
        "intent_resolved": "intent_ready",
        "tool_completed": "search_step_complete",
        "composing": "composing_reply",
    }
    return {
        "progress_schema_version": 1,
        "phase": phase,
        "intent": intent,
        "display_code": display_code or codes.get(phase, "starting"),
        "routing_snapshot_hash": routing_snapshot_hash,
        "plan_id": plan_id,
        "plan_step_id": plan_step_id,
        "request_index": request_index,
        "elapsed_ms": max(0, min(int(elapsed_ms), 600_000)),
        "step_result_count": step_result_count,
        "aggregate_result_count": aggregate_result_count,
    }


_CALLBACK_ERROR_CODES = frozenset(
    {
        "required_tool_not_called",
        "query_plan_invalid",
        "query_budget_exhausted",
        "tool_failed",
        "tool_timed_out",
        "routing_snapshot_invalid",
        "receipt_invalid",
        "callback_ack_invalid",
        "callback_seq_diverged",
        "callback_auth_rejected",
        "permission_denied",
        "intent_classifier_timeout",
        "intent_classifier_invalid",
        "cancelled",
        "agent_error",
    }
)


def _terminal_error_payload(error_code: str, *, timed_out: bool) -> dict[str, Any]:
    code = error_code if error_code in _CALLBACK_ERROR_CODES else "agent_error"
    planner_failed = code in {"intent_classifier_invalid", "intent_classifier_timeout"}
    return {
        "terminal_reason": "timed_out" if timed_out else "failed",
        "error_code": code,
        "message": (
            "本轮意图规划失败，未执行搜索；请稍后重试。"
            if planner_failed
            else "智能体处理超时，请稍后重试"
            if timed_out
            else "智能体处理遇到异常，请稍后重试"
        ),
    }


def _rebuild_session(envelope: dict[str, Any]) -> tuple[AsyncCallRequest, TrustedSession, str]:
    """Restore one trusted request/session and recheck encrypted bindings.

    Args:
        envelope: Authenticated payload decrypted from the durable ledger.

    Returns:
        The request, trusted session, and canonical thread identifier.

    Raises:
        ValueError: If identity, thread, or v4 credential binding differs.
    """
    sess = envelope["session"]
    req = AsyncCallRequest.model_validate(envelope["request"])
    principal = SessionPrincipal(
        user_id=str(sess["user_id"]),
        unionid=str(sess["unionid"]) if sess.get("unionid") is not None else None,
        user_token=sess.get("user_token"),
        group_token=sess.get("group_token"),
        source=str(sess.get("principal_source") or "durable_envelope"),
    )
    tier = CapabilityTier(str(sess["membership_tier"]))
    membership = MembershipResult(
        tier=tier,
        source=str(sess.get("membership_source") or "durable_envelope"),
    )
    permissions = sess.get("capabilities") or {}
    if not isinstance(permissions, dict):
        msg = "binding_mismatch: invalid trusted capabilities"
        raise TypeError(msg)
    capabilities = resolve_session_capabilities(
        tier=tier,
        can_write_profile=(
            permissions.get("can_write_profile") is True
            and bool(principal.user_id.strip() and (principal.unionid or "").strip())
            and profile_write_mode() != "disabled"
        ),
    )
    # Permissions come from the encrypted admission envelope, never request
    # metadata. Older envelopes without this field remain read/write denied.
    capabilities = SessionCapabilities(
        tier=tier,
        can_search=capabilities.can_search and permissions.get("can_search") is True,
        can_write_profile=capabilities.can_write_profile,
    )
    session = TrustedSession(
        principal=principal,
        membership=membership,
        group_id=str(sess["group_id"]),
        group_token=sess.get("group_token"),
        capabilities=capabilities,
    )
    if req.request_schema_version in {2, 3, 4}:
        expected_tid = thread_id(
            user_id=req.user_id,
            group_id=req.group_id,
            conversation_id=req.conversation_id,
            episode_id=req.episode_id,
            execution_mode=req.execution_mode if req.request_schema_version in {3, 4} else None,
        )
        if (
            req.user_id != principal.user_id
            or req.unionid != principal.unionid
            or req.group_id != session.group_id
            or (
                req.request_schema_version != 4
                and (
                    (req.user_token or "").strip() != (principal.user_token or "").strip()
                    or (req.group_token or "").strip() != (session.group_token or "").strip()
                )
            )
            or str(envelope["thread_id"]) != expected_tid
        ):
            msg = "binding_mismatch: invalid trusted envelope"
            raise ValueError(msg)
        if req.request_schema_version == 4:
            def _digest(value: str | None) -> str:
                text = (value or "").strip()
                return hashlib.sha256(text.encode("utf-8")).hexdigest() if text else "-"

            expected = {
                "user_token_sha256": _digest(principal.user_token),
                "group_token_sha256": _digest(session.group_token),
            }
            if req.credential_binding != expected:
                msg = "binding_mismatch: invalid v4 credential binding"
                raise ValueError(msg)
    return req, session, str(envelope["thread_id"])


def _poison_mapped_run(
    *,
    store: Any,
    cfg: Any,
    run_id: str,
    conversation_id: str,
    error_code: str,
    expected_status: str = "",
) -> dict[str, str]:
    """Unified poison → DLQ CAS for mappable non-running runs (FIX3)."""
    record = store.get(run_id)
    if record is not None and record.status == ExecutionStatus.RUNNING:
        # Never revoke an active lease from a poison/conflict delivery.
        _logger.warning(
            "poison_ignored_running run_id=%s error_code=%s",
            run_id,
            error_code,
        )
        return {"status": "ignored_running", "error_code": error_code, "run_id": run_id}
    if expected_status == "running":
        return {"status": "ignored_running", "error_code": error_code, "run_id": run_id}
    try:
        store.poison_to_dlq(
            run_id,
            conversation_id=conversation_id or run_id,
            error_code=error_code,
            expected_status=expected_status,
        )
        push_dlq_index(store, cfg, run_id)
    except ExecutionStoreError as exc:
        if exc.code == "running_protected":
            return {"status": "ignored_running", "error_code": error_code, "run_id": run_id}
        _logger.error(
            "poison_dlq_failed run_id=%s code=%s err=%s",
            run_id,
            error_code,
            exc.code,
        )
    return {"status": "dead_lettered", "error_code": error_code, "run_id": run_id}


async def _run_attempt(
    *,
    claim: LeaseClaim,
    req: AsyncCallRequest,
    session: TrustedSession,
    tid: str,
) -> str:
    """Execute one attempt; returns terminal status value or retry/lost_lease."""
    rt = get_worker_runtime()
    store = rt["store"]
    state = rt["state"]
    cfg = rt["config"]
    bp = rt["backpressure"]
    provider = (os.environ.get("GROUP_AGENT_PROVIDER") or "default").strip() or "default"
    operational_scope = lease_scope(req)

    bp.on_start_running(conversation_id=operational_scope, provider=provider)
    fence = SideEffectFence(
        store=store,
        claim=claim,
        conversation_id=operational_scope,
        abort_on_heartbeat_failures=cfg.heartbeat_fail_threshold,
    )
    active = ActiveAttemptFence(
        store=store,
        claim=claim,
        conversation_id=operational_scope,
        cancel_event=threading.Event(),
    )
    fence_tokens = set_active_fence(active, require=True)
    fence.on_abort(active.cancel)

    stop_hb = threading.Event()

    def _hb_loop() -> None:
        while not stop_hb.wait(cfg.heartbeat_interval_s):
            if fence.aborted or active.cancel_event.is_set():
                break
            fence.renew_or_abort()

    hb_thread = threading.Thread(
        target=_hb_loop,
        name=f"ga-hb-{claim.attempt_id[:8]}",
        daemon=True,
    )
    hb_thread.start()

    strict_callbacks = _strict_callback_enabled(req)

    def _build_envelope(
        seq: int, event_type: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        record = store.get(req.run_id)
        routing_hash = (
            payload.get("routing_snapshot_hash")
            if event_type == "progress"
            else (record.routing_snapshot_hash if record else None)
        )
        env = CallbackEnvelope(
            version=None if strict_callbacks else "GA-CALLBACK-V1",
            **({"callback_version": 1} if strict_callbacks else {}),
            run_id=req.run_id,
            **(
                {"idempotency_key": req.idempotency_key}
                if req.request_schema_version not in {2, 3, 4}
                else {}
            ),
            seq=seq,
            event=event_type,  # type: ignore[arg-type]
            occurred_at=datetime.datetime.now(datetime.UTC).isoformat(timespec="microseconds").replace("+00:00", "Z"),
            user_id=session.principal.user_id,
            group_id=session.group_id,
            conversation_id=req.conversation_id,
            **(
                {
                    "deep_attempt_id": claim.attempt_id,
                    "deep_attempt_number": record.attempt_count if record else 1,
                    "routing_snapshot_hash": routing_hash,
                }
                if strict_callbacks
                else {}
            ),
            payload=payload,
            **callback_bindings(req),
        )
        excluded = (
            {"idempotency_key"}
            if req.request_schema_version == 2
            else (
                {"idempotency_key", "space_id", "host_agent_id", "memory_scope_id", "input_kind", "event_name", "visit_id"}
                if req.request_schema_version in {3, 4}
                else {"episode_id", "space_id", "host_agent_id", "memory_scope_id", "input_kind", "event_name", "visit_id"}
            )
        )
        return env.model_dump(exclude=excluded, exclude_none=True)

    async def _send_body(envelope: dict[str, Any], body_bytes: bytes) -> bool:
        return await send_callback_event(
            callback_url=req.callback_url,
            envelope_dict=envelope,
            body_bytes=body_bytes,
            require_ack=True,
        )

    async def _deliver(event_type: str, payload: dict[str, Any]) -> bool:
        envelope = _build_envelope(fence.seq, event_type, payload)
        return await send_callback_event(
            callback_url=req.callback_url,
            envelope_dict=envelope,
        )

    async def _emit(event_type: str, payload: dict[str, Any]) -> bool:
        if event_type == "final":
            _assert_event_live()
        if strict_callbacks:
            if event_type == "progress" and payload.get("progress_schema_version") != 1:
                if payload.get("phase") != "started":
                    # Legacy streaming/tool hints are not part of ProgressEventV1.
                    return True
                payload = _progress_payload("started")
            return await fence.emit_durable(
                event_type,
                payload,
                build_envelope=_build_envelope,
                send_body=_send_body,
            )
        return await fence.emit(event_type, payload, send=_deliver, include_attempt_metadata=req.input_kind != "venue_event")

    def _assert_event_live() -> None:
        if req.request_schema_version == 3 and req.input_kind == "venue_event":
            expires = datetime.datetime.strptime(req.event["expires_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.UTC).timestamp()
            if store.redis_time() >= expires:
                raise VenueGreetingError("venue_event_expired")

    def _side_effect_gate(action: str) -> None:
        fence.check_side_effect(action)
        if active.cancel_event.is_set():
            raise LeaseLostError("fence_cancelled")
        _assert_event_live()

    error_code = ""
    outcome = "unknown"
    try:
        _assert_event_live()
        record = store.get(req.run_id)
        if record is not None and record.attempt_count > cfg.max_attempts:
            # Lease-expiry recovery also consumes the persisted attempt budget.
            error_code = "max_attempts_exceeded"
            raise RuntimeError(error_code)
        if strict_callbacks:
            had_pending = bool(record and record.pending_callback)
            recovered = await fence.recover_pending(send_body=_send_body)
            if had_pending and not recovered:
                error_code = "callback_uncertain"
                raise RuntimeError("pending_callback_delivery_uncertain")
            if fence.terminal_delivered:
                store.finish(
                    claim,
                    conversation_id=operational_scope,
                    status=ExecutionStatus.SUCCEEDED,
                )
                outcome = ExecutionStatus.SUCCEEDED.value
                bp.on_finish(
                    user_id=session.principal.user_id,
                    group_id=session.group_id,
                    conversation_id=operational_scope,
                    provider=provider,
                    was_queued=True,
                )
                return outcome
        await _emit(
            "progress", {"phase": "preparing_greeting"} if req.input_kind == "venue_event" else {"phase": "started", "message": "processing_started"}
        )
        await asyncio.wait_for(
            execute_async_run_core(
                req=req,
                session=session,
                state=state,
                tid=tid,
                emit_callback=_emit,
                side_effect_gate=_side_effect_gate,
            ),
            timeout=async_run_timeout_s(),
        )
        if not fence.assert_owner() or active.cancel_event.is_set():
            outcome = "lost_lease"
        elif fence.final_callback_ok is False:
            error_code = "callback_5xx"
            raise RuntimeError("final_callback_failed")
        elif fence.final_callback_ok is None:
            error_code = "callback_uncertain"
            raise RuntimeError("final_callback_missing")
        else:
            store.finish(
                claim,
                conversation_id=operational_scope,
                status=ExecutionStatus.SUCCEEDED,
            )
            outcome = ExecutionStatus.SUCCEEDED.value
    except (LeaseLostError, FenceRejectedError):
        outcome = "lost_lease"
    except CallbackRunTerminal as exc:
        terminal_status = (
            ExecutionStatus.SUCCEEDED
            if exc.run_status == "completed"
            else ExecutionStatus.FAILED
        )
        store.finish(
            claim,
            conversation_id=operational_scope,
            status=terminal_status,
            error_code="" if terminal_status == ExecutionStatus.SUCCEEDED else "run_terminal",
        )
        outcome = terminal_status.value
    except CallbackProtocolError as exc:
        error_code = exc.code
    except (TimeoutError, SoftTimeLimitExceeded):
        error_code = error_code or "provider_timeout"
    except VenueEventExecutorUnavailable:
        error_code = "venue_event_executor_unavailable"
    except VenueGreetingError as exc:
        error_code = exc.code
    except QueryPlanError as exc:
        error_code = exc.code
    except Exception as exc:
        if not error_code:
            if "mouth_ingress_rejected" in str(exc):
                error_code = "mouth_ingress_rejected"
            elif "final_callback" in str(exc):
                error_code = "callback_5xx"
            else:
                error_code = "transient_execution_error"
        _logger.exception(
            "worker_attempt_error run_id=%s attempt_id=%s error_code=%s exc=%s",
            req.run_id,
            claim.attempt_id,
            error_code,
            exc,
        )
    finally:
        stop_hb.set()
        hb_thread.join(timeout=2.0)
        clear_active_fence(fence_tokens)

    if outcome in {ExecutionStatus.SUCCEEDED.value, ExecutionStatus.FAILED.value}:
        bp.on_finish(
            user_id=session.principal.user_id,
            group_id=session.group_id,
            conversation_id=operational_scope,
            provider=provider,
            was_queued=True,
        )
        return outcome

    if outcome == "lost_lease" or not fence.assert_owner() or active.cancel_event.is_set():
        bp.on_lost_lease()
        return "lost_lease"

    record = store.get(req.run_id)
    attempt_count = record.attempt_count if record else 1
    decision = decide_retry(
        error_code=error_code,
        attempt_count=attempt_count,
        max_attempts=cfg.max_attempts,
        base_s=cfg.retry_base_s,
        max_s=cfg.retry_max_s,
    )

    if decision.dead_letter or not decision.should_retry:
        try:
            payload = (
                _terminal_error_payload(
                    error_code,
                    timed_out=error_code in {"provider_timeout", "tool_timed_out"},
                )
                if strict_callbacks
                else {
                    "error_code": error_code or "AsyncRunFailed",
                    "message": (
                        "本轮意图规划失败，未执行搜索；请稍后重试。"
                        if error_code in {"intent_classifier_invalid", "intent_classifier_timeout"}
                        else "智能体处理遇到异常或超时，请稍后重试"
                    ),
                }
            )
            await _emit("error", payload)
        except Exception:  # noqa: BLE001
            pass
        status = ExecutionStatus.DEAD_LETTERED if decision.dead_letter else ExecutionStatus.FAILED
        try:
            store.finish(
                claim,
                conversation_id=operational_scope,
                status=status,
                error_code=decision.reason_code,
            )
            if status == ExecutionStatus.DEAD_LETTERED:
                push_dlq_index(store, cfg, req.run_id)
        except ExecutionStoreError:
            bp.on_lost_lease()
            return "finish_rejected"
        bp.on_finish(
            user_id=session.principal.user_id,
            group_id=session.group_id,
            conversation_id=operational_scope,
            provider=provider,
            was_queued=True,
        )
        return status.value

    next_at = store.redis_time() + decision.delay_s
    try:
        store.schedule_retry(
            claim,
            conversation_id=operational_scope,
            next_attempt_at=next_at,
            error_code=decision.reason_code,
        )
    except ExecutionStoreError:
        bp.on_lost_lease()
        return "retry_rejected"
    # FIX2: restore queued occupancy; do not terminal-finish counters.
    bp.on_retry_wait(
        user_id=session.principal.user_id,
        group_id=session.group_id,
        conversation_id=operational_scope,
        provider=provider,
    )
    return "retry"


@shared_task(
    bind=True,
    name="group_agent.process_run",
    acks_late=True,
    reject_on_worker_lost=True,
    max_retries=20,
)
def process_run(self, delivery: dict[str, Any]) -> dict[str, str]:
    """Claim lease, decrypt envelope, execute shared core, ACK/retry/DLQ."""
    from apps.group_agent_api.execution.dlq import publish_unmappable_poison

    try:
        ref = BrokerDeliveryRef.from_dict(delivery)
    except Exception:  # noqa: BLE001
        _logger.error("poison_payload delivery_deserialize_failed")
        try:
            rt = get_worker_runtime()
            publish_unmappable_poison(
                rt["config"],
                error_code="deserialize_failed",
                raw_preview=repr(delivery)[:200],
                delivery=delivery if isinstance(delivery, dict) else None,
            )
        except Exception:  # noqa: BLE001
            pass
        # Message is parked in queryable DLQ — do not requeue.
        return {"status": "poison", "error_code": "deserialize_failed"}

    rt = get_worker_runtime()
    store = rt["store"]
    cfg = rt["config"]
    record = store.get(ref.run_id)

    if type(ref.queue_schema_version) is not int or ref.queue_schema_version not in {1, 2, 3}:
        # FIX4: schema/binding mismatch isolates the delivery only — never mutate ledger.
        publish_unmappable_poison(
            cfg,
            error_code="unsupported_queue_schema",
            delivery=delivery,
        )
        return {"status": "poison", "error_code": "unsupported_queue_schema"}

    if record is None:
        raise Reject("missing_record", requeue=True)

    if (
        record.idempotency_key != ref.idempotency_key
        or record.request_fingerprint != ref.request_fingerprint
        or record.queue_schema_version != ref.queue_schema_version
        or (record.request_schema_version, record.queue_schema_version) not in SUPPORTED_SCHEMA_PAIRS
    ):
        # FIX4: forged/stale delivery must not terminate a healthy ledger Run.
        publish_unmappable_poison(
            cfg,
            error_code="binding_conflict",
            delivery=delivery,
        )
        return {
            "status": "poison",
            "error_code": "binding_conflict",
            "run_id": ref.run_id,
        }

    if record.status in {
        ExecutionStatus.SUCCEEDED,
        ExecutionStatus.FAILED,
        ExecutionStatus.DEAD_LETTERED,
    }:
        return {"status": "terminal", "error_code": record.status.value}

    if record.payload_ciphertext is None:
        if record.status == ExecutionStatus.RUNNING:
            return {"status": "ignored_running", "error_code": "payload_missing"}
        return _poison_mapped_run(
            store=store,
            cfg=cfg,
            run_id=ref.run_id,
            conversation_id=record.conversation_id or ref.run_id,
            error_code="payload_missing",
            expected_status=record.status.value,
        )

    try:
        envelope = decrypt_envelope(
            record.payload_ciphertext,
            keys=cfg.payload_keys,
            run_id=record.run_id,
            idempotency_key=record.idempotency_key,
            request_fingerprint=record.request_fingerprint,
            schema_version=record.request_schema_version,
            mode_binding=record.mode_binding,
        )
    except PayloadCryptoError as exc:
        if record.status == ExecutionStatus.RUNNING:
            return {"status": "ignored_running", "error_code": exc.code}
        return _poison_mapped_run(
            store=store,
            cfg=cfg,
            run_id=ref.run_id,
            conversation_id=record.conversation_id or ref.run_id,
            error_code=exc.code,
            expected_status=record.status.value,
        )

    try:
        req, session, tid = _rebuild_session(envelope)
        if (
            req.run_id != record.run_id
            or req.idempotency_key != record.idempotency_key
            or req.request_fingerprint != record.request_fingerprint
            or req.request_schema_version != record.request_schema_version
            or req.queue_schema_version != record.queue_schema_version
            or (req.request_schema_version in {2, 3, 4} and (req.input_kind != record.input_kind or lease_scope(req) != record.conversation_id))
        ):
            raise ValueError("binding_mismatch: ledger and request differ")
    except (ValueError, KeyError, TypeError, HTTPException):
        return _poison_mapped_run(
            store=store,
            cfg=cfg,
            run_id=ref.run_id,
            conversation_id=record.conversation_id or ref.run_id,
            error_code="binding_conflict",
            expected_status=record.status.value,
        )
    outcome = store.claim_lease(
        run_id=ref.run_id,
        conversation_id=lease_scope(req),
        owner=cfg.worker_instance_id,
    )
    if outcome.kind == "conversation_busy":
        try:
            store.schedule_conversation_wait(ref.run_id, delay_s=2.0)
        except ExecutionStoreError:
            raise self.retry(countdown=2) from None
        return {"status": "deferred", "error_code": "conversation_busy"}

    if outcome.kind == "not_queued":
        raise self.retry(countdown=1)

    if outcome.kind != "claimed" or outcome.claim is None:
        raise self.retry(countdown=2)

    claim = outcome.claim

    result = run_on_worker_event_loop(
        _run_attempt(
            claim=claim,
            req=req,
            session=session,
            tid=tid,
        )
    )
    return {"status": result, "run_id": ref.run_id, "attempt_id": claim.attempt_id}
