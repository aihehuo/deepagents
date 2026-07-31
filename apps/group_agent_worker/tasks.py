"""Celery tasks for Group Agent durable execution (REQ-032-FIX2)."""

from __future__ import annotations

import asyncio
import datetime
import logging
import os
import threading
from typing import Any

from celery import shared_task
from celery.exceptions import Reject

from apps.group_agent_api.agent_factory.capability import CapabilityTier
from apps.group_agent_api.agent_factory.integrations.callback_client import send_callback_event
from apps.group_agent_api.agent_factory.integrations.config import async_run_timeout_s
from apps.group_agent_api.agent_factory.integrations.membership_client import MembershipResult
from apps.group_agent_api.agent_factory.integrations.principal import SessionPrincipal
from apps.group_agent_api.app.async_manager import execute_async_run_core
from apps.group_agent_api.app.models import AsyncCallRequest, CallbackEnvelope
from apps.group_agent_api.app.session import TrustedSession
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
    QUEUE_SCHEMA_VERSION,
    BrokerDeliveryRef,
    ExecutionStatus,
    LeaseClaim,
)
from apps.group_agent_api.execution.redis_store import ExecutionStoreError
from apps.group_agent_api.execution.retry import decide_retry
from apps.group_agent_worker.runtime import get_worker_runtime

_logger = logging.getLogger(__name__)


def _rebuild_session(envelope: dict[str, Any]) -> tuple[AsyncCallRequest, TrustedSession, str]:
    sess = envelope["session"]
    req = AsyncCallRequest.model_validate(envelope["request"])
    principal = SessionPrincipal(
        user_id=str(sess["user_id"]),
        unionid=str(sess["unionid"]),
        user_token=sess.get("user_token"),
        group_token=sess.get("group_token"),
        source=str(sess.get("principal_source") or "durable_envelope"),
    )
    tier = CapabilityTier(str(sess["membership_tier"]))
    membership = MembershipResult(
        tier=tier,
        source=str(sess.get("membership_source") or "durable_envelope"),
    )
    session = TrustedSession(
        principal=principal,
        membership=membership,
        group_id=str(sess["group_id"]),
        group_token=sess.get("group_token"),
    )
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

    bp.on_start_running(conversation_id=req.conversation_id, provider=provider)
    fence = SideEffectFence(
        store=store,
        claim=claim,
        conversation_id=req.conversation_id,
        abort_on_heartbeat_failures=cfg.heartbeat_fail_threshold,
    )
    active = ActiveAttemptFence(
        store=store,
        claim=claim,
        conversation_id=req.conversation_id,
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

    async def _deliver(event_type: str, payload: dict[str, Any]) -> bool:
        env = CallbackEnvelope(
            version="GA-CALLBACK-V1",
            run_id=req.run_id,
            idempotency_key=req.idempotency_key,
            seq=fence.seq,
            event=event_type,  # type: ignore[arg-type]
            occurred_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            user_id=session.principal.user_id,
            group_id=session.group_id,
            conversation_id=req.conversation_id,
            payload=payload,
        )
        return await send_callback_event(
            callback_url=req.callback_url,
            envelope_dict=env.model_dump(),
        )

    async def _emit(event_type: str, payload: dict[str, Any]) -> bool:
        return await fence.emit(event_type, payload, send=_deliver)

    def _side_effect_gate(action: str) -> None:
        fence.check_side_effect(action)
        if active.cancel_event.is_set():
            raise LeaseLostError("fence_cancelled")

    error_code = ""
    outcome = "unknown"
    try:
        await _emit("progress", {"phase": "started", "message": "processing_started"})
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
                conversation_id=req.conversation_id,
                status=ExecutionStatus.SUCCEEDED,
            )
            outcome = ExecutionStatus.SUCCEEDED.value
    except (LeaseLostError, FenceRejectedError):
        outcome = "lost_lease"
    except asyncio.TimeoutError:
        error_code = error_code or "provider_timeout"
    except Exception as exc:  # noqa: BLE001
        if not error_code:
            if "final_callback" in str(exc):
                error_code = "callback_5xx"
            else:
                error_code = "transient_execution_error"
        _logger.error(
            "worker_attempt_error run_id=%s attempt_id=%s error_code=%s",
            req.run_id,
            claim.attempt_id,
            error_code,
        )
    finally:
        stop_hb.set()
        hb_thread.join(timeout=2.0)
        clear_active_fence(fence_tokens)

    if outcome == ExecutionStatus.SUCCEEDED.value:
        bp.on_finish(
            user_id=session.principal.user_id,
            group_id=session.group_id,
            conversation_id=req.conversation_id,
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
            await _emit(
                "error",
                {
                    "error_code": error_code or "AsyncRunFailed",
                    "message": "Task execution failed",
                },
            )
        except Exception:  # noqa: BLE001
            pass
        status = (
            ExecutionStatus.DEAD_LETTERED
            if decision.dead_letter
            else ExecutionStatus.FAILED
        )
        try:
            store.finish(
                claim,
                conversation_id=req.conversation_id,
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
            conversation_id=req.conversation_id,
            provider=provider,
            was_queued=True,
        )
        return status.value

    next_at = store.redis_time() + decision.delay_s
    try:
        store.schedule_retry(
            claim,
            conversation_id=req.conversation_id,
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
        conversation_id=req.conversation_id,
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

    if ref.queue_schema_version != QUEUE_SCHEMA_VERSION:
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

    req, session, tid = _rebuild_session(envelope)
    outcome = store.claim_lease(
        run_id=ref.run_id,
        conversation_id=req.conversation_id,
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

    result = asyncio.run(
        _run_attempt(
            claim=claim,
            req=req,
            session=session,
            tid=tid,
        )
    )
    return {"status": result, "run_id": ref.run_id, "attempt_id": claim.attempt_id}
