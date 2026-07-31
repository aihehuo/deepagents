"""Leaderless recovery loop for durable execution ledger (REQ-032-FIX2)."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Callable

from apps.group_agent_api.execution.config import DurableQueueConfig
from apps.group_agent_api.execution.models import (
    QUEUE_SCHEMA_VERSION,
    BrokerDeliveryRef,
    ExecutionStatus,
)
from apps.group_agent_api.execution.redis_store import ExecutionStore, ExecutionStoreError

_logger = logging.getLogger(__name__)

EnqueueFn = Callable[[BrokerDeliveryRef], None]


@dataclass
class RecoveryReport:
    accepted_requeued: int = 0
    enqueue_failed_requeued: int = 0
    lease_expired: int = 0
    retry_promoted: int = 0
    skipped_busy: int = 0
    errors: int = 0


def _publish_then_queued(
    store: ExecutionStore,
    enqueue: EnqueueFn,
    *,
    run_id: str,
    idempotency_key: str,
    request_fingerprint: str,
    expected_status: str,
) -> str:
    """Claim recovery ownership, publish, then CAS to queued.

    Returns:
        'ok' | 'busy' | 'conflict'

    At most one new delivery per run per recovery claim window under multi-replica.
    """
    delivery_id = str(uuid.uuid4())
    kind = store.claim_publish_delivery(
        run_id,
        expected_status=expected_status,
        delivery_id=delivery_id,
    )
    if kind == "busy":
        return "busy"
    if kind != "ok":
        return "conflict"

    delivery = BrokerDeliveryRef(
        queue_schema_version=QUEUE_SCHEMA_VERSION,
        run_id=run_id,
        idempotency_key=idempotency_key,
        request_fingerprint=request_fingerprint,
        delivery_id=delivery_id,
    )
    try:
        enqueue(delivery)
        store.mark_queued(
            run_id,
            expected_status=expected_status,
            delivery_id=delivery_id,
        )
        store.release_recovery_claim(run_id, delivery_id=delivery_id)
        return "ok"
    except ExecutionStoreError as exc:
        if exc.code == "already_queued":
            # Another replica won the mark_queued race after our claim expired —
            # do not treat as success for this publisher.
            store.release_recovery_claim(run_id, delivery_id=delivery_id)
            return "busy"
        # Keep claim briefly so peers don't immediately double-publish; claim TTL expires.
        raise


def run_recovery_once(
    store: ExecutionStore,
    config: DurableQueueConfig,
    enqueue: EnqueueFn,
    *,
    accepted_timeout_s: float = 60.0,
    backpressure: object | None = None,
) -> RecoveryReport:
    """Single recovery pass — leaderless with per-run delivery claim (FIX2)."""
    report = RecoveryReport()
    now = store.redis_time()

    for status, attr in (
        (ExecutionStatus.ACCEPTED, "accepted_requeued"),
        (ExecutionStatus.ENQUEUE_FAILED, "enqueue_failed_requeued"),
    ):
        for record in store.scan_status(status, count=100):
            age = now - (record.created_at or now)
            if age < accepted_timeout_s and status == ExecutionStatus.ACCEPTED:
                continue
            try:
                result = _publish_then_queued(
                    store,
                    enqueue,
                    run_id=record.run_id,
                    idempotency_key=record.idempotency_key,
                    request_fingerprint=record.request_fingerprint,
                    expected_status=status.value,
                )
                if result == "ok":
                    setattr(report, attr, getattr(report, attr) + 1)
                    _logger.info(
                        "recovery_requeue run_id=%s reason=%s",
                        record.run_id,
                        status.value,
                    )
                elif result == "busy":
                    report.skipped_busy += 1
            except Exception:  # noqa: BLE001
                report.errors += 1
                _logger.warning(
                    "recovery_requeue_failed run_id=%s status=%s",
                    record.run_id,
                    status.value,
                )

    for record in store.scan_status(ExecutionStatus.RUNNING, count=100):
        try:
            result = store.expire_lease_if_needed(
                record.run_id,
                record.conversation_id or record.run_id,
            )
            if result == "ok":
                report.lease_expired += 1
                if backpressure is not None and hasattr(backpressure, "on_lease_expired"):
                    try:
                        backpressure.on_lease_expired(  # type: ignore[attr-defined]
                            conversation_id=record.conversation_id or record.run_id,
                            provider=(record.provider or "default"),
                            user_id_digest=record.user_id_digest or "",
                            group_id_digest=record.group_id_digest or "",
                        )
                    except Exception:  # noqa: BLE001
                        pass
                try:
                    pub = _publish_then_queued(
                        store,
                        enqueue,
                        run_id=record.run_id,
                        idempotency_key=record.idempotency_key,
                        request_fingerprint=record.request_fingerprint,
                        expected_status=ExecutionStatus.ENQUEUE_FAILED.value,
                    )
                    if pub == "busy":
                        report.skipped_busy += 1
                except Exception:  # noqa: BLE001
                    report.errors += 1
                _logger.info(
                    "recovery_lease_expired run_id=%s reason=lease_expired",
                    record.run_id,
                )
        except ExecutionStoreError:
            report.errors += 1

    for record in store.scan_status(ExecutionStatus.RETRY_WAIT, count=100):
        try:
            kind = store.promote_retry_wait(record.run_id)
            if kind != "ok":
                continue
            report.retry_promoted += 1
            if backpressure is not None and hasattr(backpressure, "on_retry_promoted"):
                try:
                    backpressure.on_retry_promoted(  # type: ignore[attr-defined]
                        user_id=getattr(record, "user_id", "") or "",
                        group_id=getattr(record, "group_id", "") or "",
                    )
                except Exception:  # noqa: BLE001
                    pass
            try:
                pub = _publish_then_queued(
                    store,
                    enqueue,
                    run_id=record.run_id,
                    idempotency_key=record.idempotency_key,
                    request_fingerprint=record.request_fingerprint,
                    expected_status=ExecutionStatus.ENQUEUE_FAILED.value,
                )
                if pub == "busy":
                    report.skipped_busy += 1
            except Exception:  # noqa: BLE001
                report.errors += 1
        except ExecutionStoreError:
            report.errors += 1

    if backpressure is not None and hasattr(backpressure, "reconcile_from_ledger"):
        try:
            backpressure.reconcile_from_ledger(store)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            report.errors += 1

    return report
