"""Dead-letter queue helpers and local admin commands (REQ-032)."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from apps.group_agent_api.execution.config import DurableQueueConfig
from apps.group_agent_api.execution.models import (
    QUEUE_SCHEMA_VERSION,
    BrokerDeliveryRef,
    ExecutionRecord,
    ExecutionStatus,
)
from apps.group_agent_api.execution.redis_store import ExecutionStore, ExecutionStoreError

_logger = logging.getLogger(__name__)

EnqueueFn = Callable[[BrokerDeliveryRef], None]


@dataclass(frozen=True)
class SafeDlqView:
    """Safe metadata for DLQ inspect (no payload / tokens / messages)."""

    run_id: str
    idempotency_key: str
    request_fingerprint: str
    status: str
    attempt_count: int
    last_error_code: str | None
    finished_at: float | None
    replay_audit_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "idempotency_key": self.idempotency_key,
            "request_fingerprint": self.request_fingerprint,
            "status": self.status,
            "attempt_count": self.attempt_count,
            "last_error_code": self.last_error_code,
            "finished_at": self.finished_at,
            "replay_audit_count": self.replay_audit_count,
        }


def to_safe_view(record: ExecutionRecord) -> SafeDlqView:
    return SafeDlqView(
        run_id=record.run_id,
        idempotency_key=record.idempotency_key,
        request_fingerprint=record.request_fingerprint,
        status=record.status.value,
        attempt_count=record.attempt_count,
        last_error_code=record.last_error_code,
        finished_at=record.finished_at,
        replay_audit_count=len(record.replay_audit),
    )


class DlqAdmin:
    """Local-only DLQ list / inspect / replay / cancel (no public HTTP)."""

    def __init__(
        self,
        store: ExecutionStore,
        config: DurableQueueConfig,
        enqueue: EnqueueFn,
    ) -> None:
        self._store = store
        self._cfg = config
        self._enqueue = enqueue

    def list_dead_lettered(self, *, limit: int = 50) -> list[SafeDlqView]:
        records = self._store.scan_status(ExecutionStatus.DEAD_LETTERED, count=limit)
        return [to_safe_view(r) for r in records]

    def inspect(self, run_id: str) -> SafeDlqView:
        record = self._store.get(run_id)
        if record is None:
            raise ExecutionStoreError("missing_record")
        return to_safe_view(record)

    def cancel(self, run_id: str, *, operator_id: str, reason: str) -> SafeDlqView:
        record = self._store.get(run_id)
        if record is None:
            raise ExecutionStoreError("missing_record")
        if record.status != ExecutionStatus.DEAD_LETTERED:
            raise ExecutionStoreError("not_dead_lettered")
        self._store.append_replay_audit(
            run_id,
            operator_id=operator_id,
            reason=f"cancel:{reason}",
            replay_id=str(uuid.uuid4()),
        )
        # Keep dead_lettered; audit-only cancel (no identity change).
        return self.inspect(run_id)

    def replay(
        self,
        run_id: str,
        *,
        operator_id: str,
        reason: str,
        replay_id: str | None = None,
    ) -> SafeDlqView:
        """Re-enqueue without changing run/key/fingerprint."""
        record = self._store.get(run_id)
        if record is None:
            raise ExecutionStoreError("missing_record")
        if record.status != ExecutionStatus.DEAD_LETTERED:
            raise ExecutionStoreError("not_dead_lettered")
        rid = replay_id or str(uuid.uuid4())
        # FIX2: dead_lettered → enqueue_failed only via operator/replay CAS.
        self._store.mark_dlq_replay_to_enqueue_failed(
            run_id,
            operator_id=operator_id,
            replay_id=rid,
            reason=reason,
        )
        delivery = BrokerDeliveryRef(
            queue_schema_version=QUEUE_SCHEMA_VERSION,
            run_id=record.run_id,
            idempotency_key=record.idempotency_key,
            request_fingerprint=record.request_fingerprint,
            delivery_id=rid,
        )
        self._enqueue(delivery)
        self._store.mark_queued(
            run_id,
            expected_status=ExecutionStatus.ENQUEUE_FAILED.value,
            delivery_id=rid,
        )
        _logger.info(
            "dlq_replay run_id=%s replay_id=%s operator=%s",
            run_id,
            rid,
            operator_id,
        )
        return self.inspect(run_id)


def push_dlq_index(store: ExecutionStore, config: DurableQueueConfig, run_id: str) -> None:
    """Maintain a lightweight DLQ index set for list operations."""
    key = config.key("dlq")
    store._r.sadd(key, run_id)  # noqa: SLF001
    store._r.expire(key, config.terminal_ttl_s)


def publish_unmappable_poison(
    config: DurableQueueConfig,
    *,
    error_code: str,
    raw_preview: str = "",
    delivery: dict[str, Any] | None = None,
) -> str:
    """Publish safe poison metadata to the dedicated DLQ queue + Redis index (FIX3).

    Redis transport has no dead-letter exchange — we explicitly enqueue inspectable
    metadata and never rely on Reject(requeue=False) alone.
    """
    poison_id = str(uuid.uuid4())
    safe = {
        "poison_id": poison_id,
        "error_code": error_code,
        "queue_schema_version": (delivery or {}).get("queue_schema_version"),
        "run_id": (delivery or {}).get("run_id"),
        "idempotency_key_digest": None,
        "raw_preview": (raw_preview or "")[:200],
        "at": str(uuid.uuid1().time),
    }
    # Digest idempotency key if present (never store raw key secrets beyond ids).
    idem = (delivery or {}).get("idempotency_key")
    if isinstance(idem, str) and idem:
        import hashlib

        safe["idempotency_key_digest"] = hashlib.sha256(idem.encode()).hexdigest()[:16]

    from apps.group_agent_worker.celery_app import get_celery_app

    app = get_celery_app(config)
    app.send_task(
        "group_agent.poison_inspect",
        kwargs={"poison": safe},
        queue=config.dlq_queue,
        ignore_result=True,
    )
    # Redis index for local admin inspect without Celery result backend.
    try:
        from redis import Redis

        r = Redis.from_url(config.redis_url, decode_responses=True)
        idx = config.key("broker_dlq")
        r.hset(idx, poison_id, json.dumps(safe, separators=(",", ":")))
        r.expire(idx, config.terminal_ttl_s)
        r.sadd(config.key("broker_dlq_ids"), poison_id)
        r.expire(config.key("broker_dlq_ids"), config.terminal_ttl_s)
    except Exception:  # noqa: BLE001
        _logger.warning("broker_dlq_index_failed poison_id=%s", poison_id)
    _logger.error(
        "unmappable_poison_published poison_id=%s error_code=%s",
        poison_id,
        error_code,
    )
    return poison_id


def list_from_index(store: ExecutionStore, config: DurableQueueConfig) -> list[str]:
    key = config.key("dlq")
    members = store._r.smembers(key)  # noqa: SLF001
    return sorted(str(m) for m in members)


def format_safe_json(views: list[SafeDlqView]) -> str:
    return json.dumps([v.to_dict() for v in views], indent=2, sort_keys=True)
