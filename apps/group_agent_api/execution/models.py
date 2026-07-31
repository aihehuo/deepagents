"""Execution ledger models and status machine (REQ-032)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ExecutionStatus(str, Enum):
    """Durable execution status values stored in Redis."""

    ACCEPTED = "accepted"
    ENQUEUE_FAILED = "enqueue_failed"
    QUEUED = "queued"
    RUNNING = "running"
    RETRY_WAIT = "retry_wait"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DEAD_LETTERED = "dead_lettered"


TERMINAL_STATUSES = frozenset(
    {
        ExecutionStatus.SUCCEEDED,
        ExecutionStatus.FAILED,
        ExecutionStatus.DEAD_LETTERED,
    }
)

QUEUE_SCHEMA_VERSION = 1
REQUEST_SCHEMA_VERSION = 1


class RetryClass(str, Enum):
    """Retry taxonomy for worker / recovery decisions."""

    PERMANENT = "permanent"
    TRANSIENT = "transient"
    UNCERTAIN = "uncertain"
    WORKER_LOST = "worker_lost"
    POISON = "poison"


@dataclass
class EncryptedPayload:
    """AES-GCM ciphertext envelope stored on the execution record."""

    key_version: str
    nonce_b64: str
    ciphertext_b64: str
    tag_b64: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EncryptedPayload:
        return cls(
            key_version=str(data["key_version"]),
            nonce_b64=str(data["nonce_b64"]),
            ciphertext_b64=str(data["ciphertext_b64"]),
            tag_b64=str(data["tag_b64"]),
        )


@dataclass
class ExecutionRecord:
    """Single Run execution mirror held by Deep Agents Redis ledger."""

    run_id: str
    idempotency_key: str
    request_schema_version: int
    request_fingerprint: str
    queue_schema_version: int
    status: ExecutionStatus
    attempt_count: int = 0
    current_attempt_id: str | None = None
    lease_owner: str | None = None
    lease_token_digest: str | None = None
    lease_expires_at: float | None = None
    heartbeat_at: float | None = None
    next_attempt_at: float | None = None
    last_error_code: str | None = None
    created_at: float = 0.0
    queued_at: float | None = None
    started_at: float | None = None
    finished_at: float | None = None
    payload_key_version: str | None = None
    payload_ciphertext: EncryptedPayload | None = None
    conversation_id: str | None = None
    user_id_digest: str | None = None
    group_id_digest: str | None = None
    provider: str | None = None
    terminal_fence: str | None = None
    fencing_token: int = 0
    replay_audit: list[dict[str, str]] = field(default_factory=list)

    def to_redis_mapping(self) -> dict[str, str]:
        """Serialize to Redis hash string fields (no secrets / plaintext)."""
        mapping: dict[str, str] = {
            "run_id": self.run_id,
            "idempotency_key": self.idempotency_key,
            "request_schema_version": str(self.request_schema_version),
            "request_fingerprint": self.request_fingerprint,
            "queue_schema_version": str(self.queue_schema_version),
            "status": self.status.value,
            "attempt_count": str(self.attempt_count),
            "created_at": str(self.created_at),
            "fencing_token": str(self.fencing_token),
        }
        optional_str = {
            "current_attempt_id": self.current_attempt_id,
            "lease_owner": self.lease_owner,
            "lease_token_digest": self.lease_token_digest,
            "last_error_code": self.last_error_code,
            "payload_key_version": self.payload_key_version,
            "conversation_id": self.conversation_id,
            "user_id_digest": self.user_id_digest,
            "group_id_digest": self.group_id_digest,
            "provider": self.provider,
            "terminal_fence": self.terminal_fence,
        }
        for key, value in optional_str.items():
            if value is not None:
                mapping[key] = value
        optional_float = {
            "lease_expires_at": self.lease_expires_at,
            "heartbeat_at": self.heartbeat_at,
            "next_attempt_at": self.next_attempt_at,
            "queued_at": self.queued_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }
        for key, value in optional_float.items():
            if value is not None:
                mapping[key] = str(value)
        if self.payload_ciphertext is not None:
            mapping["payload_nonce_b64"] = self.payload_ciphertext.nonce_b64
            mapping["payload_ciphertext_b64"] = self.payload_ciphertext.ciphertext_b64
            mapping["payload_tag_b64"] = self.payload_ciphertext.tag_b64
            mapping["payload_key_version"] = self.payload_ciphertext.key_version
        if self.replay_audit:
            import json

            mapping["replay_audit_json"] = json.dumps(self.replay_audit, separators=(",", ":"))
        return mapping

    @classmethod
    def from_redis_mapping(cls, data: dict[str, Any]) -> ExecutionRecord:
        """Deserialize from Redis hash fields."""
        import json

        if not data:
            raise ValueError("empty_execution_record")

        def _f(name: str) -> float | None:
            raw = data.get(name)
            if raw is None or raw == "":
                return None
            return float(raw)

        def _i(name: str, default: int = 0) -> int:
            raw = data.get(name)
            if raw is None or raw == "":
                return default
            return int(raw)

        ciphertext = None
        if data.get("payload_ciphertext_b64"):
            ciphertext = EncryptedPayload(
                key_version=str(data.get("payload_key_version") or ""),
                nonce_b64=str(data.get("payload_nonce_b64") or ""),
                ciphertext_b64=str(data["payload_ciphertext_b64"]),
                tag_b64=str(data.get("payload_tag_b64") or ""),
            )
        replay_audit: list[dict[str, str]] = []
        raw_audit = data.get("replay_audit_json")
        if raw_audit:
            loaded = json.loads(str(raw_audit))
            if isinstance(loaded, list):
                replay_audit = [dict(item) for item in loaded if isinstance(item, dict)]

        return cls(
            run_id=str(data["run_id"]),
            idempotency_key=str(data["idempotency_key"]),
            request_schema_version=_i("request_schema_version", REQUEST_SCHEMA_VERSION),
            request_fingerprint=str(data["request_fingerprint"]),
            queue_schema_version=_i("queue_schema_version", QUEUE_SCHEMA_VERSION),
            status=ExecutionStatus(str(data["status"])),
            attempt_count=_i("attempt_count", 0),
            current_attempt_id=(str(data["current_attempt_id"]) if data.get("current_attempt_id") else None),
            lease_owner=(str(data["lease_owner"]) if data.get("lease_owner") else None),
            lease_token_digest=(str(data["lease_token_digest"]) if data.get("lease_token_digest") else None),
            lease_expires_at=_f("lease_expires_at"),
            heartbeat_at=_f("heartbeat_at"),
            next_attempt_at=_f("next_attempt_at"),
            last_error_code=(str(data["last_error_code"]) if data.get("last_error_code") else None),
            created_at=float(data.get("created_at") or 0.0),
            queued_at=_f("queued_at"),
            started_at=_f("started_at"),
            finished_at=_f("finished_at"),
            payload_key_version=(str(data["payload_key_version"]) if data.get("payload_key_version") else None),
            payload_ciphertext=ciphertext,
            conversation_id=(str(data["conversation_id"]) if data.get("conversation_id") else None),
            user_id_digest=(str(data["user_id_digest"]) if data.get("user_id_digest") else None),
            group_id_digest=(str(data["group_id_digest"]) if data.get("group_id_digest") else None),
            provider=(str(data["provider"]) if data.get("provider") else None),
            terminal_fence=(str(data["terminal_fence"]) if data.get("terminal_fence") else None),
            fencing_token=_i("fencing_token", 0),
            replay_audit=replay_audit,
        )


@dataclass(frozen=True)
class BrokerDeliveryRef:
    """Minimal Celery broker message — no secrets or user content."""

    queue_schema_version: int
    run_id: str
    idempotency_key: str
    request_fingerprint: str
    delivery_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "queue_schema_version": self.queue_schema_version,
            "run_id": self.run_id,
            "idempotency_key": self.idempotency_key,
            "request_fingerprint": self.request_fingerprint,
            "delivery_id": self.delivery_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BrokerDeliveryRef:
        return cls(
            queue_schema_version=int(data["queue_schema_version"]),
            run_id=str(data["run_id"]),
            idempotency_key=str(data["idempotency_key"]),
            request_fingerprint=str(data["request_fingerprint"]),
            delivery_id=str(data["delivery_id"]),
        )


@dataclass(frozen=True)
class LeaseClaim:
    """In-memory lease ownership after a successful claim."""

    run_id: str
    attempt_id: str
    lease_token: str
    lease_owner: str
    lease_expires_at: float
    fencing_token: int = 0
