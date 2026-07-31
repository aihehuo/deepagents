"""Durable queue configuration and fail-closed validation (REQ-032)."""

from __future__ import annotations

import base64
import os
import re
from dataclasses import dataclass
from typing import Mapping


_TRUE = {"1", "true", "yes", "on"}
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _env_bool(name: str, default: bool = False) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in _TRUE


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    return float(raw)


def durable_queue_enabled() -> bool:
    """Startup-time feature flag (never per-request)."""
    return _env_bool("GROUP_AGENT_DURABLE_QUEUE_ENABLED", default=False)


def parse_payload_keys(raw: str | None) -> dict[str, bytes]:
    """Parse ``v1:<b64>,v2:<b64>`` into version → 32-byte key map."""
    text = (raw or "").strip()
    if not text:
        return {}
    keys: dict[str, bytes] = {}
    for part in text.split(","):
        piece = part.strip()
        if not piece:
            continue
        if ":" not in piece:
            raise ValueError("payload_key_entry_missing_version")
        version, _, encoded = piece.partition(":")
        version = version.strip()
        encoded = encoded.strip()
        if not version or not encoded:
            raise ValueError("payload_key_entry_empty")
        try:
            key = base64.b64decode(encoded, validate=True)
        except Exception as exc:  # noqa: BLE001
            raise ValueError("payload_key_b64_invalid") from exc
        if len(key) != 32:
            raise ValueError("payload_key_length_invalid")
        keys[version] = key
    return keys


@dataclass(frozen=True)
class DurableQueueConfig:
    """Validated durable-queue settings for API and worker processes."""

    redis_url: str
    redis_prefix: str
    celery_queue: str
    dlq_queue: str
    payload_keys: Mapping[str, bytes]
    payload_current_version: str
    lease_ttl_s: float
    heartbeat_interval_s: float
    visibility_timeout_s: float
    soft_time_limit_s: float
    hard_time_limit_s: float
    max_attempts: int
    retry_base_s: float
    retry_max_s: float
    queue_max_depth: int
    max_running: int
    provider_max_running: int
    user_max_queued: int
    group_max_queued: int
    admission_timeout_s: float
    record_ttl_s: int
    terminal_ttl_s: int
    heartbeat_fail_threshold: int
    worker_instance_id: str

    @property
    def current_payload_key(self) -> bytes:
        return self.payload_keys[self.payload_current_version]

    def key(self, *parts: str) -> str:
        """Build a namespaced Redis key under the configured prefix."""
        clean = [p.strip(":") for p in parts if p and str(p).strip()]
        return ":".join([self.redis_prefix.rstrip(":"), *clean])


def load_durable_queue_config(*, require_enabled: bool = False) -> DurableQueueConfig | None:
    """Load config when durable mode is on; return None when disabled.

    Args:
        require_enabled: When True, raise if durable mode is off.

    Raises:
        RuntimeError: Fail-closed on missing/illegal durable configuration.
    """
    enabled = durable_queue_enabled()
    if not enabled:
        if require_enabled:
            raise RuntimeError("GROUP_AGENT_DURABLE_QUEUE_ENABLED must be 1")
        return None

    redis_url = (os.environ.get("GROUP_AGENT_REDIS_URL") or "").strip()
    if not redis_url:
        raise RuntimeError("GROUP_AGENT_REDIS_URL is required when durable queue is enabled")

    redis_prefix = (os.environ.get("GROUP_AGENT_REDIS_PREFIX") or "ga:exec:v1").strip()
    if not redis_prefix:
        raise RuntimeError("GROUP_AGENT_REDIS_PREFIX must not be empty")

    celery_queue = (os.environ.get("GROUP_AGENT_CELERY_QUEUE") or "group_agent.runs").strip()
    if not celery_queue:
        raise RuntimeError("GROUP_AGENT_CELERY_QUEUE must not be empty")

    dlq_queue = (os.environ.get("GROUP_AGENT_DLQ_QUEUE") or "group_agent.dlq").strip()
    if not dlq_queue:
        raise RuntimeError("GROUP_AGENT_DLQ_QUEUE must not be empty")

    try:
        payload_keys = parse_payload_keys(os.environ.get("GROUP_AGENT_QUEUE_PAYLOAD_KEYS"))
    except ValueError as exc:
        raise RuntimeError(f"GROUP_AGENT_QUEUE_PAYLOAD_KEYS invalid: {exc}") from exc
    if not payload_keys:
        raise RuntimeError("GROUP_AGENT_QUEUE_PAYLOAD_KEYS is required when durable queue is enabled")

    current_version = (
        os.environ.get("GROUP_AGENT_QUEUE_PAYLOAD_CURRENT_VERSION") or ""
    ).strip()
    if not current_version or current_version not in payload_keys:
        raise RuntimeError(
            "GROUP_AGENT_QUEUE_PAYLOAD_CURRENT_VERSION must reference an entry in "
            "GROUP_AGENT_QUEUE_PAYLOAD_KEYS"
        )

    lease_ttl_s = _env_float("GROUP_AGENT_LEASE_TTL_S", 30.0)
    heartbeat_interval_s = _env_float("GROUP_AGENT_HEARTBEAT_INTERVAL_S", 10.0)
    visibility_timeout_s = _env_float("GROUP_AGENT_VISIBILITY_TIMEOUT_S", 240.0)
    soft_time_limit_s = _env_float("GROUP_AGENT_TASK_SOFT_LIMIT_S", 150.0)
    hard_time_limit_s = _env_float("GROUP_AGENT_TASK_HARD_LIMIT_S", 180.0)

    if not (0 < heartbeat_interval_s < lease_ttl_s / 2):
        raise RuntimeError("heartbeat_interval must be < lease_ttl / 2")
    if not (lease_ttl_s < visibility_timeout_s):
        raise RuntimeError("lease_ttl must be < visibility_timeout")
    if not (hard_time_limit_s < visibility_timeout_s):
        raise RuntimeError("hard_time_limit must be < visibility_timeout")
    if not (0 < soft_time_limit_s <= hard_time_limit_s):
        raise RuntimeError("soft_time_limit must be <= hard_time_limit")

    max_attempts = _env_int("GROUP_AGENT_MAX_ATTEMPTS", 5)
    if not (1 <= max_attempts <= 20):
        raise RuntimeError("GROUP_AGENT_MAX_ATTEMPTS out of bounds")

    retry_base_s = _env_float("GROUP_AGENT_RETRY_BASE_S", 2.0)
    retry_max_s = _env_float("GROUP_AGENT_RETRY_MAX_S", 120.0)
    if not (0 < retry_base_s <= retry_max_s):
        raise RuntimeError("retry base/max delay invalid")

    queue_max_depth = _env_int("GROUP_AGENT_QUEUE_MAX_DEPTH", 500)
    max_running = _env_int("GROUP_AGENT_MAX_RUNNING", 20)
    provider_max_running = _env_int("GROUP_AGENT_PROVIDER_MAX_RUNNING", 10)
    user_max_queued = _env_int("GROUP_AGENT_USER_MAX_QUEUED", 5)
    group_max_queued = _env_int("GROUP_AGENT_GROUP_MAX_QUEUED", 50)
    for name, value in (
        ("queue_max_depth", queue_max_depth),
        ("max_running", max_running),
        ("provider_max_running", provider_max_running),
        ("user_max_queued", user_max_queued),
        ("group_max_queued", group_max_queued),
    ):
        if value < 1:
            raise RuntimeError(f"{name} must be >= 1")

    worker_instance_id = (
        os.environ.get("GROUP_AGENT_WORKER_INSTANCE_ID")
        or os.environ.get("HOSTNAME")
        or "ga-worker-local"
    ).strip()
    if not worker_instance_id:
        raise RuntimeError("GROUP_AGENT_WORKER_INSTANCE_ID must not be empty")

    return DurableQueueConfig(
        redis_url=redis_url,
        redis_prefix=redis_prefix,
        celery_queue=celery_queue,
        dlq_queue=dlq_queue,
        payload_keys=dict(payload_keys),
        payload_current_version=current_version,
        lease_ttl_s=lease_ttl_s,
        heartbeat_interval_s=heartbeat_interval_s,
        visibility_timeout_s=visibility_timeout_s,
        soft_time_limit_s=soft_time_limit_s,
        hard_time_limit_s=hard_time_limit_s,
        max_attempts=max_attempts,
        retry_base_s=retry_base_s,
        retry_max_s=retry_max_s,
        queue_max_depth=queue_max_depth,
        max_running=max_running,
        provider_max_running=provider_max_running,
        user_max_queued=user_max_queued,
        group_max_queued=group_max_queued,
        admission_timeout_s=_env_float("GROUP_AGENT_ADMISSION_TIMEOUT_S", 30.0),
        record_ttl_s=_env_int("GROUP_AGENT_EXEC_RECORD_TTL_S", 7 * 24 * 3600),
        terminal_ttl_s=_env_int("GROUP_AGENT_EXEC_TERMINAL_TTL_S", 7 * 24 * 3600),
        heartbeat_fail_threshold=_env_int("GROUP_AGENT_HEARTBEAT_FAIL_THRESHOLD", 3),
        worker_instance_id=worker_instance_id,
    )


def validate_request_fingerprint(value: str) -> str:
    """Require exact 64-char lowercase SHA-256 hex — no trim/case repair."""
    if not isinstance(value, str) or not _HEX64.match(value):
        raise ValueError("request_fingerprint_invalid")
    return value
