"""INTENT-DA-004 callback ACK, pending recovery, and run-global sequence tests."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import uuid
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from redis import Redis

from apps.group_agent_api.agent_factory.integrations.callback_client import (
    CallbackProtocolError,
    CallbackRejected,
    CallbackRunTerminal,
    send_callback_event,
    serialize_callback_body,
)
from apps.group_agent_api.execution.config import (
    DurableQueueConfig,
    load_durable_queue_config,
)
from apps.group_agent_api.execution.crypto import (
    decrypt_callback_body,
    digest_id,
    encrypt_callback_body,
    encrypt_envelope,
)
from apps.group_agent_api.execution.fence import SideEffectFence
from apps.group_agent_api.execution.models import (
    ExecutionRecord,
    ExecutionStatus,
    PendingCallback,
)
from apps.group_agent_api.execution.redis_store import ExecutionStore, ExecutionStoreError


CALLBACK_URL = "http://127.0.0.1:3001/group_agent_callbacks/run-da004"
CALLBACK_SECRET = "intent-da004-callback-secret"
REDIS_URL = os.environ.get(
    "GROUP_AGENT_TEST_REDIS_URL", "redis://127.0.0.1:6379/15"
)
TEST_KEY = bytes(range(32))


class _Response:
    def __init__(self, status: int, body: dict[str, Any]) -> None:
        self.status_code = status
        self._body = body

    def json(self) -> dict[str, Any]:
        return self._body


def _ack(
    *,
    status: int = 200,
    disposition: str = "accepted",
    ignored_reason: str | None = None,
    ack_last_seq: int = 1,
    terminal: bool = False,
    run_status: str = "running",
    error_code: str | None = None,
) -> _Response:
    return _Response(
        status,
        {
            "ack_version": 1,
            "run_id": "run-da004",
            "sent_seq": 1,
            "ack_last_seq": ack_last_seq,
            "disposition": disposition,
            "ignored_reason": ignored_reason,
            "terminal": terminal,
            "run_status": run_status,
            "error_code": error_code,
        },
    )


async def _send(response: _Response, *, event: str = "progress") -> bool:
    envelope = {"run_id": "run-da004", "seq": 1, "event": event}
    with (
        patch.dict(
            os.environ,
            {
                "GROUP_AGENT_CALLBACK_ALLOWED_BASE_URLS": (
                    "http://127.0.0.1:3001/group_agent_callbacks"
                )
            },
        ),
        patch("httpx.AsyncClient.post", new=AsyncMock(return_value=response)),
    ):
        return await send_callback_event(
            callback_url=CALLBACK_URL,
            envelope_dict=envelope,
            secret=CALLBACK_SECRET,
            max_retries=0,
            require_ack=True,
        )


@pytest.mark.asyncio
async def test_strict_ack_accepts_commit_and_exact_duplicate() -> None:
    assert await _send(_ack()) is True
    assert await _send(
        _ack(disposition="ignored", ignored_reason="duplicate_seq")
    ) is True
    assert await _send(
        _ack(terminal=True, run_status="completed"),
        event="final",
    ) is True


@pytest.mark.asyncio
async def test_strict_ack_fails_closed_on_ahead_malformed_and_rejected() -> None:
    with pytest.raises(CallbackProtocolError, match="callback_seq_diverged"):
        await _send(
            _ack(
                disposition="ignored",
                ignored_reason="duplicate_seq",
                ack_last_seq=2,
            )
        )
    malformed = _ack()
    malformed._body["unknown"] = True
    with pytest.raises(CallbackProtocolError, match="callback_ack_invalid"):
        await _send(malformed)
    with pytest.raises(CallbackRejected, match="seq_gap"):
        await _send(
            _ack(
                status=503,
                disposition="rejected",
                ack_last_seq=0,
                error_code="seq_gap",
            )
        )


@pytest.mark.asyncio
async def test_strict_ack_stops_on_authoritative_terminal() -> None:
    with pytest.raises(CallbackRunTerminal) as caught:
        await _send(
            _ack(
                disposition="ignored",
                ignored_reason="run_terminal",
                ack_last_seq=7,
                terminal=True,
                run_status="completed",
            )
        )
    assert caught.value.run_status == "completed"


@pytest.mark.asyncio
async def test_strict_ack_rejects_auth_failure_and_terminal_progress() -> None:
    with pytest.raises(CallbackProtocolError, match="callback_auth_rejected"):
        await _send(_Response(401, {"error": "nonce_replay"}))
    with pytest.raises(CallbackProtocolError, match="callback_ack_invalid"):
        await _send(_ack(terminal=True, run_status="completed"))


def test_pending_callback_crypto_preserves_exact_body_bytes() -> None:
    body = b'{"callback_version":1,"run_id":"run-da004","seq":1}'
    digest = "sha256:" + hashlib.sha256(body).hexdigest()
    encrypted = encrypt_callback_body(
        body,
        key=TEST_KEY,
        key_version="v1",
        run_id="run-da004",
        request_fingerprint=hashlib.sha256(b"request").hexdigest(),
        seq=1,
        body_sha256=digest,
    )
    assert (
        decrypt_callback_body(
            encrypted,
            keys={"v1": TEST_KEY},
            run_id="run-da004",
            request_fingerprint=hashlib.sha256(b"request").hexdigest(),
            seq=1,
            body_sha256=digest,
        )
        == body
    )


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> ExecutionStore:
    prefix = f"ga:test:da004:{uuid.uuid4().hex[:10]}"
    monkeypatch.setenv("GROUP_AGENT_DURABLE_QUEUE_ENABLED", "1")
    monkeypatch.setenv("GROUP_AGENT_REDIS_URL", REDIS_URL)
    monkeypatch.setenv("GROUP_AGENT_REDIS_PREFIX", prefix)
    monkeypatch.setenv("GROUP_AGENT_CELERY_QUEUE", "group_agent.test.runs")
    monkeypatch.setenv("GROUP_AGENT_DLQ_QUEUE", "group_agent.test.dlq")
    monkeypatch.setenv(
        "GROUP_AGENT_QUEUE_PAYLOAD_KEYS",
        "v1:" + base64.b64encode(TEST_KEY).decode("ascii"),
    )
    monkeypatch.setenv("GROUP_AGENT_QUEUE_PAYLOAD_CURRENT_VERSION", "v1")
    monkeypatch.setenv("GROUP_AGENT_LEASE_TTL_S", "30")
    monkeypatch.setenv("GROUP_AGENT_HEARTBEAT_INTERVAL_S", "10")
    monkeypatch.setenv("GROUP_AGENT_VISIBILITY_TIMEOUT_S", "240")
    monkeypatch.setenv("GROUP_AGENT_TASK_SOFT_LIMIT_S", "150")
    monkeypatch.setenv("GROUP_AGENT_TASK_HARD_LIMIT_S", "180")
    monkeypatch.setenv("GROUP_AGENT_WORKER_INSTANCE_ID", "da004-test-worker")
    cfg = load_durable_queue_config(require_enabled=True)
    assert isinstance(cfg, DurableQueueConfig)
    client = Redis.from_url(REDIS_URL, decode_responses=True)
    return ExecutionStore(client, cfg)


def _claimed(store: ExecutionStore, run_id: str = "run-da004") -> Any:
    fingerprint = hashlib.sha256(b"request").hexdigest()
    encrypted = encrypt_envelope(
        {"request": "test"},
        key=TEST_KEY,
        key_version="v1",
        run_id=run_id,
        idempotency_key="idem-da004",
        request_fingerprint=fingerprint,
        schema_version=1,
    )
    store.create_or_get(
        record=ExecutionRecord(
            run_id=run_id,
            idempotency_key="idem-da004",
            request_schema_version=1,
            request_fingerprint=fingerprint,
            queue_schema_version=1,
            status=ExecutionStatus.QUEUED,
            created_at=store.redis_time(),
            payload_ciphertext=encrypted,
            conversation_id="conv-da004",
            user_id_digest=digest_id("user"),
            group_id_digest=digest_id("group"),
        )
    )
    outcome = store.claim_lease(
        run_id=run_id,
        conversation_id="conv-da004",
        owner="worker-1",
    )
    assert outcome.claim is not None
    return outcome.claim


@pytest.mark.asyncio
async def test_durable_emit_persists_before_send_and_commits_seq(store: ExecutionStore) -> None:
    claim = _claimed(store)
    fence = SideEffectFence(store=store, claim=claim, conversation_id="conv-da004")
    seen: list[bytes] = []

    def build(seq: int, event: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {"callback_version": 1, "run_id": claim.run_id, "seq": seq, "event": event, "payload": payload}

    async def send(envelope: dict[str, Any], body: bytes) -> bool:
        pending = store.get(claim.run_id).pending_callback
        assert pending is not None and pending.seq == envelope["seq"]
        assert pending.body_sha256 == "sha256:" + hashlib.sha256(body).hexdigest()
        seen.append(body)
        return True

    assert await fence.emit_durable("progress", {"phase": "started"}, build_envelope=build, send_body=send)
    record = store.get(claim.run_id)
    assert record is not None
    assert record.last_callback_seq == 1
    assert record.pending_callback is None
    assert seen == [serialize_callback_body(build(1, "progress", {"phase": "started"}))]


@pytest.mark.asyncio
async def test_new_lease_adopts_exact_pending_body_then_continues_global_seq(store: ExecutionStore) -> None:
    first = _claimed(store)
    record = store.get(first.run_id)
    assert record is not None
    body = b'{"callback_version":1,"run_id":"run-da004","seq":1,"event":"progress"}'
    digest = "sha256:" + hashlib.sha256(body).hexdigest()
    encrypted = encrypt_callback_body(
        body,
        key=TEST_KEY,
        key_version="v1",
        run_id=record.run_id,
        request_fingerprint=record.request_fingerprint,
        seq=1,
        body_sha256=digest,
    )
    store.create_pending_callback(
        first,
        attempt_number=record.attempt_count,
        pending=PendingCallback(
            seq=1,
            body_sha256=digest,
            producer_attempt_id=first.attempt_id,
            producer_attempt_number=record.attempt_count,
            routing_snapshot_hash=None,
            created_at="2026-09-15T08:00:00Z",
            ciphertext=encrypted,
        ),
    )
    # Simulate lease expiry recovery without mutating callback state.
    key = store.run_key(first.run_id)
    store._r.hset(key, mapping={"status": "queued"})  # noqa: SLF001
    store._r.hdel(  # noqa: SLF001
        key,
        "lease_owner",
        "lease_token_digest",
        "lease_expires_at",
        "current_attempt_id",
    )
    store._r.delete(store.conv_key("conv-da004"))  # noqa: SLF001
    second_outcome = store.claim_lease(
        run_id=first.run_id,
        conversation_id="conv-da004",
        owner="worker-2",
    )
    assert second_outcome.claim is not None
    second = second_outcome.claim
    fence = SideEffectFence(store=store, claim=second, conversation_id="conv-da004")

    with pytest.raises(ExecutionStoreError, match="fence_attempt_mismatch"):
        store.commit_callback_ack(first, seq=1, body_sha256=digest)

    async def send(envelope: dict[str, Any], frozen: bytes) -> bool:
        assert frozen == body
        pending = store.get(first.run_id).pending_callback
        assert pending is not None
        assert pending.producer_attempt_id == first.attempt_id
        assert pending.adopted_by_attempt_id == second.attempt_id
        return True

    assert await fence.recover_pending(send_body=send)
    final = store.get(first.run_id)
    assert final is not None
    assert final.attempt_count == 2
    assert final.last_callback_seq == 1
    assert final.pending_callback is None


@pytest.mark.asyncio
async def test_concurrent_callbacks_are_serialized_with_run_global_seq(
    store: ExecutionStore,
) -> None:
    claim = _claimed(store, run_id="run-da004-concurrent")
    fence = SideEffectFence(
        store=store,
        claim=claim,
        conversation_id="conv-da004",
    )
    observed: list[int] = []

    def build(seq: int, event: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "callback_version": 1,
            "run_id": claim.run_id,
            "seq": seq,
            "event": event,
            "payload": payload,
        }

    async def send(envelope: dict[str, Any], body: bytes) -> bool:
        del body
        observed.append(envelope["seq"])
        await asyncio.sleep(0)
        return True

    await asyncio.gather(
        fence.emit_durable(
            "progress",
            {"phase": "intent_resolving"},
            build_envelope=build,
            send_body=send,
        ),
        fence.emit_durable(
            "final",
            {"answer": "done"},
            build_envelope=build,
            send_body=send,
        ),
    )
    record = store.get(claim.run_id)
    assert record is not None
    assert observed == [1, 2]
    assert record.last_callback_seq == 2
    assert record.pending_callback is None
