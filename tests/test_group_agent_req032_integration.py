"""REQ-032 integration / fault-injection tests (isolated Redis DB, no prod)."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import Request
from redis import Redis

from apps.group_agent_api.agent_factory.capability import CapabilityTier
from apps.group_agent_api.agent_factory.integrations.membership_client import MembershipResult
from apps.group_agent_api.agent_factory.integrations.principal import SessionPrincipal
from apps.group_agent_api.app.models import AsyncCallRequest
from apps.group_agent_api.app.session import TrustedSession
from apps.group_agent_api.app.state import AppState
from apps.group_agent_api.execution.admission import admit_durable_async_call
from apps.group_agent_api.execution.backpressure import BackpressureController
from apps.group_agent_api.execution.config import load_durable_queue_config
from apps.group_agent_api.execution.crypto import digest_id, encrypt_envelope
from apps.group_agent_api.execution.dlq import DlqAdmin, push_dlq_index
from apps.group_agent_api.execution.models import (
    BrokerDeliveryRef,
    ExecutionRecord,
    ExecutionStatus,
)
from apps.group_agent_api.execution.recovery import run_recovery_once
from apps.group_agent_api.execution.redis_store import ExecutionStore, ExecutionStoreError
from apps.group_agent_api.execution.retry import decide_retry

TEST_KEY_V1 = base64.b64encode(bytes(range(32))).decode()
REDIS_URL = os.environ.get("GROUP_AGENT_TEST_REDIS_URL", "redis://127.0.0.1:6379/15")
PREFIX = f"ga:test:req032i:{uuid.uuid4().hex[:8]}"


def _fp(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()


@pytest.fixture
def env_cfg(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GROUP_AGENT_DURABLE_QUEUE_ENABLED", "1")
    monkeypatch.setenv("GROUP_AGENT_REDIS_URL", REDIS_URL)
    monkeypatch.setenv("GROUP_AGENT_REDIS_PREFIX", PREFIX)
    monkeypatch.setenv("GROUP_AGENT_CELERY_QUEUE", "group_agent.test.runs")
    monkeypatch.setenv("GROUP_AGENT_DLQ_QUEUE", "group_agent.test.dlq")
    monkeypatch.setenv("GROUP_AGENT_QUEUE_PAYLOAD_KEYS", f"v1:{TEST_KEY_V1}")
    monkeypatch.setenv("GROUP_AGENT_QUEUE_PAYLOAD_CURRENT_VERSION", "v1")
    monkeypatch.setenv("GROUP_AGENT_LEASE_TTL_S", "2")
    monkeypatch.setenv("GROUP_AGENT_HEARTBEAT_INTERVAL_S", "0.5")
    monkeypatch.setenv("GROUP_AGENT_VISIBILITY_TIMEOUT_S", "30")
    monkeypatch.setenv("GROUP_AGENT_TASK_SOFT_LIMIT_S", "10")
    monkeypatch.setenv("GROUP_AGENT_TASK_HARD_LIMIT_S", "15")
    monkeypatch.setenv("GROUP_AGENT_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("GROUP_AGENT_WORKER_INSTANCE_ID", "integ-worker-a")
    monkeypatch.setenv(
        "GROUP_AGENT_CALLBACK_ALLOWED_BASE_URLS",
        "http://127.0.0.1:9/group_agent_callbacks",
    )
    cfg = load_durable_queue_config(require_enabled=True)
    assert cfg is not None
    client = Redis.from_url(REDIS_URL, decode_responses=True)
    for key in client.scan_iter(match=f"{PREFIX}:*", count=200):
        client.delete(key)
    store = ExecutionStore(client, cfg)
    return cfg, store


def _session() -> TrustedSession:
    return TrustedSession(
        principal=SessionPrincipal(
            user_id="u1",
            unionid="union-1",
            user_token="tok-user",
            source="body_stub",
            group_token="tok-group",
        ),
        membership=MembershipResult(tier=CapabilityTier.in_group, source="stub"),
        group_id="g1",
        group_token="tok-group",
    )


def _seed_queued(store: ExecutionStore, cfg, run_id: str, idem: str, fp: str) -> None:
    enc = encrypt_envelope(
        {
            "thread_id": "tid",
            "request": {
                "run_id": run_id,
                "idempotency_key": idem,
                "user_id": "u1",
                "unionid": "union-1",
                "group_id": "g1",
                "conversation_id": "c1",
                "message": "hi",
                "callback_url": "http://127.0.0.1:9/group_agent_callbacks",
                "membership": "in_group",
            },
            "session": {
                "user_id": "u1",
                "unionid": "union-1",
                "user_token": "tok-user",
                "group_token": "tok-group",
                "group_id": "g1",
                "membership_tier": "in_group",
                "membership_source": "stub",
                "principal_source": "body_stub",
            },
        },
        key=cfg.current_payload_key,
        key_version="v1",
        run_id=run_id,
        idempotency_key=idem,
        request_fingerprint=fp,
        schema_version=1,
    )
    store.create_or_get(
        record=ExecutionRecord(
            run_id=run_id,
            idempotency_key=idem,
            request_schema_version=1,
            request_fingerprint=fp,
            queue_schema_version=1,
            status=ExecutionStatus.ACCEPTED,
            created_at=store.redis_time(),
            payload_ciphertext=enc,
            conversation_id="c1",
            user_id_digest=digest_id("u1"),
            group_id_digest=digest_id("g1"),
        )
    )
    store.mark_queued(run_id, expected_status="accepted")


def test_fault_enqueue_then_crash_before_ack_retry_same_run(env_cfg) -> None:
    """Scenario 1: ACK lost → retry same job."""
    cfg, store = env_cfg
    deliveries: list[BrokerDeliveryRef] = []
    bp = BackpressureController(store._r, cfg)  # noqa: SLF001
    req = AsyncCallRequest.model_validate(
        {
            "run_id": "ga_f1",
            "idempotency_key": "idem-f1",
            "user_id": "u1",
            "unionid": "union-1",
            "group_id": "g1",
            "conversation_id": "c1",
            "message": "hi",
            "callback_url": "http://127.0.0.1:9/group_agent_callbacks",
            "membership": "in_group",
            "request_schema_version": 1,
            "request_fingerprint": _fp("f1"),
            "queue_schema_version": 1,
        }
    )
    from apps.group_agent_api.agent_factory.integrations import callback_client as cc

    with patch.object(cc, "validate_and_normalize_callback_url", lambda u: u):
        r1 = admit_durable_async_call(
            req=req,
            session=_session(),
            thread_id="tid",
            store=store,
            config=cfg,
            backpressure=bp,
            enqueue=lambda d: deliveries.append(d),
        )
        r2 = admit_durable_async_call(
            req=req,
            session=_session(),
            thread_id="tid",
            store=store,
            config=cfg,
            backpressure=bp,
            enqueue=lambda d: deliveries.append(d),
        )
    assert r1.response.run_id == r2.response.run_id == "ga_f1"
    assert store.get("ga_f1").status == ExecutionStatus.QUEUED  # type: ignore[union-attr]


def test_fault_sigkill_after_claim_takeover(env_cfg) -> None:
    """Scenario 2/7: claim then kill → lease expire → other worker."""
    cfg, store = env_cfg
    fp = _fp("f2")
    _seed_queued(store, cfg, "ga_f2", "idem-f2", fp)
    c1_out = store.claim_lease(run_id="ga_f2", conversation_id="c1", owner="integ-worker-a")
    assert c1_out.kind == "claimed" and c1_out.claim is not None
    c1 = c1_out.claim
    store._r.hset(store.run_key("ga_f2"), "lease_expires_at", "1")  # noqa: SLF001
    assert store.expire_lease_if_needed("ga_f2", "c1") == "ok"
    assert store.get("ga_f2").status == ExecutionStatus.ENQUEUE_FAILED  # type: ignore[union-attr]
    store.mark_queued("ga_f2", expected_status="enqueue_failed")
    c2_out = store.claim_lease(run_id="ga_f2", conversation_id="c1", owner="integ-worker-b")
    assert c2_out.kind == "claimed" and c2_out.claim is not None
    c2 = c2_out.claim
    assert c2.attempt_id != c1.attempt_id
    with pytest.raises(ExecutionStoreError):
        store.finish(c1, conversation_id="c1", status=ExecutionStatus.SUCCEEDED)
    store.finish(c2, conversation_id="c1", status=ExecutionStatus.SUCCEEDED)
    rec = store.get("ga_f2")
    assert rec is not None
    assert rec.status == ExecutionStatus.SUCCEEDED
    assert rec.attempt_count == 2


def test_fault_two_workers_race_single_lease(env_cfg) -> None:
    cfg, store = env_cfg
    fp = _fp("f7")
    _seed_queued(store, cfg, "ga_f7", "idem-f7", fp)

    def _claim(owner: str):
        return store.claim_lease(run_id="ga_f7", conversation_id="c1", owner=owner)

    with ThreadPoolExecutor(max_workers=2) as pool:
        winners = [f.result() for f in [pool.submit(_claim, f"w{i}") for i in range(2)]]
    claimed = [c for c in winners if c.kind == "claimed"]
    assert len(claimed) == 1


def test_fault_max_attempts_dlq_and_replay(env_cfg) -> None:
    """Scenario 9/10: max attempts → DLQ → replay keeps identity."""
    cfg, store = env_cfg
    fp = _fp("f9")
    _seed_queued(store, cfg, "ga_f9", "idem-f9", fp)
    out = store.claim_lease(run_id="ga_f9", conversation_id="c1", owner="w1")
    assert out.kind == "claimed" and out.claim is not None
    claim = out.claim
    for _ in range(2):
        store._r.hincrby(store.run_key("ga_f9"), "attempt_count", 1)  # noqa: SLF001
    rec = store.get("ga_f9")
    assert rec is not None
    decision = decide_retry(
        error_code="llm_rate_limited",
        attempt_count=rec.attempt_count,
        max_attempts=cfg.max_attempts,
        base_s=cfg.retry_base_s,
        max_s=cfg.retry_max_s,
    )
    assert decision.dead_letter
    store.finish(
        claim,
        conversation_id="c1",
        status=ExecutionStatus.DEAD_LETTERED,
        error_code=decision.reason_code,
    )
    push_dlq_index(store, cfg, "ga_f9")
    deliveries: list[BrokerDeliveryRef] = []
    admin = DlqAdmin(store, cfg, enqueue=lambda d: deliveries.append(d))
    view = admin.replay("ga_f9", operator_id="op", reason="fault_test")
    assert view.run_id == "ga_f9"
    assert view.request_fingerprint == fp
    assert len(deliveries) == 1
    assert store.get("ga_f9").status == ExecutionStatus.QUEUED  # type: ignore[union-attr]


def test_fault_recovery_requeues_enqueue_failed(env_cfg) -> None:
    cfg, store = env_cfg
    fp = _fp("f5")
    enc = encrypt_envelope(
        {"thread_id": "t", "request": {}, "session": {}},
        key=cfg.current_payload_key,
        key_version="v1",
        run_id="ga_f5",
        idempotency_key="idem-f5",
        request_fingerprint=fp,
        schema_version=1,
    )
    store.create_or_get(
        record=ExecutionRecord(
            run_id="ga_f5",
            idempotency_key="idem-f5",
            request_schema_version=1,
            request_fingerprint=fp,
            queue_schema_version=1,
            status=ExecutionStatus.ACCEPTED,
            created_at=store.redis_time() - 120,
            payload_ciphertext=enc,
            conversation_id="c1",
            user_id_digest=digest_id("u1"),
            group_id_digest=digest_id("g1"),
        )
    )
    store.mark_accepted_enqueue_failed("ga_f5")
    deliveries: list[BrokerDeliveryRef] = []
    report = run_recovery_once(
        store, cfg, lambda d: deliveries.append(d), accepted_timeout_s=1.0
    )
    assert report.enqueue_failed_requeued >= 1
    assert store.get("ga_f5").status == ExecutionStatus.QUEUED  # type: ignore[union-attr]


def test_fault_quota_retry_after(env_cfg) -> None:
    cfg, store = env_cfg
    bp = BackpressureController(store._r, cfg)  # noqa: SLF001
    store._r.set(cfg.key("metrics", "queued_global"), cfg.queue_max_depth)  # noqa: SLF001
    d = bp.check_and_reserve(
        user_id="u9", group_id="g9", conversation_id="c9", provider="default"
    )
    assert d.allowed is False
    assert d.http_status == 503
    assert d.retry_after_s is not None


def test_durable_call_async_does_not_create_task(env_cfg, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg, store = env_cfg
    monkeypatch.setenv("GROUP_AGENT_DURABLE_QUEUE_ENABLED", "1")
    bp = BackpressureController(store._r, cfg)  # noqa: SLF001
    state = AppState(
        agent=MagicMock(),
        base_dir=Path("/tmp"),
        durable_config=cfg,
        durable_store=store,
        backpressure=bp,
    )
    req = AsyncCallRequest.model_validate(
        {
            "run_id": "ga_no_task",
            "idempotency_key": "idem-no-task",
            "user_id": "u1",
            "unionid": "union-1",
            "group_id": "g1",
            "conversation_id": "c1",
            "message": "hi",
            "callback_url": "http://127.0.0.1:9/group_agent_callbacks",
            "membership": "in_group",
            "request_schema_version": 1,
            "request_fingerprint": _fp("no-task"),
            "queue_schema_version": 1,
        }
    )
    from apps.group_agent_api.app.endpoints import call_async as ca_mod
    from apps.group_agent_api.agent_factory.integrations import callback_client as cc

    scope = {"type": "http", "method": "POST", "path": "/call_async", "headers": []}
    request = Request(scope)

    async def _resolve(*_a, **_k):
        return _session()

    async def _run() -> None:
        with patch.object(cc, "validate_and_normalize_callback_url", lambda u: u), patch(
            "apps.group_agent_api.app.endpoints.call_async.resolve_trusted_session",
            _resolve,
        ), patch(
            "apps.group_agent_api.app.endpoints.call_async.enqueue_delivery",
            lambda config, delivery: None,
        ), patch("asyncio.create_task") as ct:
            resp = await ca_mod.call_async(req, state, request)
            assert resp.accepted is True
            assert resp.execution_status == "queued"
            ct.assert_not_called()

    asyncio.run(_run())


def test_legacy_flag_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROUP_AGENT_DURABLE_QUEUE_ENABLED", "0")
    from apps.group_agent_api.execution.config import durable_queue_enabled

    assert durable_queue_enabled() is False
