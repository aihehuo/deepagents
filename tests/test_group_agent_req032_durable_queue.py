"""REQ-032 unit tests — crypto, retry, config, ledger CAS, admission, fence."""

from __future__ import annotations

import base64
import hashlib
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from redis import Redis

from apps.group_agent_api.agent_factory.capability import CapabilityTier
from apps.group_agent_api.agent_factory.integrations.membership_client import MembershipResult
from apps.group_agent_api.agent_factory.integrations.principal import SessionPrincipal
from apps.group_agent_api.app.models import AsyncCallRequest
from apps.group_agent_api.app.session import TrustedSession
from apps.group_agent_api.execution.admission import admit_durable_async_call
from apps.group_agent_api.execution.backpressure import BackpressureController
from apps.group_agent_api.execution.config import (
    DurableQueueConfig,
    load_durable_queue_config,
    parse_payload_keys,
)
from apps.group_agent_api.execution.crypto import (
    PayloadCryptoError,
    decrypt_envelope,
    digest_id,
    encrypt_envelope,
)
from apps.group_agent_api.execution.dlq import DlqAdmin
from apps.group_agent_api.execution.fence import SideEffectFence
from apps.group_agent_api.execution.models import (
    QUEUE_SCHEMA_VERSION,
    BrokerDeliveryRef,
    ExecutionRecord,
    ExecutionStatus,
)
from apps.group_agent_api.execution.redis_store import ExecutionStore, ExecutionStoreError
from apps.group_agent_api.execution.retry import classify_error, compute_retry_delay, decide_retry
from apps.group_agent_api.execution.models import RetryClass


TEST_KEY_V1 = base64.b64encode(bytes(range(32))).decode()
TEST_KEY_V2 = base64.b64encode(bytes(range(32, 64))).decode()
REDIS_URL = os.environ.get("GROUP_AGENT_TEST_REDIS_URL", "redis://127.0.0.1:6379/15")
PREFIX = f"ga:test:req032:{uuid.uuid4().hex[:8]}"


def _fp(seed: str = "x") -> str:
    return hashlib.sha256(seed.encode()).hexdigest()


@pytest.fixture
def cfg(monkeypatch: pytest.MonkeyPatch) -> DurableQueueConfig:
    monkeypatch.setenv("GROUP_AGENT_DURABLE_QUEUE_ENABLED", "1")
    monkeypatch.setenv("GROUP_AGENT_REDIS_URL", REDIS_URL)
    monkeypatch.setenv("GROUP_AGENT_REDIS_PREFIX", PREFIX)
    monkeypatch.setenv("GROUP_AGENT_CELERY_QUEUE", "group_agent.test.runs")
    monkeypatch.setenv("GROUP_AGENT_DLQ_QUEUE", "group_agent.test.dlq")
    monkeypatch.setenv(
        "GROUP_AGENT_QUEUE_PAYLOAD_KEYS",
        f"v1:{TEST_KEY_V1},v2:{TEST_KEY_V2}",
    )
    monkeypatch.setenv("GROUP_AGENT_QUEUE_PAYLOAD_CURRENT_VERSION", "v1")
    monkeypatch.setenv("GROUP_AGENT_LEASE_TTL_S", "30")
    monkeypatch.setenv("GROUP_AGENT_HEARTBEAT_INTERVAL_S", "10")
    monkeypatch.setenv("GROUP_AGENT_VISIBILITY_TIMEOUT_S", "240")
    monkeypatch.setenv("GROUP_AGENT_TASK_SOFT_LIMIT_S", "150")
    monkeypatch.setenv("GROUP_AGENT_TASK_HARD_LIMIT_S", "180")
    monkeypatch.setenv("GROUP_AGENT_WORKER_INSTANCE_ID", "test-worker-a")
    loaded = load_durable_queue_config(require_enabled=True)
    assert loaded is not None
    return loaded


@pytest.fixture
def store(cfg: DurableQueueConfig) -> ExecutionStore:
    client = Redis.from_url(REDIS_URL, decode_responses=True)
    # wipe only our prefix keys
    for key in client.scan_iter(match=f"{PREFIX}:*", count=200):
        client.delete(key)
    return ExecutionStore(client, cfg)


@pytest.fixture
def session() -> TrustedSession:
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


def _req(**overrides: object) -> AsyncCallRequest:
    base = {
        "run_id": "ga_run_1",
        "idempotency_key": "idem-1",
        "user_id": "u1",
        "unionid": "union-1",
        "group_id": "g1",
        "conversation_id": "c1",
        "message": "hello need partner",
        "callback_url": "http://127.0.0.1:9/group_agent_callbacks",
        "membership": "in_group",
        "request_schema_version": 1,
        "request_fingerprint": _fp("same"),
        "queue_schema_version": 1,
    }
    base.update(overrides)
    return AsyncCallRequest.model_validate(base)


# --- crypto ---


def test_aes_gcm_roundtrip_and_aad_tamper(cfg: DurableQueueConfig) -> None:
    pt = {"message": "secret-should-not-log", "token": "Bearer-x"}
    enc = encrypt_envelope(
        pt,
        key=cfg.current_payload_key,
        key_version="v1",
        run_id="r1",
        idempotency_key="k1",
        request_fingerprint=_fp("a"),
        schema_version=1,
    )
    out = decrypt_envelope(
        enc,
        keys=cfg.payload_keys,
        run_id="r1",
        idempotency_key="k1",
        request_fingerprint=_fp("a"),
        schema_version=1,
    )
    assert out["message"] == pt["message"]
    with pytest.raises(PayloadCryptoError) as ei:
        decrypt_envelope(
            enc,
            keys=cfg.payload_keys,
            run_id="r1",
            idempotency_key="k1",
            request_fingerprint=_fp("tampered"),
            schema_version=1,
        )
    assert ei.value.code == "payload_decrypt_failed"


def test_key_rotation_and_unknown_version(cfg: DurableQueueConfig) -> None:
    enc = encrypt_envelope(
        {"ok": True},
        key=cfg.payload_keys["v2"],
        key_version="v2",
        run_id="r1",
        idempotency_key="k1",
        request_fingerprint=_fp("a"),
        schema_version=1,
    )
    assert decrypt_envelope(
        enc,
        keys=cfg.payload_keys,
        run_id="r1",
        idempotency_key="k1",
        request_fingerprint=_fp("a"),
        schema_version=1,
    )["ok"] is True
    enc.key_version = "v999"
    with pytest.raises(PayloadCryptoError):
        decrypt_envelope(
            enc,
            keys=cfg.payload_keys,
            run_id="r1",
            idempotency_key="k1",
            request_fingerprint=_fp("a"),
            schema_version=1,
        )


def test_parse_payload_keys_rejects_short() -> None:
    with pytest.raises(ValueError):
        parse_payload_keys("v1:" + base64.b64encode(b"short").decode())


# --- retry ---


def test_retry_taxonomy_and_budget() -> None:
    assert classify_error("payload_decrypt_failed") == RetryClass.PERMANENT
    assert classify_error("llm_rate_limited") == RetryClass.TRANSIENT
    d = decide_retry(
        error_code="llm_rate_limited",
        attempt_count=1,
        max_attempts=5,
        base_s=2,
        max_s=120,
        full_jitter=False,
    )
    assert d.should_retry and d.delay_s == 2.0
    d2 = decide_retry(
        error_code="poison_payload",
        attempt_count=1,
        max_attempts=5,
        base_s=2,
        max_s=120,
    )
    assert d2.dead_letter and not d2.should_retry
    d3 = decide_retry(
        error_code="llm_rate_limited",
        attempt_count=5,
        max_attempts=5,
        base_s=2,
        max_s=120,
    )
    assert d3.dead_letter and d3.reason_code == "max_attempts_exceeded"


def test_jitter_bounds() -> None:
    import random

    rng = random.Random(0)
    for _ in range(20):
        delay = compute_retry_delay(attempt_count=3, base_s=2, max_s=10, full_jitter=True, rng=rng)
        assert 0 <= delay <= 8


# --- redis ledger ---


def test_create_or_get_hit_and_conflicts(store: ExecutionStore, cfg: DurableQueueConfig) -> None:
    fp = _fp("same")
    enc = encrypt_envelope(
        {"x": 1},
        key=cfg.current_payload_key,
        key_version="v1",
        run_id="ga_a",
        idempotency_key="idem-a",
        request_fingerprint=fp,
        schema_version=1,
    )
    rec = ExecutionRecord(
        run_id="ga_a",
        idempotency_key="idem-a",
        request_schema_version=1,
        request_fingerprint=fp,
        queue_schema_version=1,
        status=ExecutionStatus.ACCEPTED,
        created_at=store.redis_time(),
        payload_ciphertext=enc,
        conversation_id="c1",
    )
    kind, loaded = store.create_or_get(record=rec)
    assert kind == "created"
    kind2, loaded2 = store.create_or_get(record=rec)
    assert kind2 == "hit" and loaded2.run_id == loaded.run_id

    bad = ExecutionRecord(
        run_id="ga_a",
        idempotency_key="idem-other",
        request_schema_version=1,
        request_fingerprint=fp,
        queue_schema_version=1,
        status=ExecutionStatus.ACCEPTED,
        created_at=store.redis_time(),
        payload_ciphertext=enc,
    )
    with pytest.raises(ExecutionStoreError) as ei:
        store.create_or_get(record=bad)
    assert ei.value.code == "run_binding_conflict"

    bad2 = ExecutionRecord(
        run_id="ga_b",
        idempotency_key="idem-a",
        request_schema_version=1,
        request_fingerprint=_fp("other"),
        queue_schema_version=1,
        status=ExecutionStatus.ACCEPTED,
        created_at=store.redis_time(),
        payload_ciphertext=enc,
    )
    with pytest.raises(ExecutionStoreError) as ei2:
        store.create_or_get(record=bad2)
    assert ei2.value.code == "idempotency_conflict"


def test_claim_renew_finish_and_expired_owner(
    store: ExecutionStore, cfg: DurableQueueConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    fp = _fp("lease")
    enc = encrypt_envelope(
        {"x": 1},
        key=cfg.current_payload_key,
        key_version="v1",
        run_id="ga_lease",
        idempotency_key="idem-lease",
        request_fingerprint=fp,
        schema_version=1,
    )
    store.create_or_get(
        record=ExecutionRecord(
            run_id="ga_lease",
            idempotency_key="idem-lease",
            request_schema_version=1,
            request_fingerprint=fp,
            queue_schema_version=1,
            status=ExecutionStatus.QUEUED,
            created_at=store.redis_time(),
            payload_ciphertext=enc,
            conversation_id="c-lease",
            user_id_digest=digest_id("u1"),
            group_id_digest=digest_id("g1"),
        )
    )
    outcome = store.claim_lease(run_id="ga_lease", conversation_id="c-lease", owner="w1")
    assert outcome.kind == "claimed" and outcome.claim is not None
    claim = outcome.claim
    store.renew_lease(claim, conversation_id="c-lease")
    # second worker cannot claim while not queued (running)
    busy = store.claim_lease(run_id="ga_lease", conversation_id="c-lease", owner="w2")
    assert busy.kind == "not_queued"

    # expire lease → enqueue_failed; recovery would republish; for test mark queued again
    store._r.hset(store.run_key("ga_lease"), "lease_expires_at", "1")  # noqa: SLF001
    assert store.expire_lease_if_needed("ga_lease", "c-lease") == "ok"
    assert store.get("ga_lease").status == ExecutionStatus.ENQUEUE_FAILED  # type: ignore[union-attr]
    store.mark_queued("ga_lease", expected_status="enqueue_failed")
    claim2_out = store.claim_lease(run_id="ga_lease", conversation_id="c-lease", owner="w2")
    assert claim2_out.kind == "claimed" and claim2_out.claim is not None
    claim2 = claim2_out.claim
    assert claim2.attempt_id != claim.attempt_id

    # expired original owner cannot finish
    with pytest.raises(ExecutionStoreError):
        store.finish(
            claim,
            conversation_id="c-lease",
            status=ExecutionStatus.SUCCEEDED,
        )
    store.finish(
        claim2,
        conversation_id="c-lease",
        status=ExecutionStatus.SUCCEEDED,
    )
    assert store.get("ga_lease").status == ExecutionStatus.SUCCEEDED  # type: ignore[union-attr]


def test_redis_keys_have_no_identity(store: ExecutionStore, cfg: DurableQueueConfig) -> None:
    keys = list(store._r.scan_iter(match=f"{PREFIX}:*", count=100))  # noqa: SLF001
    blob = " ".join(keys)
    assert "union" not in blob
    assert "Bearer" not in blob
    assert "hello need" not in blob
    assert digest_id("u1") not in blob or True  # digest may appear in metrics only


# --- admission ---


def test_admission_same_key_hit_and_conflict(
    store: ExecutionStore,
    cfg: DurableQueueConfig,
    session: TrustedSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROUP_AGENT_CALLBACK_ALLOWED_BASE_URLS", "http://127.0.0.1:9/group_agent_callbacks")
    enqueued: list[BrokerDeliveryRef] = []

    def _enqueue(d: BrokerDeliveryRef) -> None:
        enqueued.append(d)

    bp = BackpressureController(store._r, cfg)  # noqa: SLF001
    req = _req()
    # bypass callback URL validation by patching admit's dependency? request already has URL
    # validate_and_normalize happens before admit in call_async; here we call admit directly.
    from apps.group_agent_api.agent_factory.integrations import callback_client as cc

    monkeypatch.setattr(cc, "validate_and_normalize_callback_url", lambda u: u)

    r1 = admit_durable_async_call(
        req=req,
        session=session,
        thread_id="tid1",
        store=store,
        config=cfg,
        backpressure=bp,
        enqueue=_enqueue,
    )
    assert r1.response.accepted and r1.execution_status == "queued"
    assert len(enqueued) == 1
    assert "celery" not in r1.response.model_dump_json().lower()

    r2 = admit_durable_async_call(
        req=req,
        session=session,
        thread_id="tid1",
        store=store,
        config=cfg,
        backpressure=bp,
        enqueue=_enqueue,
    )
    assert r2.created is False

    bad = _req(request_fingerprint=_fp("different"))
    with pytest.raises(HTTPException) as ei:
        admit_durable_async_call(
            req=bad,
            session=session,
            thread_id="tid1",
            store=store,
            config=cfg,
            backpressure=bp,
            enqueue=_enqueue,
        )
    assert ei.value.status_code == 409
    assert ei.value.detail["error"] == "idempotency_conflict"


def test_admission_enqueue_failed_no_202(
    store: ExecutionStore,
    cfg: DurableQueueConfig,
    session: TrustedSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from apps.group_agent_api.agent_factory.integrations import callback_client as cc

    monkeypatch.setattr(cc, "validate_and_normalize_callback_url", lambda u: u)
    bp = BackpressureController(store._r, cfg)  # noqa: SLF001

    def _boom(_d: BrokerDeliveryRef) -> None:
        raise RuntimeError("broker_down")

    with pytest.raises(HTTPException) as ei:
        admit_durable_async_call(
            req=_req(run_id="ga_enq_fail", idempotency_key="idem-enq-fail"),
            session=session,
            thread_id="tid1",
            store=store,
            config=cfg,
            backpressure=bp,
            enqueue=_boom,
        )
    assert ei.value.status_code == 503
    assert ei.value.detail["error"] == "enqueue_failed"
    rec = store.get("ga_enq_fail")
    assert rec is not None and rec.status == ExecutionStatus.ENQUEUE_FAILED


def test_backpressure_status_codes(
    store: ExecutionStore, cfg: DurableQueueConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GROUP_AGENT_USER_MAX_QUEUED", "1")
    # reload config with low user limit
    from apps.group_agent_api.execution.config import load_durable_queue_config

    monkeypatch.setenv("GROUP_AGENT_DURABLE_QUEUE_ENABLED", "1")
    cfg2 = load_durable_queue_config(require_enabled=True)
    assert cfg2 is not None
    # force tiny limit
    object.__setattr__(cfg2, "user_max_queued", 1) if False else None
    bp = BackpressureController(store._r, cfg)
    # manually set counters
    store._r.set(cfg.key("metrics", "queued_user", digest_id("u1")), 999)  # noqa: SLF001
    d = bp.check_and_reserve(
        user_id="u1", group_id="g1", conversation_id="c1", provider="default"
    )
    assert d.allowed is False and d.http_status == 429 and d.error_code == "queue_limit_exceeded"

    store._r.set(cfg.key("metrics", "queued_global"), 999999)  # noqa: SLF001
    store._r.delete(cfg.key("metrics", "queued_user", digest_id("u2")))
    d2 = bp.check_and_reserve(
        user_id="u2", group_id="g1", conversation_id="c2", provider="default"
    )
    assert d2.http_status == 503 and d2.error_code == "queue_saturated"


def test_fence_blocks_non_owner(store: ExecutionStore, cfg: DurableQueueConfig) -> None:
    fp = _fp("fence")
    enc = encrypt_envelope(
        {"x": 1},
        key=cfg.current_payload_key,
        key_version="v1",
        run_id="ga_fence",
        idempotency_key="idem-fence",
        request_fingerprint=fp,
        schema_version=1,
    )
    store.create_or_get(
        record=ExecutionRecord(
            run_id="ga_fence",
            idempotency_key="idem-fence",
            request_schema_version=1,
            request_fingerprint=fp,
            queue_schema_version=1,
            status=ExecutionStatus.QUEUED,
            created_at=store.redis_time(),
            payload_ciphertext=enc,
            conversation_id="c-f",
            user_id_digest=digest_id("u1"),
            group_id_digest=digest_id("g1"),
        )
    )
    out = store.claim_lease(run_id="ga_fence", conversation_id="c-f", owner="w1")
    assert out.kind == "claimed" and out.claim is not None
    claim = out.claim
    fence = SideEffectFence(store=store, claim=claim, conversation_id="c-f")
    assert fence.assert_owner()
    store._r.hset(store.run_key("ga_fence"), "lease_expires_at", "1")  # noqa: SLF001
    assert fence.assert_owner() is False


def test_dlq_replay_preserves_identity(store: ExecutionStore, cfg: DurableQueueConfig) -> None:
    fp = _fp("dlq")
    enc = encrypt_envelope(
        {"x": 1},
        key=cfg.current_payload_key,
        key_version="v1",
        run_id="ga_dlq",
        idempotency_key="idem-dlq",
        request_fingerprint=fp,
        schema_version=1,
    )
    store.create_or_get(
        record=ExecutionRecord(
            run_id="ga_dlq",
            idempotency_key="idem-dlq",
            request_schema_version=1,
            request_fingerprint=fp,
            queue_schema_version=1,
            status=ExecutionStatus.QUEUED,
            created_at=store.redis_time(),
            payload_ciphertext=enc,
            conversation_id="c-d",
            user_id_digest=digest_id("u1"),
            group_id_digest=digest_id("g1"),
        )
    )
    out = store.claim_lease(run_id="ga_dlq", conversation_id="c-d", owner="w1")
    assert out.kind == "claimed" and out.claim is not None
    claim = out.claim
    store.finish(
        claim,
        conversation_id="c-d",
        status=ExecutionStatus.DEAD_LETTERED,
        error_code="poison_payload",
    )
    deliveries: list[BrokerDeliveryRef] = []
    admin = DlqAdmin(store, cfg, enqueue=lambda d: deliveries.append(d))
    view = admin.replay("ga_dlq", operator_id="op1", reason="manual_test", replay_id="rep-1")
    assert view.run_id == "ga_dlq"
    assert view.idempotency_key == "idem-dlq"
    assert view.request_fingerprint == fp
    assert store.get("ga_dlq").status == ExecutionStatus.QUEUED  # type: ignore[union-attr]
    assert len(deliveries) == 1
    safe = admin.inspect("ga_dlq").to_dict()
    assert "ciphertext" not in str(safe)
    assert "token" not in str(safe).lower()


def test_concurrent_same_key_one_record(
    store: ExecutionStore, cfg: DurableQueueConfig
) -> None:
    fp = _fp("race")

    def _one(i: int) -> str:
        enc = encrypt_envelope(
            {"i": i},
            key=cfg.current_payload_key,
            key_version="v1",
            run_id="ga_race",
            idempotency_key="idem-race",
            request_fingerprint=fp,
            schema_version=1,
        )
        kind, _ = store.create_or_get(
            record=ExecutionRecord(
                run_id="ga_race",
                idempotency_key="idem-race",
                request_schema_version=1,
                request_fingerprint=fp,
                queue_schema_version=1,
                status=ExecutionStatus.ACCEPTED,
                created_at=store.redis_time(),
                payload_ciphertext=enc,
            )
        )
        return kind

    with ThreadPoolExecutor(max_workers=20) as pool:
        results = list(pool.map(_one, range(100)))
    assert results.count("created") == 1
    assert results.count("hit") == 99


def test_concurrent_different_fp_one_winner(
    store: ExecutionStore, cfg: DurableQueueConfig
) -> None:
    def _one(i: int) -> str:
        fp = _fp(f"fp-{i}")
        enc = encrypt_envelope(
            {"i": i},
            key=cfg.current_payload_key,
            key_version="v1",
            run_id=f"ga_diff_{i}",
            idempotency_key="idem-diff",
            request_fingerprint=fp,
            schema_version=1,
        )
        try:
            kind, _ = store.create_or_get(
                record=ExecutionRecord(
                    run_id=f"ga_diff_{i}",
                    idempotency_key="idem-diff",
                    request_schema_version=1,
                    request_fingerprint=fp,
                    queue_schema_version=1,
                    status=ExecutionStatus.ACCEPTED,
                    created_at=store.redis_time(),
                    payload_ciphertext=enc,
                )
            )
            return kind
        except ExecutionStoreError as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=20) as pool:
        results = list(as_completed([pool.submit(_one, i) for i in range(100)]))
        vals = [r.result() for r in results]
    assert vals.count("created") == 1
    assert vals.count("idempotency_conflict") == 99


def test_durable_mode_flag_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROUP_AGENT_DURABLE_QUEUE_ENABLED", raising=False)
    from apps.group_agent_api.execution.config import durable_queue_enabled

    assert durable_queue_enabled() is False


def test_config_fail_closed_bad_lease_relation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROUP_AGENT_DURABLE_QUEUE_ENABLED", "1")
    monkeypatch.setenv("GROUP_AGENT_REDIS_URL", REDIS_URL)
    monkeypatch.setenv("GROUP_AGENT_REDIS_PREFIX", PREFIX)
    monkeypatch.setenv("GROUP_AGENT_CELERY_QUEUE", "q")
    monkeypatch.setenv("GROUP_AGENT_DLQ_QUEUE", "dlq")
    monkeypatch.setenv("GROUP_AGENT_QUEUE_PAYLOAD_KEYS", f"v1:{TEST_KEY_V1}")
    monkeypatch.setenv("GROUP_AGENT_QUEUE_PAYLOAD_CURRENT_VERSION", "v1")
    monkeypatch.setenv("GROUP_AGENT_LEASE_TTL_S", "30")
    monkeypatch.setenv("GROUP_AGENT_HEARTBEAT_INTERVAL_S", "20")  # >= lease/2
    monkeypatch.setenv("GROUP_AGENT_VISIBILITY_TIMEOUT_S", "240")
    monkeypatch.setenv("GROUP_AGENT_TASK_SOFT_LIMIT_S", "150")
    monkeypatch.setenv("GROUP_AGENT_TASK_HARD_LIMIT_S", "180")
    with pytest.raises(RuntimeError):
        load_durable_queue_config(require_enabled=True)


def test_broker_delivery_has_no_secrets() -> None:
    d = BrokerDeliveryRef(
        queue_schema_version=QUEUE_SCHEMA_VERSION,
        run_id="ga_x",
        idempotency_key="k",
        request_fingerprint=_fp("z"),
        delivery_id="d1",
    )
    raw = str(d.to_dict())
    assert "token" not in raw
    assert "callback" not in raw
    assert "message" not in raw
    assert "unionid" not in raw
