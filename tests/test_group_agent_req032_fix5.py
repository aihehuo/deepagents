"""REQ-032-FIX5 tests for atomic quota reconciliation and Deep fencing."""

from __future__ import annotations

import base64
import hashlib
import uuid

import pytest
from redis import Redis

from apps.group_agent_api.execution.backpressure import BackpressureController
from apps.group_agent_api.execution.config import load_durable_queue_config
from apps.group_agent_api.execution.crypto import digest_id
from apps.group_agent_api.execution.models import ExecutionRecord, ExecutionStatus
from apps.group_agent_api.execution.redis_store import ExecutionStore

TEST_KEY_V1 = base64.b64encode(bytes(range(32))).decode()
REDIS_URL = "redis://127.0.0.1:6379/15"
PREFIX = f"ga:test:fix5:{uuid.uuid4().hex[:8]}"


def _fp(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()


@pytest.fixture
def cfg_store(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GROUP_AGENT_DURABLE_QUEUE_ENABLED", "1")
    monkeypatch.setenv("GROUP_AGENT_REDIS_URL", REDIS_URL)
    monkeypatch.setenv("GROUP_AGENT_REDIS_PREFIX", PREFIX)
    monkeypatch.setenv("GROUP_AGENT_CELERY_QUEUE", "group_agent.fix5.runs")
    monkeypatch.setenv("GROUP_AGENT_DLQ_QUEUE", "group_agent.fix5.dlq")
    monkeypatch.setenv("GROUP_AGENT_QUEUE_PAYLOAD_KEYS", f"v1:{TEST_KEY_V1}")
    monkeypatch.setenv("GROUP_AGENT_QUEUE_PAYLOAD_CURRENT_VERSION", "v1")
    monkeypatch.setenv("GROUP_AGENT_LEASE_TTL_S", "5")
    monkeypatch.setenv("GROUP_AGENT_HEARTBEAT_INTERVAL_S", "2")
    monkeypatch.setenv("GROUP_AGENT_VISIBILITY_TIMEOUT_S", "240")
    monkeypatch.setenv("GROUP_AGENT_TASK_SOFT_LIMIT_S", "150")
    monkeypatch.setenv("GROUP_AGENT_TASK_HARD_LIMIT_S", "180")
    monkeypatch.setenv("GROUP_AGENT_ADMISSION_TIMEOUT_S", "5")
    monkeypatch.setenv("GROUP_AGENT_WORKER_INSTANCE_ID", "fix5-worker")
    cfg = load_durable_queue_config(require_enabled=True)
    assert cfg is not None
    client = Redis.from_url(REDIS_URL, decode_responses=True)
    for key in client.scan_iter(match=f"{PREFIX}:*", count=200):
        client.delete(key)
    return cfg, ExecutionStore(client, cfg)


def _record(
    store: ExecutionStore,
    run_id: str,
    *,
    user_id: str = "u1",
    group_id: str = "g1",
    conversation_id: str = "c1",
) -> ExecutionRecord:
    record = ExecutionRecord(
        run_id=run_id,
        idempotency_key=f"idem-{run_id}",
        request_schema_version=1,
        request_fingerprint=_fp(run_id),
        queue_schema_version=1,
        status=ExecutionStatus.QUEUED,
        created_at=store.redis_time(),
        conversation_id=conversation_id,
        user_id_digest=digest_id(user_id),
        group_id_digest=digest_id(group_id),
        provider="default",
    )
    store.create_or_get(record=record)
    loaded = store.get(run_id)
    assert loaded is not None
    return loaded


def test_fix5_reconcile_compare_and_apply_rejects_last_moment_admission(
    cfg_store, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, store = cfg_store
    _record(store, "ga_q1")
    counter = cfg.key("metrics", "queued_global")
    version = cfg.key("metrics", "metrics_version")
    store._r.set(counter, 1)  # noqa: SLF001
    store._r.set(version, 10)  # noqa: SLF001

    original_eval = store._r.eval  # noqa: SLF001
    concurrent_client = Redis.from_url(REDIS_URL, decode_responses=True)
    concurrent_bp = BackpressureController(concurrent_client, cfg)
    raced = False

    def _eval_with_admission(script, numkeys, *args):  # noqa: ANN001
        nonlocal raced
        if "version_conflict" in script and not raced:
            raced = True
            _record(store, "ga_q2", conversation_id="c2")
            decision = concurrent_bp.check_and_reserve(
                user_id="u1",
                group_id="g1",
                conversation_id="c2",
                provider="default",
            )
            assert decision.allowed
        return original_eval(script, numkeys, *args)

    monkeypatch.setattr(store._r, "eval", _eval_with_admission)  # noqa: SLF001
    applied = BackpressureController(store._r, cfg).reconcile_from_ledger(store)  # noqa: SLF001

    assert raced
    assert applied is False
    assert int(store._r.get(counter) or 0) == 2  # noqa: SLF001
    assert int(store._r.get(version) or 0) == 11  # noqa: SLF001


def test_fix5_reconcile_compare_and_apply_rejects_finish_race(
    cfg_store, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finish between last version check and Lua apply must not be overwritten."""
    cfg, store = cfg_store
    _record(store, "ga_fin", conversation_id="c-fin")
    claim = store.claim_lease(
        run_id="ga_fin", conversation_id="c-fin", owner="w-fin"
    ).claim
    assert claim is not None

    running = cfg.key("metrics", "running_global")
    version = cfg.key("metrics", "metrics_version")
    store._r.set(running, 1)  # noqa: SLF001
    store._r.set(version, 20)  # noqa: SLF001

    original_eval = store._r.eval  # noqa: SLF001
    raced = False

    def _eval_with_finish(script, numkeys, *args):  # noqa: ANN001
        nonlocal raced
        if "version_conflict" in script and not raced:
            raced = True
            BackpressureController(store._r, cfg).on_finish(  # noqa: SLF001
                user_id="u1",
                group_id="g1",
                conversation_id="c-fin",
                provider="default",
            )
        return original_eval(script, numkeys, *args)

    monkeypatch.setattr(store._r, "eval", _eval_with_finish)  # noqa: SLF001
    applied = BackpressureController(store._r, cfg).reconcile_from_ledger(store)  # noqa: SLF001

    assert raced
    assert applied is False
    assert int(store._r.get(running) or 0) == 0  # noqa: SLF001
    assert int(store._r.get(version) or 0) == 21  # noqa: SLF001


def test_fix5_reconcile_compare_and_apply_rejects_retry_wait_race(
    cfg_store, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retry_wait between last version check and Lua apply must keep its counters."""
    cfg, store = cfg_store
    _record(store, "ga_rw", conversation_id="c-rw")
    claim = store.claim_lease(
        run_id="ga_rw", conversation_id="c-rw", owner="w-rw"
    ).claim
    assert claim is not None

    running = cfg.key("metrics", "running_global")
    queued = cfg.key("metrics", "queued_global")
    version = cfg.key("metrics", "metrics_version")
    store._r.set(running, 1)  # noqa: SLF001
    store._r.set(queued, 0)  # noqa: SLF001
    store._r.set(version, 30)  # noqa: SLF001

    original_eval = store._r.eval  # noqa: SLF001
    raced = False

    def _eval_with_retry_wait(script, numkeys, *args):  # noqa: ANN001
        nonlocal raced
        if "version_conflict" in script and not raced:
            raced = True
            BackpressureController(store._r, cfg).on_retry_wait(  # noqa: SLF001
                user_id="u1",
                group_id="g1",
                conversation_id="c-rw",
                provider="default",
            )
        return original_eval(script, numkeys, *args)

    monkeypatch.setattr(store._r, "eval", _eval_with_retry_wait)  # noqa: SLF001
    applied = BackpressureController(store._r, cfg).reconcile_from_ledger(store)  # noqa: SLF001

    assert raced
    assert applied is False
    assert int(store._r.get(running) or 0) == 0  # noqa: SLF001
    assert int(store._r.get(queued) or 0) == 1  # noqa: SLF001
    assert int(store._r.get(version) or 0) == 31  # noqa: SLF001


def test_fix5_all_counter_transitions_bump_version_in_their_lua_operation(
    cfg_store, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, store = cfg_store
    bp = BackpressureController(store._r, cfg)  # noqa: SLF001
    calls: list[str] = []
    original_eval = store._r.eval  # noqa: SLF001

    def _record_eval(script, numkeys, *args):  # noqa: ANN001
        calls.append(script)
        return original_eval(script, numkeys, *args)

    monkeypatch.setattr(store._r, "eval", _record_eval)  # noqa: SLF001
    bp.on_start_running(conversation_id="c1", provider="default")
    bp.on_finish(
        user_id="u1",
        group_id="g1",
        conversation_id="c1",
        provider="default",
    )
    bp.on_retry_wait(
        user_id="u1",
        group_id="g1",
        conversation_id="c1",
        provider="default",
    )
    bp.on_lease_expired(
        conversation_id="c1",
        provider="default",
        user_id_digest=digest_id("u1"),
        group_id_digest=digest_id("g1"),
    )
    bp.release_queued_reservation(user_id="u1", group_id="g1")

    assert len(calls) == 5
    assert int(store._r.get(cfg.key("metrics", "metrics_version")) or 0) == 5  # noqa: SLF001


def test_fix5_claim_fails_closed_without_user_group_identity(cfg_store) -> None:
    cfg, store = cfg_store
    _record(store, "ga_identity")
    store._r.hdel(store.run_key("ga_identity"), "user_id_digest", "group_id_digest")  # noqa: SLF001

    outcome = store.claim_lease(
        run_id="ga_identity",
        conversation_id="c1",
        owner="worker",
    )

    assert outcome.kind == "fence_identity_missing"
    assert store.get("ga_identity").status == ExecutionStatus.QUEUED  # type: ignore[union-attr]
    assert not list(store._r.scan_iter(match=cfg.key("profile_epoch", "*")))  # noqa: SLF001


def test_fix5_latest_issued_epoch_invalidates_older_active_run(cfg_store) -> None:
    _cfg, store = cfg_store
    _record(store, "ga_old_active", conversation_id="c-old")
    old = store.claim_lease(
        run_id="ga_old_active",
        conversation_id="c-old",
        owner="old-worker",
    ).claim
    assert old is not None

    _record(store, "ga_new_active", conversation_id="c-new")
    new = store.claim_lease(
        run_id="ga_new_active",
        conversation_id="c-new",
        owner="new-worker",
    ).claim
    assert new is not None
    assert new.fencing_token > old.fencing_token

    assert store.cas_profile_write_fence(user_id="u1", group_id="g1", claim=old) == "stale"
    assert store.cas_profile_write_fence(user_id="u1", group_id="g1", claim=new) == "ok"


def test_fix5_profile_fence_rejects_identity_mismatch_and_bounds_audit_keys(cfg_store) -> None:
    cfg, store = cfg_store
    _record(store, "ga_fence_identity")
    claim = store.claim_lease(
        run_id="ga_fence_identity",
        conversation_id="c1",
        owner="worker",
    ).claim
    assert claim is not None

    assert (
        store.cas_profile_write_fence(user_id="u2", group_id="g1", claim=claim)
        == "fence_identity_mismatch"
    )
    assert store.cas_profile_write_fence(user_id="u1", group_id="g1", claim=claim) == "ok"

    ud = digest_id("u1")
    gd = digest_id("g1")
    assert store._r.ttl(cfg.key("profile_fence", ud, gd)) > 0  # noqa: SLF001
    assert store._r.ttl(cfg.key("profile_fence_meta", ud, gd)) > 0  # noqa: SLF001
