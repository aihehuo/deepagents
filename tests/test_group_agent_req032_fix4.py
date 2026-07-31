"""REQ-032-FIX4 tests for FIX3-BLOCKER-1～5."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest
from redis import Redis

from apps.group_agent_api.execution.active_fence import (
    ActiveAttemptFence,
    clear_active_fence,
    commit_profile_write_allowed,
    set_active_fence,
)
from apps.group_agent_api.execution.backpressure import BackpressureController
from apps.group_agent_api.execution.config import load_durable_queue_config
from apps.group_agent_api.execution.crypto import digest_id, encrypt_envelope
from apps.group_agent_api.execution.dlq import publish_unmappable_poison
from apps.group_agent_api.execution.models import (
    QUEUE_SCHEMA_VERSION,
    BrokerDeliveryRef,
    ExecutionRecord,
    ExecutionStatus,
    LeaseClaim,
)
from apps.group_agent_api.execution.redis_store import ExecutionStore
from apps.group_agent_worker.tasks import process_run

TEST_KEY_V1 = base64.b64encode(bytes(range(32))).decode()
REDIS_URL = os.environ.get("GROUP_AGENT_TEST_REDIS_URL", "redis://127.0.0.1:6379/15")
PREFIX = f"ga:test:fix4:{uuid.uuid4().hex[:8]}"


def _fp(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()


@pytest.fixture
def cfg_store(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GROUP_AGENT_DURABLE_QUEUE_ENABLED", "1")
    monkeypatch.setenv("GROUP_AGENT_REDIS_URL", REDIS_URL)
    monkeypatch.setenv("GROUP_AGENT_REDIS_PREFIX", PREFIX)
    monkeypatch.setenv("GROUP_AGENT_CELERY_QUEUE", "group_agent.fix4.runs")
    monkeypatch.setenv("GROUP_AGENT_DLQ_QUEUE", "group_agent.fix4.dlq")
    monkeypatch.setenv("GROUP_AGENT_QUEUE_PAYLOAD_KEYS", f"v1:{TEST_KEY_V1}")
    monkeypatch.setenv("GROUP_AGENT_QUEUE_PAYLOAD_CURRENT_VERSION", "v1")
    monkeypatch.setenv("GROUP_AGENT_LEASE_TTL_S", "5")
    monkeypatch.setenv("GROUP_AGENT_HEARTBEAT_INTERVAL_S", "2")
    monkeypatch.setenv("GROUP_AGENT_VISIBILITY_TIMEOUT_S", "240")
    monkeypatch.setenv("GROUP_AGENT_TASK_SOFT_LIMIT_S", "150")
    monkeypatch.setenv("GROUP_AGENT_TASK_HARD_LIMIT_S", "180")
    monkeypatch.setenv("GROUP_AGENT_ADMISSION_TIMEOUT_S", "5")
    monkeypatch.setenv("GROUP_AGENT_WORKER_INSTANCE_ID", "fix4-worker")
    monkeypatch.setenv("GROUP_AGENT_MAX_ATTEMPTS", "3")
    monkeypatch.setenv(
        "GROUP_AGENT_CALLBACK_ALLOWED_BASE_URLS",
        "http://127.0.0.1:9/group_agent_callbacks",
    )
    cfg = load_durable_queue_config(require_enabled=True)
    assert cfg is not None
    client = Redis.from_url(REDIS_URL, decode_responses=True)
    for key in client.scan_iter(match=f"{PREFIX}:*", count=200):
        client.delete(key)
    return cfg, ExecutionStore(client, cfg)


def _record(store, cfg, run_id: str, status: ExecutionStatus, **kwargs) -> ExecutionRecord:
    fp = kwargs.get("fingerprint") or _fp(run_id)
    idem = kwargs.get("idempotency_key") or f"idem-{run_id}"
    user_id = kwargs.get("user_id", "u1")
    group_id = kwargs.get("group_id", "g1")
    conv = kwargs.get("conversation_id", "c1")
    enc = encrypt_envelope(
        {
            "thread_id": "t",
            "request": {
                "run_id": run_id,
                "idempotency_key": idem,
                "user_id": user_id,
                "unionid": "un",
                "group_id": group_id,
                "message": "hi",
                "callback_url": "http://127.0.0.1:9/group_agent_callbacks",
                "conversation_id": conv,
                "request_schema_version": 1,
                "request_fingerprint": fp,
                "queue_schema_version": 1,
            },
            "session": {
                "user_id": user_id,
                "unionid": "un",
                "group_id": group_id,
                "membership_tier": "in_group",
                "membership_source": "test",
                "principal_source": "test",
            },
        },
        key=cfg.current_payload_key,
        key_version="v1",
        run_id=run_id,
        idempotency_key=idem,
        request_fingerprint=fp,
        schema_version=1,
    )
    rec = ExecutionRecord(
        run_id=run_id,
        idempotency_key=idem,
        request_schema_version=1,
        request_fingerprint=fp,
        queue_schema_version=1,
        status=status,
        created_at=store.redis_time(),
        payload_ciphertext=enc,
        conversation_id=conv,
        user_id_digest=digest_id(user_id),
        group_id_digest=digest_id(group_id),
        provider="default",
    )
    store.create_or_get(record=rec)
    loaded = store.get(run_id)
    assert loaded is not None
    return loaded


def _expire_and_requeue(store: ExecutionStore, run_id: str, conversation_id: str = "c1") -> None:
    store._r.hset(store.run_key(run_id), "lease_expires_at", "1")  # noqa: SLF001
    assert store.expire_lease_if_needed(run_id, conversation_id) == "ok"
    store.mark_queued(run_id, expected_status="enqueue_failed")


def test_fix4_1_user_group_epoch_new_run_after_old_takeover(cfg_store) -> None:
    cfg, store = cfg_store
    _record(store, cfg, "ga_old", ExecutionStatus.QUEUED, conversation_id="c-old")
    out1 = store.claim_lease(run_id="ga_old", conversation_id="c-old", owner="w1")
    assert out1.claim is not None
    tok1 = out1.claim.fencing_token
    _expire_and_requeue(store, "ga_old", "c-old")
    out2 = store.claim_lease(run_id="ga_old", conversation_id="c-old", owner="w1b")
    assert out2.claim is not None
    tok2 = out2.claim.fencing_token
    assert tok2 > tok1

    store.finish(out2.claim, conversation_id="c-old", status=ExecutionStatus.SUCCEEDED)

    _record(store, cfg, "ga_new", ExecutionStatus.QUEUED, conversation_id="c-new")
    out3 = store.claim_lease(run_id="ga_new", conversation_id="c-new", owner="w2")
    assert out3.claim is not None
    assert out3.claim.fencing_token > tok2

    fresh = ActiveAttemptFence(
        store=store, claim=out3.claim, conversation_id="c-new", cancel_event=threading.Event()
    )
    tok = set_active_fence(fresh, require=True)
    try:
        commit_profile_write_allowed(user_id="u1", group_id="g1")
    finally:
        clear_active_fence(tok)

    kind = store.cas_profile_write_fence(user_id="u1", group_id="g1", claim=out2.claim)
    assert kind in {"stale", "fence_not_running", "fence_attempt_mismatch", "fence_epoch_mismatch"}


def test_fix4_1_same_epoch_different_attempt_rejected(cfg_store) -> None:
    cfg, store = cfg_store
    _record(store, cfg, "ga_same", ExecutionStatus.QUEUED)
    out = store.claim_lease(run_id="ga_same", conversation_id="c1", owner="w1")
    claim = out.claim
    assert claim is not None
    fence = ActiveAttemptFence(
        store=store, claim=claim, conversation_id="c1", cancel_event=threading.Event()
    )
    tokens = set_active_fence(fence, require=True)
    try:
        commit_profile_write_allowed(user_id="u1", group_id="g1")
        spoof = LeaseClaim(
            run_id=claim.run_id,
            attempt_id="other-attempt",
            lease_token=claim.lease_token,
            lease_owner=claim.lease_owner,
            lease_expires_at=claim.lease_expires_at,
            fencing_token=claim.fencing_token,
        )
        kind = store.cas_profile_write_fence(user_id="u1", group_id="g1", claim=spoof)
        assert kind in {"fence_attempt_mismatch", "fence_attempt_conflict"}
    finally:
        clear_active_fence(tokens)


def test_fix4_1_fail_closed_non_positive_token(cfg_store) -> None:
    cfg, store = cfg_store
    _record(store, cfg, "ga_tok", ExecutionStatus.QUEUED)
    out = store.claim_lease(run_id="ga_tok", conversation_id="c1", owner="w1")
    claim = out.claim
    assert claim is not None
    bad = LeaseClaim(
        run_id=claim.run_id,
        attempt_id=claim.attempt_id,
        lease_token=claim.lease_token,
        lease_owner=claim.lease_owner,
        lease_expires_at=claim.lease_expires_at,
        fencing_token=0,
    )
    assert store.cas_profile_write_fence(user_id="u1", group_id="g1", claim=bad) == "fence_token_required"


def test_fix4_1_late_old_run_rejected_after_newer_epoch(cfg_store) -> None:
    cfg, store = cfg_store
    _record(store, cfg, "ga_a", ExecutionStatus.QUEUED, conversation_id="ca")
    a1 = store.claim_lease(run_id="ga_a", conversation_id="ca", owner="wa").claim
    assert a1 is not None
    store.finish(a1, conversation_id="ca", status=ExecutionStatus.SUCCEEDED)

    _record(store, cfg, "ga_b", ExecutionStatus.QUEUED, conversation_id="cb")
    b1 = store.claim_lease(run_id="ga_b", conversation_id="cb", owner="wb").claim
    assert b1 is not None
    fence = ActiveAttemptFence(
        store=store, claim=b1, conversation_id="cb", cancel_event=threading.Event()
    )
    tokens = set_active_fence(fence, require=True)
    try:
        commit_profile_write_allowed(user_id="u1", group_id="g1")
    finally:
        clear_active_fence(tokens)

    assert b1.fencing_token > a1.fencing_token
    kind = store.cas_profile_write_fence(user_id="u1", group_id="g1", claim=a1)
    assert kind in {"stale", "fence_not_running", "fence_epoch_mismatch", "fence_attempt_mismatch"}


def test_fix4_2_micro_fencing_artifacts_absent() -> None:
    micro = Path("/Users/yc/workspace/aihehuo/aihehuo_total/backend/aihehuomicro")
    if not micro.exists():
        pytest.skip("micro not in workspace")
    upserter = (micro / "app/services/group_agent/profile_upserter.rb").read_text(encoding="utf-8")
    assert "source_fencing_token" not in upserter
    assert "fence_stale" not in upserter
    migrations = list((micro / "db/migrate").glob("*fencing*"))
    assert migrations == []
    schema = (micro / "db/schema.rb").read_text(encoding="utf-8")
    assert "source_fencing_token" not in schema
    assert "source_attempt_id" not in schema


def test_fix4_3_reconcile_aborts_and_keeps_exact_concurrent_two(cfg_store) -> None:
    cfg, store = cfg_store
    bp = BackpressureController(store._r, cfg)  # noqa: SLF001
    _record(store, cfg, "ga_q1", ExecutionStatus.QUEUED, user_id="uA", group_id="gA")
    gkey = cfg.key("metrics", "queued_global")
    ver_key = cfg.key("metrics", "metrics_version")
    store._r.set(gkey, 1)  # noqa: SLF001
    store._r.set(ver_key, 10)  # noqa: SLF001

    original = store.scan_status

    def _scan_and_race(status, count=100, limit=None):  # noqa: ANN001
        rows = original(status, count=count, limit=limit)
        if status == ExecutionStatus.QUEUED and not getattr(_scan_and_race, "_done", False):
            _scan_and_race._done = True  # type: ignore[attr-defined]
            _record(store, cfg, "ga_q2", ExecutionStatus.QUEUED, user_id="uA", group_id="gA")
            store._r.incr(gkey)
            store._r.incr(ver_key)
        return rows

    store.scan_status = _scan_and_race  # type: ignore[method-assign]
    try:
        applied = bp.reconcile_from_ledger(store)
    finally:
        store.scan_status = original  # type: ignore[method-assign]

    assert applied is False
    assert int(store._r.get(gkey) or 0) == 2  # noqa: SLF001


def test_fix4_3_scan_status_full_cursor_beyond_page_hint(cfg_store) -> None:
    cfg, store = cfg_store
    # Prove full cursor coverage past former silent 5000 cap (bare hashes for speed).
    n = 5100
    pipe = store._r.pipeline(transaction=False)  # noqa: SLF001
    now = str(store.redis_time())
    for i in range(n):
        rid = f"ga_scan_{i}"
        key = store.run_key(rid)
        pipe.hset(
            key,
            mapping={
                "run_id": rid,
                "status": ExecutionStatus.QUEUED.value,
                "idempotency_key": f"idem-{rid}",
                "request_schema_version": "1",
                "request_fingerprint": _fp(rid),
                "queue_schema_version": "1",
                "created_at": now,
                "conversation_id": f"c_scan_{i}",
                "user_id_digest": digest_id("uScan"),
                "group_id_digest": digest_id("gScan"),
                "provider": "default",
                "attempt_count": "0",
            },
        )
        if i % 200 == 199:
            pipe.execute()
            pipe = store._r.pipeline(transaction=False)  # noqa: SLF001
    pipe.execute()
    found = store.scan_status(ExecutionStatus.QUEUED, count=200)
    ids = {r.run_id for r in found}
    assert len(ids) >= n
    assert f"ga_scan_0" in ids and f"ga_scan_{n-1}" in ids


def test_fix4_3_prod_task_has_no_test_hooks() -> None:
    src = Path("apps/group_agent_worker/tasks.py").read_text(encoding="utf-8")
    assert "GROUP_AGENT_TEST_HOLD_AFTER_CLAIM_S" not in src
    assert "GROUP_AGENT_TEST_CLAIM_MARKER" not in src


def test_fix4_5_binding_conflict_does_not_kill_queued_run(
    cfg_store, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, store = cfg_store
    _record(store, cfg, "ga_bind", ExecutionStatus.QUEUED)
    sent: list[dict] = []

    class _App:
        def send_task(self, name, kwargs=None, queue=None, ignore_result=True):
            sent.append({"name": name, "kwargs": kwargs, "queue": queue})
            return type("R", (), {"id": "1"})()

    monkeypatch.setattr(
        "apps.group_agent_worker.celery_app.get_celery_app",
        lambda *a, **k: _App(),
    )
    monkeypatch.setattr(
        "apps.group_agent_worker.tasks.get_worker_runtime",
        lambda: {
            "store": store,
            "config": cfg,
            "state": None,
            "backpressure": BackpressureController(store._r, cfg),  # noqa: SLF001
        },
    )
    delivery = BrokerDeliveryRef(
        queue_schema_version=QUEUE_SCHEMA_VERSION,
        run_id="ga_bind",
        idempotency_key="wrong-idem",
        request_fingerprint="0" * 64,
        delivery_id=str(uuid.uuid4()),
    ).to_dict()
    out = process_run.run(delivery)
    assert out["status"] == "poison"
    assert out["error_code"] == "binding_conflict"
    rec = store.get("ga_bind")
    assert rec is not None
    assert rec.status == ExecutionStatus.QUEUED
    assert sent and sent[0]["queue"] == cfg.dlq_queue


def test_fix4_5_schema_mismatch_does_not_kill_queued_run(
    cfg_store, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, store = cfg_store
    _record(store, cfg, "ga_sch", ExecutionStatus.QUEUED)
    sent: list[dict] = []

    class _App:
        def send_task(self, name, kwargs=None, queue=None, ignore_result=True):
            sent.append({"name": name, "kwargs": kwargs, "queue": queue})
            return type("R", (), {"id": "1"})()

    monkeypatch.setattr(
        "apps.group_agent_worker.celery_app.get_celery_app",
        lambda *a, **k: _App(),
    )
    monkeypatch.setattr(
        "apps.group_agent_worker.tasks.get_worker_runtime",
        lambda: {
            "store": store,
            "config": cfg,
            "state": None,
            "backpressure": BackpressureController(store._r, cfg),  # noqa: SLF001
        },
    )
    delivery = BrokerDeliveryRef(
        queue_schema_version=QUEUE_SCHEMA_VERSION + 99,
        run_id="ga_sch",
        idempotency_key="idem-ga_sch",
        request_fingerprint=_fp("ga_sch"),
        delivery_id=str(uuid.uuid4()),
    ).to_dict()
    out = process_run.run(delivery)
    assert out["status"] == "poison"
    assert store.get("ga_sch").status == ExecutionStatus.QUEUED  # type: ignore[union-attr]
    assert sent


def test_fix4_5_ledger_payload_missing_still_poisonable(cfg_store) -> None:
    cfg, store = cfg_store
    _record(store, cfg, "ga_miss", ExecutionStatus.QUEUED)
    store._r.hdel(  # noqa: SLF001
        store.run_key("ga_miss"),
        "payload_key_version",
        "payload_nonce_b64",
        "payload_ciphertext_b64",
        "payload_tag_b64",
    )
    store.poison_to_dlq(
        "ga_miss",
        conversation_id="c1",
        error_code="payload_missing",
        expected_status="queued",
    )
    assert store.get("ga_miss").status == ExecutionStatus.DEAD_LETTERED  # type: ignore[union-attr]


def test_fix4_4_sigkill_beat_redelivery_terminal_old_write_rejected(
    cfg_store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real worker + beat recovery; no parent expire/recovery; no prod TEST_* hooks."""
    from celery import Celery as C

    cfg, store = cfg_store
    queue = cfg.celery_queue
    claim_marker = tmp_path / "claimed.json"
    worker_log = tmp_path / "w1.log"
    beat_log = tmp_path / "beat.log"
    w2_log = tmp_path / "w2.log"

    _record(store, cfg, "ga_rd4", ExecutionStatus.QUEUED)

    env_common = {
        "GROUP_AGENT_DURABLE_QUEUE_ENABLED": "1",
        "GROUP_AGENT_REDIS_URL": REDIS_URL,
        "GROUP_AGENT_REDIS_PREFIX": PREFIX,
        "GROUP_AGENT_CELERY_QUEUE": queue,
        "GROUP_AGENT_DLQ_QUEUE": cfg.dlq_queue,
        "GROUP_AGENT_QUEUE_PAYLOAD_KEYS": f"v1:{TEST_KEY_V1}",
        "GROUP_AGENT_QUEUE_PAYLOAD_CURRENT_VERSION": "v1",
        "GROUP_AGENT_LEASE_TTL_S": "6",
        "GROUP_AGENT_VISIBILITY_TIMEOUT_S": "60",
        "GROUP_AGENT_TASK_SOFT_LIMIT_S": "20",
        "GROUP_AGENT_TASK_HARD_LIMIT_S": "25",
        "GROUP_AGENT_HEARTBEAT_INTERVAL_S": "2",
        "GROUP_AGENT_RECOVERY_INTERVAL_S": "2",
        "GROUP_AGENT_MAX_ATTEMPTS": "3",
        "GROUP_AGENT_MODEL_MODE": "stub",
        "GROUP_AGENT_INTEGRATION": "stub",
        "GROUP_AGENT_CALLBACK_HMAC_SECRET": "test-secret",
        "GROUP_AGENT_CALLBACK_ALLOWED_BASE_URLS": "http://127.0.0.1:9/group_agent_callbacks",
        "PYTHONPATH": str(Path.cwd()) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }

    hold_script = tmp_path / "hold_worker.py"
    hold_script.write_text(
        "\n".join(
            [
                "import json, os, time",
                "from apps.group_agent_api.execution.redis_store import ExecutionStore",
                "_orig = ExecutionStore.claim_lease",
                f"_marker = r'{claim_marker}'",
                "def _claim(self, *a, **k):",
                "    out = _orig(self, *a, **k)",
                "    if out.kind == 'claimed' and out.claim is not None:",
                "        c = out.claim",
                "        open(_marker, 'w').write(json.dumps({",
                "            'attempt_id': c.attempt_id,",
                "            'fencing_token': c.fencing_token,",
                "            'lease_token': c.lease_token,",
                "            'lease_owner': c.lease_owner,",
                "            'lease_expires_at': c.lease_expires_at,",
                "            'run_id': c.run_id,",
                "        }))",
                "        time.sleep(120)",
                "    return out",
                "ExecutionStore.claim_lease = _claim",
                "import apps.group_agent_api.agent_factory.integrations.callback_client as cc",
                "async def _ok(*a, **k):",
                "    return True",
                "cc.send_callback_event = _ok",
                "from apps.group_agent_worker.celery_app import get_celery_app",
                "app = get_celery_app(for_worker=True)",
                f"app.worker_main(['worker','--loglevel=info','--pool=solo','--concurrency=1',"
                f"'--without-heartbeat','--without-mingle','--without-gossip','-Q','{queue}'])",
                "",
            ]
        ),
        encoding="utf-8",
    )

    fast_script = tmp_path / "fast_worker.py"
    fast_script.write_text(
        "\n".join(
            [
                "import apps.group_agent_api.agent_factory.integrations.callback_client as cc",
                "async def _ok(*a, **k):",
                "    return True",
                "cc.send_callback_event = _ok",
                "import apps.group_agent_api.app.async_manager as am",
                "async def _fast(**kwargs):",
                "    emit = kwargs.get('emit_callback')",
                "    if emit:",
                "        await emit('final', {'reply': 'ok', 'match_status': 'skipped'})",
                "am.execute_async_run_core = _fast",
                "from apps.group_agent_worker.celery_app import get_celery_app",
                "app = get_celery_app(for_worker=True)",
                f"app.worker_main(['worker','--loglevel=info','--pool=solo','--concurrency=1',"
                f"'--without-heartbeat','--without-mingle','--without-gossip','-Q','{queue}'])",
                "",
            ]
        ),
        encoding="utf-8",
    )

    def _env(worker_id: str, beat: str = "0") -> dict:
        env = os.environ.copy()
        env.update(env_common)
        env["GROUP_AGENT_WORKER_INSTANCE_ID"] = worker_id
        env["GROUP_AGENT_WORKER_BEAT"] = beat
        return env

    w1 = subprocess.Popen(  # noqa: S603
        [sys.executable, str(hold_script)],
        stdout=worker_log.open("w"),
        stderr=subprocess.STDOUT,
        cwd=str(Path.cwd()),
        env=_env("fix4-w1", "0"),
    )
    beat = subprocess.Popen(  # noqa: S603
        [
            sys.executable,
            "-c",
            "from apps.group_agent_worker.celery_app import get_celery_app;"
            "app=get_celery_app(for_worker=True);"
            "app.Beat(logfile=None).run()",
        ],
        stdout=beat_log.open("w"),
        stderr=subprocess.STDOUT,
        cwd=str(Path.cwd()),
        env=_env("fix4-beat", "1"),
    )
    try:
        pub = C("fix4pub", broker=REDIS_URL)
        pub.conf.update(
            task_serializer="json",
            accept_content=["json"],
            task_default_queue=queue,
            broker_transport_options={"visibility_timeout": 5},
        )
        delivery = BrokerDeliveryRef(
            queue_schema_version=QUEUE_SCHEMA_VERSION,
            run_id="ga_rd4",
            idempotency_key="idem-ga_rd4",
            request_fingerprint=_fp("ga_rd4"),
            delivery_id=str(uuid.uuid4()),
        )
        pub.send_task(
            "group_agent.process_run",
            kwargs={"delivery": delivery.to_dict()},
            queue=queue,
        )

        for _ in range(120):
            if claim_marker.exists():
                break
            time.sleep(0.25)
        assert claim_marker.exists(), worker_log.read_text(encoding="utf-8")[-4000:]
        old = json.loads(claim_marker.read_text(encoding="utf-8"))
        old_attempt = old["attempt_id"]
        assert store.get("ga_rd4").status == ExecutionStatus.RUNNING  # type: ignore[union-attr]

        os.kill(w1.pid, signal.SIGKILL)
        w1.wait(timeout=10)

        w2 = subprocess.Popen(  # noqa: S603
            [sys.executable, str(fast_script)],
            stdout=w2_log.open("w"),
            stderr=subprocess.STDOUT,
            cwd=str(Path.cwd()),
            env=_env("fix4-w2", "0"),
        )
        try:
            terminal = None
            deadline = time.time() + 90
            while time.time() < deadline:
                rec = store.get("ga_rd4")
                if rec and rec.status in {
                    ExecutionStatus.SUCCEEDED,
                    ExecutionStatus.FAILED,
                    ExecutionStatus.DEAD_LETTERED,
                }:
                    terminal = rec
                    break
                time.sleep(0.5)
            assert terminal is not None, (
                f"status={store.get('ga_rd4') and store.get('ga_rd4').status} "
                f"w1={worker_log.read_text()[-1500:]} "
                f"w2={w2_log.read_text()[-1500:]} "
                f"beat={beat_log.read_text()[-1500:]}"
            )
            assert terminal.attempt_count >= 2
            assert terminal.current_attempt_id != old_attempt

            old_claim = LeaseClaim(
                run_id="ga_rd4",
                attempt_id=old["attempt_id"],
                lease_token=old["lease_token"],
                lease_owner=old["lease_owner"],
                lease_expires_at=float(old["lease_expires_at"]),
                fencing_token=int(old["fencing_token"]),
            )
            kind = store.cas_profile_write_fence(user_id="u1", group_id="g1", claim=old_claim)
            assert kind != "ok"
        finally:
            if w2.poll() is None:
                os.kill(w2.pid, signal.SIGKILL)
                w2.wait(timeout=5)
    finally:
        for proc in (w1, beat):
            if proc.poll() is None:
                try:
                    os.kill(proc.pid, signal.SIGKILL)
                except OSError:
                    pass
                try:
                    proc.wait(timeout=5)
                except Exception:  # noqa: BLE001
                    pass


def test_fix4_2_unmappable_poison_still_works(cfg_store, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg, store = cfg_store
    sent: list[dict] = []

    class _App:
        def send_task(self, name, kwargs=None, queue=None, ignore_result=True):
            sent.append({"name": name, "kwargs": kwargs, "queue": queue})
            return type("R", (), {"id": "1"})()

    monkeypatch.setattr(
        "apps.group_agent_worker.celery_app.get_celery_app",
        lambda *a, **k: _App(),
    )
    poison_id = publish_unmappable_poison(
        cfg,
        error_code="deserialize_failed",
        raw_preview="bad",
        delivery={"run_id": "x"},
    )
    assert poison_id
    assert sent and sent[0]["queue"] == cfg.dlq_queue
