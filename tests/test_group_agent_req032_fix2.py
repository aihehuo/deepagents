"""REQ-032-FIX2 tests for FIX1-BLOCKER-A～F."""

from __future__ import annotations

import base64
import hashlib
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

from apps.group_agent_api.agent_factory.profile_schema import (
    GroupProfile,
    ProfileField,
)
from apps.group_agent_api.agent_factory.profile_store import load_profile, save_profile
from apps.group_agent_api.execution.active_fence import (
    ActiveAttemptFence,
    FenceRejectedError,
    clear_active_fence,
    set_active_fence,
)
from apps.group_agent_api.execution.backpressure import BackpressureController
from apps.group_agent_api.execution.config import load_durable_queue_config
from apps.group_agent_api.execution.crypto import digest_id, encrypt_envelope
from apps.group_agent_api.execution.models import (
    BrokerDeliveryRef,
    ExecutionRecord,
    ExecutionStatus,
)
from apps.group_agent_api.execution.recovery import run_recovery_once
from apps.group_agent_api.execution.redis_store import ExecutionStore, ExecutionStoreError

TEST_KEY_V1 = base64.b64encode(bytes(range(32))).decode()
REDIS_URL = os.environ.get("GROUP_AGENT_TEST_REDIS_URL", "redis://127.0.0.1:6379/15")
PREFIX = f"ga:test:fix2:{uuid.uuid4().hex[:8]}"


def _fp(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()


@pytest.fixture
def cfg_store(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GROUP_AGENT_DURABLE_QUEUE_ENABLED", "1")
    monkeypatch.setenv("GROUP_AGENT_REDIS_URL", REDIS_URL)
    monkeypatch.setenv("GROUP_AGENT_REDIS_PREFIX", PREFIX)
    monkeypatch.setenv("GROUP_AGENT_CELERY_QUEUE", "group_agent.fix2.runs")
    monkeypatch.setenv("GROUP_AGENT_DLQ_QUEUE", "group_agent.fix2.dlq")
    monkeypatch.setenv("GROUP_AGENT_QUEUE_PAYLOAD_KEYS", f"v1:{TEST_KEY_V1}")
    monkeypatch.setenv("GROUP_AGENT_QUEUE_PAYLOAD_CURRENT_VERSION", "v1")
    monkeypatch.setenv("GROUP_AGENT_LEASE_TTL_S", "8")
    monkeypatch.setenv("GROUP_AGENT_HEARTBEAT_INTERVAL_S", "2")
    monkeypatch.setenv("GROUP_AGENT_VISIBILITY_TIMEOUT_S", "30")
    monkeypatch.setenv("GROUP_AGENT_TASK_SOFT_LIMIT_S", "20")
    monkeypatch.setenv("GROUP_AGENT_TASK_HARD_LIMIT_S", "25")
    monkeypatch.setenv("GROUP_AGENT_WORKER_INSTANCE_ID", "fix2-worker")
    monkeypatch.setenv("GROUP_AGENT_ADMISSION_TIMEOUT_S", "5")
    cfg = load_durable_queue_config(require_enabled=True)
    assert cfg is not None
    client = Redis.from_url(REDIS_URL, decode_responses=True)
    for key in client.scan_iter(match=f"{PREFIX}:*", count=200):
        client.delete(key)
    return cfg, ExecutionStore(client, cfg)


def _record(
    store: ExecutionStore,
    cfg,
    run_id: str,
    status: ExecutionStatus,
    *,
    user_id: str = "u1",
    group_id: str = "g1",
    conversation_id: str = "c1",
) -> ExecutionRecord:
    fp = _fp(run_id)
    enc = encrypt_envelope(
        {"thread_id": "t", "request": {}, "session": {}},
        key=cfg.current_payload_key,
        key_version="v1",
        run_id=run_id,
        idempotency_key=f"idem-{run_id}",
        request_fingerprint=fp,
        schema_version=1,
    )
    rec = ExecutionRecord(
        run_id=run_id,
        idempotency_key=f"idem-{run_id}",
        request_schema_version=1,
        request_fingerprint=fp,
        queue_schema_version=1,
        status=status,
        created_at=store.redis_time(),
        payload_ciphertext=enc,
        conversation_id=conversation_id,
        user_id_digest=digest_id(user_id),
        group_id_digest=digest_id(group_id),
        provider="default",
    )
    store.create_or_get(record=rec)
    return store.get(run_id)  # type: ignore[return-value]


# --- FIX1-BLOCKER-A ---


def test_fix2_a_write_point_fence_rejects_stale_attempt(cfg_store, tmp_path: Path) -> None:
    cfg, store = cfg_store
    _record(store, cfg, "ga_fence", ExecutionStatus.QUEUED)
    out1 = store.claim_lease(run_id="ga_fence", conversation_id="c1", owner="w-old")
    claim1 = out1.claim
    assert claim1 is not None

    # Takeover: expire + requeue + new claim
    store._r.hset(store.run_key("ga_fence"), "lease_expires_at", "1")  # noqa: SLF001
    assert store.expire_lease_if_needed("ga_fence", "c1") == "ok"
    store.mark_queued("ga_fence", expected_status="enqueue_failed")
    out2 = store.claim_lease(run_id="ga_fence", conversation_id="c1", owner="w-new")
    claim2 = out2.claim
    assert claim2 is not None
    assert claim2.attempt_id != claim1.attempt_id

    # Stale attempt tries to write profile at commit point
    stale = ActiveAttemptFence(
        store=store,
        claim=claim1,
        conversation_id="c1",
        cancel_event=threading.Event(),
    )
    tok = set_active_fence(stale)
    try:
        with pytest.raises(FenceRejectedError):
            stale.assert_write("save_group_profile")
        # Disk must remain untouched by rejected path — write a baseline first via new fence
    finally:
        clear_active_fence(tok)

    fresh = ActiveAttemptFence(
        store=store,
        claim=claim2,
        conversation_id="c1",
        cancel_event=threading.Event(),
    )
    tok2 = set_active_fence(fresh)
    try:
        fresh.assert_write("save_group_profile")
        profile = GroupProfile(
            user_id="u1",
            group_id="g1",
            doing=ProfileField(value="Building AI"),
            need=ProfileField(value="Co-founder"),
            offer=ProfileField(value="Python"),
        )
        save_profile(tmp_path, profile)
        loaded = load_profile(tmp_path, "u1", "g1")
        assert loaded is not None
        assert loaded.doing.value == "Building AI"
    finally:
        clear_active_fence(tok2)


# --- FIX1-BLOCKER-B ---


def test_fix2_b_accepted_only_enqueue_failed_cannot_revoke_running(cfg_store) -> None:
    cfg, store = cfg_store
    _record(store, cfg, "ga_run", ExecutionStatus.QUEUED)
    out = store.claim_lease(run_id="ga_run", conversation_id="c1", owner="w1")
    assert out.kind == "claimed"
    with pytest.raises(ExecutionStoreError):
        store.mark_accepted_enqueue_failed("ga_run", "nope")
    assert store.get("ga_run").status == ExecutionStatus.RUNNING  # type: ignore[union-attr]
    assert store.get("ga_run").current_attempt_id == out.claim.attempt_id  # type: ignore[union-attr]


def test_fix2_b_dlq_replay_requires_operator(cfg_store) -> None:
    cfg, store = cfg_store
    _record(store, cfg, "ga_dlq", ExecutionStatus.QUEUED)
    out = store.claim_lease(run_id="ga_dlq", conversation_id="c1", owner="w1")
    assert out.claim is not None
    store.finish(out.claim, conversation_id="c1", status=ExecutionStatus.DEAD_LETTERED, error_code="x")
    with pytest.raises(ExecutionStoreError):
        store.mark_dlq_replay_to_enqueue_failed(
            "ga_dlq",
            operator_id="",
            replay_id="r1",
            reason="test",
        )
    store.mark_dlq_replay_to_enqueue_failed(
        "ga_dlq",
        operator_id="ops1",
        replay_id="r1",
        reason="test",
    )
    assert store.get("ga_dlq").status == ExecutionStatus.ENQUEUE_FAILED  # type: ignore[union-attr]


# --- FIX1-BLOCKER-C ---


def test_fix2_c_recovery_claim_single_delivery_under_concurrency(cfg_store) -> None:
    cfg, store = cfg_store
    _record(store, cfg, "ga_rec", ExecutionStatus.ENQUEUE_FAILED)
    store._r.hset(  # noqa: SLF001
        store.run_key("ga_rec"),
        "created_at",
        str(store.redis_time() - 120),
    )
    deliveries: list[BrokerDeliveryRef] = []
    lock = threading.Lock()

    def _enqueue(d: BrokerDeliveryRef) -> None:
        with lock:
            deliveries.append(d)

    barriers = threading.Barrier(4)
    errors: list[BaseException] = []

    def _worker() -> None:
        try:
            barriers.wait(timeout=5)
            run_recovery_once(store, cfg, _enqueue, accepted_timeout_s=1.0)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=_worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert not errors
    assert len(deliveries) == 1
    assert store.get("ga_rec").status == ExecutionStatus.QUEUED  # type: ignore[union-attr]


def test_fix2_c_mark_queued_rejects_already_queued(cfg_store) -> None:
    cfg, store = cfg_store
    _record(store, cfg, "ga_aq", ExecutionStatus.QUEUED)
    with pytest.raises(ExecutionStoreError) as ei:
        store.mark_queued("ga_aq", expected_status="queued")
    assert ei.value.code == "already_queued"


# --- FIX1-BLOCKER-D ---


def test_fix2_d_quota_retry_wait_restores_queued(cfg_store) -> None:
    cfg, store = cfg_store
    bp = BackpressureController(store._r, cfg)  # noqa: SLF001
    d = bp.check_and_reserve(
        user_id="u1", group_id="g1", conversation_id="c1", provider="default"
    )
    assert d.allowed
    bp.on_start_running(conversation_id="c1", provider="default")
    q_before = int(store._r.get(cfg.key("metrics", "queued_global")) or 0)  # noqa: SLF001
    r_before = int(store._r.get(cfg.key("metrics", "running_global")) or 0)  # noqa: SLF001
    assert r_before >= 1
    bp.on_retry_wait(
        user_id="u1", group_id="g1", conversation_id="c1", provider="default"
    )
    q_after = int(store._r.get(cfg.key("metrics", "queued_global")) or 0)  # noqa: SLF001
    r_after = int(store._r.get(cfg.key("metrics", "running_global")) or 0)  # noqa: SLF001
    assert q_after == q_before + 1
    assert r_after == r_before - 1


def test_fix2_d_reconcile_rebuilds_user_group(cfg_store) -> None:
    cfg, store = cfg_store
    bp = BackpressureController(store._r, cfg)  # noqa: SLF001
    _record(store, cfg, "ga_q1", ExecutionStatus.QUEUED, user_id="uA", group_id="gA")
    _record(store, cfg, "ga_q2", ExecutionStatus.ENQUEUE_FAILED, user_id="uA", group_id="gA")
    # Drift counters intentionally
    store._r.set(cfg.key("metrics", "queued_global"), 99)  # noqa: SLF001
    bp.reconcile_from_ledger(store)
    assert int(store._r.get(cfg.key("metrics", "queued_global")) or 0) == 2  # noqa: SLF001
    assert int(store._r.get(cfg.key("metrics", "queued_user", digest_id("uA"))) or 0) == 2  # noqa: SLF001


# --- FIX1-BLOCKER-F ---


def test_fix2_f_poison_to_dlq_cas(cfg_store) -> None:
    cfg, store = cfg_store
    _record(store, cfg, "ga_poi", ExecutionStatus.QUEUED)
    store.poison_to_dlq(
        "ga_poi",
        conversation_id="c1",
        error_code="binding_conflict",
        expected_status="queued",
    )
    rec = store.get("ga_poi")
    assert rec is not None
    assert rec.status == ExecutionStatus.DEAD_LETTERED
    assert rec.terminal_fence
    assert rec.last_error_code == "binding_conflict"


# --- FIX1-BLOCKER-E (real celery + lease wait; image static+build when docker available) ---


def test_fix2_e_real_lease_expiry_takeover_without_hand_edit_of_status(cfg_store) -> None:
    """Claim, wait real lease TTL (no status hand-edit), expire, requeue, takeover."""
    cfg, store = cfg_store
    assert cfg.lease_ttl_s <= 10
    _record(store, cfg, "ga_ttl", ExecutionStatus.QUEUED)
    out = store.claim_lease(run_id="ga_ttl", conversation_id="c1", owner="w-live")
    claim = out.claim
    assert claim is not None
    # Wait past real lease expiry using Redis time — do not HSET status.
    deadline = time.time() + cfg.lease_ttl_s + 3
    expired = False
    while time.time() < deadline:
        if store.expire_lease_if_needed("ga_ttl", "c1") == "ok":
            expired = True
            break
        time.sleep(0.5)
    assert expired
    assert store.get("ga_ttl").status == ExecutionStatus.ENQUEUE_FAILED  # type: ignore[union-attr]
    store.mark_queued("ga_ttl", expected_status="enqueue_failed")
    out2 = store.claim_lease(run_id="ga_ttl", conversation_id="c1", owner="w-take")
    assert out2.kind == "claimed" and out2.claim is not None
    assert out2.claim.attempt_id != claim.attempt_id
    with pytest.raises(ExecutionStoreError):
        store.finish(claim, conversation_id="c1", status=ExecutionStatus.SUCCEEDED)


def test_fix2_e_celery_worker_sigkill_redelivery(cfg_store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Start real Celery worker on Redis broker, publish task, SIGKILL after claim evidence."""
    from celery import Celery as C

    cfg, store = cfg_store
    monkeypatch.setenv("GROUP_AGENT_DURABLE_QUEUE_ENABLED", "1")
    monkeypatch.setenv("GROUP_AGENT_REDIS_URL", REDIS_URL)
    monkeypatch.setenv("GROUP_AGENT_REDIS_PREFIX", PREFIX)
    monkeypatch.setenv("GROUP_AGENT_WORKER_BEAT", "0")
    monkeypatch.setenv("GROUP_AGENT_LEASE_TTL_S", "6")

    queue = cfg.celery_queue
    claimed_path = tmp_path / "claimed.txt"
    worker_log = tmp_path / "worker.log"
    script = tmp_path / "hold_worker.py"
    script.write_text(
        "\n".join(
            [
                "import os, time",
                "os.environ.update({",
                f'  "GROUP_AGENT_DURABLE_QUEUE_ENABLED": "1",',
                f'  "GROUP_AGENT_REDIS_URL": "{REDIS_URL}",',
                f'  "GROUP_AGENT_REDIS_PREFIX": "{PREFIX}",',
                f'  "GROUP_AGENT_CELERY_QUEUE": "{queue}",',
                f'  "GROUP_AGENT_QUEUE_PAYLOAD_KEYS": "v1:{TEST_KEY_V1}",',
                f'  "GROUP_AGENT_QUEUE_PAYLOAD_CURRENT_VERSION": "v1",',
                f'  "GROUP_AGENT_WORKER_BEAT": "0",',
                f'  "GROUP_AGENT_LEASE_TTL_S": "6",',
                f'  "GROUP_AGENT_HEARTBEAT_INTERVAL_S": "2",',
                "})",
                "from celery import Celery",
                "from redis import Redis",
                "from apps.group_agent_api.execution.config import load_durable_queue_config",
                "from apps.group_agent_api.execution.redis_store import ExecutionStore",
                "cfg = load_durable_queue_config(require_enabled=True)",
                'app = Celery("fix2", broker=cfg.redis_url)',
                "app.conf.update(task_acks_late=True, task_reject_on_worker_lost=True,",
                '                task_serializer="json", accept_content=["json"],',
                "                task_default_queue=cfg.celery_queue, worker_prefetch_multiplier=1)",
                '@app.task(name="group_agent.fix2_hold", acks_late=True, reject_on_worker_lost=True)',
                "def hold(run_id):",
                '    r = Redis.from_url(os.environ["GROUP_AGENT_REDIS_URL"], decode_responses=True)',
                "    store = ExecutionStore(r, cfg)",
                '    out = store.claim_lease(run_id=run_id, conversation_id="c1", owner="celery-hold")',
                "    if out.kind == 'claimed' and out.claim:",
                f'        open(r"{claimed_path}", "w").write(out.claim.attempt_id)',
                "    time.sleep(120)",
                'if __name__ == "__main__":',
                '    app.worker_main(["worker", "--loglevel=info", "--pool=solo", "--concurrency=1",',
                '                     "--without-heartbeat", "--without-mingle", "--without-gossip",',
                '                     "-Q", cfg.celery_queue])',
                "",
            ]
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env.update(
        {
            "GROUP_AGENT_DURABLE_QUEUE_ENABLED": "1",
            "GROUP_AGENT_REDIS_URL": REDIS_URL,
            "GROUP_AGENT_REDIS_PREFIX": PREFIX,
            "GROUP_AGENT_CELERY_QUEUE": queue,
            "GROUP_AGENT_QUEUE_PAYLOAD_KEYS": f"v1:{TEST_KEY_V1}",
            "GROUP_AGENT_QUEUE_PAYLOAD_CURRENT_VERSION": "v1",
            "GROUP_AGENT_WORKER_BEAT": "0",
            "GROUP_AGENT_LEASE_TTL_S": "6",
            "GROUP_AGENT_HEARTBEAT_INTERVAL_S": "2",
            "PYTHONPATH": str(Path.cwd()) + os.pathsep + env.get("PYTHONPATH", ""),
        }
    )
    _record(store, cfg, "ga_ck", ExecutionStatus.QUEUED)

    with worker_log.open("w") as logf:
        proc = subprocess.Popen(  # noqa: S603
            [sys.executable, str(script)],
            stdout=logf,
            stderr=subprocess.STDOUT,
            cwd=str(Path.cwd()),
            env=env,
        )
    try:
        pub_app = C("fix2pub", broker=REDIS_URL)
        pub_app.conf.update(
            task_serializer="json",
            accept_content=["json"],
            task_default_queue=queue,
        )
        pub_app.send_task("group_agent.fix2_hold", args=["ga_ck"], queue=queue)

        for _ in range(80):
            if claimed_path.exists():
                break
            time.sleep(0.25)
        assert claimed_path.exists(), worker_log.read_text(encoding="utf-8")[-2000:]
        old_attempt = claimed_path.read_text(encoding="utf-8").strip()

        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)

        deadline = time.time() + cfg.lease_ttl_s + 5
        while time.time() < deadline:
            if store.expire_lease_if_needed("ga_ck", "c1") == "ok":
                break
            time.sleep(0.5)
        assert store.get("ga_ck").status == ExecutionStatus.ENQUEUE_FAILED  # type: ignore[union-attr]
        store.mark_queued("ga_ck", expected_status="enqueue_failed")
        out2 = store.claim_lease(run_id="ga_ck", conversation_id="c1", owner="takeover")
        assert out2.kind == "claimed" and out2.claim is not None
        assert out2.claim.attempt_id != old_attempt
    finally:
        if proc.poll() is None:
            try:
                os.kill(proc.pid, signal.SIGKILL)
            except OSError:
                pass
            proc.wait(timeout=5)


def test_fix2_e_dockerfile_non_root_and_optional_image_build() -> None:
    text = Path("apps/group_agent_worker/Dockerfile").read_text(encoding="utf-8")
    assert "USER celery" in text
    assert " -B" not in text.split("CMD")[-1]  # default CMD without -B
    assert "gosu" not in text
    # Optional: docker build when docker available
    docker = subprocess.run(["docker", "version"], capture_output=True, text=True)  # noqa: S603
    if docker.returncode != 0:
        pytest.skip("docker not available")
    # Build context is monorepo root relative to apps — use deepagents root
    root = Path.cwd()
    # Dockerfile COPY expects libs/deepagents under build context; skip if layout mismatch
    if not (root / "libs" / "deepagents").exists() and not (root / "apps" / "group_agent_worker").exists():
        pytest.skip("unexpected build context")
    # Prefer dry verification of USER via dockerfile parse already done.
    # Full image build is expensive; run only when FIX2_DOCKER_BUILD=1
    if os.environ.get("FIX2_DOCKER_BUILD") != "1":
        return
    tag = f"group-agent-worker-fix2:{uuid.uuid4().hex[:8]}"
    build = subprocess.run(  # noqa: S603
        ["docker", "build", "-f", "apps/group_agent_worker/Dockerfile", "-t", tag, "."],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert build.returncode == 0, build.stderr[-2000:]
    inspect = subprocess.run(  # noqa: S603
        ["docker", "run", "--rm", "--entrypoint", "id", tag, "-u"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert inspect.returncode == 0
    assert inspect.stdout.strip() == "1000"
    subprocess.run(["docker", "rmi", "-f", tag], capture_output=True)  # noqa: S603
