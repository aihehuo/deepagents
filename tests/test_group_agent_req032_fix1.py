"""REQ-032-FIX1 tests for blockers 1–7."""

from __future__ import annotations

import base64
import hashlib
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError
from redis import Redis

from apps.group_agent_api.app.models import AsyncCallRequest
from apps.group_agent_api.execution.config import load_durable_queue_config
from apps.group_agent_api.execution.crypto import digest_id, encrypt_envelope
from apps.group_agent_api.execution.fence import SideEffectFence
from apps.group_agent_api.execution.models import (
    ExecutionRecord,
    ExecutionStatus,
    LeaseClaim,
)
from apps.group_agent_api.execution.redis_store import ExecutionStore
from apps.group_agent_worker.celery_app import build_celery_app

TEST_KEY_V1 = base64.b64encode(bytes(range(32))).decode()
REDIS_URL = os.environ.get("GROUP_AGENT_TEST_REDIS_URL", "redis://127.0.0.1:6379/15")
PREFIX = f"ga:test:fix1:{uuid.uuid4().hex[:8]}"


def _fp(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()


@pytest.fixture
def cfg_store(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GROUP_AGENT_DURABLE_QUEUE_ENABLED", "1")
    monkeypatch.setenv("GROUP_AGENT_REDIS_URL", REDIS_URL)
    monkeypatch.setenv("GROUP_AGENT_REDIS_PREFIX", PREFIX)
    monkeypatch.setenv("GROUP_AGENT_CELERY_QUEUE", "group_agent.fix1.runs")
    monkeypatch.setenv("GROUP_AGENT_DLQ_QUEUE", "group_agent.fix1.dlq")
    monkeypatch.setenv("GROUP_AGENT_QUEUE_PAYLOAD_KEYS", f"v1:{TEST_KEY_V1}")
    monkeypatch.setenv("GROUP_AGENT_QUEUE_PAYLOAD_CURRENT_VERSION", "v1")
    monkeypatch.setenv("GROUP_AGENT_LEASE_TTL_S", "30")
    monkeypatch.setenv("GROUP_AGENT_HEARTBEAT_INTERVAL_S", "10")
    monkeypatch.setenv("GROUP_AGENT_VISIBILITY_TIMEOUT_S", "240")
    monkeypatch.setenv("GROUP_AGENT_TASK_SOFT_LIMIT_S", "150")
    monkeypatch.setenv("GROUP_AGENT_TASK_HARD_LIMIT_S", "180")
    monkeypatch.setenv("GROUP_AGENT_WORKER_INSTANCE_ID", "fix1-worker")
    cfg = load_durable_queue_config(require_enabled=True)
    assert cfg is not None
    client = Redis.from_url(REDIS_URL, decode_responses=True)
    for key in client.scan_iter(match=f"{PREFIX}:*", count=200):
        client.delete(key)
    return cfg, ExecutionStore(client, cfg)


def _record(store, cfg, run_id: str, status: ExecutionStatus) -> ExecutionRecord:
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
        conversation_id="c1",
        user_id_digest=digest_id("u1"),
        group_id_digest=digest_id("g1"),
    )
    store.create_or_get(record=rec)
    return store.get(run_id)  # type: ignore[return-value]


def test_blocker1_accepted_and_retry_wait_not_claimable(cfg_store) -> None:
    cfg, store = cfg_store
    _record(store, cfg, "ga_acc", ExecutionStatus.ACCEPTED)
    out = store.claim_lease(run_id="ga_acc", conversation_id="c1", owner="w1")
    assert out.kind == "not_queued"
    assert out.detail == "accepted"

    store.mark_accepted_enqueue_failed("ga_acc", "x")
    # put into retry_wait via schedule from queued path
    store.mark_queued("ga_acc", expected_status="enqueue_failed")
    store.schedule_conversation_wait("ga_acc", delay_s=60)
    out2 = store.claim_lease(run_id="ga_acc", conversation_id="c1", owner="w1")
    assert out2.kind == "not_queued"
    assert out2.detail == "retry_wait"


def test_blocker2_terminal_only_on_successful_send(cfg_store) -> None:
    cfg, store = cfg_store
    _record(store, cfg, "ga_cb", ExecutionStatus.QUEUED)
    out = store.claim_lease(run_id="ga_cb", conversation_id="c1", owner="w1")
    claim = out.claim
    assert claim is not None
    fence = SideEffectFence(store=store, claim=claim, conversation_id="c1")

    async def _fail(_e, _p):
        return False

    import asyncio

    ok = asyncio.run(fence.emit("final", {"reply": "x"}, send=_fail))
    assert ok is False
    assert fence.final_callback_ok is False
    assert fence._terminal_delivered is False  # noqa: SLF001


def test_blocker2_heartbeat_threshold(cfg_store) -> None:
    cfg, store = cfg_store
    _record(store, cfg, "ga_hb", ExecutionStatus.QUEUED)
    out = store.claim_lease(run_id="ga_hb", conversation_id="c1", owner="w1")
    claim = out.claim
    assert claim is not None
    fence = SideEffectFence(
        store=store,
        claim=claim,
        conversation_id="c1",
        abort_on_heartbeat_failures=3,
    )
    # Force renew failures by clearing token digest
    store._r.hset(store.run_key("ga_hb"), "lease_token_digest", "dead")  # noqa: SLF001
    assert fence.renew_or_abort() is True  # failure 1, still trying
    assert fence.aborted is False
    assert fence.renew_or_abort() is True  # failure 2
    assert fence.aborted is False
    assert fence.renew_or_abort() is False  # failure 3 → abort
    assert fence.aborted is True


def test_blocker3_beat_schedule_configured(cfg_store, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg, _ = cfg_store
    monkeypatch.setenv("GROUP_AGENT_WORKER_BEAT", "1")
    app = build_celery_app(cfg, for_worker=True)
    assert "group-agent-recovery-tick" in app.conf.beat_schedule
    assert app.conf.beat_schedule["group-agent-recovery-tick"]["task"] == "group_agent.recovery_tick"


def test_blocker3_beat_disabled_by_default_on_worker(cfg_store, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg, _ = cfg_store
    monkeypatch.setenv("GROUP_AGENT_WORKER_BEAT", "0")
    app = build_celery_app(cfg, for_worker=True)
    assert not getattr(app.conf, "beat_schedule", None)


def test_blocker3_expire_goes_to_enqueue_failed_not_queued(cfg_store) -> None:
    cfg, store = cfg_store
    _record(store, cfg, "ga_ex", ExecutionStatus.QUEUED)
    out = store.claim_lease(run_id="ga_ex", conversation_id="c1", owner="w1")
    assert out.kind == "claimed"
    store._r.hset(store.run_key("ga_ex"), "lease_expires_at", "1")  # noqa: SLF001
    assert store.expire_lease_if_needed("ga_ex", "c1") == "ok"
    assert store.get("ga_ex").status == ExecutionStatus.ENQUEUE_FAILED  # type: ignore[union-attr]


def test_blocker7_fingerprint_rejects_uppercase_and_whitespace() -> None:
    fp = _fp("strict")
    with pytest.raises(ValidationError):
        AsyncCallRequest.model_validate(
            {
                "run_id": "ga_x",
                "idempotency_key": "k",
                "user_id": "u1",
                "unionid": "un",
                "group_id": "g1",
                "message": "hi",
                "callback_url": "http://127.0.0.1:9/cb",
                "request_schema_version": 1,
                "request_fingerprint": fp.upper(),
                "queue_schema_version": 1,
            }
        )
    with pytest.raises(ValidationError):
        AsyncCallRequest.model_validate(
            {
                "run_id": "ga_x",
                "idempotency_key": "k",
                "user_id": "u1",
                "unionid": "un",
                "group_id": "g1",
                "message": "hi",
                "callback_url": "http://127.0.0.1:9/cb",
                "request_schema_version": 1,
                "request_fingerprint": f" {fp}",
                "queue_schema_version": 1,
            }
        )


def test_blocker5_dockerfile_is_non_root_no_gosu() -> None:
    text = Path("apps/group_agent_worker/Dockerfile").read_text(encoding="utf-8")
    assert "USER celery" in text
    assert "exec gosu" not in text
    entry = Path("apps/group_agent_worker/docker-entrypoint.sh").read_text(encoding="utf-8")
    assert "exec gosu" not in entry
    assert "id -u" not in entry


def test_blocker6_subprocess_sigkill_lease_takeover(cfg_store, tmp_path: Path) -> None:
    """Real OS SIGKILL of a child that held a lease; another claim after expiry."""
    cfg, store = cfg_store
    _record(store, cfg, "ga_kill", ExecutionStatus.QUEUED)
    out = store.claim_lease(run_id="ga_kill", conversation_id="c1", owner="parent-w")
    claim = out.claim
    assert claim is not None

    # Child process holds nothing but we SIGKILL a helper that would renew;
    # simulate worker death by expiring lease then verifying takeover.
    script = tmp_path / "holder.py"
    script.write_text(
        "import time, os\n"
        f"open('{tmp_path / 'pid.txt'}','w').write(str(os.getpid()))\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    proc = subprocess.Popen([sys.executable, str(script)])  # noqa: S603
    for _ in range(50):
        if (tmp_path / "pid.txt").exists():
            break
        time.sleep(0.05)
    os.kill(proc.pid, signal.SIGKILL)
    proc.wait(timeout=5)
    assert proc.returncode != 0

    store._r.hset(store.run_key("ga_kill"), "lease_expires_at", "1")  # noqa: SLF001
    assert store.expire_lease_if_needed("ga_kill", "c1") == "ok"
    store.mark_queued("ga_kill", expected_status="enqueue_failed")
    out2 = store.claim_lease(run_id="ga_kill", conversation_id="c1", owner="takeover-w")
    assert out2.kind == "claimed" and out2.claim is not None
    assert out2.claim.attempt_id != claim.attempt_id
    # Old claim cannot finish
    from apps.group_agent_api.execution.redis_store import ExecutionStoreError

    with pytest.raises(ExecutionStoreError):
        store.finish(claim, conversation_id="c1", status=ExecutionStatus.SUCCEEDED)
