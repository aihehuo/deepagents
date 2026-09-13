"""V3 durable admission/recovery with synthetic requests and localhost Redis."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import urlparse

import pytest
from fastapi import HTTPException

from apps.group_agent_api.app.execution_modes import execution_binding
from apps.group_agent_api.app.models import AsyncCallRequest, CallbackEnvelope
from apps.group_agent_api.app.utils import thread_id
from apps.group_agent_api.app.venue_protocol import calculate_v2_fingerprint, lease_scope
from apps.group_agent_api.execution.admission import admit_durable_async_call
from apps.group_agent_api.execution.backpressure import BackpressureController
from apps.group_agent_api.execution.crypto import PayloadCryptoError, decrypt_envelope
from apps.group_agent_api.execution.models import ExecutionRecord, ExecutionStatus
from apps.group_agent_api.execution.recovery import run_recovery_once
from apps.group_agent_worker import tasks
from tests.test_group_agent_mode_da001 import _body
from tests.test_group_agent_req032_integration import env_cfg  # noqa: F401
from tests.test_group_agent_venue_da001 import _session

pytestmark = pytest.mark.skipif(urlparse(os.environ.get("GROUP_AGENT_TEST_REDIS_URL", "")).hostname not in {"127.0.0.1", "localhost"}, reason="requires explicitly selected isolated localhost Redis")


def _request(index: int = 0, **updates: Any) -> AsyncCallRequest:
    body = _body(index)
    body.update(updates)
    body["context_snapshot"]["episode_id"] = body["episode_id"]
    body["request_fingerprint"] = calculate_v2_fingerprint(body)
    return AsyncCallRequest.model_validate(body)


def _admit(req: AsyncCallRequest, config: Any, store: Any, enqueue: Any) -> Any:
    tid = thread_id(user_id=req.user_id, group_id=req.group_id, conversation_id=req.conversation_id, episode_id=req.episode_id, execution_mode=req.execution_mode)
    return admit_durable_async_call(req=req, session=_session(req), thread_id=tid, store=store, config=config, backpressure=BackpressureController(store._r, config), enqueue=enqueue)


@pytest.mark.parametrize("index", [0, 1, 2, 3])
def test_admission_ciphertext_ack_restore_and_recovery(env_cfg: Any, index: int) -> None:
    config, store = env_cfg
    req = _request(index)
    enqueue = MagicMock()
    result = _admit(req, config, store, enqueue)
    assert result.response.execution_binding == execution_binding(req)
    record = store.get(req.run_id)
    restored = ExecutionRecord.from_redis_mapping(record.to_redis_mapping())
    assert restored.mode_binding == record.mode_binding
    assert restored.conversation_id == lease_scope(req)
    envelope = decrypt_envelope(restored.payload_ciphertext, keys=config.payload_keys, run_id=req.run_id, idempotency_key=req.idempotency_key, request_fingerprint=req.request_fingerprint, schema_version=3, mode_binding=restored.mode_binding)
    restored_req, session, tid = tasks._rebuild_session(envelope)
    assert restored_req.model_dump() == req.model_dump()
    assert tid == thread_id(user_id=req.user_id, group_id="global", conversation_id=req.conversation_id, episode_id=req.episode_id, execution_mode=req.execution_mode)
    assert _admit(req, config, store, enqueue).created is False
    assert enqueue.call_count == 1
    assert set(enqueue.call_args.args[0].to_dict()) == {"queue_schema_version", "run_id", "idempotency_key", "request_fingerprint", "delivery_id"}
    modified = {**record.mode_binding, "episode_id": "other"}
    with pytest.raises(PayloadCryptoError):
        decrypt_envelope(record.payload_ciphertext, keys=config.payload_keys, run_id=req.run_id, idempotency_key=req.idempotency_key, request_fingerprint=req.request_fingerprint, schema_version=3, mode_binding=modified)
    broken = copy.deepcopy(envelope)
    broken["thread_id"] = "ga::101::global::legacy"
    with pytest.raises(ValueError, match="binding_mismatch"):
        tasks._rebuild_session(broken)
    report = run_recovery_once(store, config, MagicMock())
    assert report.errors == 0


def test_modes_and_episodes_have_independent_leases_but_shared_profile_epoch(env_cfg: Any) -> None:
    config, store = env_cfg
    cafe = _request(0)
    group = _request(2)
    second = _request(2, run_id="00000000-0000-4000-8000-000000000099", idempotency_key="synthetic-second-episode", episode_id="ep_other")
    for req in (cafe, group, second):
        _admit(req, config, store, MagicMock())
    group_record = store.get(group.run_id)
    epoch = store.epoch_key_for_record(group_record, conversation_id=lease_scope(group))
    cafe_epoch = store.epoch_key_for_record(store.get(cafe.run_id), conversation_id=lease_scope(cafe))
    assert cafe_epoch != epoch
    assert epoch == store.epoch_key_for_record(store.get(second.run_id), conversation_id=lease_scope(second))
    claims = [store.claim_lease(run_id=req.run_id, conversation_id=lease_scope(req), owner="synthetic-worker") for req in (group, cafe, second)]
    assert all(claim.kind == "claimed" for claim in claims)
    assert claims[0].claim.fencing_token < claims[2].claim.fencing_token
    assert store.cas_profile_write_fence(claim=claims[0].claim, user_id="101", group_id="global") == "stale"


def test_expired_lease_recovery_and_cancel_reject_late_effects(env_cfg: Any) -> None:
    from apps.group_agent_api.execution.dlq import DlqAdmin
    from apps.group_agent_api.execution.fence import SideEffectFence, LeaseLostError
    config, store = env_cfg
    req = _request(2)
    enqueue = MagicMock()
    _admit(req, config, store, enqueue)
    claim = store.claim_lease(run_id=req.run_id, conversation_id=lease_scope(req), owner="lost-worker").claim
    original = store.get(req.run_id)
    store._r.hset(store.run_key(req.run_id), "lease_expires_at", store.redis_time() - 1)
    report = run_recovery_once(store, config, enqueue)
    assert report.lease_expired == 1
    recovered = store.get(req.run_id)
    assert recovered.mode_binding == original.mode_binding
    assert recovered.request_fingerprint == original.request_fingerprint
    assert recovered.payload_ciphertext.to_dict() == original.payload_ciphertext.to_dict()
    assert enqueue.call_args.args[0].queue_schema_version == 3
    with pytest.raises(LeaseLostError):
        SideEffectFence(store=store, claim=claim, conversation_id=lease_scope(req)).check_side_effect("late-profile")
    new_claim = store.claim_lease(run_id=req.run_id, conversation_id=lease_scope(req), owner="recovered-worker").claim
    assert new_claim.attempt_id != claim.attempt_id
    store.finish(new_claim, conversation_id=lease_scope(req), status=ExecutionStatus.DEAD_LETTERED, error_code="synthetic-cancel-fence")
    DlqAdmin(store, config, enqueue).cancel(req.run_id, operator_id="synthetic-operator", reason="synthetic-cancel")
    with pytest.raises(LeaseLostError):
        SideEffectFence(store=store, claim=new_claim, conversation_id=lease_scope(req)).check_side_effect("late-callback")


@pytest.mark.parametrize("index", [0, 2])
def test_worker_output_binding_and_lease_fence(env_cfg: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, index: int) -> None:
    config, store = env_cfg
    req = _request(index)
    enqueue = MagicMock()
    _admit(req, config, store, enqueue)
    callback = AsyncMock(return_value=True)
    captured: list[Any] = []

    async def core(**kwargs: Any) -> None:
        captured.append(kwargs)
        await kwargs["emit_callback"]("final", {"reply": "合成回复"})

    monkeypatch.setattr(tasks, "execute_async_run_core", core)
    monkeypatch.setattr(tasks, "send_callback_event", callback)
    monkeypatch.setattr(tasks, "get_worker_runtime", lambda: {"store": store, "config": config, "state": SimpleNamespace(base_dir=tmp_path), "backpressure": BackpressureController(store._r, config)})
    result = tasks.process_run.run(enqueue.call_args.args[0].to_dict())
    assert result["status"] == "succeeded"
    assert captured[0]["req"].model_dump() == req.model_dump()
    events = [call.kwargs["envelope_dict"] for call in callback.await_args_list]
    allowed = {"version", "run_id", "user_id", "group_id", "conversation_id", "episode_id", "execution_binding", "seq", "event", "occurred_at", "payload"}
    assert all(set(event) <= allowed for event in events)
    assert all("idempotency_key" not in event for event in events)
    for event in events:
        assert event["execution_binding"] == execution_binding(req)
        assert event["episode_id"] == req.episode_id
        assert "input_kind" not in event
        CallbackEnvelope.model_validate(event)
    assert [event["seq"] for event in events] == [1, 2]
    assert tasks.process_run.run(enqueue.call_args.args[0].to_dict())["status"] == "terminal"
    assert len(captured) == 1


def test_expired_entry_never_calls_model(env_cfg: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, store = env_cfg
    req = _request(3)
    enqueue = MagicMock()
    _admit(req, config, store, enqueue)
    core = AsyncMock(side_effect=AssertionError("expired event must not execute"))
    callback = AsyncMock(return_value=True)
    monkeypatch.setattr(tasks, "execute_async_run_core", core)
    monkeypatch.setattr(tasks, "send_callback_event", callback)
    monkeypatch.setattr(tasks, "get_worker_runtime", lambda: {"store": store, "config": config, "state": SimpleNamespace(base_dir=tmp_path), "backpressure": BackpressureController(store._r, config)})
    assert tasks.process_run.run(enqueue.call_args.args[0].to_dict())["status"] == "dead_lettered"
    assert not core.await_count
    assert callback.await_args.kwargs["envelope_dict"]["payload"]["error_code"] == "venue_event_expired"


def test_cross_mode_idempotency_conflict_and_forged_delivery(env_cfg: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    config, store = env_cfg
    cafe = _request(0)
    enqueue = MagicMock()
    _admit(cafe, config, store, enqueue)
    group = _request(2, idempotency_key=cafe.idempotency_key)
    with pytest.raises(HTTPException) as exc:
        _admit(group, config, store, MagicMock())
    assert exc.value.status_code == 409
    monkeypatch.setattr(tasks, "get_worker_runtime", lambda: {"store": store, "config": config})
    import apps.group_agent_api.execution.dlq as dlq
    monkeypatch.setattr(dlq, "publish_unmappable_poison", MagicMock())
    for updates in ({"queue_schema_version": 4}, {"request_fingerprint": "0" * 64}):
        delivery = {**enqueue.call_args.args[0].to_dict(), **updates}
        assert tasks.process_run.run(delivery)["status"] == "poison"
        assert store.get(cafe.run_id).status == ExecutionStatus.QUEUED


async def test_endpoint_uses_durable_v3_ack(env_cfg: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from starlette.requests import Request
    import apps.group_agent_api.app.endpoints.call_async as endpoint
    from apps.group_agent_api.app.state import AppState
    from apps.group_agent_api.app.venue_protocol import canonical_json
    config, store = env_cfg
    req = _request(2)
    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": canonical_json(req.model_dump()).encode()}
    request = Request({"type": "http", "method": "POST", "path": "/call_async", "headers": []}, receive=receive)
    monkeypatch.setattr(endpoint, "validate_and_normalize_callback_url", lambda value: value)
    monkeypatch.setattr(endpoint, "resolve_trusted_session", AsyncMock(return_value=_session(req)))
    publish = MagicMock()
    monkeypatch.setattr(endpoint, "enqueue_delivery", publish)
    state = AppState(agent=None, base_dir=tmp_path, durable_store=store, durable_config=config, backpressure=BackpressureController(store._r, config))
    response = await endpoint.call_async(req, state, request)
    assert response.execution_binding == execution_binding(req)
    assert publish.call_args.args[1].queue_schema_version == 3


def test_private_review_and_profile_transport_keep_frozen_binding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    from apps.group_agent_api.agent_factory.cafe_review_audit import review_audit
    from apps.group_agent_api.agent_factory.host_context import HostContext
    from apps.group_agent_api.agent_factory.integrations.profile_client import persist_group_profile, ProfileHttpError
    from tests.test_group_agent_host_da002 import _profile
    monkeypatch.setenv("GROUP_AGENT_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("GROUP_AGENT_CAFE_REVIEW_AUDIT", "1")
    profile = _profile(tmp_path)
    cafe, group = _request(0), _request(2)
    host = HostContext.from_request(cafe, principal=_session(cafe).principal, base_dir=tmp_path)
    with review_audit(host, "合成审核正文", []):
        pass
    audit = json.loads(next((tmp_path / "cafe_review_audits").glob("review-*.json")).read_text())
    assert audit["execution_binding"] == execution_binding(cafe)
    assert audit["executor"] == "cafe_agent"
    captured = MagicMock(return_value=SimpleNamespace(status_code=409))
    monkeypatch.setattr("apps.group_agent_api.agent_factory.integrations.profile_client.requests.post", captured)
    with pytest.raises(ProfileHttpError, match="http_409"):
        persist_group_profile(profile=profile, run_id=group.run_id, execution_binding=execution_binding(group), secret="synthetic-mode-test-key", base_url="http://127.0.0.1:9")
    payload = json.loads(captured.call_args.kwargs["data"])
    assert payload["execution_binding"] == execution_binding(group)
    with pytest.raises(ProfileHttpError, match="execution_binding_mismatch"):
        persist_group_profile(profile=profile, run_id=cafe.run_id, execution_binding=execution_binding(cafe), secret="synthetic-mode-test-key")
