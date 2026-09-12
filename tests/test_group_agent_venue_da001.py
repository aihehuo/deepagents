"""DA-001 protocol tests: synthetic fixtures, no LLM or external services."""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from starlette.requests import Request

from apps.group_agent_api.agent_factory.capability import CapabilityTier
from apps.group_agent_api.agent_factory.integrations.membership_client import MembershipResult
from apps.group_agent_api.agent_factory.integrations.principal import (
    SessionPrincipal,
    canonical_principal_payload,
    canonical_request_payload,
    clear_nonce_cache,
    resolve_session_principal,
    sign_principal,
)
from apps.group_agent_api.app.async_manager import (
    VenueEventExecutorUnavailable,
    calculate_request_fingerprint,
    execute_async_run_core,
)
from apps.group_agent_api.app.models import AsyncCallRequest, CallbackEnvelope
from apps.group_agent_api.app.session import TrustedSession
from apps.group_agent_api.app.utils import thread_id
from apps.group_agent_api.app.venue_protocol import (
    calculate_v2_fingerprint,
    callback_bindings,
    canonical_json,
    lease_scope,
)
from apps.group_agent_api.execution.admission import admit_durable_async_call, build_trusted_envelope
from apps.group_agent_api.execution.crypto import decrypt_envelope, encrypt_envelope
from apps.group_agent_api.execution.models import BrokerDeliveryRef, ExecutionRecord, ExecutionStatus
from apps.group_agent_api.execution.recovery import _publish_then_queued
from apps.group_agent_api.execution.retry import classify_error
from apps.group_agent_worker.tasks import _rebuild_session

FIXTURES = Path(__file__).parent / "fixtures" / "venue_entry"
VECTORS = json.loads((FIXTURES / "durable-vectors.json").read_text())["vectors"]
SIGNATURES = json.loads((FIXTURES / "signatures.json").read_text())
REQUESTS = [item for item in VECTORS if "request" in item]


def _session(req: AsyncCallRequest) -> TrustedSession:
    return TrustedSession(principal=SessionPrincipal(user_id=req.user_id, unionid=req.unionid, user_token=req.user_token, group_token=req.group_token, source="signed_oauth_principal"), membership=MembershipResult(tier=CapabilityTier.unknown, source="synthetic"), group_id=req.group_id, group_token=req.group_token)


def _request(headers: dict[str, str], *, path: str = "/call_async") -> Request:
    return Request({"type": "http", "method": "POST", "path": path, "headers": [(key.lower().encode(), value.encode()) for key, value in headers.items()], "scheme": "https", "server": ("example.test", 443), "query_string": b""})


def _refingerprint(body: dict[str, Any]) -> dict[str, Any]:
    body["request_fingerprint"] = calculate_v2_fingerprint(body)
    return body


@pytest.mark.parametrize("vector", REQUESTS, ids=[item["name"] for item in REQUESTS])
def test_frozen_vectors_survive_model_queue_and_worker(vector: dict[str, Any]) -> None:
    raw = vector["request"]
    req = AsyncCallRequest.model_validate(raw)
    assert req.model_dump() == raw
    assert calculate_v2_fingerprint(raw) == vector["expected_sha256"]
    session = _session(req)
    assert calculate_request_fingerprint(req, session) == vector["expected_sha256"]
    tid = thread_id(user_id=req.user_id, group_id=req.group_id, conversation_id=req.conversation_id, episode_id=req.episode_id)
    envelope = build_trusted_envelope(req=req, session=session, thread_id=tid)
    key = bytes(range(32))
    encrypted = encrypt_envelope(envelope, key=key, key_version="fixture", run_id=req.run_id, idempotency_key=req.idempotency_key, request_fingerprint=req.request_fingerprint, schema_version=2)
    restored = decrypt_envelope(encrypted, keys={"fixture": key}, run_id=req.run_id, idempotency_key=req.idempotency_key, request_fingerprint=req.request_fingerprint, schema_version=2)
    rebuilt, rebuilt_session, rebuilt_tid = _rebuild_session(restored)
    assert rebuilt.model_dump() == raw
    assert rebuilt_session.principal == session.principal
    assert rebuilt_tid == tid


@pytest.mark.parametrize(("path", "value"), [
    (("unknown",), "ignored?"),
    (("metadata", "profile"), {"doing": "injected"}),
    (("metadata", "number"), 1.25),
    (("metadata", "source_message_id"), 123),
    (("venue_context", "visitor_user_id"), "102"),
    (("venue_context", "space_id"), "office_102"),
    (("venue_context", "owner_user_id"), "102"),
    (("venue_context", "host_agent_id"), "community_bartender"),
    (("venue_context", "host_config_version"), True),
    (("venue_context", "timezone"), "invalid/timezone"),
    (("venue_context", "unknown"), 1),
    (("event", "name"), "profile_saved"),
    (("event", "visit_id"), "not_a_visit"),
    (("event", "unknown"), True),
    (("context_snapshot", "episode_id"), "ep_other"),
    (("context_snapshot", "user_turn_version"), True),
    (("context_snapshot", "user_turn_version"), 9007199254740992),
    (("rollout_context", "extra"), "hidden"),
    (("message",), "pretend user hello"),
    (("run_match",), True),
    (("run_invite",), True),
    (("willing_to_at",), True),
    (("user_id",), "0101"),
    (("membership",), "in_group"),
])
def test_v2_rejects_invalid_even_with_matching_fingerprint(path: tuple[str, ...], value: Any) -> None:
    body = copy.deepcopy(REQUESTS[0]["request"])
    parent = body
    for part in path[:-1]:
        parent = parent[part]
    parent[path[-1]] = value
    with pytest.raises(ValidationError):
        AsyncCallRequest.model_validate(_refingerprint(body))


def test_fingerprint_binds_all_fields_and_is_rechecked() -> None:
    body = copy.deepcopy(REQUESTS[0]["request"])
    for field in body:
        if field == "request_fingerprint":
            continue
        changed = copy.deepcopy(body)
        changed[field] = "different" if not isinstance(body[field], str) else body[field] + "_changed"
        assert calculate_v2_fingerprint(changed) != body["request_fingerprint"]
    body["metadata"]["source"] = "changed"
    with pytest.raises(ValidationError, match="fingerprint mismatch"):
        AsyncCallRequest.model_validate(body)


def test_unknown_or_mixed_versions_are_explicit_400() -> None:
    for versions in ((2, 1), (2, 3), (3, 3), ("2", 2)):
        body = copy.deepcopy(REQUESTS[0]["request"])
        body["request_schema_version"], body["queue_schema_version"] = versions
        with pytest.raises(HTTPException) as exc:
            AsyncCallRequest.model_validate(body)
        assert exc.value.status_code == 400
        assert exc.value.detail["error"] == "schema_invalid"


def test_v1_user_text_and_restoration_still_work() -> None:
    body = {"run_id": "legacy_run", "idempotency_key": "legacy_key", "user_id": "u1", "unionid": "union1", "group_id": "g1", "conversation_id": "default", "message": "真实消息", "callback_url": "https://micro.example.test/callback", "metadata": {"episode_id": "legacy_episode"}, "request_schema_version": 1, "queue_schema_version": 1, "request_fingerprint": "a" * 64}
    req = AsyncCallRequest.model_validate(body)
    envelope = build_trusted_envelope(req=req, session=_session(req), thread_id="legacy_tid")
    restored, _, tid = _rebuild_session(envelope)
    assert restored.message == "真实消息"
    assert restored.metadata["episode_id"] == "legacy_episode"
    assert tid == "legacy_tid"
    for message in (None, "", "   "):
        with pytest.raises(ValidationError):
            AsyncCallRequest.model_validate({**body, "message": message})
    with pytest.raises(ValidationError):
        AsyncCallRequest.model_validate({**body, "input_kind": "venue_event"})


def test_golden_principal_and_raw_body_signatures() -> None:
    vector = SIGNATURES["vectors"][0]
    body = vector["raw_body"].encode()
    headers = sign_principal(user_id="101", unionid="union_fixture_101", ts=SIGNATURES["timestamp"], nonce=SIGNATURES["nonce"], path="/call_async", secret=SIGNATURES["test_key"], request_schema_version=2, request_body=body)
    assert headers["X-GA-Signature"] == SIGNATURES["principal_v2"]["expected_signature"]
    assert headers["X-GA-Request-Signature"] == vector["expected_signature"]
    assert hashlib.sha256(body).hexdigest() == vector["expected_body_sha256"]
    assert canonical_principal_payload(user_id="101", unionid="union_fixture_101", user_token=None, group_token=None, ts=SIGNATURES["timestamp"], nonce=SIGNATURES["nonce"], method="POST", path="/call_async", version="GA-PRINCIPAL-V2") == SIGNATURES["principal_v2"]["canonical_payload"]
    payload = canonical_request_payload(method="POST", path="/call_async", body=body, ts=SIGNATURES["timestamp"], nonce=SIGNATURES["nonce"])
    assert hmac.new(SIGNATURES["test_key"].encode(), payload.encode(), hashlib.sha256).hexdigest() == vector["expected_signature"]


def test_v2_body_auth_and_principal_claim_nonce_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROUP_AGENT_PRINCIPAL_HMAC_SECRET", SIGNATURES["test_key"])
    clear_nonce_cache()
    body = canonical_json(REQUESTS[0]["request"]).encode()
    headers = sign_principal(user_id="101", unionid="union_fixture_101", ts=int(time.time()), nonce="fixture_v2_nonce_single", path="/call_async", secret=SIGNATURES["test_key"], request_schema_version=2, request_body=body)
    request = _request(headers)
    kwargs = {"body_user_id": "101", "body_unionid": "union_fixture_101", "force_mode": "http", "request_schema_version": 2, "request_body": body}
    with pytest.raises(HTTPException, match="request_signature_invalid"):
        resolve_session_principal(request, **{**kwargs, "request_body": body + b" "})
    # Failed body authentication did not consume the nonce; both signatures
    # now succeed using one nonce claim, not two competing claims.
    principal = resolve_session_principal(request, **kwargs)
    assert principal.user_id == "101"
    with pytest.raises(HTTPException, match="principal_nonce_replay"):
        resolve_session_principal(request, **kwargs)
    with pytest.raises(HTTPException, match="request_version_mismatch"):
        resolve_session_principal(request, **{**kwargs, "request_schema_version": 1})
    clear_nonce_cache()
    stripped = {key: value for key, value in headers.items() if not key.startswith("X-GA-Request-")}
    with pytest.raises(HTTPException, match="principal_signature_invalid"):
        resolve_session_principal(_request(stripped), **{**kwargs, "request_schema_version": 1})
    clear_nonce_cache()


@pytest.mark.parametrize("vector", REQUESTS, ids=[item["name"] for item in REQUESTS])
def test_admission_and_ack_use_run_version(vector: dict[str, Any]) -> None:
    req = AsyncCallRequest.model_validate(vector["request"])
    session = _session(req)
    store = MagicMock()
    store.get_by_idempotency.return_value = None
    store.get.return_value = None
    store.redis_time.return_value = 123.0
    store.create_or_get.side_effect = lambda *, record: ("created", record)
    store.claim_publish_delivery.return_value = "ok"
    store.mark_queued.return_value = SimpleNamespace(status=ExecutionStatus.QUEUED)
    bp = MagicMock()
    bp.check_and_reserve.return_value = SimpleNamespace(allowed=True)
    config = SimpleNamespace(current_payload_key=bytes(range(32)), payload_current_version="fixture")
    enqueue = MagicMock()
    result = admit_durable_async_call(req=req, session=session, thread_id="fixture_tid", store=store, config=config, backpressure=bp, enqueue=enqueue)
    record = store.create_or_get.call_args.kwargs["record"]
    assert record.request_schema_version == record.queue_schema_version == 2
    assert record.conversation_id == lease_scope(req)
    assert result.response.queue_schema_version == 2
    assert enqueue.call_args.args[0].queue_schema_version == 2
    assert bp.check_and_reserve.call_args.kwargs["conversation_id"] == lease_scope(req)
    assert lease_scope(req) != req.conversation_id if req.input_kind == "venue_event" else lease_scope(req) == req.conversation_id
    # Idempotent admission is hit-first and keeps the original run version.
    store.get_by_idempotency.return_value = record
    record.status = ExecutionStatus.QUEUED
    bp.reset_mock()
    again = admit_durable_async_call(req=req, session=session, thread_id="fixture_tid", store=store, config=config, backpressure=bp, enqueue=enqueue)
    assert again.response.queue_schema_version == 2
    bp.check_and_reserve.assert_not_called()


def test_worker_rejects_forged_session_or_thread_binding() -> None:
    req = AsyncCallRequest.model_validate(REQUESTS[0]["request"])
    tid = thread_id(user_id=req.user_id, group_id=req.group_id, conversation_id=req.conversation_id, episode_id=req.episode_id)
    envelope = build_trusted_envelope(req=req, session=_session(req), thread_id=tid)
    for changed in ({**envelope, "thread_id": "other_thread"}, {**envelope, "session": {**envelope["session"], "user_id": "102"}}):
        with pytest.raises(ValueError, match="binding_mismatch"):
            _rebuild_session(changed)


@pytest.mark.parametrize("schema", [1, 2])
def test_redis_model_and_recovery_preserve_versions(schema: int) -> None:
    record = ExecutionRecord(run_id="fixture_run", idempotency_key="fixture_key", request_schema_version=schema, queue_schema_version=schema, request_fingerprint="a" * 64, status=ExecutionStatus.ENQUEUE_FAILED)
    restored = ExecutionRecord.from_redis_mapping(record.to_redis_mapping())
    assert restored.queue_schema_version == restored.request_schema_version == schema
    store = MagicMock()
    store.get.return_value = restored
    store.claim_publish_delivery.return_value = "ok"
    enqueue = MagicMock()
    assert _publish_then_queued(store, enqueue, run_id=record.run_id, idempotency_key=record.idempotency_key, request_fingerprint=record.request_fingerprint, expected_status="enqueue_failed") == "ok"
    ref = enqueue.call_args.args[0]
    assert BrokerDeliveryRef.from_dict(ref.to_dict()).queue_schema_version == schema
    assert ref.queue_schema_version == schema


def test_v2_callback_fixture_and_constraints() -> None:
    callback = json.loads(SIGNATURES["vectors"][1]["raw_body"])
    env = CallbackEnvelope.model_validate(callback)
    assert env.model_dump(exclude={"idempotency_key"}) == callback
    for event, payload in (("chunk", {"text": "late partial"}), ("progress", {"phase": "started"}), ("final", {"result_kind": "greeting", "reply": "hello", "candidates": []})):
        with pytest.raises(ValidationError):
            CallbackEnvelope.model_validate({**callback, "event": event, "payload": payload})
    for change in ({"seq": True}, {"seq": 1.5}, {"unknown": 1}, {"episode_id": None}, {"visit_id": None}, {"occurred_at": "2026-02-30T00:00:00.000000Z"}):
        with pytest.raises(ValidationError):
            CallbackEnvelope.model_validate({**callback, **change})
    req = AsyncCallRequest.model_validate(REQUESTS[0]["request"])
    assert callback_bindings(req)["visit_id"] == req.event["visit_id"]


async def test_event_executor_hold_never_calls_agent_or_emits_human_turn() -> None:
    req = AsyncCallRequest.model_validate(REQUESTS[0]["request"])
    state = MagicMock()
    emit = AsyncMock()
    with pytest.raises(VenueEventExecutorUnavailable):
        await execute_async_run_core(req=req, session=_session(req), state=state, tid="fixture_tid", emit_callback=emit)
    state.assert_not_called()
    assert not state.mock_calls
    emit.assert_not_called()
    assert classify_error("venue_event_executor_unavailable").value == "permanent"


async def test_v2_ordinary_dispatches_to_original_checkpoint_core(monkeypatch: pytest.MonkeyPatch) -> None:
    from apps.group_agent_api.app import async_manager

    req = AsyncCallRequest.model_validate(REQUESTS[2]["request"])
    core = AsyncMock()
    monkeypatch.setattr(async_manager, "_execute_core_agent", core)
    tid = thread_id(user_id=req.user_id, group_id=req.group_id, conversation_id=req.conversation_id, episode_id=req.episode_id)
    await execute_async_run_core(req=req, session=_session(req), state=MagicMock(), tid=tid, emit_callback=AsyncMock())
    core.assert_awaited_once()
    assert core.await_args.kwargs["req"] is req
    assert core.await_args.kwargs["tid"].endswith(req.episode_id)
    assert core.await_args.kwargs["req"].message == "我有一个新想法"
