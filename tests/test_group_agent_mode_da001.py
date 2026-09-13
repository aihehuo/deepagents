"""Synthetic mode admission, crypto and real stub-graph routing; no network."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from apps.group_agent_api.agent_factory.host_context import HostContext, active_host_context, current_host_context
from apps.group_agent_api.agent_factory.host_policy import HostActionRejected
from apps.group_agent_api.agent_factory.profile_store import load_profile, save_profile
from apps.group_agent_api.agent_factory.integrations.principal import clear_nonce_cache, resolve_session_principal, sign_principal
from apps.group_agent_api.app.async_manager import calculate_request_fingerprint, execute_async_run_core
from apps.group_agent_api.app.execution_modes import execution_binding, validate_v3_request, validate_v3_wire
from apps.group_agent_api.app.models import AsyncCallRequest, AsyncCallResponse
from apps.group_agent_api.app.state import AppState
from apps.group_agent_api.app.utils import thread_id
from apps.group_agent_api.app.venue_protocol import calculate_v2_fingerprint, canonical_json
from apps.group_agent_api.execution.admission import build_trusted_envelope
from apps.group_agent_api.execution.crypto import PayloadCryptoError, build_aad, decrypt_envelope, encrypt_envelope
from tests.test_group_agent_venue_da001 import _request, _session
from tests.test_group_agent_host_da002 import ChatModel, _profile

FIXTURE = json.loads((Path(__file__).parent / "fixtures/venue_entry/execution-modes.json").read_text())
VECTORS = FIXTURE["vectors"]


def _body(index: int = 0) -> dict[str, Any]:
    return copy.deepcopy(VECTORS[index]["request"])


def _refingerprint(body: dict[str, Any]) -> dict[str, Any]:
    body["request_fingerprint"] = calculate_v2_fingerprint(body)
    return body


@pytest.mark.parametrize("vector", VECTORS, ids=lambda vector: vector["name"])
def test_frozen_vectors_and_signature_domains(vector: dict[str, Any]) -> None:
    body, expected = vector["request"], vector["expected"]
    req = AsyncCallRequest.model_validate(body)
    assert req.model_dump() == body
    wire = canonical_json(body).encode("utf-8")
    assert calculate_request_fingerprint(req, _session(req)) == expected["fingerprint"]
    assert hashlib.sha256(wire).hexdigest() == expected["body_sha256"]
    headers = sign_principal(user_id=req.user_id, unionid=req.unionid, user_token=req.user_token, group_token=None, ts=FIXTURE["timestamp"], nonce=FIXTURE["nonce"], method="POST", path="/call_async", secret=FIXTURE["test_key"], request_schema_version=3, request_body=wire)
    assert headers["X-GA-Signature"] == expected["principal_hmac"]
    assert headers["X-GA-Request-Signature"] == expected["body_hmac"]
    binding = {name: body[name] for name in ("user_id", "execution_mode", "conversation_id", "episode_id")}
    aad = build_aad(run_id=req.run_id, idempotency_key=req.idempotency_key, request_fingerprint=req.request_fingerprint, schema_version=3, mode_binding=binding)
    assert hashlib.sha256(aad).hexdigest() == expected["aad_sha256"]


@pytest.mark.parametrize(("path", "value"), [
    (("queue_schema_version",), 2), (("mode_contract_version",), True),
    (("execution_mode",), "admin"), (("executor_policy_version",), 2),
    (("profile_scope_id",), "admin"), (("conversation_id",), "ga_global_101"),
    (("memory_scope_id",), "community_cafe_mode_group"), (("run_match",), True),
    (("user_id",), "0101"), (("message",), ""),
    (("venue_context", "visitor_user_id"), "102"),
    (("venue_context", "host_config_version"), 2),
    (("metadata", "admin_mode"), True), (("metadata", "executor"), "admin"),
    (("context_snapshot", "user_turn_version"), 1.5),
    (("context_snapshot", "episode_id"), "other"),
])
def test_fail_closed_even_when_body_resigned(path: tuple[str, ...], value: Any) -> None:
    body = _body()
    target = body
    for field in path[:-1]:
        target = target[field]
    target[path[-1]] = value
    with pytest.raises(HTTPException) as error:
        AsyncCallRequest.model_validate(_refingerprint(body))
    assert error.value.status_code == 400
    assert error.value.detail["error"] == "schema_invalid"


def test_fingerprint_raw_body_tamper_and_domain_downgrade(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROUP_AGENT_PRINCIPAL_HMAC_SECRET", FIXTURE["test_key"])
    req = AsyncCallRequest.model_validate(_body())
    wire = canonical_json(req.model_dump()).encode("utf-8")
    headers = sign_principal(user_id=req.user_id, unionid=req.unionid, ts=int(time.time()), nonce="synthetic-mode-nonce", secret=FIXTURE["test_key"], method="POST", path="/call_async", request_schema_version=3, request_body=wire)
    clear_nonce_cache()
    principal = resolve_session_principal(_request(headers), body_user_id=req.user_id, body_unionid=req.unionid, force_mode="http", request_schema_version=3, request_body=wire)
    assert principal.user_id == req.user_id
    for version, changed_wire in ((3, wire + b" "), (2, wire), (1, wire)):
        clear_nonce_cache()
        with pytest.raises(HTTPException) as error:
            resolve_session_principal(_request(headers), body_user_id=req.user_id, body_unionid=req.unionid, force_mode="http", request_schema_version=version, request_body=changed_wire)
        assert error.value.status_code == 401
    body = _body()
    body["message"] += "changed"
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_v3_request(body)


def test_encrypted_roundtrip_requires_full_binding(tmp_path: Path) -> None:
    req = AsyncCallRequest.model_validate(_body())
    binding = {name: getattr(req, name) for name in ("user_id", "execution_mode", "conversation_id", "episode_id")}
    tid = thread_id(user_id=req.user_id, group_id="global", conversation_id=req.conversation_id, episode_id=req.episode_id, execution_mode=req.execution_mode)
    assert tid == "ga::101::global::cafe_host::ga_mode_cafe_101::ep_synthetic_cafe"
    envelope = build_trusted_envelope(req=req, session=_session(req), thread_id=tid)
    args = {"run_id": req.run_id, "idempotency_key": req.idempotency_key, "request_fingerprint": req.request_fingerprint, "schema_version": 3, "mode_binding": binding}
    encrypted = encrypt_envelope(envelope, key=bytes(range(32)), key_version="synthetic", **args)
    assert decrypt_envelope(encrypted, keys={"synthetic": bytes(range(32))}, **args) == envelope
    for field in binding:
        changed = {**binding, field: "group_agent" if field == "execution_mode" else "other"}
        with pytest.raises(PayloadCryptoError):
            decrypt_envelope(encrypted, keys={"synthetic": bytes(range(32))}, **{**args, "mode_binding": changed})
    with pytest.raises(PayloadCryptoError):
        build_aad(run_id=req.run_id, idempotency_key=req.idempotency_key, request_fingerprint=req.request_fingerprint, schema_version=3)
    assert execution_binding(req)["run_id"] == req.run_id
    mismatched = copy.deepcopy(envelope)
    mismatched["request"]["episode_id"] = "wrong"
    poisoned = encrypt_envelope(mismatched, key=bytes(range(32)), key_version="synthetic", **args)
    with pytest.raises(PayloadCryptoError):
        decrypt_envelope(poisoned, keys={"synthetic": bytes(range(32))}, **args)


def test_legacy_aad_encoding_and_new_fields_cannot_downgrade() -> None:
    values = {"run_id": "旧任务", "idempotency_key": "键", "request_fingerprint": "a" * 64}
    for schema in (1, 2):
        assert build_aad(**values, schema_version=schema) == json.dumps({**values, "schema_version": schema}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    for version in (None, 1, 2):
        body = _body()
        body["request_schema_version"] = version
        with pytest.raises(HTTPException):
            AsyncCallRequest.model_validate(body)


@pytest.mark.parametrize("schema", [None, 1, 2])
def test_old_ack_dump_bytes_do_not_add_binding(schema: int | None) -> None:
    original = {"success": True, "run_id": "synthetic_run", "session_id": "synthetic_thread", "accepted": True, "message": "accepted", "idempotency_key": "synthetic_key", "execution_status": "queued", "queue_schema_version": schema}
    response = AsyncCallResponse.model_validate(original)
    assert response.model_dump() == original
    assert response.model_dump_json().encode("utf-8") == json.dumps(original, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    req = AsyncCallRequest.model_validate(_body())
    bound = AsyncCallResponse.model_validate({**original, "execution_binding": execution_binding(req)})
    assert bound.model_dump()["execution_binding"] == execution_binding(req)


def test_group_store_write_is_seen_by_next_cafe_snapshot_only(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    req = AsyncCallRequest.model_validate(_body())
    principal = _session(req).principal
    before = HostContext.from_request(req, principal=principal, base_dir=tmp_path)
    changed = profile.model_copy(update={"doing": profile.doing.model_copy(update={"value": "合成新项目"})})
    # The unmodified ordinary-member store boundary still permits the same
    # global profile. Accepted café evidence is copied, not hot-mutated.
    assert current_host_context() is None
    save_profile(tmp_path, changed)
    after = HostContext.from_request(req, principal=principal, base_dir=tmp_path)
    assert before.profile["doing"] == profile.doing.value
    assert after.profile["doing"] == "合成新项目"


async def test_real_cafe_graph_remains_tool_free_and_profile_shared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROUP_AGENT_INTEGRATION_MODE", "stub")
    monkeypatch.setenv("GROUP_AGENT_DURABLE_QUEUE_ENABLED", "0")
    profile = _profile(tmp_path)
    req = AsyncCallRequest.model_validate(_body())
    model = ChatModel(captured=[], bindings=[])
    state = AppState(agent=create_agent(model=model, tools=[], checkpointer=InMemorySaver()), base_dir=tmp_path, greeting_model=model)
    state.admin_agent = AsyncMock(side_effect=AssertionError("must not route admin"))
    emit = AsyncMock(return_value=True)
    await execute_async_run_core(req=req, session=_session(req), state=state, tid="mode-cafe", emit_callback=emit)
    assert "tools" not in state.cafe_agent.get_graph().nodes
    assert not any(model.bindings)
    final = next(call.args[1] for call in emit.await_args_list if call.args[0] == "final")
    assert final["profile_persisted"] is False
    assert final["profile_status"] == "host_read_only"
    host = HostContext.from_request(req, principal=_session(req).principal, base_dir=tmp_path)
    assert host.is_cafe_host and host.memory_scope_id == req.memory_scope_id
    assert host.profile["doing"] == profile.doing.value
    with active_host_context(host):
        with pytest.raises(HostActionRejected):
            save_profile(tmp_path, profile)
    assert load_profile(tmp_path, "101", "global") == profile
    assert current_host_context() is None


def test_duplicate_keys_and_json_nonfinite_rejected() -> None:
    req = AsyncCallRequest.model_validate(_body())
    wire = canonical_json(req.model_dump()).encode("utf-8")
    validate_v3_wire(wire, req)
    for changed in (wire[:-1] + b',"execution_mode":"cafe_host"}', wire.replace(b'"user_turn_version":0', b'"user_turn_version":NaN')):
        with pytest.raises(ValueError):
            validate_v3_wire(changed, req)


@pytest.mark.parametrize(("field", "maximum"), [("user_id", 64), ("unionid", 128), ("message", 8192), ("user_token", 1024), ("callback_url", 1024)])
def test_frozen_scalar_bounds(field: str, maximum: int) -> None:
    def make(length: int) -> dict[str, Any]:
        body = _body()
        if field == "user_id":
            body[field] = "1" * length
            body["conversation_id"] = "ga_mode_cafe_" + body[field]
            body["venue_context"]["visitor_user_id"] = body[field]
        elif field == "callback_url":
            body[field] = "https://" + "x" * (length - 8)
        else:
            body[field] = "x" * length
        return _refingerprint(body)

    validate_v3_request(make(maximum))
    with pytest.raises(ValueError):
        validate_v3_request(make(maximum + 1))


def test_frozen_snapshot_bounds() -> None:
    def make(*, count: int = 20, text_length: int = 2000, previous_length: int = 1000) -> dict[str, Any]:
        body = _body()
        body["context_snapshot"]["recent_messages"] = [{"message_id": index + 1, "role": "user", "text": "x" * text_length, "created_at": "2026-09-13T00:00:00Z", "result_kind": "conversation"} for index in range(count)]
        body["context_snapshot"]["previous_greeting"] = "x" * previous_length
        return _refingerprint(body)

    validate_v3_request(make())
    for invalid in (make(count=21), make(text_length=2001), make(previous_length=1001)):
        with pytest.raises(ValueError):
            validate_v3_request(invalid)
    oversized_utf8 = make()
    for message in oversized_utf8["context_snapshot"]["recent_messages"]:
        message["text"] = "☕" * 2000
    oversized_utf8["message"] = "☕" * 8192
    with pytest.raises(ValueError, match="budget exceeded"):
        validate_v3_request(_refingerprint(oversized_utf8))
    wrong_rollout = _body()
    wrong_rollout["rollout_context"]["rollout_version"] = "arbitrary-policy"
    with pytest.raises(ValueError):
        validate_v3_request(_refingerprint(wrong_rollout))
    req = AsyncCallRequest.model_validate(_body())
    with pytest.raises(ValueError, match="budget exceeded"):
        validate_v3_wire(b" " * 131073, req)


async def test_v3_endpoint_holds_until_durable_recovery_ready(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import apps.group_agent_api.app.endpoints.call_async as endpoint
    from starlette.requests import Request

    req = AsyncCallRequest.model_validate(_body())
    wire = canonical_json(req.model_dump()).encode("utf-8")

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": wire}

    request = Request({"type": "http", "method": "POST", "path": "/call_async", "headers": []}, receive=receive)
    monkeypatch.setattr(endpoint, "validate_and_normalize_callback_url", lambda value: value)
    monkeypatch.setattr(endpoint, "resolve_trusted_session", AsyncMock(return_value=_session(req)))
    with pytest.raises(HTTPException) as error:
        await endpoint.call_async(req, AppState(agent=None, base_dir=tmp_path), request)
    assert error.value.status_code == 503
    assert error.value.detail["error"] == "execution_mode_unavailable"


async def test_group_original_executor_is_not_cafe_or_admin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import apps.group_agent_api.app.async_manager as manager

    monkeypatch.setenv("GROUP_AGENT_DURABLE_QUEUE_ENABLED", "0")
    raw = copy.deepcopy(next(vector["request"] for vector in VECTORS if vector["name"] == "group_message"))
    raw["run_match"] = False
    req = AsyncCallRequest.model_validate(_refingerprint(raw))
    _profile(tmp_path)
    model = ChatModel(captured=[], bindings=[])
    member = create_agent(model=model, tools=[], checkpointer=InMemorySaver())
    state = AppState(agent=member, base_dir=tmp_path, greeting_model=model, admin_agent=AsyncMock(side_effect=AssertionError("admin must not run")))
    # Inspect the actual graph input/config without executing match/action IO.
    original = manager.aget_agent_state
    selected: list[Any] = []

    async def inspect(agent: Any, config: dict[str, Any]) -> Any:
        selected.append(agent)
        assert current_host_context() is None
        return await original(agent, config)

    monkeypatch.setattr(manager, "aget_agent_state", inspect)
    await execute_async_run_core(req=req, session=_session(req), state=state, tid="mode-group", emit_callback=AsyncMock(return_value=True))
    assert selected and all(agent is member for agent in selected)
    assert state.cafe_agent is None
    assert not any("日间 Barista" in str(messages[0].content) for messages in model.captured)
    assert current_host_context() is None
