"""Cafe Host v4 protocol, durable binding, and access-gate tests."""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from urllib.parse import urlparse

import pytest
from fastapi import HTTPException
from langchain_core.messages import AIMessage

from apps.group_agent_api.agent_factory.host_context import HostContext
from apps.group_agent_api.agent_factory.integrations.principal import (
    clear_nonce_cache,
    resolve_session_principal,
    sign_principal,
)
from apps.group_agent_api.app.async_manager import execute_async_run_core
from apps.group_agent_api.app.chat_access import (
    ChatAccessContext,
    ChatAccessRejected,
    active_chat_access,
    current_chat_access,
)
from apps.group_agent_api.app.execution_modes import (
    calculate_v4_fingerprint,
    v4_queue_binding,
    validate_v3_request,
    validate_v4_request,
    validate_v4_wire,
)
from apps.group_agent_api.app.models import AsyncCallRequest
from apps.group_agent_api.app.session import resolve_trusted_session
from apps.group_agent_api.app.utils import thread_id
from apps.group_agent_api.app.venue_protocol import callback_bindings, canonical_json
from apps.group_agent_api.execution.admission import (
    _v4_admission_enabled,
    _validate_v4_principal,
    admit_durable_async_call,
    build_trusted_envelope,
)
from apps.group_agent_api.execution.backpressure import BackpressureController
from apps.group_agent_api.execution.crypto import (
    PayloadCryptoError,
    build_aad,
    decrypt_envelope,
    encrypt_envelope,
)
from apps.group_agent_worker import tasks
from skillpacks.aihehuo.integration import _enforce_chat_access
from tests.test_group_agent_host_da002 import ChatModel, _profile, _state
from tests.test_group_agent_venue_da001 import _request, _session
from tests.unit_tests.group_agent_api.planner_fixtures import (
    PlannerModel,
    StructuredPlanner,
    people_query,
)

pytest_plugins = ["tests.test_group_agent_req032_integration"]

MODE_FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures/venue_entry/execution-modes.json").read_text()
)
BASE = next(
    vector["request"]
    for vector in MODE_FIXTURE["vectors"]
    if vector["name"] == "cafe_day"
)
KEY = "synthetic-cafe-access-contract-key-not-live"
TIMESTAMP = "1790000000"
NONCE = "synthetic-cafe-access-nonce"
PAID = [
    "community_search",
    "opportunity_search",
    "people_search",
    "project_partnership_search",
    "startup_digest",
]
LEGAL_EXPECTED = {
    "ordinary": {
        "fingerprint": "b024efe4eae48a2d7e37dd932ddd82dd870d77abb9e5c862d5cae0252d9a373b",
        "body_sha256": "3f3996fe5af6f62e1f921770fcb95500997734f386c626bf59e63ba4350f37c9",
        "principal_hmac": "82694b7c4e47ab7f6351f483dc5b2677a074e80b0c1d3a6d8e37b7f1fa6f5a9f",
        "body_hmac": "7d962f3c19fd7869734ec8457f889025da7421b77a952d9e3f39cf1638b05115",
        "aad_sha256": "874afdd453a49f2252b4244f45195f8b1e09106852a3f6aed10c31fcc83a81d9",
    },
    "active_people": {
        "fingerprint": "c25baa0fbef5cfb86c1030673c4d608fa86fb8f1c2a8ba839e2b12e602ac59ba",
        "body_sha256": "c435478bdecb7c86bc2c0f396a9a67b9eaa250e75659b10bf442200d48c1cbd8",
        "principal_hmac": "9c265ca7fabab4bb9f9b2c37a192596ee381ea7f4d170501b4b8fdc5b17fd95d",
        "body_hmac": "8d3c7e4bac290b856e81a8469a870233bdf7647bc1c27ddf338ec5c5bb3dc4b9",
        "aad_sha256": "3bea1b2775d2999077edb00eaf36fb70868fbfb088cd8f4bb97b66a9e37f83d9",
    },
    "active_global": {
        "fingerprint": "0e8006d7f6527c8b10e36b0055c2463b448a44194b1eedf3bdb8d607793652bb",
        "body_sha256": "23636237fcb6335a321b6443ab032361fec145405b9ae48176434ff6e528430e",
        "principal_hmac": "3a0ec7efdd81724b6f30c90a1e17766d80699f8028044bf9ccfd866603cfc730",
        "body_hmac": "a7eea5828d818d55122cd895a3d73de1c1363eca1b7ffbbbbdc5ade90948d1cc",
        "aad_sha256": "4554215a2003be0fe2c3a156f101b29d9863f882fb7da1da90d7e987d7a74fb2",
    },
}


def test_worker_preflight_accepts_v4_queue_schema() -> None:
    """Guard the Cafe Host v4 delivery before it reaches durable record lookup."""
    assert tasks._supported_queue_schema_version(4) is True
    assert tasks._supported_queue_schema_version(5) is True
    assert tasks._supported_queue_schema_version(6) is False
    assert tasks._supported_queue_schema_version("4") is False


def _v4_body(*, active: bool = False, confirmed: bool = False) -> dict[str, Any]:
    """Compose one strict request from the frozen local v3 Cafe Host vector."""
    body = copy.deepcopy(BASE)
    for name in ("group_token", "user_token", "request_fingerprint"):
        body.pop(name)
    body.update(
        request_schema_version=4,
        queue_schema_version=4,
        access_contract_version=1,
        run_id="00000000-0000-4000-8000-000000000202"
        if active
        else "00000000-0000-4000-8000-000000000201",
        idempotency_key="synthetic-access-cafe-active"
        if active
        else "synthetic-access-cafe-ordinary",
        message="可以，帮我找上海的餐饮合伙人" if active else "帮我找上海的餐饮合伙人",
        credential_binding=(
            {"user_token_sha256": "1" * 64, "group_token_sha256": "2" * 64}
            if active
            else {"user_token_sha256": "-", "group_token_sha256": "-"}
        ),
        chat_access={
            "version": 2,
            "mode": "active" if active else "ordinary",
            "policy_version": "chat-access-v2",
            "revision": 9 if active else 7,
            "drink_session_id": 501 if active else None,
            "pending_search_ref": "psr_synthetic_0001",
            "search_confirmation_id": "scf_synthetic_0001" if confirmed else None,
            "confirmed_search_intent": "people_search" if confirmed else None,
            "allowed_search_intents": PAID if active else [],
            "restricted_search_intents": [] if active else PAID,
        },
    )
    body["request_fingerprint"] = calculate_v4_fingerprint(body)
    return body


def _active_with_token(token: str) -> tuple[AsyncCallRequest, Any]:
    """Bind an active request to one synthetic trusted header token."""
    body = _v4_body(active=True, confirmed=True)
    body["credential_binding"] = {
        "user_token_sha256": hashlib.sha256(token.encode()).hexdigest(),
        "group_token_sha256": "-",
    }
    body["request_fingerprint"] = calculate_v4_fingerprint(body)
    req = AsyncCallRequest.model_validate(body)
    session = _session(req)
    principal = replace(session.principal, user_token=token, group_token=None)
    return req, replace(session, principal=principal, group_token=None)


def _phone_only_body(*, active: bool = False) -> dict[str, Any]:
    """Represent one Micro-signed App account with no WeChat unionid."""
    body = _v4_body(active=active)
    body["unionid"] = None
    body["request_fingerprint"] = calculate_v4_fingerprint(body)
    return body


@pytest.mark.parametrize("active", [False, True])
def test_android_v4_phone_only_source_is_admitted(active: bool) -> None:
    """A signed Android App principal may use the same Cafe Host v4 contract as iOS."""
    body = _phone_only_body(active=active)
    body["metadata"] = {"source": "android_app", "client_surface": "android_app"}
    body["request_fingerprint"] = calculate_v4_fingerprint(body)
    assert validate_v4_request(body)["metadata"]["source"] == "android_app"
    AsyncCallRequest.model_validate(body)


def test_android_v4_rejects_mismatched_surface() -> None:
    """The new Android source cannot impersonate another transport in the envelope."""
    body = _phone_only_body()
    body["metadata"] = {"source": "android_app", "client_surface": "ios_app"}
    body["request_fingerprint"] = calculate_v4_fingerprint(body)
    with pytest.raises(ValueError, match="source/client_surface mismatch"):
        validate_v4_request(body)


def _legal_body(name: str) -> dict[str, Any]:
    """Compose one of the three frozen cross-service legal vectors."""
    if name == "ordinary":
        return _v4_body()
    if name == "active_people":
        return _v4_body(active=True, confirmed=True)
    body = _v4_body(active=True, confirmed=True)
    body.update(
        run_id="00000000-0000-4000-8000-000000000203",
        idempotency_key="synthetic-access-cafe-active-global",
        message="可以，帮我查最近的人工智能教育机会",
        credential_binding={
            "user_token_sha256": "3" * 64,
            "group_token_sha256": "-",
        },
    )
    body["chat_access"].update(
        revision=12,
        drink_session_id=502,
        pending_search_ref=None,
        search_confirmation_id="scf_synthetic_0002",
        confirmed_search_intent="opportunity_search",
    )
    body["request_fingerprint"] = calculate_v4_fingerprint(body)
    return body


@pytest.mark.parametrize("name", tuple(LEGAL_EXPECTED))
def test_all_legal_cross_service_vectors(name: str) -> None:
    """Match fingerprint, both HMAC domains, body digest, and queue AAD."""
    body = _legal_body(name)
    expected = LEGAL_EXPECTED[name]
    req = AsyncCallRequest.model_validate(body)
    wire = canonical_json(body).encode()
    body_sha = hashlib.sha256(wire).hexdigest()
    credential = body["credential_binding"]
    principal_input = "\n".join(
        [
            "CAFE-HOST-PRINCIPAL-V4",
            f"user_id={body['user_id']}",
            f"unionid={body['unionid']}",
            f"token_sha256={credential['user_token_sha256']}",
            f"group_token_sha256={credential['group_token_sha256']}",
            f"ts={TIMESTAMP}",
            f"nonce={NONCE}",
            "method=POST",
            "path=/call_async",
        ]
    )
    request_input = "\n".join(
        [
            "CAFE-HOST-REQUEST-V4",
            "method=POST",
            "path=/call_async",
            f"body_sha256={body_sha}",
            f"ts={TIMESTAMP}",
            f"nonce={NONCE}",
        ]
    )

    def digest(value: str) -> str:
        """Sign one frozen fixture input with its synthetic test key."""
        return hmac.new(KEY.encode(), value.encode(), hashlib.sha256).hexdigest()

    aad = build_aad(
        run_id=req.run_id,
        idempotency_key=req.idempotency_key,
        request_fingerprint=req.request_fingerprint,
        schema_version=4,
        mode_binding=v4_queue_binding(req),
    )
    assert body["request_fingerprint"] == expected["fingerprint"]
    assert body_sha == expected["body_sha256"]
    assert digest(principal_input) == expected["principal_hmac"]
    assert digest(request_input) == expected["body_hmac"]
    assert hashlib.sha256(aad).hexdigest() == expected["aad_sha256"]


def test_ordinary_vector_matches_frozen_fingerprint_hmac_and_aad() -> None:
    body = _v4_body()
    req = AsyncCallRequest.model_validate(body)
    assert req.model_dump() == body
    assert (
        body["request_fingerprint"]
        == "b024efe4eae48a2d7e37dd932ddd82dd870d77abb9e5c862d5cae0252d9a373b"
    )
    wire = canonical_json(body).encode()
    assert (
        hashlib.sha256(wire).hexdigest()
        == "3f3996fe5af6f62e1f921770fcb95500997734f386c626bf59e63ba4350f37c9"
    )
    headers = sign_principal(
        user_id=req.user_id,
        unionid=req.unionid,
        ts=TIMESTAMP,
        nonce=NONCE,
        method="POST",
        path="/call_async",
        secret=KEY,
        request_schema_version=4,
        request_body=wire,
    )
    assert (
        headers["X-GA-Signature"]
        == "82694b7c4e47ab7f6351f483dc5b2677a074e80b0c1d3a6d8e37b7f1fa6f5a9f"
    )
    assert (
        headers["X-GA-Request-Signature"]
        == "7d962f3c19fd7869734ec8457f889025da7421b77a952d9e3f39cf1638b05115"
    )
    aad = build_aad(
        run_id=req.run_id,
        idempotency_key=req.idempotency_key,
        request_fingerprint=req.request_fingerprint,
        schema_version=4,
        mode_binding=v4_queue_binding(req),
    )
    assert (
        hashlib.sha256(aad).hexdigest()
        == "874afdd453a49f2252b4244f45195f8b1e09106852a3f6aed10c31fcc83a81d9"
    )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("execution_mode",), "group_agent"),
        (("credential_binding", "user_token_sha256"), "1" * 64),
        (("chat_access", "allowed_search_intents"), ["people_search"]),
        (("chat_access", "drink_session_id"), 1),
        (("chat_access", "search_confirmation_id"), "scf_synthetic_0001"),
        (("chat_access", "confirmed_search_intent"), "people_search"),
    ],
)
def test_v4_invalid_access_mutations_fail_closed(
    path: tuple[str, ...], value: Any
) -> None:
    body = _v4_body()
    target = body
    for name in path[:-1]:
        target = target[name]
    target[path[-1]] = value
    body["request_fingerprint"] = calculate_v4_fingerprint(body)
    with pytest.raises((HTTPException, ValueError)):
        AsyncCallRequest.model_validate(body)


def test_v4_accepts_legacy_paid_partition_for_rolling_deploy_replay() -> None:
    body = _v4_body()
    body["chat_access"]["restricted_search_intents"] = [
        "opportunity_search",
        "people_search",
        "project_partnership_search",
    ]
    body["request_fingerprint"] = calculate_v4_fingerprint(body)
    req = AsyncCallRequest.model_validate(body)
    assert req.chat_access["restricted_search_intents"] == body["chat_access"][
        "restricted_search_intents"
    ]


def test_v4_wire_rejects_duplicate_keys_and_model_drift() -> None:
    body = _v4_body()
    req = AsyncCallRequest.model_validate(body)
    validate_v4_wire(canonical_json(body).encode(), req)
    duplicate = canonical_json(body).replace(
        '"request_schema_version":4',
        '"request_schema_version":4,"request_schema_version":4',
        1,
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        validate_v4_wire(duplicate.encode(), req)
    with pytest.raises(ValueError, match="not canonical JSON"):
        validate_v4_wire((canonical_json(body) + "\n").encode(), req)


def test_v4_hmac_verification_binds_raw_body_and_active_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the actual HTTP principal parser, not only signature generation."""
    token = "synthetic-v4-user-token"
    req, _ = _active_with_token(token)
    wire = canonical_json(req.model_dump()).encode()
    headers = sign_principal(
        user_id=req.user_id,
        unionid=req.unionid,
        user_token=token,
        ts=int(time.time()),
        nonce="synthetic-v4-http-verification",
        method="POST",
        path="/call_async",
        secret=KEY,
        request_schema_version=4,
        request_body=wire,
    )
    monkeypatch.setenv("GROUP_AGENT_PRINCIPAL_HMAC_SECRET", KEY)
    clear_nonce_cache()
    principal = resolve_session_principal(
        _request(headers),
        body_user_id=req.user_id,
        body_unionid=req.unionid,
        body_user_token=None,
        force_mode="http",
        request_schema_version=4,
        request_body=wire,
    )
    assert principal.user_token == token
    clear_nonce_cache()
    with pytest.raises(HTTPException) as error:
        resolve_session_principal(
            _request(headers),
            body_user_id=req.user_id,
            body_unionid=req.unionid,
            body_user_token=None,
            force_mode="http",
            request_schema_version=4,
            request_body=wire + b" ",
        )
    assert error.value.detail["error"] == "request_signature_invalid"


@pytest.mark.parametrize("active", (False, True))
def test_phone_only_v4_signed_principal_and_frozen_request(
    active: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A signed App user_id authenticates v4; active still binds its token."""
    body = _phone_only_body(active=active)
    token = "synthetic-phone-user-token" if active else None
    if token:
        body["credential_binding"]["user_token_sha256"] = hashlib.sha256(
            token.encode()
        ).hexdigest()
        body["credential_binding"]["group_token_sha256"] = "-"
        body["request_fingerprint"] = calculate_v4_fingerprint(body)
    req = AsyncCallRequest.model_validate(body)
    wire = canonical_json(body).encode()
    validate_v4_wire(wire, req)
    headers = sign_principal(
        user_id=req.user_id,
        unionid=None,
        user_token=token,
        ts=int(time.time()),
        nonce=f"synthetic-phone-v4-{active}",
        method="POST",
        path="/call_async",
        secret=KEY,
        request_schema_version=4,
        request_body=wire,
    )
    assert "X-GA-Unionid" not in headers
    monkeypatch.setenv("GROUP_AGENT_PRINCIPAL_HMAC_SECRET", KEY)
    clear_nonce_cache()
    principal = resolve_session_principal(
        _request(headers),
        body_user_id=req.user_id,
        body_unionid=None,
        force_mode="http",
        request_schema_version=4,
        request_body=wire,
    )
    assert principal.user_id == req.user_id
    assert principal.unionid is None
    assert principal.user_token == token
    session = replace(
        _session(req),
        principal=principal,
    )
    _validate_v4_principal(req, session)
    canonical_thread = thread_id(
        user_id=req.user_id,
        group_id=req.group_id,
        conversation_id=req.conversation_id,
        episode_id=req.episode_id,
        execution_mode=req.execution_mode,
    )
    envelope = build_trusted_envelope(
        req=req, session=session, thread_id=canonical_thread
    )
    assert envelope["session"]["unionid"] is None
    restored_req, restored_session, restored_thread = tasks._rebuild_session(envelope)
    assert restored_req == req
    assert restored_session.principal.unionid is None
    assert restored_thread == canonical_thread
    assert HostContext.from_request(
        restored_req, principal=restored_session.principal, base_dir=tmp_path
    ).visitor_user_id == req.user_id


def test_phone_only_principal_rejects_legacy_schema_and_identity_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The v4 exception must not weaken v3 or accept another signed user."""
    with pytest.raises(ValueError, match="unionid required"):
        sign_principal(user_id="123", unionid=None, secret=KEY, request_schema_version=3)
    legacy = copy.deepcopy(BASE)
    legacy["unionid"] = None
    with pytest.raises(ValueError):
        validate_v3_request(legacy)

    body = _phone_only_body()
    wire = canonical_json(body).encode()
    headers = sign_principal(
        user_id="999",
        unionid=None,
        ts=int(time.time()),
        nonce="synthetic-phone-wrong-user",
        method="POST",
        path="/call_async",
        secret=KEY,
        request_schema_version=4,
        request_body=wire,
    )
    monkeypatch.setenv("GROUP_AGENT_PRINCIPAL_HMAC_SECRET", KEY)
    clear_nonce_cache()
    with pytest.raises(HTTPException) as error:
        resolve_session_principal(
            _request(headers),
            body_user_id=body["user_id"],
            body_unionid=None,
            force_mode="http",
            request_schema_version=4,
            request_body=wire,
        )
    assert error.value.detail == {
        "error": "identity_injection_forbidden",
        "field": "user_id",
    }


@pytest.mark.parametrize("active", (False, True))
async def test_phone_only_v4_http_session_resolves_without_group_membership(
    active: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The HTTP path uses signed App identity even without a WeChat bind."""
    body = _phone_only_body(active=active)
    token = "synthetic-phone-active-token" if active else None
    if token:
        body["credential_binding"]["user_token_sha256"] = hashlib.sha256(
            token.encode()
        ).hexdigest()
        body["credential_binding"]["group_token_sha256"] = "-"
        body["request_fingerprint"] = calculate_v4_fingerprint(body)
    req = AsyncCallRequest.model_validate(body)
    wire = canonical_json(body).encode()
    headers = sign_principal(
        user_id=req.user_id,
        unionid=None,
        user_token=token,
        ts=int(time.time()),
        nonce=f"synthetic-phone-session-{active}",
        method="POST",
        path="/call_async",
        secret=KEY,
        request_schema_version=4,
        request_body=wire,
    )
    monkeypatch.setenv("GROUP_AGENT_INTEGRATION", "http")
    monkeypatch.setenv("GROUP_AGENT_PRINCIPAL_HMAC_SECRET", KEY)
    clear_nonce_cache()
    session = await resolve_trusted_session(
        _request(headers),
        body_user_id=req.user_id,
        body_group_id=req.group_id,
        body_membership=req.membership,
        body_unionid=req.unionid,
        body_group_token=req.group_token,
        body_user_token=req.user_token,
        request_schema_version=4,
        request_body=wire,
    )
    assert session.principal.user_id == req.user_id
    assert session.principal.unionid is None
    assert session.group_id == "global"
    assert session.capabilities.can_write_profile is False
    _validate_v4_principal(req, session)


def test_active_header_digest_binding_and_default_off_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    req, session = _active_with_token("synthetic-user-bearer")
    _validate_v4_principal(req, session)
    with pytest.raises(HTTPException) as error:
        _validate_v4_principal(
            req,
            replace(session, principal=replace(session.principal, user_token="other")),
        )
    assert error.value.detail["error"] == "binding_mismatch"
    monkeypatch.delenv("CAFE_HOST_V4_ADMISSION_ENABLED", raising=False)
    monkeypatch.delenv("CAFE_HOST_V4_USER_ALLOWLIST", raising=False)
    assert not _v4_admission_enabled(req.user_id)
    monkeypatch.setenv("CAFE_HOST_V4_ADMISSION_ENABLED", "1")
    monkeypatch.setenv("CAFE_HOST_V4_USER_ALLOWLIST", req.user_id)
    assert _v4_admission_enabled(req.user_id)


def test_v4_encrypted_envelope_roundtrip_binds_access_and_hides_token() -> None:
    req, session = _active_with_token("synthetic-user-bearer")
    binding = v4_queue_binding(req)
    envelope = build_trusted_envelope(
        req=req, session=session, thread_id="synthetic-thread"
    )
    encrypted = encrypt_envelope(
        envelope,
        key=bytes(range(32)),
        key_version="synthetic",
        run_id=req.run_id,
        idempotency_key=req.idempotency_key,
        request_fingerprint=req.request_fingerprint,
        schema_version=4,
        mode_binding=binding,
    )
    assert "synthetic-user-bearer" not in json.dumps(encrypted.to_dict())
    restored = decrypt_envelope(
        encrypted,
        keys={"synthetic": bytes(range(32))},
        run_id=req.run_id,
        idempotency_key=req.idempotency_key,
        request_fingerprint=req.request_fingerprint,
        schema_version=4,
        mode_binding=binding,
    )
    assert restored == envelope
    changed = {**binding, "chat_access_revision": binding["chat_access_revision"] + 1}
    with pytest.raises(PayloadCryptoError):
        decrypt_envelope(
            encrypted,
            keys={"synthetic": bytes(range(32))},
            run_id=req.run_id,
            idempotency_key=req.idempotency_key,
            request_fingerprint=req.request_fingerprint,
            schema_version=4,
            mode_binding=changed,
        )


def test_access_context_blocks_each_paid_intent_resource_and_false_final() -> None:
    req = AsyncCallRequest.model_validate(_v4_body())
    access = ChatAccessContext.from_request(req)
    assert access is not None
    assert current_chat_access() is None
    with active_chat_access(access):
        assert current_chat_access() is access
        for intent, resource in (
            ("people_search", "users"),
            ("opportunity_search", "ideas"),
            ("project_partnership_search", "projects"),
            ("community_search", "groups"),
            ("startup_digest", "weekly"),
        ):
            assert access.control_for_intent(intent) == "chat_session_required"
            with pytest.raises(ChatAccessRejected, match="chat_session_required"):
                access.assert_resource_allowed(intent=intent, resource=resource)
            with pytest.raises(ChatAccessRejected, match="chat_session_required"):
                _enforce_chat_access(resource)
        with pytest.raises(ChatAccessRejected, match="search_receipt_required"):
            access.guard_final(reply="本次公开查询已完成。")
        with pytest.raises(ChatAccessRejected, match="search_receipt_required"):
            access.guard_final(reply="我们继续聊。", has_paid_results=True)
    assert current_chat_access() is None


def test_active_allowlist_is_frozen_but_da001_tools_remain_unavailable(
    tmp_path: Path,
) -> None:
    req = AsyncCallRequest.model_validate(_v4_body(active=True, confirmed=True))
    access = ChatAccessContext.from_request(req)
    assert access is not None
    assert access.control_for_intent("people_search") == "search_capability_unavailable"
    assert (
        access.control_for_intent("opportunity_search")
        == "search_capability_unavailable"
    )
    host = HostContext.from_request(
        req, principal=_session(req).principal, base_dir=tmp_path
    )
    assert host.is_cafe_host
    assert host.execution_binding["run_id"] == req.run_id
    assert callback_bindings(req)["execution_binding"] == host.execution_binding


async def test_ordinary_paid_intent_returns_typed_control_without_search_output(
    tmp_path: Path,
) -> None:
    """Exercise the real Cafe Host graph and deterministic final guard."""
    req = AsyncCallRequest.model_validate(_v4_body())
    model = ChatModel(
        captured=[],
        bindings=[],
        output=AIMessage(content="我已经搜索完成并找到了三位候选人。"),
    )
    state = _state(tmp_path, model)
    state.quality_model = PlannerModel(
        StructuredPlanner([people_query("上海 餐饮 合伙人")])
    )
    _profile(tmp_path)
    emitted: list[tuple[str, dict[str, Any]]] = []

    async def emit(event: str, payload: dict[str, Any]) -> bool:
        emitted.append((event, payload))
        return True

    await execute_async_run_core(
        req=req,
        session=_session(req),
        state=state,
        tid=thread_id(
            user_id=req.user_id,
            group_id=req.group_id,
            conversation_id=req.conversation_id,
            episode_id=req.episode_id,
            execution_mode=req.execution_mode,
        ),
        emit_callback=emit,
    )
    final = next(payload for event, payload in emitted if event == "final")
    assert final["chat_access_control"] == {
        "code": "chat_session_required",
        "intent": "people_search",
    }
    assert final["candidates"] == []
    assert final["search_log"] is None
    assert "搜索完成" not in final["reply"]
    assert not any(model.bindings)


@pytest.mark.skipif(
    urlparse(os.environ.get("GROUP_AGENT_TEST_REDIS_URL", "")).hostname
    not in {"127.0.0.1", "localhost"},
    reason="requires explicitly selected isolated localhost Redis",
)
def test_v4_durable_admission_replay_restore_and_conflict(
    env_cfg: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove one v4 snapshot survives ledger, replay, and worker restore."""
    config, store = env_cfg
    req = AsyncCallRequest.model_validate(_v4_body())
    session = _session(req)
    tid = thread_id(
        user_id=req.user_id,
        group_id=req.group_id,
        conversation_id=req.conversation_id,
        episode_id=req.episode_id,
        execution_mode=req.execution_mode,
    )
    monkeypatch.setenv("CAFE_HOST_V4_ADMISSION_ENABLED", "1")
    monkeypatch.setenv("CAFE_HOST_V4_USER_ALLOWLIST", req.user_id)
    enqueue = MagicMock()
    backpressure = BackpressureController(store._r, config)

    admitted = admit_durable_async_call(
        req=req,
        session=session,
        thread_id=tid,
        store=store,
        config=config,
        backpressure=backpressure,
        enqueue=enqueue,
    )
    assert admitted.created is True
    assert admitted.response.queue_schema_version == 4
    assert admitted.response.execution_binding["run_id"] == req.run_id
    record = store.get(req.run_id)
    assert record.mode_binding == v4_queue_binding(req)
    assert "synthetic-user-bearer" not in json.dumps(record.to_redis_mapping())

    envelope = decrypt_envelope(
        record.payload_ciphertext,
        keys=config.payload_keys,
        run_id=req.run_id,
        idempotency_key=req.idempotency_key,
        request_fingerprint=req.request_fingerprint,
        schema_version=4,
        mode_binding=record.mode_binding,
    )
    restored_req, _, restored_tid = tasks._rebuild_session(envelope)
    assert restored_req.model_dump() == req.model_dump()
    assert restored_tid == tid

    replay = admit_durable_async_call(
        req=req,
        session=session,
        thread_id=tid,
        store=store,
        config=config,
        backpressure=backpressure,
        enqueue=enqueue,
    )
    assert replay.created is False
    assert enqueue.call_count == 1

    changed = _v4_body()
    changed["chat_access"]["revision"] += 1
    changed["request_fingerprint"] = calculate_v4_fingerprint(changed)
    with pytest.raises(HTTPException) as error:
        admit_durable_async_call(
            req=AsyncCallRequest.model_validate(changed),
            session=session,
            thread_id=tid,
            store=store,
            config=config,
            backpressure=backpressure,
            enqueue=enqueue,
        )
    assert error.value.status_code == 409
