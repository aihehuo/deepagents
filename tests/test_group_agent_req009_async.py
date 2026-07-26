"""Full REQ-009 / RESP-009-FIX5 cumulative unit test matrix for group_agent_api async endpoint & callback integration."""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from apps.group_agent_api.agent_factory.capability import CapabilityTier
from apps.group_agent_api.agent_factory.integrations.callback_client import (
    HEADER_CALLBACK_NONCE,
    HEADER_CALLBACK_SIGNATURE,
    HEADER_CALLBACK_TS,
    HEADER_CALLBACK_VERSION,
    send_callback_event,
    sign_callback_payload,
    validate_and_normalize_callback_url,
    validate_callback_url,
)
from apps.group_agent_api.agent_factory.integrations.config import (
    assert_startup_security,
    callback_allowed_base_urls,
    callback_hmac_secret,
    principal_hmac_secret,
)
from apps.group_agent_api.agent_factory.integrations.membership_client import MembershipResult
from apps.group_agent_api.agent_factory.integrations.principal import (
    HEADER_GROUP_TOKEN,
    HEADER_NONCE,
    HEADER_SIGNATURE,
    HEADER_TS,
    HEADER_UNIONID,
    HEADER_USER_ID,
    HEADER_USER_TOKEN,
    SessionPrincipal,
    clear_nonce_cache,
    sign_principal,
)
from apps.group_agent_api.app import app
from apps.group_agent_api.app.async_manager import (
    IdempotencySlot,
    _idempotency_store,
    _run_id_store,
    clear_async_idempotency_cache,
    complete_idempotency_reservation,
    execute_async_run,
    reserve_idempotency_slot,
    rollback_idempotency_reservation,
)
from apps.group_agent_api.app.models import AsyncCallRequest, AsyncCallResponse
from apps.group_agent_api.app.session import TrustedSession
from apps.group_agent_api.app.state import AppState
from apps.group_agent_api.app.utils import thread_id

PRINCIPAL_SECRET = "principal_secret_32bytes_key_01"
CALLBACK_SECRET = "callback_secret_32bytes_key_02"


@pytest.fixture(autouse=True)
def _setup_env(tmp_path: Path):
    old_env = dict(os.environ)
    os.environ["GROUP_AGENT_ENV"] = "development"
    os.environ["GROUP_AGENT_INTEGRATION"] = "stub"
    os.environ["GROUP_AGENT_MODEL_MODE"] = "stub"
    os.environ["GROUP_AGENT_PRINCIPAL_HMAC_SECRET"] = PRINCIPAL_SECRET
    os.environ["GROUP_AGENT_CALLBACK_HMAC_SECRET"] = CALLBACK_SECRET
    os.environ["GROUP_AGENT_CALLBACK_ALLOWED_BASE_URLS"] = "http://micro-web.example.invalid:3000/group_agent_callbacks"
    os.environ["GROUP_AGENT_RUNTIME_DIR"] = str(tmp_path / "runtime")
    clear_nonce_cache()
    clear_async_idempotency_cache()
    yield
    os.environ.clear()
    os.environ.update(old_env)
    clear_nonce_cache()
    clear_async_idempotency_cache()


# ---------------------------------------------------------------------------
# 1. Fast ACK & Principal Header Rejections Tests
# ---------------------------------------------------------------------------


def test_call_async_endpoint_returns_202_ack_fast_before_llm_finishes():
    """Verify POST /call_async returns 202 ACK fast before background LLM execution completes."""
    body = {
        "run_id": "run_fast_1",
        "idempotency_key": "idem_fast_1",
        "user_id": "u1",
        "unionid": "u_union_1",
        "group_id": "g1",
        "conversation_id": "c1",
        "message": "Fast ACK test",
        "callback_url": "http://micro-web.example.invalid:3000/group_agent_callbacks/run_fast_1",
        "membership": "in_group",
    }

    sig_headers = sign_principal(
        user_id="u1",
        unionid="u_union_1",
        method="POST",
        path="/call_async",
        secret=PRINCIPAL_SECRET,
    )

    llm_completed = False

    async def slow_exec(*args, **kwargs):
        nonlocal llm_completed
        await asyncio.sleep(0.2)
        llm_completed = True

    with TestClient(app) as client:
        with patch("apps.group_agent_api.app.endpoints.call_async.execute_async_run", side_effect=slow_exec):
            start_t = time.time()
            resp = client.post("/call_async", json=body, headers=sig_headers)
            elapsed = time.time() - start_t

            assert resp.status_code == 202
            assert elapsed < 0.1, f"Fast ACK took too long: {elapsed:.3f}s"
            assert llm_completed is False, "ACK returned after LLM completed!"


def test_call_async_principal_header_rejections():
    """Verify POST /call_async rejects missing headers, invalid signature, expired timestamp, or replayed nonce."""
    body = {
        "run_id": "run_sec_1",
        "idempotency_key": "idem_sec_1",
        "user_id": "u1",
        "unionid": "u_union_1",
        "group_id": "g1",
        "conversation_id": "c1",
        "message": "Security check",
        "callback_url": "http://micro-web.example.invalid:3000/group_agent_callbacks/run_sec_1",
    }

    os.environ["GROUP_AGENT_INTEGRATION"] = "http"
    os.environ.pop("GROUP_AGENT_MODEL_MODE", None)
    os.environ["DEEPSEEK_API_KEY"] = "sk-fake-test-key-32bytes-long!"

    with TestClient(app) as client:
        # 1. Missing principal headers
        resp_no_hdr = client.post("/call_async", json=body)
        assert resp_no_hdr.status_code == 401

        # 2. Bad signature
        sig_headers = sign_principal(user_id="u1", unionid="u_union_1", method="POST", path="/call_async", secret=PRINCIPAL_SECRET)
        sig_headers[HEADER_SIGNATURE] = "bad_signature_hash"
        resp_bad_sig = client.post("/call_async", json=body, headers=sig_headers)
        assert resp_bad_sig.status_code == 401

        # 3. Expired timestamp (>300s skew)
        old_ts_headers = sign_principal(user_id="u1", unionid="u_union_1", method="POST", path="/call_async", secret=PRINCIPAL_SECRET, ts=int(time.time()) - 400)
        resp_old_ts = client.post("/call_async", json=body, headers=old_ts_headers)
        assert resp_old_ts.status_code == 401

        # 4. Replayed nonce
        valid_headers = sign_principal(user_id="u1", unionid="u_union_1", method="POST", path="/call_async", secret=PRINCIPAL_SECRET, nonce="nonce_replay_999")
        resp_valid1 = client.post("/call_async", json=body, headers=valid_headers)
        assert resp_valid1.status_code == 202

        resp_replay = client.post("/call_async", json=body, headers=valid_headers)
        assert resp_replay.status_code == 401

        # 5. Body/header identity mismatch
        mismatch_headers = sign_principal(user_id="other_user", unionid="u_union_1", method="POST", path="/call_async", secret=PRINCIPAL_SECRET)
        resp_mismatch = client.post("/call_async", json=body, headers=mismatch_headers)
        assert resp_mismatch.status_code in {400, 401}


# ---------------------------------------------------------------------------
# 2. Callback HMAC Tampering Rejections Test
# ---------------------------------------------------------------------------


def test_callback_hmac_tampering_rejections():
    """Verify sign_callback_payload HMAC signature rejects tampered method, path, body, or timestamp."""
    body = json.dumps({"test": "data"}).encode("utf-8")
    headers = sign_callback_payload(
        secret=CALLBACK_SECRET,
        method="POST",
        path="/group_agent_callbacks/run1",
        body_bytes=body,
        ts="1770000000",
        nonce="nonce12345",
    )

    sig = headers[HEADER_CALLBACK_SIGNATURE]

    # Tamper body
    tampered_body = json.dumps({"test": "tampered_data"}).encode("utf-8")
    h_tampered_body = sign_callback_payload(
        secret=CALLBACK_SECRET, method="POST", path="/group_agent_callbacks/run1", body_bytes=tampered_body, ts="1770000000", nonce="nonce12345"
    )
    assert h_tampered_body[HEADER_CALLBACK_SIGNATURE] != sig

    # Tamper path
    h_tampered_path = sign_callback_payload(
        secret=CALLBACK_SECRET, method="POST", path="/group_agent_callbacks/run2", body_bytes=body, ts="1770000000", nonce="nonce12345"
    )
    assert h_tampered_path[HEADER_CALLBACK_SIGNATURE] != sig

    # Tamper method
    h_tampered_method = sign_callback_payload(
        secret=CALLBACK_SECRET, method="GET", path="/group_agent_callbacks/run1", body_bytes=body, ts="1770000000", nonce="nonce12345"
    )
    assert h_tampered_method[HEADER_CALLBACK_SIGNATURE] != sig

    # Tamper timestamp
    h_tampered_ts = sign_callback_payload(
        secret=CALLBACK_SECRET, method="POST", path="/group_agent_callbacks/run1", body_bytes=body, ts="1770000099", nonce="nonce12345"
    )
    assert h_tampered_ts[HEADER_CALLBACK_SIGNATURE] != sig


# ---------------------------------------------------------------------------
# 3. SSRF Allowlist Matrix & 3xx Redirect Tests
# ---------------------------------------------------------------------------


def test_callback_ssrf_allowlist_validation_matrix():
    """Verify SSRF allowlist strictly checks scheme, host, port, path prefix, rejecting prefix bypasses & traversals."""
    # Valid exact path & subpath
    validate_callback_url("http://micro-web.example.invalid:3000/group_agent_callbacks")
    validate_callback_url("http://micro-web.example.invalid:3000/group_agent_callbacks/run123")

    # 1. Adjacent path prefix bypass attempt (.evil) -> MUST REJECT
    with pytest.raises(HTTPException) as exc1:
        validate_callback_url("http://micro-web.example.invalid:3000/group_agent_callbacks.evil/run")
    assert exc1.value.status_code == 400
    assert exc1.value.detail["error"] == "callback_url_not_allowed"

    # 2. Host prefix bypass attempt (.evil.com) -> MUST REJECT
    with pytest.raises(HTTPException) as exc2:
        validate_callback_url("http://micro-web.example.invalid.evil.com:3000/group_agent_callbacks/run")
    assert exc2.value.status_code == 400
    assert exc2.value.detail["error"] == "callback_url_not_allowed"

    # 3. Wrong port (:3001 vs :3000) -> MUST REJECT
    with pytest.raises(HTTPException) as exc3:
        validate_callback_url("http://micro-web.example.invalid:3001/group_agent_callbacks/run")
    assert exc3.value.status_code == 400
    assert exc3.value.detail["error"] == "callback_url_not_allowed"

    # 4. Path traversal (.. or %2e%2e) -> MUST REJECT
    with pytest.raises(HTTPException) as exc4:
        validate_callback_url("http://micro-web.example.invalid:3000/group_agent_callbacks/../evil")
    assert exc4.value.status_code == 400
    assert exc4.value.detail["error"] == "callback_url_path_traversal"

    # 5. Userinfo in URL -> MUST REJECT
    with pytest.raises(HTTPException) as exc5:
        validate_callback_url("http://user:pass@micro-web.example.invalid:3000/group_agent_callbacks")
    assert exc5.value.status_code == 400
    assert exc5.value.detail["error"] == "callback_url_userinfo_forbidden"

    # 6. Fragment in URL -> MUST REJECT
    with pytest.raises(HTTPException) as exc6:
        validate_callback_url("http://micro-web.example.invalid:3000/group_agent_callbacks#fragment")
    assert exc6.value.status_code == 400
    assert exc6.value.detail["error"] == "callback_url_fragment_forbidden"

    # 7. Query in URL -> MUST REJECT
    with pytest.raises(HTTPException) as exc7:
        validate_callback_url("http://micro-web.example.invalid:3000/group_agent_callbacks?token=secret")
    assert exc7.value.status_code == 400
    assert exc7.value.detail["error"] == "callback_url_query_forbidden"

    # 8. Backslash in path -> MUST REJECT
    with pytest.raises(HTTPException) as exc8:
        validate_callback_url("http://micro-web.example.invalid:3000/group_agent_callbacks\\path")
    assert exc8.value.status_code == 400
    assert exc8.value.detail["error"] == "callback_url_invalid_path"


@pytest.mark.asyncio
async def test_callback_3xx_redirect_handling():
    """Verify send_callback_event treats 3xx redirect response as non-retryable failure (returns False without follow)."""
    class Mock302Response:
        status_code = 302

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=Mock302Response()):
        res = await send_callback_event(
            callback_url="http://micro-web.example.invalid:3000/group_agent_callbacks/run1",
            envelope_dict={"test": "data"},
            secret=CALLBACK_SECRET,
            max_retries=2,
        )
        assert res is False


# ---------------------------------------------------------------------------
# 4. Startup Security Fail-Closed Matrix Tests
# ---------------------------------------------------------------------------


def test_startup_security_fail_closed_checks():
    """Verify assert_startup_security enforces directional secret isolation, explicit allowlist, and bounds."""
    # 1. Missing callback secret in http mode -> Fail Closed
    os.environ["GROUP_AGENT_INTEGRATION"] = "http"
    os.environ.pop("GROUP_AGENT_MODEL_MODE", None)
    os.environ["GROUP_AGENT_CALLBACK_HMAC_SECRET"] = ""
    with pytest.raises(RuntimeError, match="GROUP_AGENT_CALLBACK_HMAC_SECRET is required"):
        assert_startup_security()

    # 2. Identical callback and principal secret -> Fail Closed
    os.environ["GROUP_AGENT_CALLBACK_HMAC_SECRET"] = PRINCIPAL_SECRET
    with pytest.raises(RuntimeError, match="directional isolation required"):
        assert_startup_security()

    # 3. Missing explicit allowlist in production mode -> Fail Closed
    os.environ["GROUP_AGENT_CALLBACK_HMAC_SECRET"] = CALLBACK_SECRET
    os.environ["GROUP_AGENT_ENV"] = "production"
    os.environ["GROUP_AGENT_CALLBACK_ALLOWED_BASE_URLS"] = ""
    with pytest.raises(RuntimeError, match="must be explicitly configured"):
        assert_startup_security()

    # 4. Out of bounds numeric config -> Fail Closed
    os.environ["GROUP_AGENT_CALLBACK_ALLOWED_BASE_URLS"] = "http://micro-web.example.invalid:3000/group_agent_callbacks"
    os.environ["GROUP_AGENT_ASYNC_MAX_ACTIVE"] = "5000"
    with pytest.raises(RuntimeError, match="Invalid async/callback numeric configuration"):
        assert_startup_security()


# ---------------------------------------------------------------------------
# 5. TrustedSession Fingerprint Token & Canonical URL Mismatch Tests
# ---------------------------------------------------------------------------


def test_trusted_session_fingerprint_token_mismatch():
    """Verify header token changes (with empty body tokens) or canonical callback URL variations generate distinct fingerprints."""
    os.environ["GROUP_AGENT_INTEGRATION"] = "http"
    os.environ.pop("GROUP_AGENT_MODEL_MODE", None)

    body = {
        "run_id": "run_fp_tok_1",
        "idempotency_key": "idem_fp_tok_1",
        "user_id": "u1",
        "unionid": "u_union_1",
        "group_id": "g1",
        "conversation_id": "c1",
        "message": "Token fingerprint test",
        "callback_url": "http://MICRO-WEB.EXAMPLE.INVALID:3000/group_agent_callbacks/run_fp_tok_1",
        # Empty body tokens
        "group_token": None,
        "user_token": None,
    }

    headers_tokA = sign_principal(
        user_id="u1",
        unionid="u_union_1",
        group_token="group_token_AAA",
        method="POST",
        path="/call_async",
        secret=PRINCIPAL_SECRET,
    )

    headers_tokB = sign_principal(
        user_id="u1",
        unionid="u_union_1",
        group_token="group_token_BBB",
        method="POST",
        path="/call_async",
        secret=PRINCIPAL_SECRET,
    )

    os.environ["GROUP_AGENT_INTEGRATION"] = "http"
    os.environ["DEEPSEEK_API_KEY"] = "sk-fake-test-key-32bytes-long!"

    with TestClient(app) as client:
        with patch("apps.group_agent_api.app.endpoints.call_async.execute_async_run", new_callable=AsyncMock):
            # Request 1 with group_token_AAA
            res1 = client.post("/call_async", json=body, headers=headers_tokA)
            assert res1.status_code == 202

            # Request 2 with group_token_BBB on same idempotency_key -> MUST REJECT WITH 409
            res2 = client.post("/call_async", json=body, headers=headers_tokB)
            assert res2.status_code == 409
            assert res2.json()["detail"]["error"] == "idempotency_conflict"

            # Request 3 with identical trusted session & equivalent uppercase/trailing-slash URL -> MUST MATCH HIT
            body_url_var = dict(body)
            body_url_var["callback_url"] = "http://micro-web.example.invalid:3000/group_agent_callbacks/run_fp_tok_1/"
            headers_tokA_replay = sign_principal(
                user_id="u1",
                unionid="u_union_1",
                group_token="group_token_AAA",
                method="POST",
                path="/call_async",
                secret=PRINCIPAL_SECRET,
                nonce="nonce_unique_tokA_2",
            )
            res3 = client.post("/call_async", json=body_url_var, headers=headers_tokA_replay)
            assert res3.status_code == 202


# ---------------------------------------------------------------------------
# 6. Deterministic Async Waiter & Rollback Decision Gate Regression Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deterministic_waiter_commit_delivers_same_ack():
    """Verify waiter B awaiting pending reservation A receives exact same committed ACK response when A completes."""
    req = AsyncCallRequest(
        run_id="run_waiter_commit_1",
        idempotency_key="idem_waiter_commit_1",
        user_id="u1",
        unionid="u_union_1",
        group_id="g1",
        conversation_id="c1",
        message="Waiter commit test",
        callback_url="http://micro-web.example.invalid:3000/group_agent_callbacks/run_waiter_commit_1",
    )
    session = TrustedSession(
        principal=SessionPrincipal(user_id="u1", unionid="u_union_1", user_token=None, source="signed_oauth_principal"),
        group_id="g1",
        group_token=None,
        membership=MembershipResult(tier=CapabilityTier.in_group, source="stub"),
    )

    # 1. Request A reserves slot
    status_a, resp_a, slot_a = await reserve_idempotency_slot(req, session)
    assert status_a == "RESERVED"
    assert slot_a is not None

    # 2. Concurrent Request B enters waiter loop in background task
    async def _waiter_b():
        return await reserve_idempotency_slot(req, session)

    task_b = asyncio.create_task(_waiter_b())
    await asyncio.sleep(0.02)

    # 3. Request A completes commit with exact slot reference
    from apps.group_agent_api.app.models import AsyncCallResponse
    ack_response = AsyncCallResponse(success=True, run_id=req.run_id, session_id="ga::u1::g1::c1", accepted=True, message="accepted")
    committed = await complete_idempotency_reservation(slot_a, ack_response)
    assert committed is True

    # 4. Request B wakes up and gets HIT with identical response
    status_b, resp_b, slot_b = await task_b
    assert status_b == "HIT"
    assert resp_b == ack_response


@pytest.mark.asyncio
async def test_deterministic_waiter_rollback_returns_425_without_ghost_202():
    """Verify waiter B awaiting pending reservation A returns HTTP 425 request_initializing when A rolls back (0 ghost 202s)."""
    req = AsyncCallRequest(
        run_id="run_waiter_rb_1",
        idempotency_key="idem_waiter_rb_1",
        user_id="u1",
        unionid="u_union_1",
        group_id="g1",
        conversation_id="c1",
        message="Waiter rollback test",
        callback_url="http://micro-web.example.invalid:3000/group_agent_callbacks/run_waiter_rb_1",
    )
    session = TrustedSession(
        principal=SessionPrincipal(user_id="u1", unionid="u_union_1", user_token=None, source="signed_oauth_principal"),
        group_id="g1",
        group_token=None,
        membership=MembershipResult(tier=CapabilityTier.in_group, source="stub"),
    )

    # 1. Request A reserves slot
    status_a, _, slot_a = await reserve_idempotency_slot(req, session)
    assert status_a == "RESERVED"
    assert slot_a is not None

    # 2. Concurrent Request B enters waiter loop
    task_b = asyncio.create_task(reserve_idempotency_slot(req, session))
    await asyncio.sleep(0.02)

    # 3. Request A rolls back reservation with exact slot reference
    rolled_back = await rollback_idempotency_reservation(slot_a)
    assert rolled_back is True

    # 4. Request B wakes up, sees slot rolled back, returns INITIALIZING (HTTP 425, 0 ghost 202s)
    status_b, resp_b, _ = await task_b
    assert status_b == "INITIALIZING"
    assert resp_b is None


@pytest.mark.asyncio
async def test_deterministic_waiter_conflict_returns_409_immediately():
    """Verify request C with different fingerprint on pending idempotency_key returns CONFLICT (409) immediately."""
    req_a = AsyncCallRequest(
        run_id="run_waiter_conf_1",
        idempotency_key="idem_waiter_conf_1",
        user_id="u1",
        unionid="u_union_1",
        group_id="g1",
        conversation_id="c1",
        message="Original request",
        callback_url="http://micro-web.example.invalid:3000/group_agent_callbacks/run_waiter_conf_1",
        run_match=True,
    )
    req_c = AsyncCallRequest(
        run_id="run_waiter_conf_1",
        idempotency_key="idem_waiter_conf_1",
        user_id="u1",
        unionid="u_union_1",
        group_id="g1",
        conversation_id="c1",
        message="Conflict request",
        callback_url="http://micro-web.example.invalid:3000/group_agent_callbacks/run_waiter_conf_1",
        run_match=False,  # Mismatched run_match
    )
    session = TrustedSession(
        principal=SessionPrincipal(user_id="u1", unionid="u_union_1", user_token=None, source="signed_oauth_principal"),
        group_id="g1",
        group_token=None,
        membership=MembershipResult(tier=CapabilityTier.in_group, source="stub"),
    )

    # 1. Request A reserves slot
    status_a, _, _ = await reserve_idempotency_slot(req_a, session)
    assert status_a == "RESERVED"

    # 2. Request C with different fingerprint returns CONFLICT immediately without waiting
    start_t = time.time()
    status_c, resp_c, _ = await reserve_idempotency_slot(req_c, session)
    elapsed = time.time() - start_t

    assert status_c == "CONFLICT"
    assert elapsed < 0.05, f"Conflict check took too long: {elapsed:.3f}s"


@pytest.mark.asyncio
async def test_deterministic_waiter_timeout_preserves_owner_slot():
    """Verify waiter timeout returns INITIALIZING (425) without modifying or deleting owner slot."""
    req = AsyncCallRequest(
        run_id="run_waiter_to_1",
        idempotency_key="idem_waiter_to_1",
        user_id="u1",
        unionid="u_union_1",
        group_id="g1",
        conversation_id="c1",
        message="Waiter timeout test",
        callback_url="http://micro-web.example.invalid:3000/group_agent_callbacks/run_waiter_to_1",
    )
    session = TrustedSession(
        principal=SessionPrincipal(user_id="u1", unionid="u_union_1", user_token=None, source="signed_oauth_principal"),
        group_id="g1",
        group_token=None,
        membership=MembershipResult(tier=CapabilityTier.in_group, source="stub"),
    )

    # 1. Request A reserves slot and remains pending
    status_a, _, slot_a = await reserve_idempotency_slot(req, session)
    assert status_a == "RESERVED"
    assert slot_a is not None

    # 2. Mock decision_event.wait on slot_a to raise TimeoutError
    async def _mock_wait_timeout(*args, **kwargs):
        raise asyncio.TimeoutError()

    with patch.object(slot_a.decision_event, "wait", side_effect=_mock_wait_timeout):
        status_b, resp_b, _ = await reserve_idempotency_slot(req, session)
        assert status_b == "INITIALIZING"

    # 3. Assert owner slot remains in _idempotency_store untouched
    assert req.idempotency_key in _idempotency_store
    assert _idempotency_store[req.idempotency_key] is slot_a
    assert slot_a.status == "reserved"


@pytest.mark.asyncio
async def test_rollback_gate_decision_prevents_core_execution():
    """PERMANENT REGRESSION TEST: Verify rollback decision gate prevents task from entering core execution (CORE_AFTER_ROLLBACK == 0)."""
    req = AsyncCallRequest(
        run_id="run_rb_gate_1",
        idempotency_key="idem_rb_gate_1",
        user_id="u1",
        unionid="u_union_1",
        group_id="g1",
        conversation_id="c1",
        message="Rollback decision gate test",
        callback_url="http://micro-web.example.invalid:3000/group_agent_callbacks/run_rb_gate_1",
    )
    session = TrustedSession(
        principal=SessionPrincipal(user_id="u1", unionid="u_union_1", user_token=None, source="signed_oauth_principal"),
        group_id="g1",
        group_token=None,
        membership=MembershipResult(tier=CapabilityTier.in_group, source="stub"),
    )

    from apps.group_agent_api.app.startup import startup
    state_ref = {}
    await startup(state_ref)
    app_state: AppState = state_ref["state"]

    # 1. Reserve slot and launch task waiting on decision_event
    status_a, _, slot_a = await reserve_idempotency_slot(req, session)
    assert status_a == "RESERVED"
    assert slot_a is not None

    core_exec_mock = AsyncMock()
    callback_mock = AsyncMock(return_value=True)

    with patch("apps.group_agent_api.app.async_manager._execute_core_agent", core_exec_mock), \
         patch("apps.group_agent_api.app.async_manager.send_callback_event", callback_mock):

        task = asyncio.create_task(
            execute_async_run(
                req=req,
                session=session,
                state=app_state,
                tid="ga::u1::g1::c1",
                slot=slot_a,
            )
        )
        await asyncio.sleep(0.02)  # Task starts and waits on slot.decision_event.wait()

        # 2. Rollback reservation (simulating spawn failure / timeout / lock rejection)
        await rollback_idempotency_reservation(slot_a)

        # 3. Wait for task to finish after rollback
        await task

        # 4. ASSERT CORE_AFTER_ROLLBACK == 0 and 0 progress callbacks emitted!
        assert core_exec_mock.call_count == 0, f"CORE_AFTER_ROLLBACK must be 0, got {core_exec_mock.call_count}"
        assert callback_mock.call_count == 0, f"Callbacks after rollback must be 0, got {callback_mock.call_count}"


# ---------------------------------------------------------------------------
# 7. Task Spawn, Commit Failure, Request Cancellation & Shutdown Cleanup Tests
# ---------------------------------------------------------------------------


def test_task_spawn_and_register_failure_compensation():
    """Verify task registration failure cancels & awaits task before core LLM runs and clears active tasks/runs & stores."""
    body = {
        "run_id": "run_fail_spawn_1",
        "idempotency_key": "idem_fail_spawn_1",
        "user_id": "u1",
        "unionid": "u_union_1",
        "group_id": "g1",
        "conversation_id": "c1",
        "message": "Task spawn failure test",
        "callback_url": "http://micro-web.example.invalid:3000/group_agent_callbacks/run_fail_spawn_1",
    }

    sig_headers = sign_principal(user_id="u1", unionid="u_union_1", method="POST", path="/call_async", secret=PRINCIPAL_SECRET)
    core_exec_mock = AsyncMock()

    with TestClient(app) as client:
        from apps.group_agent_api.app import _state
        assert _state is not None

        with patch.object(_state, "register_task", side_effect=RuntimeError("Task registration failure")), \
             patch("apps.group_agent_api.app.async_manager._execute_core_agent", core_exec_mock):
            res = client.post("/call_async", json=body, headers=sig_headers)
            assert res.status_code == 500
            assert res.json()["detail"]["error"] in {"idempotency_commit_failed", "task_spawn_failed"}

            # Assert core LLM execution was CANCELLED before running (_execute_core_agent call_count == 0)
            assert core_exec_mock.call_count == 0

            # Assert active run locks, task registry, and idempotency stores are empty
            tid = thread_id(user_id="u1", group_id="g1", conversation_id="c1")
            assert tid not in _state.active_agent_runs
            assert tid not in _state.active_tasks
            assert "idem_fail_spawn_1" not in _idempotency_store
            assert "run_fail_spawn_1" not in _run_id_store


def test_task_commit_failure_compensation():
    """Verify complete_idempotency_reservation failure cancels & awaits task and clears active tasks/runs & stores."""
    body = {
        "run_id": "run_fail_commit_1",
        "idempotency_key": "idem_fail_commit_1",
        "user_id": "u1",
        "unionid": "u_union_1",
        "group_id": "g1",
        "conversation_id": "c1",
        "message": "Commit failure test",
        "callback_url": "http://micro-web.example.invalid:3000/group_agent_callbacks/run_fail_commit_1",
    }

    sig_headers = sign_principal(user_id="u1", unionid="u_union_1", method="POST", path="/call_async", secret=PRINCIPAL_SECRET)
    core_exec_mock = AsyncMock()

    with TestClient(app) as client:
        from apps.group_agent_api.app import _state
        assert _state is not None

        with patch("apps.group_agent_api.app.endpoints.call_async.complete_idempotency_reservation", return_value=False), \
             patch("apps.group_agent_api.app.async_manager._execute_core_agent", core_exec_mock):
            res = client.post("/call_async", json=body, headers=sig_headers)
            assert res.status_code == 500
            assert res.json()["detail"]["error"] == "idempotency_commit_failed"

            # Assert core LLM execution was CANCELLED before running
            assert core_exec_mock.call_count == 0

            # Assert active tasks, runs, and idempotency stores are empty
            tid = thread_id(user_id="u1", group_id="g1", conversation_id="c1")
            assert tid not in _state.active_agent_runs
            assert tid not in _state.active_tasks
            assert "idem_fail_commit_1" not in _idempotency_store
            assert "run_fail_commit_1" not in _run_id_store


@pytest.mark.asyncio
async def test_request_cancellation_compensation():
    """Verify endpoint coroutine cancellation cleans active tasks/runs and idempotency stores while re-raising CancelledError."""
    req = AsyncCallRequest(
        run_id="run_cancel_comp_1",
        idempotency_key="idem_cancel_comp_1",
        user_id="u1",
        unionid="u_union_1",
        group_id="g1",
        conversation_id="c1",
        message="Request cancellation test",
        callback_url="http://micro-web.example.invalid:3000/group_agent_callbacks/run_cancel_comp_1",
    )

    from apps.group_agent_api.app import _state
    assert _state is not None
    from apps.group_agent_api.app.endpoints.call_async import call_async

    class MockRequest:
        headers = {}

    with patch("apps.group_agent_api.app.endpoints.call_async.complete_idempotency_reservation", side_effect=asyncio.CancelledError):
        with pytest.raises(asyncio.CancelledError):
            await call_async(req, _state, MockRequest())

        tid = thread_id(user_id="u1", group_id="g1", conversation_id="c1")
        assert tid not in _state.active_agent_runs
        assert tid not in _state.active_tasks
        assert "idem_cancel_comp_1" not in _idempotency_store
        assert "run_cancel_comp_1" not in _run_id_store


@pytest.mark.asyncio
async def test_app_state_shutdown_cancellation_and_task_cleanup():
    """Verify AppState.shutdown cancels all in-flight background tasks and clears active registries cleanly."""
    from apps.group_agent_api.app.startup import startup
    state_ref = {}
    await startup(state_ref)
    app_state: AppState = state_ref["state"]

    task_run = False
    async def _long_task():
        nonlocal task_run
        task_run = True
        await asyncio.sleep(10.0)

    tid = "ga::u1::g1::c_shutdown"
    app_state.try_start_agent_run(tid, "call_async")
    task = asyncio.create_task(_long_task())
    app_state.register_task(tid, task)

    assert tid in app_state.active_tasks

    await app_state.shutdown()

    assert task.cancelled() or task.done()
    assert len(app_state.active_tasks) == 0
    assert len(app_state.active_agent_runs) == 0


# ---------------------------------------------------------------------------
# 8. Callback Retry Identity, Status Codes, Sequence & Timeout Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_callback_retry_identity_and_status_codes():
    """Verify callback retry behavior for 500, 429, non-429 4xx, and network errors, asserting identical payload & headers."""
    posted_headers: list[dict[str, str]] = []
    posted_bodies: list[bytes] = []

    class MockResp:
        def __init__(self, code: int):
            self.status_code = code

    # 1. 500 Server Error: retries max_retries (3 times = 4 attempts per event)
    # Patch asyncio.sleep to avoid waiting 15s in test execution
    with patch("asyncio.sleep", new_callable=AsyncMock):

        # Test 500
        count_500 = 0
        async def mock_post_500(url, content=None, headers=None, **kwargs):
            nonlocal count_500
            count_500 += 1
            posted_headers.append(dict(headers))
            posted_bodies.append(content)
            return MockResp(500)

        with patch("httpx.AsyncClient.post", side_effect=mock_post_500):
            res500 = await send_callback_event(
                callback_url="http://micro-web.example.invalid:3000/group_agent_callbacks/run1",
                envelope_dict={"event": "progress", "seq": 1},
                secret=CALLBACK_SECRET,
                max_retries=3,
            )
            assert res500 is False
            assert count_500 == 4  # 1 try + 3 retries

            # Assert retry headers and payload body are strictly preserved across all 4 attempts
            sig0 = posted_headers[0][HEADER_CALLBACK_SIGNATURE]
            nonce0 = posted_headers[0][HEADER_CALLBACK_NONCE]
            ts0 = posted_headers[0][HEADER_CALLBACK_TS]
            body0 = posted_bodies[0]

            for i in range(1, 4):
                assert posted_headers[i][HEADER_CALLBACK_SIGNATURE] == sig0
                assert posted_headers[i][HEADER_CALLBACK_NONCE] == nonce0
                assert posted_headers[i][HEADER_CALLBACK_TS] == ts0
                assert posted_bodies[i] == body0

        # Test 429 Too Many Requests: retries max_retries times
        count_429 = 0
        async def mock_post_429(url, content=None, headers=None, **kwargs):
            nonlocal count_429
            count_429 += 1
            return MockResp(429)

        with patch("httpx.AsyncClient.post", side_effect=mock_post_429):
            res429 = await send_callback_event(
                callback_url="http://micro-web.example.invalid:3000/group_agent_callbacks/run1",
                envelope_dict={"event": "progress", "seq": 1},
                secret=CALLBACK_SECRET,
                max_retries=3,
            )
            assert res429 is False
            assert count_429 == 4

        # Test 400 Bad Request (non-429 4xx): fails fast on 1st attempt, NO retries
        count_400 = 0
        async def mock_post_400(url, content=None, headers=None, **kwargs):
            nonlocal count_400
            count_400 += 1
            return MockResp(400)

        with patch("httpx.AsyncClient.post", side_effect=mock_post_400):
            res400 = await send_callback_event(
                callback_url="http://micro-web.example.invalid:3000/group_agent_callbacks/run1",
                envelope_dict={"event": "progress", "seq": 1},
                secret=CALLBACK_SECRET,
                max_retries=3,
            )
            assert res400 is False
            assert count_400 == 1  # No retries for 400!

        # Test Network Exception (ConnectError): retries max_retries times
        count_net = 0
        import httpx
        async def mock_post_net(*args, **kwargs):
            nonlocal count_net
            count_net += 1
            raise httpx.ConnectError("Network unreachable")

        with patch("httpx.AsyncClient.post", side_effect=mock_post_net):
            res_net = await send_callback_event(
                callback_url="http://micro-web.example.invalid:3000/group_agent_callbacks/run1",
                envelope_dict={"event": "progress", "seq": 1},
                secret=CALLBACK_SECRET,
                max_retries=3,
            )
            assert res_net is False
            assert count_net == 4


@pytest.mark.asyncio
async def test_callback_sequence_monotonic_terminal_guarantees():
    """Verify callback envelope sequence numbers are strictly monotonic and terminal events forbid subsequent emissions."""
    req = AsyncCallRequest(
        run_id="run_seq_mono_1",
        idempotency_key="idem_seq_mono_1",
        user_id="u1",
        unionid="u_union_1",
        group_id="g1",
        conversation_id="c1",
        message="Sequence test",
        callback_url="http://micro-web.example.invalid:3000/group_agent_callbacks/run_seq_mono_1",
        membership="in_group",
    )

    from apps.group_agent_api.app.startup import startup
    state_ref = {}
    await startup(state_ref)
    app_state = state_ref["state"]

    class _FakeCheckpointer:
        def flush(self): pass
    class _FakeAgent:
        def __init__(self, base_dir: Path):
            self.base_dir = base_dir
            self.checkpointer = _FakeCheckpointer()
        async def aget_state(self, _config):
            class _S: values = {"messages": []}
            return _S()
        async def ainvoke(self, payload, config):
            meta = config.get("metadata") or {}
            from apps.group_agent_api.agent_factory.profile_schema import profile_from_flat
            from apps.group_agent_api.agent_factory.profile_store import save_profile
            prof = profile_from_flat(user_id=str(meta["user_id"]), group_id=str(meta["group_id"]), doing="A", need="B", offer="C")
            save_profile(self.base_dir, prof)
            return {"messages": [payload["messages"][0], AIMessage(content="ACK")]}

    app_state.agent = _FakeAgent(app_state.base_dir)

    session = TrustedSession(
        principal=SessionPrincipal(user_id="u1", unionid="u_union_1", user_token=None, source="signed_oauth_principal"),
        group_id="g1",
        group_token=None,
        membership=MembershipResult(tier=CapabilityTier.in_group, source="stub"),
    )

    sent_envelopes = []
    async def mock_send_callback(*, callback_url: str, envelope_dict: dict[str, Any], **kwargs):
        sent_envelopes.append(envelope_dict)
        return True

    with patch("apps.group_agent_api.app.async_manager.send_callback_event", side_effect=mock_send_callback):
        await execute_async_run(req=req, session=session, state=app_state, tid="ga::u1::g1::c1")

    assert len(sent_envelopes) == 2
    assert sent_envelopes[0]["seq"] == 1
    assert sent_envelopes[0]["event"] == "progress"
    assert sent_envelopes[1]["seq"] == 2
    assert sent_envelopes[1]["event"] == "final"


@pytest.mark.asyncio
async def test_async_run_timeout_and_error_callback():
    """Verify async run timeout emits safe AsyncRunTimeout error callback and releases active run lock."""
    req = AsyncCallRequest(
        run_id="run_to_callback_1",
        idempotency_key="idem_to_callback_1",
        user_id="u1",
        unionid="u_union_1",
        group_id="g1",
        conversation_id="c1",
        message="Timeout test",
        callback_url="http://micro-web.example.invalid:3000/group_agent_callbacks/run_to_callback_1",
        membership="in_group",
    )

    from apps.group_agent_api.app.startup import startup
    state_ref = {}
    await startup(state_ref)
    app_state = state_ref["state"]

    session = TrustedSession(
        principal=SessionPrincipal(user_id="u1", unionid="u_union_1", user_token=None, source="signed_oauth_principal"),
        group_id="g1",
        group_token=None,
        membership=MembershipResult(tier=CapabilityTier.in_group, source="stub"),
    )

    sent_envelopes = []
    async def mock_send_callback(*, callback_url: str, envelope_dict: dict[str, Any], **kwargs):
        sent_envelopes.append(envelope_dict)
        return True

    async def slow_core(*args, **kwargs):
        raise asyncio.TimeoutError()

    tid = "ga::u1::g1::c1"
    app_state.try_start_agent_run(tid, "call_async")

    with patch("apps.group_agent_api.app.async_manager.send_callback_event", side_effect=mock_send_callback), \
         patch("apps.group_agent_api.app.async_manager._execute_core_agent", side_effect=slow_core):
        await execute_async_run(req=req, session=session, state=app_state, tid=tid)

    assert len(sent_envelopes) == 2
    assert sent_envelopes[1]["event"] == "error"
    assert sent_envelopes[1]["payload"]["error_code"] == "AsyncRunTimeout"
    # Assert active run lock was released in finally block
    assert tid not in app_state.active_agent_runs


@pytest.mark.asyncio
async def test_log_sanitization_and_secret_masking(caplog):
    """Verify exceptions, observer metrics, and callback error payloads NEVER leak sensitive secrets or raw tokens."""
    req = AsyncCallRequest(
        run_id="run_sanitized_log_1",
        idempotency_key="idem_sanitized_log_1",
        user_id="u1",
        unionid="u_union_1",
        group_id="g1",
        conversation_id="c1",
        message="Secret test",
        callback_url="http://micro-web.example.invalid:3000/group_agent_callbacks/run_sanitized_log_1",
        membership="in_group",
    )

    from apps.group_agent_api.app.startup import startup
    state_ref = {}
    await startup(state_ref)
    app_state = state_ref["state"]

    session = TrustedSession(
        principal=SessionPrincipal(user_id="u1", unionid="u_union_1", user_token="secret_user_token_999", source="signed_oauth_principal"),
        group_id="g1",
        group_token="secret_group_token_888",
        membership=MembershipResult(tier=CapabilityTier.in_group, source="stub"),
    )

    sent_envelopes = []
    async def mock_send_callback(*, callback_url: str, envelope_dict: dict[str, Any], **kwargs):
        sent_envelopes.append(envelope_dict)
        return True

    secret_exc = RuntimeError("Failure with secret_key_9999_inside_exception")

    with caplog.at_level(logging.ERROR), \
         patch("apps.group_agent_api.app.async_manager.send_callback_event", side_effect=mock_send_callback), \
         patch("apps.group_agent_api.app.async_manager._execute_core_agent", side_effect=secret_exc):
        await execute_async_run(req=req, session=session, state=app_state, tid="ga::u1::g1::c1")

    # Assert error callback payload masks sensitive error details
    assert len(sent_envelopes) == 2
    err_payload = sent_envelopes[1]["payload"]
    assert err_payload["error_code"] == "AsyncRunFailed"
    assert "secret_key_9999" not in json.dumps(err_payload)

    # Assert caplog logs only desensitized error_type
    log_text = caplog.text
    assert "secret_key_9999" not in log_text
    assert "secret_user_token_999" not in log_text
    assert "secret_group_token_888" not in log_text


def test_async_call_request_field_boundaries():
    """Verify AsyncCallRequest Pydantic field boundary validations for length, patterns, and types."""
    base_valid = {
        "run_id": "run_bound_1",
        "idempotency_key": "idem_bound_1",
        "user_id": "u1",
        "unionid": "u_union_1",
        "group_id": "g1",
        "conversation_id": "c1",
        "message": "Boundary test",
        "callback_url": "http://micro-web.example.invalid:3000/group_agent_callbacks/run_bound_1",
    }

    # 1. Empty message -> ValidationError
    invalid_empty_msg = dict(base_valid)
    invalid_empty_msg["message"] = ""
    with pytest.raises(ValidationError):
        AsyncCallRequest(**invalid_empty_msg)

    # 2. Oversized message (>8192 chars) -> ValidationError
    invalid_large_msg = dict(base_valid)
    invalid_large_msg["message"] = "A" * 10_000
    with pytest.raises(ValidationError):
        AsyncCallRequest(**invalid_large_msg)

    # 3. Oversized callback_url (>1024 chars) -> ValidationError
    invalid_url_len = dict(base_valid)
    invalid_url_len["callback_url"] = "http://micro-web.example.invalid:3000/" + ("a" * 2000)
    with pytest.raises(ValidationError):
        AsyncCallRequest(**invalid_url_len)

    # 4. Invalid metadata non-scalar type -> ValidationError
    invalid_meta = dict(base_valid)
    invalid_meta["metadata"] = {"invalid_list": [1, 2, 3]}
    with pytest.raises(ValidationError):
        AsyncCallRequest(**invalid_meta)

    # 5. Invalid callback URL scheme -> HTTPException(400) via validate_and_normalize_callback_url
    with pytest.raises(HTTPException) as exc_info:
        validate_and_normalize_callback_url("ftp://invalid-scheme.com")
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["error"] == "callback_url_invalid_scheme"


# ---------------------------------------------------------------------------
# 9. RESP-009-FIX5 Capacity Eviction & Precise Slot Identity Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_eviction_only_evicts_completed_slots_and_preserves_reserved():
    """RESP-009-FIX5: Verify cache eviction ONLY evicts completed slots and NEVER evicts reserved/pending slots."""
    req_a = AsyncCallRequest(
        run_id="run_evict_pending_a",
        idempotency_key="idem_evict_pending_a",
        user_id="u1", unionid="u_union_1", group_id="g1", conversation_id="c1",
        message="Req A", callback_url="http://micro-web.example.invalid:3000/group_agent_callbacks/run_evict_pending_a",
    )
    req_b = AsyncCallRequest(
        run_id="run_evict_pending_b",
        idempotency_key="idem_evict_pending_b",
        user_id="u1", unionid="u_union_1", group_id="g1", conversation_id="c1",
        message="Req B", callback_url="http://micro-web.example.invalid:3000/group_agent_callbacks/run_evict_pending_b",
    )
    session = TrustedSession(
        principal=SessionPrincipal(user_id="u1", unionid="u_union_1", user_token=None, source="signed_oauth_principal"),
        group_id="g1", group_token=None, membership=MembershipResult(tier=CapabilityTier.in_group, source="stub"),
    )

    # Set cache capacity to 1
    with patch("apps.group_agent_api.app.async_manager._MAX_IDEMPOTENCY_CACHE", 1):
        status_a, _, slot_a = await reserve_idempotency_slot(req_a, session)
        assert status_a == "RESERVED"
        assert slot_a is not None
        assert slot_a.status == "reserved"

        # Attempt to reserve B when capacity is full of reserved slots
        status_b, resp_b, slot_b = await reserve_idempotency_slot(req_b, session)
        assert status_b == "INITIALIZING"  # Overload / rejected because A is reserved!
        assert slot_b is None

        # Assert slot A remains in store untouched
        assert req_a.idempotency_key in _idempotency_store
        assert _idempotency_store[req_a.idempotency_key] is slot_a


@pytest.mark.asyncio
async def test_eviction_evicts_completed_slot_for_new_reservation():
    """RESP-009-FIX5: Verify cache eviction successfully evicts completed slot A to make room for new slot B."""
    req_a = AsyncCallRequest(
        run_id="run_evict_comp_a",
        idempotency_key="idem_evict_comp_a",
        user_id="u1", unionid="u_union_1", group_id="g1", conversation_id="c1",
        message="Req A", callback_url="http://micro-web.example.invalid:3000/group_agent_callbacks/run_evict_comp_a",
    )
    req_b = AsyncCallRequest(
        run_id="run_evict_comp_b",
        idempotency_key="idem_evict_comp_b",
        user_id="u1", unionid="u_union_1", group_id="g1", conversation_id="c1",
        message="Req B", callback_url="http://micro-web.example.invalid:3000/group_agent_callbacks/run_evict_comp_b",
    )
    session = TrustedSession(
        principal=SessionPrincipal(user_id="u1", unionid="u_union_1", user_token=None, source="signed_oauth_principal"),
        group_id="g1", group_token=None, membership=MembershipResult(tier=CapabilityTier.in_group, source="stub"),
    )

    with patch("apps.group_agent_api.app.async_manager._MAX_IDEMPOTENCY_CACHE", 1):
        status_a, _, slot_a = await reserve_idempotency_slot(req_a, session)
        assert status_a == "RESERVED"
        assert slot_a is not None

        # Complete slot A
        ack_res = AsyncCallResponse(success=True, run_id=req_a.run_id, session_id="ga::u1::g1::c1", accepted=True, message="accepted")
        committed = await complete_idempotency_reservation(slot_a, ack_res)
        assert committed is True
        assert slot_a.status == "completed"

        # Now reserve B -> slot A should be evicted
        status_b, _, slot_b = await reserve_idempotency_slot(req_b, session)
        assert status_b == "RESERVED"
        assert slot_b is not None

        assert req_a.idempotency_key not in _idempotency_store
        assert req_a.run_id not in _run_id_store
        assert _idempotency_store[req_b.idempotency_key] is slot_b


@pytest.mark.asyncio
async def test_complete_reservation_slot_mismatch_fails_commit():
    """RESP-009-FIX5: Verify completion or rollback with wrong/mismatched slot identity fails and does not alter store."""
    req_real = AsyncCallRequest(
        run_id="run_mismatch_real",
        idempotency_key="idem_mismatch_real",
        user_id="u1", unionid="u_union_1", group_id="g1", conversation_id="c1",
        message="Real request", callback_url="http://micro-web.example.invalid:3000/group_agent_callbacks/run_mismatch_real",
    )
    session = TrustedSession(
        principal=SessionPrincipal(user_id="u1", unionid="u_union_1", user_token=None, source="signed_oauth_principal"),
        group_id="g1", group_token=None, membership=MembershipResult(tier=CapabilityTier.in_group, source="stub"),
    )

    status_real, _, slot_real = await reserve_idempotency_slot(req_real, session)
    assert status_real == "RESERVED"
    assert slot_real is not None

    # Dummy slot with fake key & run_id
    dummy_slot = IdempotencySlot(
        idempotency_key="idem_mismatch_fake",
        run_id="run_mismatch_fake",
        fingerprint="fake_fp",
        status="reserved",
        response=None,
        created_at=time.time(),
    )
    ack_res = AsyncCallResponse(success=True, run_id="run_mismatch_fake", session_id="ga::u1::g1::c1", accepted=True, message="accepted")

    # Complete dummy slot -> returns False
    comp_dummy = await complete_idempotency_reservation(dummy_slot, ack_res)
    assert comp_dummy is False

    # Rollback dummy slot -> returns False, but dummy_slot status becomes rolled_back & event set
    rb_dummy = await rollback_idempotency_reservation(dummy_slot)
    assert rb_dummy is False
    assert dummy_slot.status == "rolled_back"
    assert dummy_slot.decision_event.is_set()

    # Real slot in store remains untouched!
    assert _idempotency_store[req_real.idempotency_key] is slot_real
    assert slot_real.status == "reserved"


@pytest.mark.asyncio
async def test_commit_failure_compensation_cancels_task_and_returns_500():
    """RESP-009-FIX5: Verify complete_idempotency_reservation returning False triggers full task cancellation and returns 500."""
    body = {
        "run_id": "run_commit_fail_comp_1",
        "idempotency_key": "idem_commit_fail_comp_1",
        "user_id": "u1",
        "unionid": "u_union_1",
        "group_id": "g1",
        "conversation_id": "c1",
        "message": "Commit failure test",
        "callback_url": "http://micro-web.example.invalid:3000/group_agent_callbacks/run_commit_fail_comp_1",
    }

    sig_headers = sign_principal(user_id="u1", unionid="u_union_1", method="POST", path="/call_async", secret=PRINCIPAL_SECRET)
    core_exec_mock = AsyncMock()

    with TestClient(app) as client:
        from apps.group_agent_api.app import _state
        assert _state is not None

        with patch("apps.group_agent_api.app.endpoints.call_async.complete_idempotency_reservation", return_value=False), \
             patch("apps.group_agent_api.app.async_manager._execute_core_agent", core_exec_mock):
            res = client.post("/call_async", json=body, headers=sig_headers)
            assert res.status_code == 500
            assert res.json()["detail"]["error"] == "idempotency_commit_failed"

            # Assert core LLM execution was CANCELLED before running
            assert core_exec_mock.call_count == 0

            # Assert active tasks, runs, and idempotency stores are empty
            tid = thread_id(user_id="u1", group_id="g1", conversation_id="c1")
            assert tid not in _state.active_agent_runs
            assert tid not in _state.active_tasks
            assert "idem_commit_fail_comp_1" not in _idempotency_store
            assert "run_commit_fail_comp_1" not in _run_id_store


# ---------------------------------------------------------------------------
# 10. Non In Group Final Callback Test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_not_in_group_final_callback_zero_candidates_and_at():
    """Verify not_in_group user gets capability=not_in_group, 0 candidates, and 0 @ in final callback payload."""
    req = AsyncCallRequest(
        run_id="run_not_in_group_1",
        idempotency_key="idem_not_in_group_1",
        user_id="u1",
        unionid="u_union_1",
        group_id="g1",
        conversation_id="c1",
        message="Hello not in group",
        callback_url="http://micro-web.example.invalid:3000/group_agent_callbacks/run_not_in_group_1",
        membership="not_in_group",
    )

    from apps.group_agent_api.app.startup import startup
    state_ref = {}
    await startup(state_ref)
    app_state = state_ref["state"]

    class _FakeCheckpointer:
        def flush(self): pass
    class _FakeAgent:
        def __init__(self, base_dir: Path):
            self.base_dir = base_dir
            self.checkpointer = _FakeCheckpointer()
        async def aget_state(self, _config):
            class _S: values = {"messages": []}
            return _S()
        async def ainvoke(self, payload, config):
            meta = config.get("metadata") or {}
            from apps.group_agent_api.agent_factory.profile_schema import profile_from_flat
            from apps.group_agent_api.agent_factory.profile_store import save_profile
            prof = profile_from_flat(user_id=str(meta["user_id"]), group_id=str(meta["group_id"]), doing="X", need="Y", offer="Z")
            save_profile(self.base_dir, prof)
            return {"messages": [payload["messages"][0], AIMessage(content="General reply")]}

    app_state.agent = _FakeAgent(app_state.base_dir)

    session = TrustedSession(
        principal=SessionPrincipal(user_id="u1", unionid="u_union_1", user_token=None, source="signed_oauth_principal"),
        group_id="g1",
        group_token=None,
        membership=MembershipResult(tier=CapabilityTier.not_in_group, source="stub"),
    )

    sent_envelopes = []
    async def mock_send_callback(*, callback_url: str, envelope_dict: dict[str, Any], **kwargs):
        sent_envelopes.append(envelope_dict)
        return True

    with patch("apps.group_agent_api.app.async_manager.send_callback_event", side_effect=mock_send_callback):
        await execute_async_run(req=req, session=session, state=app_state, tid="ga::u1::g1::c1")

    assert len(sent_envelopes) == 2
    final_payload = sent_envelopes[1]["payload"]
    assert final_payload["capability"] == "not_in_group"
    assert final_payload["candidates"] == []
    assert final_payload["mentioned_user_ids"] == []


# ---------------------------------------------------------------------------
# 11. Same Conversation Concurrency Lock Test
# ---------------------------------------------------------------------------


def test_same_conversation_concurrency_lock():
    """Verify concurrent requests for the same conversation thread_id return HTTP 409 run_in_progress."""
    body2 = {
        "run_id": "run_lock_2",
        "idempotency_key": "idem_lock_2",
        "user_id": "u1",
        "unionid": "u_union_1",
        "group_id": "g1",
        "conversation_id": "c1",  # Same conversation
        "message": "Second message",
        "callback_url": "http://micro-web.example.invalid:3000/group_agent_callbacks/run_lock_2",
    }

    sig_headers1 = sign_principal(user_id="u1", unionid="u_union_1", method="POST", path="/call_async", secret=PRINCIPAL_SECRET)

    with TestClient(app) as client:
        from apps.group_agent_api.app import _state
        assert _state is not None

        # Lock thread ga::u1::g1::c1 inside lifespan
        tid = thread_id(user_id="u1", group_id="g1", conversation_id="c1")
        assert _state.try_start_agent_run(tid, "test_owner") is True

        res = client.post("/call_async", json=body2, headers=sig_headers1)
        assert res.status_code == 409
        assert res.json()["detail"]["error"] == "run_in_progress"


# ---------------------------------------------------------------------------
# 12. Atomic Idempotency Gather Execution Count == 1 Test
# ---------------------------------------------------------------------------


def test_atomic_idempotency_gather_concurrent_execution_once():
    """Verify 10 concurrent calls with identical idempotency_key produce exact 1 execution of _execute_core_agent."""
    body = {
        "run_id": "run_concurrent_gather_1",
        "idempotency_key": "idem_concurrent_gather_1",
        "user_id": "u1",
        "unionid": "u_union_1",
        "group_id": "g1",
        "conversation_id": "c1",
        "message": "Gather test",
        "callback_url": "http://micro-web.example.invalid:3000/group_agent_callbacks/run_concurrent_gather_1",
    }

    sig_headers = sign_principal(user_id="u1", unionid="u_union_1", method="POST", path="/call_async", secret=PRINCIPAL_SECRET)

    core_exec_mock = AsyncMock()

    with TestClient(app) as client:
        from apps.group_agent_api.app import _state
        assert _state is not None

        with patch("apps.group_agent_api.app.async_manager.send_callback_event", new_callable=AsyncMock, return_value=True), \
             patch("apps.group_agent_api.app.async_manager._execute_core_agent", core_exec_mock):
            def _make_req():
                return client.post("/call_async", json=body, headers=sig_headers)

            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(_make_req) for _ in range(10)]
                results = [f.result() for f in concurrent.futures.as_completed(futures)]

            statuses = [r.status_code for r in results]
            assert all(s == 202 for s in statuses)

            time.sleep(0.3)
            assert core_exec_mock.call_count == 1, f"Expected _execute_core_agent call count == 1, got {core_exec_mock.call_count}"


# ---------------------------------------------------------------------------
# 13. Sanitized Allowlist Error Detail Test
# ---------------------------------------------------------------------------


def test_sanitized_allowlist_error_detail():
    """Verify invalid SSRF callback URL detail is sanitized and hides internal allowlist array and raw URL query."""
    body = {
        "run_id": "run_ssrf_sanitized_1",
        "idempotency_key": "idem_ssrf_sanitized_1",
        "user_id": "u1",
        "unionid": "u_union_1",
        "group_id": "g1",
        "conversation_id": "c1",
        "message": "SSRF test",
        "callback_url": "http://evil-external-domain.com/callback",
    }

    sig_headers = sign_principal(user_id="u1", unionid="u_union_1", method="POST", path="/call_async", secret=PRINCIPAL_SECRET)

    with TestClient(app) as client:
        res = client.post("/call_async", json=body, headers=sig_headers)
        assert res.status_code == 400
        detail = res.json()["detail"]
        assert detail["error"] == "callback_url_not_allowed"
        assert detail["message"] == "callback_url is not allowed by allowlist"
        assert "allowed_bases" not in json.dumps(detail)
