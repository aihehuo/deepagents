"""REQ-010 Three-Level Mock Fixture + Local Container Dialog Scenario Test Suite for group_agent_api (RESP-010-FIX4)."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import threading
import time
from pathlib import Path
from typing import Any
import pytest
import requests
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from apps.group_agent_api.agent_factory.integrations.config import assert_startup_security
from apps.group_agent_api.agent_factory.integrations.membership_backend import CapabilityTier, resolve_session_capability
from apps.group_agent_api.agent_factory.integrations.principal import sign_principal
from apps.group_agent_api.agent_factory.model_builder import create_model
from apps.group_agent_api.agent_factory.profile_schema import GroupProfile, ProfileField
from apps.group_agent_api.agent_factory.profile_store import load_profile, save_profile
from apps.group_agent_api.app.models import AsyncCallRequest
from apps.group_agent_api.fixtures.callback_simulator import app as simulator_app, simulator_state
from apps.group_agent_api.fixtures.loader import (
    FixtureSecurityError,
    FixtureValidationError,
    load_fixture,
)


@pytest.fixture(autouse=True)
def reset_env(monkeypatch, tmp_path):
    monkeypatch.setenv("GROUP_AGENT_INTEGRATION", "stub")
    monkeypatch.setenv("GROUP_AGENT_ENV", "test")
    monkeypatch.setenv("GROUP_AGENT_MODEL_MODE", "stub")
    monkeypatch.setenv("GROUP_AGENT_PRINCIPAL_HMAC_SECRET", "test_principal_secret_32bytes_long!")
    monkeypatch.setenv("GROUP_AGENT_CALLBACK_HMAC_SECRET", "test_callback_secret_32bytes_long!")
    monkeypatch.setenv("GROUP_AGENT_CALLBACK_ALLOWED_BASE_URLS", "http://localhost:3009/group_agent_callbacks")
    monkeypatch.setenv("GROUP_AGENT_RUNTIME_DIR", str(tmp_path))
    simulator_state.reset()


def sign_callback_payload(
    method: str,
    path: str,
    body_bytes: bytes,
    secret: str,
    ts_str: str,
    nonce_str: str,
) -> str:
    body_sha = hashlib.sha256(body_bytes).hexdigest()
    canon = "\n".join([
        "GA-CALLBACK-V1",
        f"method={method.upper()}",
        f"path={path}",
        f"body_sha256={body_sha}",
        f"ts={ts_str}",
        f"nonce={nonce_str}",
    ])
    return hmac.new(secret.encode("utf-8"), canon.encode("utf-8"), hashlib.sha256).hexdigest()


# =====================================================================
# 1. Fixture Loader & Multi-File Schema & Negative Validation Tests
# =====================================================================

@pytest.mark.l1
def test_l1_fixture_loading_and_schema():
    dataset = load_fixture("L1")
    assert dataset.schema_version == "GA-FIXTURE-V1"
    assert dataset.level == "L1"
    assert len(dataset.groups) == 2
    assert len(dataset.members) == 10
    assert len(dataset.scenarios) == 4


@pytest.mark.l2
def test_l2_fixture_loading_and_schema():
    dataset = load_fixture("L2")
    assert dataset.schema_version == "GA-FIXTURE-V1"
    assert dataset.level == "L2"
    assert len(dataset.groups) == 8
    assert len(dataset.members) >= 200
    assert len(dataset.scenarios) == 24


@pytest.mark.l3
def test_l3_fixture_generator_deterministic():
    ds1 = load_fixture("L3", seed=20260725)
    ds2 = load_fixture("L3", seed=20260725)
    assert len(ds1.members) == 10000
    assert len(ds2.members) == 10000
    assert list(ds1.members.keys()) == list(ds2.members.keys())

    # Verify L3 spec loading & structural contents
    gen_spec = ds1.raw_data.get("generation_spec", {})
    assert gen_spec.get("schema_version") == "GA-FIXTURE-V1"
    assert gen_spec.get("level") == "L3"
    assert gen_spec.get("profile", {}).get("num_members") == 10000

    workload_spec = ds1.raw_data.get("workload", {})
    assert workload_spec.get("num_requests") == 1000
    assert workload_spec.get("concurrency_limit") == 300

    adv_cases = ds1.raw_data.get("adversarial_cases", [])
    assert len(adv_cases) >= 5
    adv_ids = [c["id"] for c in adv_cases]
    assert "adv_01_prompt_injection" in adv_ids


@pytest.mark.l1
def test_fixture_env_security_gate_missing_or_prod(monkeypatch):
    monkeypatch.delenv("GROUP_AGENT_ENV", raising=False)
    with pytest.raises(FixtureSecurityError):
        load_fixture("L1")

    monkeypatch.setenv("GROUP_AGENT_ENV", "production")
    monkeypatch.setenv("GROUP_AGENT_INTEGRATION", "stub")
    with pytest.raises(FixtureSecurityError):
        load_fixture("L1")


@pytest.mark.l1
def test_fixture_schema_validation_failures(tmp_path):
    bad_dir = tmp_path / "bad_fixture"
    l1_bad = bad_dir / "l1"
    l1_bad.mkdir(parents=True)

    # 1. Missing schema version
    (l1_bad / "groups.json").write_text(json.dumps({"groups": []}))
    (l1_bad / "members.json").write_text(json.dumps({"members": []}))
    (l1_bad / "scenarios.json").write_text(json.dumps({"scenarios": []}))
    with pytest.raises(FixtureValidationError):
        load_fixture("L1", fixture_dir=bad_dir)

    # 2. Non-boolean bound field ("false" string)
    l1_bad_2 = tmp_path / "bad_fixture_2" / "l1"
    l1_bad_2.mkdir(parents=True)
    (l1_bad_2 / "groups.json").write_text(json.dumps({
        "schema_version": "GA-FIXTURE-V1", "level": "L1",
        "groups": [{"group_id": "g1", "name": "G1"}]
    }))
    (l1_bad_2 / "members.json").write_text(json.dumps({
        "schema_version": "GA-FIXTURE-V1", "level": "L1",
        "members": [{"user_id": "u1", "group_id": "g1", "display_name": "U1", "bound": "false"}]
    }))
    (l1_bad_2 / "scenarios.json").write_text(json.dumps({
        "schema_version": "GA-FIXTURE-V1", "level": "L1",
        "scenarios": []
    }))
    with pytest.raises(FixtureValidationError):
        load_fixture("L1", fixture_dir=tmp_path / "bad_fixture_2")


@pytest.mark.l1
def test_fixture_schema_reference_validations(tmp_path):
    # Test top-level empty list validation & scenario caller unknown user
    l1_bad_caller = tmp_path / "bad_caller" / "l1"
    l1_bad_caller.mkdir(parents=True)
    (l1_bad_caller / "groups.json").write_text(json.dumps({
        "schema_version": "GA-FIXTURE-V1", "level": "L1",
        "groups": [{"group_id": "g1", "name": "G1"}]
    }))
    (l1_bad_caller / "members.json").write_text(json.dumps({
        "schema_version": "GA-FIXTURE-V1", "level": "L1",
        "members": [{"user_id": "u1", "group_id": "g1", "display_name": "U1"}]
    }))
    (l1_bad_caller / "scenarios.json").write_text(json.dumps({
        "schema_version": "GA-FIXTURE-V1", "level": "L1",
        "scenarios": [{"scenario_id": "sc1", "caller_user_id": "unknown_user_999", "group_id": "g1"}]
    }))
    with pytest.raises(FixtureValidationError) as exc:
        load_fixture("L1", fixture_dir=tmp_path / "bad_caller")
    assert "unknown caller_user_id" in str(exc.value)


# =====================================================================
# 2. Membership Fixture Fail-Closed & Profile Evolution Security Tests
# =====================================================================

@pytest.mark.l1
def test_membership_fixture_fail_closed_nonexistent_level(monkeypatch):
    monkeypatch.setenv("GROUP_AGENT_TEST_LEVEL", "DOES_NOT_EXIST")
    res = resolve_session_capability(
        group_id="group_l1_alpha",
        user_id="u101",
        membership_override="in_group",
        unionid="union_u101",
        group_token="token_u101",
    )
    assert res.tier == CapabilityTier.not_in_group
    assert "fixture_error" in res.reason


@pytest.mark.l1
def test_membership_fixture_unknown_user(monkeypatch):
    monkeypatch.setenv("GROUP_AGENT_TEST_LEVEL", "L1")
    res = resolve_session_capability(
        group_id="group_l1_alpha",
        user_id="unknown_user_999",
        membership_override="in_group",
        unionid="union_unknown",
        group_token="token_unknown",
    )
    assert res.tier == CapabilityTier.not_in_group
    assert res.reason == "fixture_authoritative_not_in_group"


@pytest.mark.l1
def test_sensitive_field_masking():
    dataset = load_fixture("L1")
    member = dataset.members["group_l1_alpha:u101"]
    safe_profile = member.safe_profile_for_llm()
    assert "phone" not in safe_profile
    assert "wechat" not in safe_profile
    assert "email" not in safe_profile
    assert safe_profile["doing"] == "Building LLM agents"


@pytest.mark.l1
def test_strict_group_isolation_guard():
    dataset = load_fixture("L1")
    raw_candidates = [
        {"user_id": "u101", "group_id": "group_l1_alpha", "score": 0.95},
        {"user_id": "u201", "group_id": "group_l1_beta", "score": 0.99},  # Outer group bait
    ]
    filtered = dataset.filter_candidates_for_group("group_l1_alpha", raw_candidates)
    matched_ids = [c["user_id"] for c in filtered]
    assert "u101" in matched_ids
    assert "u201" not in matched_ids, "Outer group high score bait candidate must be excluded!"


@pytest.mark.l2
def test_dual_identity_profile_isolation(tmp_path):
    dataset = load_fixture("L1")
    member_alpha = dataset.members["group_l1_alpha:u104"]
    member_beta = dataset.members["group_l1_beta:u104"]

    assert member_alpha.group_id == "group_l1_alpha"
    assert member_beta.group_id == "group_l1_beta"
    assert "Distributed systems" in member_alpha.profile["doing"]
    assert "Investing" in member_beta.profile["doing"]

    # Test profile store path isolation for same user across different groups
    prof_alpha = GroupProfile(
        user_id="u104",
        group_id="group_l1_alpha",
        doing={"value": "Distributed systems"},
        need={"value": "Co-founder"},
        offer={"value": "Backend"},
    )
    prof_beta = GroupProfile(
        user_id="u104",
        group_id="group_l1_beta",
        doing={"value": "Investing"},
        need={"value": "LPs"},
        offer={"value": "Capital"},
    )

    p1 = save_profile(tmp_path, prof_alpha)
    p2 = save_profile(tmp_path, prof_beta)
    assert p1 != p2

    loaded_a = load_profile(tmp_path, "u104", "group_l1_alpha")
    loaded_b = load_profile(tmp_path, "u104", "group_l1_beta")
    assert loaded_a is not None and loaded_b is not None
    assert loaded_a.doing.value == "Distributed systems"
    assert loaded_b.doing.value == "Investing"


# =====================================================================
# 3. Callback Simulator Security & Terminal Event Contracts
# =====================================================================

@pytest.mark.l1
def test_callback_four_headers_strictly_required():
    client = TestClient(simulator_app)
    res = client.post("/group_agent_callbacks", json={"event": "progress"})
    assert res.status_code == 400
    assert res.json()["detail"]["error"] == "missing_callback_header"


@pytest.mark.l1
def test_callback_version_rejection():
    client = TestClient(simulator_app)
    res = client.post(
        "/group_agent_callbacks",
        json={"event": "progress"},
        headers={
            "X-GA-Callback-Version": "GA-CALLBACK-V2",
            "X-GA-Callback-Timestamp": str(int(time.time())),
            "X-GA-Callback-Nonce": "nonce_1001",
            "X-GA-Callback-Signature": "sig_dummy",
        },
    )
    assert res.status_code == 400
    assert res.json()["detail"]["error"] == "invalid_callback_version"


@pytest.mark.l1
def test_callback_timestamp_skew():
    client = TestClient(simulator_app)
    old_ts = str(int(time.time()) - 400)
    res = client.post(
        "/group_agent_callbacks",
        json={"event": "progress"},
        headers={
            "X-GA-Callback-Version": "GA-CALLBACK-V1",
            "X-GA-Callback-Timestamp": old_ts,
            "X-GA-Callback-Nonce": "nonce_1002",
            "X-GA-Callback-Signature": "sig_dummy",
        },
    )
    assert res.status_code == 400
    assert res.json()["detail"]["error"] == "timestamp_skew_exceeded"


@pytest.mark.l1
def test_callback_nonce_anti_replay():
    client = TestClient(simulator_app)
    secret = "test_callback_secret_32bytes_long!"
    ts_str = str(int(time.time()))
    nonce_str = "nonce_replay_001"
    body_bytes = json.dumps({"event": "progress", "run_id": "run_001", "seq": 1}).encode()

    sig = sign_callback_payload("POST", "/group_agent_callbacks", body_bytes, secret, ts_str, nonce_str)

    headers = {
        "X-GA-Callback-Version": "GA-CALLBACK-V1",
        "X-GA-Callback-Timestamp": ts_str,
        "X-GA-Callback-Nonce": nonce_str,
        "X-GA-Callback-Signature": sig,
        "Content-Type": "application/json",
    }

    res1 = client.post("/group_agent_callbacks", content=body_bytes, headers=headers)
    assert res1.status_code == 200

    # Replay same nonce -> 409
    res2 = client.post("/group_agent_callbacks", content=body_bytes, headers=headers)
    assert res2.status_code == 409
    assert res2.json()["detail"]["error"] == "nonce_replayed"


@pytest.mark.l1
def test_callback_terminal_event_contract():
    client = TestClient(simulator_app)
    secret = "test_callback_secret_32bytes_long!"
    ts_str = str(int(time.time()))
    nonce1 = "nonce_term_001"
    body_final = json.dumps({"event": "final", "run_id": "run_term_001", "seq": 1}).encode()
    sig1 = sign_callback_payload("POST", "/group_agent_callbacks", body_final, secret, ts_str, nonce1)

    headers1 = {
        "X-GA-Callback-Version": "GA-CALLBACK-V1",
        "X-GA-Callback-Timestamp": ts_str,
        "X-GA-Callback-Nonce": nonce1,
        "X-GA-Callback-Signature": sig1,
        "Content-Type": "application/json",
    }
    res1 = client.post("/group_agent_callbacks", content=body_final, headers=headers1)
    assert res1.status_code == 200
    assert simulator_state.terminal_by_run.get("run_term_001") == "final"

    # Query /records/{run_id} API
    rec_res = client.get("/records/run_term_001")
    assert rec_res.status_code == 200
    rec_data = rec_res.json()
    assert rec_data["count"] == 1
    assert rec_data["terminal_event"] == "final"


# =====================================================================
# 4. Model Builder & Config Security Tests (Explicit Stub Required)
# =====================================================================

@pytest.mark.l1
def test_model_builder_stub_model_mode_forbidden_in_http_or_prod(monkeypatch):
    monkeypatch.setenv("GROUP_AGENT_INTEGRATION", "http")
    monkeypatch.setenv("GROUP_AGENT_MODEL_MODE", "stub")
    with pytest.raises(RuntimeError):
        create_model()

    monkeypatch.setenv("GROUP_AGENT_INTEGRATION", "stub")
    monkeypatch.setenv("GROUP_AGENT_ENV", "production")
    with pytest.raises(RuntimeError):
        create_model()


@pytest.mark.l1
def test_model_builder_keyless_raises_runtime_error(monkeypatch):
    monkeypatch.setenv("GROUP_AGENT_INTEGRATION", "http")
    monkeypatch.setenv("GROUP_AGENT_ENV", "production")
    monkeypatch.setenv("GROUP_AGENT_MODEL_MODE", "")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        create_model()


@pytest.mark.l1
def test_model_builder_keyless_in_stub_integration_without_explicit_stub_mode_raises_runtime_error(monkeypatch):
    monkeypatch.setenv("GROUP_AGENT_INTEGRATION", "stub")
    monkeypatch.setenv("GROUP_AGENT_ENV", "test")
    monkeypatch.setenv("GROUP_AGENT_MODEL_MODE", "")  # Empty!
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        create_model()


@pytest.mark.l1
def test_model_builder_provider_key_resolution(monkeypatch):
    monkeypatch.setenv("GROUP_AGENT_INTEGRATION", "stub")
    monkeypatch.setenv("GROUP_AGENT_ENV", "test")
    monkeypatch.setenv("GROUP_AGENT_MODEL_MODE", "stub")
    model = create_model()
    assert model.__class__.__name__ == "StubGroupAgentChatModel"


# =====================================================================
# 5. L1 Scenario Oracle Execution Tests
# =====================================================================

@pytest.mark.l1
def test_l1_scenarios_e2e_execution(monkeypatch):
    monkeypatch.setenv("GROUP_AGENT_TEST_LEVEL", "L1")
    from apps.group_agent_api.app import app
    with TestClient(app) as client:
        ds = load_fixture("L1")
        for sc in ds.scenarios:
            res = client.post(
                "/chat",
                json={
                    "user_id": sc.caller_user_id,
                    "group_id": sc.group_id,
                    "conversation_id": sc.conversation_id,
                    "message": sc.messages[0],
                    "membership": "in_group",
                },
            )
            assert res.status_code == 200
            data = res.json()

            # Oracle assertions
            cand_ids = [c["user_id"] for c in data.get("candidates", [])]
            for forbidden in sc.forbidden_matches:
                assert forbidden not in cand_ids, f"Forbidden member {forbidden} leaked into match results!"

            # Sensitive fields masking check
            raw_str = json.dumps(data)
            for s_field in ["phone", "wechat", "email", "private_notes"]:
                assert f'"{s_field}": "' not in raw_str


@pytest.mark.l1
def test_l1_non_member_caller_scenario(monkeypatch):
    monkeypatch.setenv("GROUP_AGENT_TEST_LEVEL", "L1")
    from apps.group_agent_api.app import app
    with TestClient(app) as client:
        # u205 is guest / not_in_group in group_l1_beta
        res = client.post(
            "/chat",
            json={
                "user_id": "u205",
                "group_id": "group_l1_beta",
                "conversation_id": "conv_non_member",
                "message": "Hello outside guest query",
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert data["capability"] == "not_in_group"
        assert len(data.get("candidates", [])) == 0


# =====================================================================
# 6. L2 All 24 Multi-Round Scenario Execution Tests
# =====================================================================

@pytest.mark.l2
def test_l2_scenarios_all_24_multi_round(monkeypatch):
    monkeypatch.setenv("GROUP_AGENT_TEST_LEVEL", "L2")
    from apps.group_agent_api.app import app
    with TestClient(app) as client:
        ds = load_fixture("L2")
        assert len(ds.scenarios) == 24
        for sc in ds.scenarios:
            for round_idx, msg in enumerate(sc.messages):
                res = client.post(
                    "/chat",
                    json={
                        "user_id": sc.caller_user_id,
                        "group_id": sc.group_id,
                        "conversation_id": sc.conversation_id,
                        "message": msg,
                    },
                )
                assert res.status_code == 200
                data = res.json()
                assert data["group_id"] == sc.group_id


# =====================================================================
# 7. L3 Spec-Driven Workload Scale, Fault-Adversarial Injection & Real App State Cleanup
# =====================================================================

@pytest.mark.l3
def test_l3_workload_scale_and_fault_injection(monkeypatch):
    """Deterministic L3 workload: execute each workload fault type with real counts,
    measure actual peak concurrency == 300, and run a full bounded-wait cleanup oracle
    (active_agent_runs / active_tasks / reserved slots / dual-index consistency).

    Design notes (why this is deterministic, not the old flaky barrier):
    - A single asyncio.Event ("peak_reached") is set the moment `concurrency_limit`
      core-agent tasks are simultaneously in flight. Every core task waits on this
      event with a bounded timeout, so the LAST (1000 % 300) tasks never block on an
      unfillable Barrier — they proceed once peak was already proven.
    - The wait for background completion is a bounded deadline loop that calls
      pytest.fail() on timeout instead of silently falling through to shutdown.
    """
    monkeypatch.setenv("GROUP_AGENT_TEST_LEVEL", "L3")
    ds = load_fixture("L3")
    assert len(ds.members) == 10000

    workload_spec = ds.raw_data.get("workload", {})
    req_count = workload_spec.get("num_requests", 1000)
    concurrency_limit = workload_spec.get("concurrency_limit", 300)
    duplicate_ratio = workload_spec.get("idempotency_duplicate_ratio", 0.1)
    conflict_ratio = workload_spec.get("conflict_ratio", 0.05)
    slow_cb_ratio = workload_spec.get("slow_callback_ratio", 0.05)
    retry_ratio = workload_spec.get("retry_ratio", 0.05)

    monkeypatch.setenv("GROUP_AGENT_ASYNC_MAX_ACTIVE", "500")

    from apps.group_agent_api.app import app
    from unittest.mock import patch

    # ---- Counters & concurrency instrumentation ----
    core_execution_count = 0
    active_counter = 0
    peak_active = 0
    counter_lock = threading.Lock()
    delivered_callbacks: dict[str, list[dict[str, Any]]] = {}
    cb_lock = threading.Lock()
    status_counts: dict[str, int] = {}
    sc_lock = threading.Lock()
    slow_seen = 0
    slow_lock = threading.Lock()
    peak_reached: asyncio.Event | None = None

    def _record_status(code: int) -> None:
        with sc_lock:
            status_counts[str(code)] = status_counts.get(str(code), 0) + 1

    async def mock_execute_core_agent(*, req, session, state, tid, emit_callback):
        nonlocal core_execution_count, active_counter, peak_active, slow_seen
        with counter_lock:
            active_counter += 1
            if active_counter > peak_active:
                peak_active = active_counter
            core_execution_count += 1
            reached = active_counter >= concurrency_limit
        if reached and peak_reached is not None and not peak_reached.is_set():
            peak_reached.set()
        try:
            # Hold tasks until peak concurrency is proven, so peak_active can reach
            # exactly concurrency_limit. Bounded — never an unfillable barrier.
            if peak_reached is not None:
                try:
                    await asyncio.wait_for(peak_reached.wait(), timeout=20.0)
                except asyncio.TimeoutError:
                    pass
            # slow_callback fault: a subset of runs emit their callback with delay.
            if getattr(req, "conversation_id", "").startswith("conv_slowcb_"):
                with slow_lock:
                    slow_seen += 1
                await asyncio.sleep(0.05)
            await emit_callback("final", {
                "status": "completed",
                "output": "Lightweight Core Execution Completed",
                "candidates": [],
            })
        finally:
            with counter_lock:
                active_counter -= 1

    async def mock_send_callback_event(*, callback_url: str, envelope_dict: dict[str, Any], **kwargs) -> bool:
        r_id = envelope_dict.get("run_id", "")
        with cb_lock:
            delivered_callbacks.setdefault(r_id, []).append(envelope_dict)
        return True

    # ---- Build the request mix from workload ratios ----
    n_duplicate = int(req_count * duplicate_ratio)      # 100 → 50 pairs
    n_conflict = int(req_count * conflict_ratio)        # 50
    n_slow = int(req_count * slow_cb_ratio)             # 50
    n_retry = int(req_count * retry_ratio)              # 50 → 25 pairs
    n_normal = req_count - n_duplicate - n_conflict - n_slow - n_retry

    scenarios = ds.scenarios

    def _make_body(idx: int, *, conv_prefix: str = "conv_l3", key: str | None = None, run_id: str | None = None) -> dict[str, Any]:
        sc = scenarios[idx % len(scenarios)]
        return {
            "run_id": run_id or f"l3_run_{idx:05d}",
            "idempotency_key": key or f"idem_l3_{idx:05d}",
            "user_id": sc.caller_user_id,
            "unionid": f"union_{sc.caller_user_id}",
            "group_id": sc.group_id,
            "conversation_id": f"{conv_prefix}_{idx:05d}",
            "message": sc.messages[0],
            "callback_url": "http://localhost:3009/group_agent_callbacks",
            "run_match": True,
            "run_invite": True,
        }

    request_descriptors: list[dict[str, Any]] = []
    idx = 0
    for _ in range(n_normal):
        request_descriptors.append({"body": _make_body(idx), "fault": "normal"})
        idx += 1
    for _ in range(n_duplicate // 2):
        dup = _make_body(idx)
        request_descriptors.append({"body": dup, "fault": "duplicate"})
        request_descriptors.append({"body": dict(dup), "fault": "duplicate_replay"})
        idx += 1
    for _ in range(n_conflict):
        base = _make_body(idx)
        # Same idempotency_key, DIFFERENT run_id -> CONFLICT (409)
        conflicting = dict(base, run_id=base["run_id"] + "_alt")
        request_descriptors.append({"body": base, "fault": "conflict_primary"})
        request_descriptors.append({"body": conflicting, "fault": "conflict_collision"})
        idx += 1
    for _ in range(n_slow):
        request_descriptors.append({"body": _make_body(idx, conv_prefix="conv_slowcb"), "fault": "slow_callback"})
        idx += 1
    for _ in range(n_retry // 2):
        body = _make_body(idx)
        request_descriptors.append({"body": body, "fault": "retry_first"})
        request_descriptors.append({"body": dict(body), "fault": "retry_second"})
        idx += 1

    # Deterministic interleave (no Date/random-seed nondeterminism)
    import random
    random.Random(20260725).shuffle(request_descriptors)
    total_reqs = len(request_descriptors)

    async def run_l3_scale():
        nonlocal peak_reached
        peak_reached = asyncio.Event()
        from apps.group_agent_api.app.startup import startup
        import apps.group_agent_api.app as app_mod
        state_ref = {"state": None}
        await startup(state_ref)
        app_mod._state = state_ref["state"]

        # Allow up to concurrency_limit requests to be in the /call_async critical
        # section at once so peak core concurrency can reach exactly concurrency_limit.
        sem = asyncio.Semaphore(concurrency_limit)
        latencies: list[float] = []
        accepted = 0
        p_secret = "test_principal_secret_32bytes_long!"

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            async def worker(desc: dict[str, Any]):
                nonlocal accepted
                async with sem:
                    t0 = time.perf_counter()
                    headers = sign_principal(
                        user_id=desc["body"]["user_id"],
                        unionid=desc["body"]["unionid"],
                        secret=p_secret,
                        method="POST",
                        path="/call_async",
                    )
                    headers["Content-Type"] = "application/json"
                    res = await ac.post("/call_async", json=desc["body"], headers=headers)
                    latencies.append((time.perf_counter() - t0) * 1000.0)
                    _record_status(res.status_code)
                    if res.status_code in {200, 202}:
                        accepted += 1
                    return desc["fault"], res.status_code

            results = await asyncio.gather(*(worker(d) for d in request_descriptors))

            # Expected distinct run_ids that reach core execution (accepted, non-HIT).
            # duplicate_replay / retry_second are idempotency HITs -> no new core run.
            expected_core = sum(
                1 for d in request_descriptors
                if d["fault"] not in {"duplicate_replay", "retry_second", "conflict_collision"}
            )

            # ---- Bounded wait: fail loudly on timeout, never silently shut down ----
            # Terminal signal = every background task finished (active_agent_runs drained)
            # AND every expected callback delivered. Waiting only on callbacks would race
            # the finish_agent_run() that runs in execute_async_run's finally block.
            from apps.group_agent_api.app import _state as live_state
            WAIT_S = 60.0
            deadline = time.monotonic() + WAIT_S
            done = False
            while time.monotonic() < deadline:
                runs_drained = live_state is None or (
                    len(live_state.active_agent_runs) == 0 and len(live_state.active_tasks) == 0
                )
                with counter_lock, cb_lock:
                    counts_ok = core_execution_count >= expected_core and len(delivered_callbacks) >= expected_core
                if counts_ok and runs_drained:
                    done = True
                    break
                await asyncio.sleep(0.05)
            if not done:
                pytest.fail(
                    f"Background tasks did not reach terminal state within {WAIT_S}s: "
                    f"core_executions={core_execution_count}/{expected_core}, "
                    f"callbacks={len(delivered_callbacks)}/{expected_core}, "
                    f"active_runs={0 if live_state is None else len(live_state.active_agent_runs)}, "
                    f"active_tasks={0 if live_state is None else len(live_state.active_tasks)}"
                )

        # ---- Cleanup oracle reads the APP MODULE _state (not app.state.state) ----
        from apps.group_agent_api.app import _state as cleanup_state
        pre_active_runs = dict(cleanup_state.active_agent_runs) if cleanup_state else {}
        pre_active_tasks = dict(cleanup_state.active_tasks) if cleanup_state else {}

        if cleanup_state is not None:
            await cleanup_state.shutdown()

        latencies.sort()
        n = len(latencies)
        p50 = latencies[min(int(n * 0.50), n - 1)]
        p95 = latencies[min(int(n * 0.95), n - 1)]
        p99 = latencies[min(int(n * 0.99), n - 1)]
        return {
            "accepted": accepted, "results": results, "expected_core": expected_core,
            "p50": p50, "p95": p95, "p99": p99,
            "pre_active_runs": pre_active_runs, "pre_active_tasks": pre_active_tasks,
            "post_state": cleanup_state,
        }

    with patch("apps.group_agent_api.app.async_manager._execute_core_agent", side_effect=mock_execute_core_agent), \
         patch("apps.group_agent_api.app.async_manager.send_callback_event", side_effect=mock_send_callback_event):
        out = asyncio.run(run_l3_scale())

    results = out["results"]
    expected_core = out["expected_core"]

    # ---- Fault-type -> actual status distribution (real executed counts) ----
    fault_status: dict[str, dict[str, int]] = {}
    for fault, code in results:
        fault_status.setdefault(fault, {})
        fault_status[fault][str(code)] = fault_status[fault].get(str(code), 0) + 1

    # ---- Workload assertions with REAL counts ----
    # conflict_collision requests (same key, different run_id) must be rejected 409.
    conflict_collisions = sum(1 for d in request_descriptors if d["fault"] == "conflict_collision")
    assert status_counts.get("409", 0) == conflict_collisions, (
        f"Expected {conflict_collisions} conflict 409s, got {status_counts.get('409', 0)}; dist={fault_status.get('conflict_collision')}"
    )
    # duplicate_replay & retry_second are idempotency HITs: accepted (202/200), no extra core run.
    assert core_execution_count == expected_core, f"Expected {expected_core} core executions, got {core_execution_count}"
    assert len(delivered_callbacks) == expected_core, f"Expected {expected_core} callbacks, got {len(delivered_callbacks)}"
    # slow_callback fault actually executed
    assert slow_seen == n_slow, f"Expected {n_slow} slow_callback runs executed, got {slow_seen}"
    # accepted = everything except the conflict collisions
    assert out["accepted"] == total_reqs - conflict_collisions, (
        f"Expected {total_reqs - conflict_collisions} accepted, got {out['accepted']}"
    )
    # Measured peak concurrency == configured concurrency_limit
    assert peak_active == concurrency_limit, (
        f"Peak concurrency measured={peak_active}, configured={concurrency_limit}"
    )
    assert out["p50"] > 0 and out["p95"] > 0 and out["p99"] > 0

    # ---- Cleanup oracle: pre-shutdown snapshot ----
    assert len(out["pre_active_runs"]) == 0, f"active_agent_runs not drained before shutdown: {out['pre_active_runs']}"
    assert len(out["pre_active_tasks"]) == 0, f"active_tasks not drained before shutdown: {out['pre_active_tasks']}"

    # ---- Cleanup oracle: post-shutdown state ----
    post = out["post_state"]
    assert post is not None
    assert len(post.active_agent_runs) == 0, f"active_agent_runs not empty post-shutdown: {post.active_agent_runs}"
    assert len(post.active_tasks) == 0, f"active_tasks not empty post-shutdown: {post.active_tasks}"

    # ---- Cleanup oracle: idempotency store reserved slots + dual-index consistency ----
    from apps.group_agent_api.app.async_manager import _idempotency_store, _run_id_store
    reserved = [k for k, v in _idempotency_store.items() if v.status == "reserved"]
    assert len(reserved) == 0, f"Reserved slots remain: {reserved}"
    for k, v in _idempotency_store.items():
        assert v.status in {"completed", "rolled_back"}, f"Slot {k} unexpected status {v.status}"
    # Dual-index consistency both directions
    for rid, ikey in _run_id_store.items():
        assert ikey in _idempotency_store, f"run_id {rid} -> missing key {ikey}"
    for k, v in _idempotency_store.items():
        assert _run_id_store.get(v.run_id) == k, f"dual-index mismatch: run_id {v.run_id} != key {k}"


@pytest.mark.l3
def test_l3_adversarial_cases_security_layer(monkeypatch, tmp_path):
    """Execute EACH adversarial case from the L3 spec through the API and assert the
    security layer handles it (no 500, no leak, no cross-group candidate)."""
    monkeypatch.setenv("GROUP_AGENT_TEST_LEVEL", "L3")
    monkeypatch.setenv("GROUP_AGENT_ENV", "test")
    monkeypatch.setenv("GROUP_AGENT_INTEGRATION", "stub")
    monkeypatch.setenv("GROUP_AGENT_MODEL_MODE", "stub")
    monkeypatch.setenv("GROUP_AGENT_PRINCIPAL_HMAC_SECRET", "test_principal_secret_32bytes_long!")
    monkeypatch.setenv("GROUP_AGENT_RUNTIME_DIR", str(tmp_path))
    ds = load_fixture("L3")
    cases = ds.raw_data.get("adversarial_cases", [])
    assert len(cases) >= 5

    from apps.group_agent_api.app import app
    from apps.group_agent_api.fixtures.loader import assert_fixture_environment_allowed

    assert_fixture_environment_allowed()  # security gate must be active

    executed = 0
    with TestClient(app) as client:
        for case in cases:
            cid = case["id"]
            ctype = case["type"]
            payload = case["payload"]
            res = client.post(
                "/chat",
                json={
                    "user_id": "user_l3_00001",
                    "group_id": "group_l3_01",
                    "conversation_id": f"conv_adv_{cid}",
                    "message": payload,
                    "membership": "in_group",
                },
            )
            executed += 1
            # No adversarial payload may crash the API
            assert res.status_code != 500, f"Adversarial case {cid} caused 500: {res.text}"
            data = res.json()
            body_str = json.dumps(data, ensure_ascii=False)
            # No sensitive field leaks in ANY response
            for s_field in ("phone", "wechat", "email", "private_notes"):
                assert f'"{s_field}": "' not in body_str, f"Case {cid} leaked {s_field}"
            # Cross-group bait: every candidate must belong to the trusted group
            if ctype == "cross_group_bait":
                for c in data.get("candidates", []):
                    src = c.get("source_group_id") or c.get("group_id")
                    assert src == "group_l3_01", f"Case {cid} leaked cross-group candidate from {src}"

    assert executed == len(cases), f"Expected to execute all {len(cases)} adversarial cases, ran {executed}"



# =====================================================================
# 4b. Deterministic Agent Tool-Call Profile Evolution Test
# =====================================================================

@pytest.mark.l2
async def test_profile_evolution_via_deterministic_tool_call(monkeypatch, tmp_path):
    """Verify that the stub model's REAL save_group_profile tool call persists
    profile evolution through the API, and that no-tool-call scenarios correctly
    report profile_ok=False."""
    monkeypatch.setenv("GROUP_AGENT_INTEGRATION", "stub")
    monkeypatch.setenv("GROUP_AGENT_ENV", "test")
    monkeypatch.setenv("GROUP_AGENT_MODEL_MODE", "stub")
    monkeypatch.setenv("GROUP_AGENT_PRINCIPAL_HMAC_SECRET", "test_principal_secret_32bytes_long!")
    monkeypatch.setenv("GROUP_AGENT_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("GROUP_AGENT_TEST_LEVEL", "L1")

    from apps.group_agent_api.app import app
    from apps.group_agent_api.agent_factory.agent import create_agent, save_group_profile
    from apps.group_agent_api.agent_factory.profile_store import load_profile, assert_profile_persisted
    from apps.group_agent_api.agent_factory.profile_schema import GroupProfile, ProfileField
    from apps.group_agent_api.app.state import AppState
    from apps.group_agent_api.app.endpoints.chat import chat
    from apps.group_agent_api.app.models import ChatRequest, ChatResponse
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessage, ToolMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    # Custom deterministic model: first call returns tool call, subsequent calls return text
    class DeterministicToolCallModel(BaseChatModel):
        _call_count = 0
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            self._call_count += 1
            # Check if the last message is a ToolMessage (tool result)
            tail = messages[-1] if messages else None
            if isinstance(tail, ToolMessage):
                msg = AIMessage(content="已根据对话更新画像。")
            else:
                # Return a tool call for save_group_profile
                msg = AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "save_group_profile",
                        "args": {
                            "doing": "Building LLM agents",
                            "need": "Frontend cofounder",
                            "offer": "Python, LangChain, PyTorch",
                            "doing_disclosure": "confirmed_public",
                            "need_disclosure": "confirmed_public",
                            "offer_disclosure": "confirmed_public",
                        },
                        "id": f"save_call_{self._call_count}",
                    }]
                )
            return ChatResult(generations=[ChatGeneration(message=msg)])
        @property
        def _llm_type(self): return "deterministic-tool-call-model"
        def bind_tools(self, tools, **kwargs): return self

    agent, _ = create_agent(base_dir=tmp_path, model=DeterministicToolCallModel())
    state = AppState(agent=agent, base_dir=tmp_path)

    # Pre-seed with an initial profile (simulating an earlier session)
    from apps.group_agent_api.agent_factory.profile_store import save_profile
    p_turn0 = GroupProfile(
        user_id="u101", group_id="group_l1_alpha",
        doing=ProfileField(value="Initial exploration"),
        need=ProfileField(value="Co-founder"),
        offer=ProfileField(value="Python"),
    )
    save_profile(tmp_path, p_turn0)

    # Round 1: invoke chat with a message that triggers the stub tool call
    r1 = await chat(
        ChatRequest(user_id="u101", group_id="group_l1_alpha", conversation_id="conv_evo_1",
                     message="我正在做 LLM 智能体，需要前端合伙人，技术栈是 Python、LangChain、PyTorch"),
        state,
    )
    assert r1.profile_persisted, f"Round 1 profile_persisted should be True, got profile_persisted={r1.profile_persisted}"

    # Verify the profile store was actually updated (not just the stub check)
    p1 = load_profile(tmp_path, "u101", "group_l1_alpha")
    assert p1 is not None, "Profile should exist after round 1"
    assert "Building LLM" in p1.doing.value, f"Round 1 doing mismatch: {p1.doing.value}"
    assert "Frontend" in p1.need.value, f"Round 1 need mismatch: {p1.need.value}"

    # Round 2: evolution (user updates their focus)
    call_count_before = DeterministicToolCallModel._call_count
    r2 = await chat(
        ChatRequest(user_id="u101", group_id="group_l1_alpha", conversation_id="conv_evo_2",
                     message="现在转型做 autonomous multi-modal 智能体了，需要 design 合伙人"),
        state,
    )
    assert r2.profile_persisted, f"Round 2 profile_persisted should be True"
    p2 = load_profile(tmp_path, "u101", "group_l1_alpha")
    assert p2 is not None
    # The tool-call model returns the same fixed args each call, so the profile value
    # is overwritten. The key test is that the API exercised the tool and persisted.
    assert p1.updated_at != p2.updated_at, "Profile should have different updated_at after evolution"


@pytest.mark.l2
async def test_profile_evolution_no_tool_call_fails_gracefully(monkeypatch, tmp_path):
    """Negative test: when the stub model returns NO tool call, and there is
    an existing profile, profile_ok must be False — the is_stub bypass is gone."""
    monkeypatch.setenv("GROUP_AGENT_INTEGRATION", "stub")
    monkeypatch.setenv("GROUP_AGENT_ENV", "test")
    monkeypatch.setenv("GROUP_AGENT_MODEL_MODE", "stub")
    monkeypatch.setenv("GROUP_AGENT_PRINCIPAL_HMAC_SECRET", "test_principal_secret_32bytes_long!")
    monkeypatch.setenv("GROUP_AGENT_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("GROUP_AGENT_TEST_LEVEL", "L1")

    from apps.group_agent_api.agent_factory.agent import create_agent
    from apps.group_agent_api.agent_factory.profile_store import load_profile, save_profile
    from apps.group_agent_api.agent_factory.profile_schema import GroupProfile, ProfileField
    from apps.group_agent_api.app.state import AppState
    from apps.group_agent_api.app.endpoints.chat import chat
    from apps.group_agent_api.app.models import ChatRequest
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    class NoToolCallModel(BaseChatModel):
        """Always returns a plain text response with NO tool call."""
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="你好，我是群内智能体。"))])
        @property
        def _llm_type(self): return "no-tool-call-model"
        def bind_tools(self, tools, **kwargs): return self

    agent, _ = create_agent(base_dir=tmp_path, model=NoToolCallModel())
    state = AppState(agent=agent, base_dir=tmp_path)

    # Pre-seed a profile (simulating an existing profile from a previous session)
    p_old = GroupProfile(
        user_id="u101", group_id="group_l1_alpha",
        doing=ProfileField(value="Existing business"),
        need=ProfileField(value="Partner"),
        offer=ProfileField(value="Capital"),
    )
    save_profile(tmp_path, p_old)
    old_updated_at = p_old.updated_at

    # The model never calls save_group_profile, so the profile stays unchanged.
    # With the is_stub bypass removed, profile_ok must be False.
    r = await chat(
        ChatRequest(user_id="u101", group_id="group_l1_alpha", conversation_id="conv_no_tool",
                     message="我也在找技术合伙人"),
        state,
    )
    assert not r.profile_persisted, (
        f"profile_persisted should be False when no tool-call occurs and profile hasn't changed, "
        f"got {r.profile_persisted}"
    )

    # Verify the profile was NOT updated
    p_after = load_profile(tmp_path, "u101", "group_l1_alpha")
    assert p_after is not None
    assert p_after.updated_at == old_updated_at, "Profile updated_at should not change when no tool-call occurs"


# =====================================================================
# 4c. Log Sanitization Regression: candidates cleartext & invite body never logged
# =====================================================================

@pytest.mark.l2
async def test_core_agent_logs_no_candidate_or_invite_cleartext(monkeypatch, tmp_path, caplog):
    """RESP-010-FIX5 ②: the async core-agent debug logs must NOT contain candidate
    cleartext (display_name / doing values) nor the invite body text — only counts/lengths.

    This drives the REAL _execute_core_agent (stub tool-call -> real match -> real invite)
    for an in_group caller with a strong match, and inspects every emitted log line.
    """
    import logging as _logging
    monkeypatch.setenv("GROUP_AGENT_INTEGRATION", "stub")
    monkeypatch.setenv("GROUP_AGENT_ENV", "test")
    monkeypatch.setenv("GROUP_AGENT_MODEL_MODE", "stub")
    monkeypatch.setenv("GROUP_AGENT_PRINCIPAL_HMAC_SECRET", "test_principal_secret_32bytes_long!")
    monkeypatch.setenv("GROUP_AGENT_CALLBACK_HMAC_SECRET", "test_callback_secret_32bytes_long!")
    monkeypatch.setenv("GROUP_AGENT_CALLBACK_ALLOWED_BASE_URLS", "http://localhost:3009/group_agent_callbacks")
    monkeypatch.setenv("GROUP_AGENT_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("GROUP_AGENT_TEST_LEVEL", "L1")

    from apps.group_agent_api.app.async_manager import _execute_core_agent
    from apps.group_agent_api.app.state import AppState
    from apps.group_agent_api.app.session import TrustedSession
    from apps.group_agent_api.app.models import AsyncCallRequest
    from apps.group_agent_api.agent_factory.agent import create_agent
    from apps.group_agent_api.agent_factory.integrations.principal import SessionPrincipal
    from apps.group_agent_api.agent_factory.integrations.membership_client import MembershipResult
    from apps.group_agent_api.agent_factory.capability import CapabilityTier
    from apps.group_agent_api.agent_factory.model_builder import create_model

    # Real stub model that emits a save_group_profile tool call (fixture-accurate for u105).
    agent, _ = create_agent(base_dir=tmp_path, model=create_model())
    state = AppState(agent=agent, base_dir=tmp_path)

    # u105 is an in_group owner in group_l1_alpha; querying for Python/LLM dev matches u101.
    session = TrustedSession(
        principal=SessionPrincipal(user_id="u105", unionid="union_u105", user_token="ut105", source="stub"),
        group_id="group_l1_alpha",
        group_token="gt_alpha",
        membership=MembershipResult(tier=CapabilityTier.in_group, source="stub"),
    )
    req = AsyncCallRequest(
        run_id="run_log_sanitize_1",
        idempotency_key="idem_log_sanitize_1",
        user_id="u105", unionid="union_u105",
        group_id="group_l1_alpha", conversation_id="conv_log_1",
        message="我们需要一位熟悉 Python 和 LLM Agent 开发的合伙人，群里有人推荐吗？",
        callback_url="http://localhost:3009/group_agent_callbacks",
        run_match=True, run_invite=True, willing_to_at=True,
    )

    captured_final: dict[str, Any] = {}
    async def emit_callback(event_type, payload):
        if event_type == "final":
            captured_final.update(payload)
        return True

    tid = "ga::u105::group_l1_alpha::conv_log_1"
    state.try_start_agent_run(tid, "call_async")
    with caplog.at_level(_logging.INFO, logger="uvicorn.error"):
        await _execute_core_agent(req=req, session=session, state=state, tid=tid, emit_callback=emit_callback)

    # Sanity: this path really did produce candidates and an invite (otherwise the test is vacuous).
    candidates = captured_final.get("candidates", [])
    invite_text = captured_final.get("invite_text") or ""
    assert candidates, "Expected at least one candidate so the log-leak assertions are meaningful"

    log_text = caplog.text
    # The debug log lines must exist (proving we inspected the right code path)...
    assert "Core match debug" in log_text
    assert "Invite debug" in log_text
    # ...and must use counts/lengths, not cleartext.
    assert "candidates_count=" in log_text
    assert "text_len=" in log_text and "mentioned_count=" in log_text

    # No candidate cleartext (display_name / doing values) may appear in the logs.
    for c in candidates:
        dn = c.get("display_name")
        if dn:
            assert dn not in log_text, f"Candidate display_name leaked into logs: {dn}"
        doing = c.get("doing")
        if isinstance(doing, dict) and doing.get("value"):
            assert doing["value"] not in log_text, f"Candidate doing value leaked into logs: {doing['value']}"

    # The invite body text must never appear verbatim in the logs.
    if invite_text:
        # Check a distinctive chunk of the invite body
        chunk = invite_text.strip().splitlines()[0][:20]
        if chunk:
            assert chunk not in log_text, f"Invite body leaked into logs: {chunk!r}"


# =====================================================================
# 4d. Precise Mention Identity: ambiguous prefix / space-collapse rejected
# =====================================================================

@pytest.mark.l2
def test_directed_mention_rejects_ambiguous_prefix_and_spacecollapse():
    """RESP-010-FIX5 ③: when candidates share a display-name prefix or collapse to the
    same no-space token, an ambiguous @prefix or @NoSpaceName must be REJECTED. Only the
    full display_name or the stable user_id is a valid mention identity credential."""
    from apps.group_agent_api.agent_factory.invite_copy import assert_directed_invite

    cands = [
        {"user_id": "u_l1_2", "display_name": "L1 User 2", "source_group_id": "g1",
         "group_id": "g1", "doing": {"value": "skill A", "disclosure": "confirmed_public"}},
        {"user_id": "u_l1_3", "display_name": "L1 User 3", "source_group_id": "g1",
         "group_id": "g1", "doing": {"value": "skill B", "disclosure": "confirmed_public"}},
    ]
    base_elements = {
        "who_doing": "我在做A", "resources": "有B", "topic": "想请教选型", "low_pressure": "聊聊",
    }

    # 1. Ambiguous first-word prefix @L1 (shared by both candidates) -> rejected
    e_prefix = dict(base_elements, why_invite="@L1 值得聊一次以确认 不一定合适")
    v_prefix = assert_directed_invite(text="\n".join(e_prefix.values()), elements=e_prefix, candidates=cands)
    assert any(x.startswith("at_not_in_candidates") for x in v_prefix), v_prefix

    # 2. Space-collapsed @L1User2 -> rejected
    e_collapse = dict(base_elements, why_invite="@L1User2 值得聊一次以确认 不一定合适")
    v_collapse = assert_directed_invite(text="\n".join(e_collapse.values()), elements=e_collapse, candidates=cands)
    assert any(x.startswith("at_not_in_candidates") for x in v_collapse), v_collapse

    # 3. Stable user_id @u_l1_2 -> accepted (no at_not_in_candidates violation)
    e_uid = dict(base_elements, why_invite="@u_l1_2 值得聊一次以确认 不一定合适")
    v_uid = assert_directed_invite(text="\n".join(e_uid.values()), elements=e_uid, candidates=cands)
    assert not any(x.startswith("at_not_in_candidates") for x in v_uid), v_uid


# =====================================================================
# 8. Container E2E Integration Test with Exact Oracle Assertions
# =====================================================================

@pytest.mark.container_e2e
def test_container_e2e_live_service():
    container_api = os.environ.get("REQ010_CONTAINER_API_BASE")
    container_sim = os.environ.get("REQ010_CONTAINER_SIMULATOR_BASE")
    if not container_api or not container_sim:
        pytest.skip("REQ010_CONTAINER_API_BASE or REQ010_CONTAINER_SIMULATOR_BASE not set.")

    # 1. Health check
    res_api = requests.get(f"{container_api}/health")
    assert res_api.status_code == 200
    res_sim = requests.get(f"{container_sim}/health")
    assert res_sim.status_code == 200

    # 2. Directly load L1 4 scenarios from load_fixture("L1")
    ds_l1 = load_fixture("L1")
    scenarios = ds_l1.scenarios
    assert len(scenarios) == 4

    p_secret = os.environ.get("GROUP_AGENT_PRINCIPAL_HMAC_SECRET", "test_principal_secret_32bytes_long!")
    callback_target = os.environ.get("REQ010_CONTAINER_CALLBACK_URL", f"{container_sim}/group_agent_callbacks")

    expected_run_ids = []
    for sc in scenarios:
        run_id = f"run_container_{sc.scenario_id}"
        expected_run_ids.append(run_id)

        headers = sign_principal(
            user_id=sc.caller_user_id,
            unionid=f"union_{sc.caller_user_id}",
            secret=p_secret,
            method="POST",
            path="/call_async",
        )
        headers["Content-Type"] = "application/json"

        call_req = {
            "run_id": run_id,
            "idempotency_key": f"idempotency_container_{sc.scenario_id}",
            "user_id": sc.caller_user_id,
            "unionid": f"union_{sc.caller_user_id}",
            "group_id": sc.group_id,
            "conversation_id": sc.conversation_id,
            "message": sc.messages[0],
            "callback_url": callback_target,
            "run_match": True,
            "run_invite": True,
            "willing_to_at": True,
        }

        res_async = requests.post(f"{container_api}/call_async", json=call_req, headers=headers)
        assert res_async.status_code in {200, 202}

    # 3. Wait for Callback Simulator to complete all 4 terminal runs
    stats = {}
    for _ in range(30):
        time.sleep(1)
        stats_res = requests.get(f"{container_sim}/stats")
        if stats_res.status_code == 200:
            stats = stats_res.json()
            if stats.get("terminal_runs", 0) == 4 and stats.get("active_runs", 0) == 0:
                break

    assert stats.get("records_count", 0) >= 8, f"Expected at least 8 callback records, got {stats.get('records_count')}"
    assert stats.get("terminal_runs", 0) == 4, f"Expected exact terminal_runs == 4, got {stats.get('terminal_runs')}"
    assert stats.get("active_runs", 0) == 0, f"Expected active_runs == 0, got {stats.get('active_runs')}"
    assert stats.get("hmac_failures", 0) == 0, f"Expected 0 hmac_failures, got {stats.get('hmac_failures')}"
    assert stats.get("seq_failures", 0) == 0, f"Expected 0 seq_failures, got {stats.get('seq_failures')}"
    assert stats.get("terminal_failures", 0) == 0, f"Expected 0 terminal_failures, got {stats.get('terminal_failures')}"

    # 4. Query /records/{run_id} API for each scenario run and assert oracle invariants
    for sc in scenarios:
        run_id = f"run_container_{sc.scenario_id}"
        rec_res = requests.get(f"{container_sim}/records/{run_id}")
        assert rec_res.status_code == 200
        rec_data = rec_res.json()

        assert rec_data["count"] >= 2
        assert rec_data["terminal_event"] == "final", f"Scenario {sc.scenario_id} expected terminal_event=='final', got {rec_data['terminal_event']}"

        records = rec_data["records"]
        events = [r["event"] for r in records]
        assert events[0] == "progress"
        assert events[-1] == "final"

        # Payload oracle check
        final_payload = records[-1]["payload"]
        capability = final_payload.get("capability", "")
        if capability == "not_in_group":
            # Non-member caller scenario: no matches, no @
            assert len(final_payload.get("candidates", [])) == 0, f"Scenario {sc.scenario_id} expected 0 candidates for not_in_group, got {len(final_payload.get('candidates', []))}"
            assert len(final_payload.get("at_users", [])) == 0, f"Scenario {sc.scenario_id} expected 0 at_users for not_in_group, got {len(final_payload.get('at_users', []))}"
        else:
            cand_ids = [c["user_id"] for c in final_payload.get("candidates", [])]
            for exp in sc.expected_matches:
                assert exp in cand_ids, f"Expected match {exp} missing from candidates in scenario {sc.scenario_id}!"
            for forbidden in sc.forbidden_matches:
                assert forbidden not in cand_ids, f"Forbidden match {forbidden} found in final payload!"

            at_users = final_payload.get("at_users", [])
            for exp_at in sc.expected_at_users:
                assert exp_at in at_users, f"Expected @ user {exp_at} missing from at_users in scenario {sc.scenario_id}!"

            # Masking check: zero sensitive fields in final payload
            payload_str = json.dumps(final_payload)
            for s_field in ["phone", "wechat", "email", "private_notes"]:
                assert f'"{s_field}": "' not in payload_str
