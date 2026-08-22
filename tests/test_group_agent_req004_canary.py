"""REQ-XCUT-004: Multi-stage canary routing, protocol stickiness and allowlist tests."""

import hashlib
import json
import os
from pathlib import Path
import pytest
from pydantic import ValidationError
from unittest.mock import MagicMock

from apps.group_agent_api.agent_factory.integrations.config import (
    v2_canary_enabled,
    v2_force_off,
    v2_user_allowlist,
    v2_enabled_for_user,
)
from apps.group_agent_api.app.models import (
    AsyncCallRequest,
    ChatRequest,
    RolloutContext,
)
from apps.group_agent_api.app.async_manager import (
    _match_contract_for_run,
    _trusted_rollout_context,
    calculate_request_fingerprint,
)

CANARY_FIXTURE_SHA256 = "cd770e1fcdb4eff9b75d6d02836445c656046f718261e9c1e16447f10cdbd4b6"
NON_CANARY_FIXTURE_SHA256 = "716fa9dc67c99943d30a199319a491c8ea6f6d5f12463201f107546bb7e2a2a9"


def test_shared_fixtures_sha256():
    fixture_dir = Path(__file__).parent / "fixtures" / "group_agent"
    canary_file = fixture_dir / "canary_user_1_v2_run.json"
    non_canary_file = fixture_dir / "non_canary_user_2_v1_run.json"

    assert canary_file.exists(), f"Missing {canary_file}"
    assert non_canary_file.exists(), f"Missing {non_canary_file}"

    canary_bytes = canary_file.read_bytes()
    non_canary_bytes = non_canary_file.read_bytes()

    assert hashlib.sha256(canary_bytes).hexdigest() == CANARY_FIXTURE_SHA256
    assert hashlib.sha256(non_canary_bytes).hexdigest() == NON_CANARY_FIXTURE_SHA256


@pytest.mark.parametrize(
    ("fixture_name", "expected_mode"),
    [
        ("canary_user_1_v2_run.json", "grounded_v2"),
        ("non_canary_user_2_v1_run.json", "legacy_v1"),
    ],
)
def test_shared_fixture_deep_request_is_consumed_by_real_model(
    fixture_name: str,
    expected_mode: str,
) -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "group_agent" / fixture_name
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

    req = AsyncCallRequest.model_validate(fixture["deep_async_request"])

    assert _trusted_rollout_context(req) == (
        expected_mode,
        "ga-v2-canary-v1",
    )
    assert "protocol_mode" not in req.metadata
    assert "rollout_context" not in req.metadata


def test_v2_allowlist_parsing(monkeypatch):
    monkeypatch.setenv("GROUP_AGENT_V2_USER_ALLOWLIST", "1, 42, 489880")
    assert v2_user_allowlist() == {1, 42, 489880}

    # Strict parsing: any non-positive-integer fails closed to empty set
    for bad_entry in ["1,,2", "1.0", "*", "-1", "0", "foo", "1,bar,2", ""]:
        monkeypatch.setenv("GROUP_AGENT_V2_USER_ALLOWLIST", bad_entry)
        assert v2_user_allowlist() == set(), f"Expected empty set for {bad_entry}"


def test_v2_enabled_for_user_gating(monkeypatch):
    monkeypatch.setenv("GROUP_AGENT_GROUNDED_FINAL_ENABLED", "1")
    monkeypatch.setenv("GROUP_AGENT_MATCH_V2_ENABLED", "1")
    monkeypatch.setenv("GROUP_AGENT_V2_CANARY_ENABLED", "1")
    monkeypatch.setenv("GROUP_AGENT_V2_USER_ALLOWLIST", "1,42")
    monkeypatch.setenv("GROUP_AGENT_V2_FORCE_OFF", "0")

    assert v2_enabled_for_user(1) is True
    assert v2_enabled_for_user("1") is True
    assert v2_enabled_for_user(42) is True
    assert v2_enabled_for_user(2) is False
    assert v2_enabled_for_user("2") is False
    assert v2_enabled_for_user(-1) is False
    assert v2_enabled_for_user("invalid") is False

    # Force off overrides everything
    monkeypatch.setenv("GROUP_AGENT_V2_FORCE_OFF", "1")
    assert v2_enabled_for_user(1) is False
    monkeypatch.setenv("GROUP_AGENT_V2_FORCE_OFF", "0")

    # Missing canary flag fails closed
    monkeypatch.setenv("GROUP_AGENT_V2_CANARY_ENABLED", "0")
    assert v2_enabled_for_user(1) is False
    monkeypatch.setenv("GROUP_AGENT_V2_CANARY_ENABLED", "1")

    # Missing grounded_final flag fails closed
    monkeypatch.setenv("GROUP_AGENT_GROUNDED_FINAL_ENABLED", "0")
    assert v2_enabled_for_user(1) is False
    monkeypatch.setenv("GROUP_AGENT_GROUNDED_FINAL_ENABLED", "1")

    # Missing match_v2 flag fails closed
    monkeypatch.setenv("GROUP_AGENT_MATCH_V2_ENABLED", "0")
    assert v2_enabled_for_user(1) is False


def test_metadata_forbids_protocol_spoofing():
    forbidden_keys = [
        "protocol_mode",
        "rollout_version",
        "contract_version",
        "rollout_context",
    ]
    for key in forbidden_keys:
        with pytest.raises(ValidationError):
            AsyncCallRequest(
                run_id="run_123",
                idempotency_key="idemp_123",
                user_id="2",
                group_id="123",
                message="hello",
                callback_url="http://test.local/callback",
                metadata={key: "grounded_v2"},
            )

        with pytest.raises(ValidationError):
            ChatRequest(
                user_id="2",
                group_id="123",
                message="hello",
                metadata={key: "grounded_v2"},
            )


def test_rollout_context_model():
    ctx = RolloutContext(protocol_mode="grounded_v2", rollout_version="ga-v2-canary-v1")
    assert ctx.protocol_mode == "grounded_v2"
    assert ctx.rollout_version == "ga-v2-canary-v1"

    # Default values
    default_ctx = RolloutContext()
    assert default_ctx.protocol_mode == "legacy_v1"
    assert default_ctx.rollout_version == "ga-v2-canary-v1"

    req = AsyncCallRequest(
        run_id="run_123",
        idempotency_key="idemp_123",
        user_id="1",
        group_id="123",
        message="hello",
        callback_url="http://test.local/callback",
        rollout_context=ctx,
    )
    assert req.rollout_context.protocol_mode == "grounded_v2"

    with pytest.raises(ValidationError):
        RolloutContext(
            protocol_mode="grounded_v2",
            rollout_version="ga-v2-canary-v2",
        )


def test_grounded_v2_never_downgrades_to_match_v1(monkeypatch):
    monkeypatch.setenv("GROUP_AGENT_MATCH_V2_ENABLED", "0")
    assert _match_contract_for_run("grounded_v2") == (
        None,
        "v2_capability_unavailable",
    )

    monkeypatch.setenv("GROUP_AGENT_MATCH_V2_ENABLED", "1")
    assert _match_contract_for_run("grounded_v2") == ("ga-match-v2", None)
    assert _match_contract_for_run("legacy_v1") == (None, None)


def test_rollout_context_is_bound_into_deep_request_fingerprint(monkeypatch):
    fixture_path = (
        Path(__file__).parent
        / "fixtures/group_agent/canary_user_1_v2_run.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    v2_req = AsyncCallRequest.model_validate(fixture["deep_async_request"])
    v1_req = v2_req.model_copy(
        update={"rollout_context": RolloutContext(protocol_mode="legacy_v1")}
    )
    session = MagicMock()
    session.principal.user_id = "1"
    session.principal.unionid = "unionid_canary_user_1"
    session.principal.user_token = ""
    session.group_id = "123"
    session.group_token = ""
    session.membership.tier.value = "member"
    session.membership.source = "test"
    monkeypatch.setattr(
        "apps.group_agent_api.app.async_manager.validate_and_normalize_callback_url",
        lambda value: value,
    )

    assert calculate_request_fingerprint(v2_req, session) != calculate_request_fingerprint(
        v1_req,
        session,
    )
