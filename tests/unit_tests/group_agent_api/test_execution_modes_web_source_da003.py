"""REQ-CAFE-WEB-003: web_cafe source admission on frozen v3.

Widens the trusted-source enum in place so iOS golden vectors stay byte-identical
(two-set _choice still admits ios_app) while web_cafe unlocks cafe_host only.
Taken from the shared admission fixture: a web source must never unlock group_agent.
"""

import copy

import pytest

from apps.group_agent_api.app.execution_modes import (
    calculate_v2_fingerprint,
    validate_v3_request,
)


def _minimal_body(*, source: str, mode: str) -> dict:
    cafe = mode == "cafe_host"
    body = {
        "request_schema_version": 3,
        "queue_schema_version": 3,
        "mode_contract_version": 1,
        "executor_policy_version": 1,
        "execution_mode": mode,
        "group_id": "global",
        "profile_scope_id": "global",
        "group_token": None,
        "user_token": None,
        "user_id": "42",
        "unionid": "u_abc",
        "conversation_id": f"ga_mode_{'cafe' if cafe else 'group'}_42",
        "memory_scope_id": f"community_cafe_mode_{'cafe' if cafe else 'group'}",
        "episode_id": "epi",
        "run_id": "11111111-2222-3333-4444-555555555555",
        "idempotency_key": "k",
        "callback_url": "https://x.example/cb",
        "membership": "unknown",
        "run_match": False,
        "run_invite": False,
        "willing_to_at": False,
        "input_kind": "user_message",
        "message": "hi",
        "event": None,
        "venue_context": {
            "space_id": "community_cafe",
            "space_kind": "cafe_bar",
            "owner_user_id": None,
            "visitor_user_id": "42",
            "host_agent_id": "community_barista",
            "effective_mode": "day",
            "mode_override": "auto",
            "timezone": "Asia/Shanghai",
        },
        "context_snapshot": {
            "episode_id": "epi",
            "user_turn_version": 1,
            "recent_messages": [],
            "previous_greeting": None,
            "previous_visit_at": None,
        },
        "metadata": {"source": source, "client_surface": source},
        "rollout_context": {"protocol_mode": "legacy_v1", "rollout_version": "ga-v2-canary-v1"},
    }
    body["request_fingerprint"] = calculate_v2_fingerprint(body)
    return body


def test_ios_golden_vector_still_admitted() -> None:
    # W08: iOS source unchanged — the two-set _choice still accepts ios_app.
    req = _minimal_body(source="ios_app", mode="cafe_host")
    assert validate_v3_request(req)["metadata"]["source"] == "ios_app"


def test_web_cafe_unlocks_cafe_host() -> None:
    req = _minimal_body(source="web_cafe", mode="cafe_host")
    assert validate_v3_request(req)["metadata"]["source"] == "web_cafe"


def test_web_cafe_never_unlocks_group_agent() -> None:
    req = _minimal_body(source="web_cafe", mode="group_agent")
    with pytest.raises(ValueError, match="web source limited to cafe_host"):
        validate_v3_request(req)


def test_ios_group_agent_still_admitted() -> None:
    req = _minimal_body(source="ios_app", mode="group_agent")
    assert validate_v3_request(req)["metadata"]["source"] == "ios_app"


def test_source_client_surface_mismatch_rejected() -> None:
    req = _minimal_body(source="ios_app", mode="cafe_host")
    req["metadata"] = {"source": "ios_app", "client_surface": "web_cafe"}
    req["request_fingerprint"] = calculate_v2_fingerprint(req)
    with pytest.raises(ValueError, match="source/client_surface mismatch"):
        validate_v3_request(req)