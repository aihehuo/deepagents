"""Personal background is usable for dialogue without granting write authority."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from fastapi import HTTPException

from apps.group_agent_api.app.assistant_protocol import fingerprint
from apps.group_agent_api.app.endpoints.ai_assistant import Input, prompt, validate_result
from apps.group_agent_api.app.models import AsyncCallRequest


def context() -> dict[str, object]:
    """Return a synthetic host-owned background snapshot."""
    return {
        "schema_version": 1, "user_id": 12, "observed_at": "2026-10-13T00:00:00Z",
        "name": "测试用户", "profile": {"id": 7, "headline": "品牌经营者", "bio": "我经营门店",
        "goal": "寻找合作", "updated_at": "2026-10-13T00:00:00Z"},
        "ideas": [{"id": 8, "title": "试点", "description": "连锁试点", "updated_at": "2026-10-13T00:00:00Z"}],
        "projects": [], "partnership_needs": [], "service_brief": {"generated_at": "2026-10-01T00:00:00Z",
        "status": "historical_unconfirmed", "summary": "历史客服判断"},
        "counts": {"ideas": 1, "projects": 0, "partnership_needs": 0}, "truncated": False,
    }


def test_context_survives_main_input_and_is_described_as_background() -> None:
    payload = Input(kind="conversation", user_id=12, user_context=context())
    assert payload.user_context.profile.bio == "我经营门店"
    instructions = prompt(payload)
    assert "不重复问" in instructions and "本轮更正优先" in instructions
    assert "不得把user_context当作聊天消息" in instructions


@pytest.mark.parametrize("kind", ["suggestion", "compose", "match"])
def test_personal_context_is_forbidden_in_group_or_matching_inputs(kind: str) -> None:
    with pytest.raises(ValidationError):
        Input(kind=kind, user_id=12, user_context=context())


def test_personal_context_rejects_identity_mismatch_and_unknown_private_fields() -> None:
    with pytest.raises(ValidationError):
        Input(kind="conversation", user_id=13, user_context=context())
    with pytest.raises(ValidationError):
        Input(kind="conversation", user_id=12, user_context={**context(), "token": "credential"})


def test_background_does_not_become_profile_update_evidence() -> None:
    payload = Input(kind="conversation", user_id=12, user_context=context(),
        messages=[{"id": 9, "author_id": 12, "role": "user", "text": "今天聊聊合作"}])
    output = {"reply": "可以聊聊你的门店", "profile": {"doing": "经营门店", "need": "", "offer": ""},
        "profile_evidence": {"doing": [{"message_id": 7, "quote": "我经营门店"}]}, "proposals": []}
    with pytest.raises(ValueError, match="invalid_profile_evidence"):
        validate_result(payload, json.dumps(output))


def test_context_is_included_in_signed_request_fingerprint() -> None:
    vector = json.loads((Path(__file__).parents[2] / "fixtures/ai_assistant/v5-ruby.json").read_text())
    body = json.loads(vector["body"])
    value = context(); value["user_id"] = int(body["user_id"])
    body["assistant_input"]["user_context"] = value
    body["request_fingerprint"] = fingerprint(body)
    assert AsyncCallRequest.model_validate(body).assistant_input["user_context"] == value
    body["assistant_input"]["user_context"]["name"] = "tampered"
    with pytest.raises(HTTPException) as error:
        AsyncCallRequest.model_validate(body)
    assert error.value.status_code == 400
    assert error.value.detail["error"] == "schema_invalid"


def test_legacy_assistant_inputs_remain_valid_without_background() -> None:
    assert Input(kind="conversation", user_id=12).user_context is None


def test_context_enforces_total_utf8_budget() -> None:
    value = context()
    publication = {"id": 8, "title": "项目", "description": "中" * 1600, "updated_at": "2026-10-13T00:00:00Z"}
    value["ideas"] = [publication] * 3
    value["projects"] = [publication] * 3
    with pytest.raises(ValidationError, match="assistant_context_too_large"):
        Input(kind="conversation", user_id=12, user_context=value)
