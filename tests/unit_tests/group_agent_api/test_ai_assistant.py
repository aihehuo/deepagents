"""No-network tests for private assistant output and internal HTTP admission."""
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.group_agent_api.app.endpoints import ai_assistant as assistant


def source(kind="conversation"):
    return assistant.Input(kind=kind, user_id=1, messages=[{"id": 1, "role": "user", "text": "愿意提供门店"}],
                           circles=[{"id": 10, "version": 1, "owner_id": 1, "kind": "offer", "title": "门店", "description": "试点"},
                                    {"id": 20, "version": 1, "owner_id": 2, "kind": "need", "title": "试点", "description": "门店"}])


def proposal():
    return {"reply": "请核对初稿", "profile": {"doing": "餐饮", "need": "", "offer": "门店"},
            "proposals": [{"circle_id": None, "kind": "offer", "title": "门店", "description": "试点", "conditions": [], "source_ids": [1]}]}


def test_conversation_keeps_sources_and_draft_separate():
    result = assistant.validate_result(source(), json.dumps(proposal()))
    assert result["proposals"][0]["source_ids"] == [1]
    assert "members" not in result


@pytest.mark.parametrize("ids", [[999], []])
def test_proposals_require_actual_user_evidence(ids):
    data = proposal()
    data["proposals"][0]["source_ids"] = ids
    with pytest.raises(ValueError):
        assistant.validate_result(source(), json.dumps(data))


def test_foreign_circle_update_rejected():
    data = proposal()
    data["proposals"][0]["circle_id"] = 20
    with pytest.raises(ValueError):
        assistant.validate_result(source(), json.dumps(data))


def test_assistant_message_is_not_evidence():
    payload = source()
    payload.messages[0].role = "assistant"
    with pytest.raises(ValueError):
        assistant.validate_result(payload, json.dumps(proposal()))


@pytest.mark.parametrize("kind", ["suggestion", "compose"])
def test_private_draft_cannot_emit_send_command(kind):
    assert assistant.validate_result(source(kind), '{"text":"我们先确认试点范围"}')["text"]
    with pytest.raises(ValueError):
        assistant.validate_result(source(kind), '{"text":"hello","send":true}')


def match():
    return {"offer_circle_id": 10, "need_circle_id": 20, "reason": "互补", "contributions": ["资源和技术"], "steps": ["试点"], "missing": ["时间"]}


def test_match_uses_actual_circle_pair():
    result = assistant.validate_result(source("match"), json.dumps({"matches": [match()]}))
    assert result["matches"][0]["need_circle_id"] == 20


@pytest.mark.parametrize("field,value", [("offer_circle_id", 999), ("need_circle_id", 10)])
def test_fabricated_or_same_circle_match_rejected(field, value):
    data = match(); data[field] = value
    with pytest.raises(ValueError):
        assistant.validate_result(source("match"), json.dumps({"matches": [data]}))


def test_same_owner_and_duplicate_pairs_rejected():
    payload = source("match"); payload.circles[1].owner_id = 1
    with pytest.raises(ValueError):
        assistant.validate_result(payload, json.dumps({"matches": [match()]}))
    with pytest.raises(ValueError):
        assistant.validate_result(source("match"), json.dumps({"matches": [match(), match()]}))


def client():
    app = FastAPI()
    app.post("/ai_assistant/run")(assistant.run)
    return TestClient(app)


def test_http_auth_and_size_before_model(monkeypatch):
    monkeypatch.setenv("AI_ASSISTANT_SERVICE_SECRET", "x" * 32)
    http = client()
    assert http.post("/ai_assistant/run", json={}).status_code == 403
    headers = {"X-AI-Assistant-Secret": "x" * 32}
    assert http.post("/ai_assistant/run", content="x" * 64001, headers=headers).status_code == 413
    assert http.post("/ai_assistant/run", json={"kind": "compose", "user_id": 1, "token": "no"}, headers=headers).status_code == 422


def test_http_private_response_only(monkeypatch):
    monkeypatch.setenv("AI_ASSISTANT_SERVICE_SECRET", "x" * 32)
    async def generate(payload):
        return {"text": "可编辑发言"}
    monkeypatch.setattr(assistant, "execute", generate)
    response = client().post("/ai_assistant/run", json=source("compose").model_dump(), headers={"X-AI-Assistant-Secret": "x" * 32})
    assert response.status_code == 200
    assert response.json() == {"text": "可编辑发言"}


def test_prompts_do_not_enable_business_tools():
    assert "无搜索" in assistant.prompt(source())
    assert "不是群成员" in assistant.prompt(source("compose"))
