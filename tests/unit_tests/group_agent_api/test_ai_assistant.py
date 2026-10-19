"""No-network tests for private assistant output and internal HTTP admission."""
import json

import pytest

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


def test_prompts_do_not_enable_business_tools():
    assert "无搜索" in assistant.prompt(source())
    assert "不是群成员" in assistant.prompt(source("compose"))


def test_profile_updates_need_current_user_quotes():
    data = proposal()
    data["profile_evidence"] = {"offer": [{"message_id": 1, "quote": "门店"}]}
    assert assistant.validate_result(source(), json.dumps(data))["profile_evidence"]["offer"][0]["quote"] == "门店"
    data["profile_evidence"]["offer"][0]["quote"] = "原话里没有的内容"
    with pytest.raises(ValueError):
        assistant.validate_result(source(), json.dumps(data))


def test_old_user_message_cannot_prove_this_turn_profile_change():
    payload = source()
    payload.messages.append(assistant.Message(id=2, role="user", text="你好"))
    data = proposal()
    data["profile_evidence"] = {"offer": [{"message_id": 1, "quote": "门店"}]}
    with pytest.raises(ValueError):
        assistant.validate_result(payload, json.dumps(data))


def test_no_profile_evidence_means_no_automatic_update_authority():
    result = assistant.validate_result(source(), json.dumps(proposal()))
    assert result["profile_evidence"] == {}

@pytest.mark.asyncio
async def test_execution_ignores_obsolete_profile_but_keeps_circle_evidence_strict(monkeypatch):
    from unittest.mock import AsyncMock
    from langchain_core.messages import AIMessage
    from apps.group_agent_api.agent_factory import model_builder
    data = proposal()
    data['profile_evidence'] = {'offer': 'obsolete malformed extraction'}
    model = AsyncMock()
    model.ainvoke.return_value = AIMessage(content=json.dumps(data))
    monkeypatch.setattr(model_builder, 'create_model', lambda **kwargs: model)
    result = await assistant.execute(source())
    assert result['profile'] == {'doing': '', 'need': '', 'offer': ''}
    assert result['profile_evidence'] == {}
    assert result['proposals'][0]['source_ids'] == [1]
    data['proposals'][0]['source_ids'] = [999]
    model.ainvoke.return_value = AIMessage(content=json.dumps(data))
    with pytest.raises(ValueError):
        await assistant.execute(source())


def test_optional_reply_options_reuse_group_agent_structure():
    data = proposal()
    data['suggested_replies'] = [
        {'label': '聊合作方式', 'reply': '我想先聊聊合作方式。'},
        {'label': '补充需求', 'reply': '我想先补充需求。'},
    ]
    result = assistant.validate_result(source(), json.dumps(data))
    assert result['suggested_replies'] == data['suggested_replies']
    assert 'suggested_replies' in assistant.prompt(source())
    assert assistant.validate_result(source(), json.dumps(proposal()))['suggested_replies'] == []


def test_invalid_optional_options_never_drop_the_answer():
    data = proposal()
    data['suggested_replies'] = {'tool': 'delete'}
    result = assistant.validate_result(source(), json.dumps(data))
    assert result['reply'] == data['reply']
    assert result['suggested_replies'] == []
