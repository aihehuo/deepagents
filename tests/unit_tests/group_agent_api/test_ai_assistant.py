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


def test_user_options_keep_user_voice_and_drop_ai_service_promises():
    data = proposal()
    data['suggested_replies'] = [
        {'label': '分析合作方案', 'reply': '我来帮你分析合作方案。'},
        {'label': '明确合作方式', 'reply': '我想先明确合作方式。'},
        {'label': '分析合作方案', 'reply': '请帮我分析合作方案。'},
    ]
    result = assistant.validate_result(source(), json.dumps(data))
    assert result['reply'] == data['reply']
    assert result['suggested_replies'] == data['suggested_replies'][1:]
    assert '说话人绝不是AI' in assistant.prompt(source())


@pytest.mark.parametrize('reply', ['我可以提供创业者资源。', '请帮我找合作伙伴。', '我想先了解匹配方式。'])
def test_user_resources_and_requests_are_not_treated_as_ai_voice(reply):
    from apps.group_agent_api.agent_factory.suggested_replies import is_assistant_voice_option
    assert not is_assistant_voice_option({'label': '进一步聊聊', 'reply': reply})


@pytest.mark.parametrize('reply', ['我可以为你匹配合适的人。', '请告诉我你的需求。', 'I can help you find a partner.'])
def test_obvious_ai_voice_is_identified(reply):
    from apps.group_agent_api.agent_factory.suggested_replies import is_assistant_voice_option
    assert is_assistant_voice_option({'label': '进一步聊聊', 'reply': reply})


@pytest.mark.parametrize("purpose", ["greeting", "attachment"])
@pytest.mark.parametrize("proposals", [proposal()["proposals"], [{"source_ids": [999]}]])
async def test_staff_reply_survives_forbidden_optional_extraction(monkeypatch: pytest.MonkeyPatch, purpose: str, proposals: list[dict]) -> None:
    """A non-utterance must deliver the reply without creating user facts."""
    from unittest.mock import AsyncMock
    from langchain_core.messages import AIMessage
    from apps.group_agent_api.agent_factory import model_builder

    payload = source().model_copy(update={"purpose": purpose})
    data = proposal()
    data["reply"] = "产品目前有样品和供应商报价了吗？"
    data["proposals"] = proposals
    data["profile_evidence"] = {"need": "malformed historical extraction"}
    model = AsyncMock()
    model.ainvoke.return_value = AIMessage(content=json.dumps(data))
    monkeypatch.setattr(model_builder, "create_model", lambda **kwargs: model)

    result = await assistant.execute(payload)
    assert result["reply"] == data["reply"]
    assert result["proposals"] == []
    assert result["profile"] == {"doing": "", "need": "", "offer": ""}
    assert result["profile_evidence"] == {}


@pytest.mark.parametrize("purpose", ["greeting", "attachment"])
def test_non_utterance_validator_remains_strict(purpose: str) -> None:
    """Callers cannot bypass provenance rules through the raw validator."""
    payload = source().model_copy(update={"purpose": purpose})
    data = proposal()
    data["profile"] = {"doing": "", "need": "", "offer": ""}
    with pytest.raises(ValueError, match="non_utterance_cannot_create_facts"):
        assistant.validate_result(payload, json.dumps(data))


def test_circle_prompt_uses_host_auto_activation_and_keeps_source_boundaries() -> None:
    """The host starts permitted new circles; extraction still excludes quoted/private data."""
    text = assistant.prompt(source())
    assert "自动开启招募" in text
    assert "无需用户确认草稿" in text
    assert "最终创建和启动结果由服务端决定" in text
    assert "示例、假设、引用、不要公开的内容不得变成圈子" in text
    assert "用户核对后才生效" not in text
