"""Core channel parity, identity binding and draft-only WeCom delivery."""
import asyncio
from unittest.mock import AsyncMock
import pytest
from fastapi import HTTPException
from apps.group_agent_api.app.models import AsyncCallRequest
from apps.group_agent_api.app.async_manager import execute_async_run_core
from apps.group_agent_api.app.shared_dialogue import wecom_input
from apps.group_agent_api.app.endpoints.ai_assistant import Input, prompt


def request_data():
    """Create a synthetic, host-owned legacy relay envelope."""
    return dict(run_id="test", idempotency_key="shared-test", user_id="12", unionid="synthetic",
                group_id="global", conversation_id="ga_wecom_12_7", message="legacy prepared context",
                callback_url="http://micro:3001/group_agent_callbacks/test", request_schema_version=1,
                metadata={"source": "wecom_service", "wecom_service_turn_id": 7,
                          "wecom_service_action": "conversation", "wecom_service_shared_input":
                          {"kind": "conversation", "user_id": 12, "messages":
                           [{"id": 8, "role": "user", "author_id": 12, "text": "愿意提供门店"}]}})


def test_same_prompt_and_executor_with_only_delivery_adapted(monkeypatch):
    from apps.group_agent_api.app.endpoints import ai_assistant
    body = request_data()
    wecom = wecom_input(body)
    assert prompt(wecom) == prompt(Input.model_validate(body["metadata"]["wecom_service_shared_input"]))
    execute = AsyncMock(return_value={"reply": "共同核心回复", "profile": {}, "proposals": []})
    monkeypatch.setattr(ai_assistant, "execute", execute)
    callback = AsyncMock()
    asyncio.run(execute_async_run_core(req=AsyncCallRequest.model_validate(body), session=None,
                                      state=None, tid="isolated", emit_callback=callback))
    execute.assert_awaited_once_with(wecom)
    callback.assert_awaited_once_with("final", {"reply": "共同核心回复", "candidates": []})


@pytest.mark.parametrize("change", ["user", "conversation", "source"])
def test_forged_channel_binding_is_rejected(change):
    body = request_data()
    if change == "user":
        body["metadata"]["wecom_service_shared_input"]["user_id"] = 13
    elif change == "conversation":
        body["conversation_id"] = "ga_wecom_group_12_7"
    else:
        body["metadata"]["source"] = "ios_app"
    with pytest.raises(HTTPException):
        AsyncCallRequest.model_validate(body)


def test_greeting_is_explicit_and_not_a_fabricated_customer_utterance():
    body = request_data()
    body["metadata"]["wecom_service_action"] = "greeting"
    body["metadata"]["wecom_service_shared_input"]["purpose"] = "greeting"
    assert "不是用户新发言" in prompt(wecom_input(body))
