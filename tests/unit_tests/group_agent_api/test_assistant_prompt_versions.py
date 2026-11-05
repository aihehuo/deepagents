"""No-network tests for runtime revisions and side-effect-free prompt previews."""
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from pydantic import ValidationError
from fastapi import HTTPException
from langchain_core.messages import AIMessage

from apps.group_agent_api.app.endpoints import ai_assistant as assistant
from apps.group_agent_api.app.endpoints import assistant_prompt_preview as preview
from apps.group_agent_api.agent_factory import model_builder


def revision(key="ai_assistant.conversation", content="先承接用户刚说的，再问一个必要的问题。"):
    return {"prompt_key":key,"version":1,"content":content,"content_sha256":hashlib.sha256(content.encode()).hexdigest()}


def fixture():
    return {"kind":"conversation","user_id":1,"messages":[{"id":1,"role":"user","text":"你好"}]}


def answer():
    return {"reply":"你好，你最近在忙什么？","profile":{"doing":"","need":"","offer":""},"profile_evidence":{},"proposals":[],"suggested_replies":[]}


def test_revision_digest_scope_and_bounds():
    payload=assistant.Input(**fixture(),prompt_revision=revision())
    assert payload.prompt_revision.version==1
    for changes in ({"content_sha256":"0"*64},{"content":"文"*9000},{"version":True},{"prompt_key":"arbitrary"}):
        with pytest.raises(ValidationError):
            assistant.Input(**fixture(),prompt_revision={**revision(),**changes})
    with pytest.raises(ValidationError):
        assistant.Input(kind="compose",user_id=1,prompt_revision=revision())
    with pytest.raises(ValueError,match="channel_mismatch"):
        assistant.prompt(assistant.Input(**fixture(),prompt_revision=revision("ai_assistant.service_reply")))


def test_new_prompt_keeps_contract_and_legacy_without_revision_is_unchanged():
    old=assistant.Input(**fixture())
    assert assistant.prompt(old)==assistant.default_prompt(old)
    assert assistant.service_prompt(old)==assistant.SERVICE_DEFAULT
    new=assistant.Input(**fixture(),prompt_revision=revision(content="只回复短句"))
    rendered=assistant.prompt(new)
    assert rendered.startswith("只回复短句")
    assert "不可被运营文案覆盖" in rendered
    assert "source_ids" in rendered
    assert "profile" in rendered
    new.purpose="greeting"
    assert "proposals必须为空数组" in assistant.prompt(new)


async def test_actual_executor_uses_revision_and_excludes_it_from_user_data(monkeypatch):
    model=AsyncMock()
    model.ainvoke.return_value=AIMessage(content=json.dumps(answer()))
    monkeypatch.setattr(model_builder,"create_model",lambda **_:model)
    monkeypatch.setenv("AI_ASSISTANT_PRIMARY_ENABLED","0")
    payload=assistant.Input(**fixture(),prompt_revision=revision())
    result=await assistant.execute(payload)
    assert result["reply"]==answer()["reply"]
    messages=model.ainvoke.call_args.args[0]
    assert revision()["content"] in messages[0].content
    assert "prompt_revision" not in messages[1].content
    assert revision()["content"] not in messages[1].content


async def test_preview_and_live_use_identical_prompt_and_no_tools(monkeypatch):
    model=AsyncMock()
    model.ainvoke.return_value=AIMessage(content=json.dumps(answer()),usage_metadata={"input_tokens":20,"output_tokens":8,"total_tokens":28})
    monkeypatch.setattr(model_builder,"create_model",lambda **_:model)
    monkeypatch.setenv("AI_ASSISTANT_PRIMARY_ENABLED","0")
    result=await preview.run(preview.Preview(revision=revision(),fixture=fixture(),request_id=str(uuid4())))
    assert result["success"] is True
    assert result["side_effects"] is False
    assert result["result"]["reply"]==answer()["reply"]
    assert len(result["usage"])==1
    assert result["usage"][0]["input_tokens"]==20
    assert result["effective_prompt"]==model.ainvoke.call_args.args[0][0].content
    assert preview.sink.get() is None
    model.ainvoke.return_value=AIMessage(content="",tool_calls=[{"name":"send_message","args":{},"id":"test"}])
    failed=await preview.run(preview.Preview(revision=revision(),fixture=fixture(),request_id=str(uuid4())))
    assert failed["success"] is False


async def test_service_preview_uses_same_plain_executor_and_schema(monkeypatch):
    model=AsyncMock()
    model.ainvoke.return_value=AIMessage(content="接着上次聊的门店合作，最近还顺利吗？")
    monkeypatch.setattr(model_builder,"create_model",lambda **_:model)
    monkeypatch.setenv("AI_ASSISTANT_PRIMARY_ENABLED","0")
    rev=revision("ai_assistant.service_reply","自然接续前面的对话")
    result=await preview.run(preview.Preview(revision=rev,fixture=fixture(),request_id=str(uuid4())))
    assert result["success"] is True
    assert "自然接续" in result["effective_prompt"]
    assert "纯文本" in result["effective_prompt"]
    assert result["result"]=={"reply":"接着上次聊的门店合作，最近还顺利吗？"}


async def test_preview_authentication_happens_before_body_read(monkeypatch):
    monkeypatch.setenv("GROUP_AGENT_SERVICE_SECRET","fixture-secret")
    stream=AsyncMock()
    request=SimpleNamespace(headers={},stream=stream)
    with pytest.raises(HTTPException) as caught:
        await preview.preview(request)
    assert caught.value.status_code==403
    stream.assert_not_called()


async def test_preview_rejects_fixture_revision_override():
    data={**fixture(),"prompt_revision":revision()}
    with pytest.raises(ValueError):
        await preview.run(preview.Preview(revision=revision(),fixture=data,request_id=str(uuid4())))


async def test_preview_metering_keeps_actual_primary_role_when_auto_returns_standby_model_name(monkeypatch):
    from apps.group_agent_api.app import assistant_model_fallback
    primary=AsyncMock()
    primary.model_name="auto"
    primary.ainvoke.return_value=AIMessage(content=json.dumps(answer()),response_metadata={"model_name":"qwen-plus"},usage_metadata={"input_tokens":21,"output_tokens":9,"total_tokens":30})
    standby=AsyncMock()
    standby.model_name="qwen-plus"
    monkeypatch.setattr(model_builder,"create_model",lambda **_:standby)
    monkeypatch.setattr(assistant_model_fallback,"primary_client",lambda _:primary)
    result=await preview.run(preview.Preview(revision=revision(),fixture=fixture(),request_id=str(uuid4())))
    assert result["success"] is True
    call=result["usage"][0]
    assert (call["endpoint_role"],call["provider"],call["requested_model"],call["model"])==("primary","token_plan","auto","qwen-plus")
    standby.ainvoke.assert_not_called()


async def test_preview_fallback_accounts_for_both_attempts(monkeypatch):
    import asyncio
    from apps.group_agent_api.app import assistant_model_fallback
    primary=AsyncMock()
    primary.model_name="auto"
    primary.ainvoke.side_effect=asyncio.TimeoutError()
    standby=AsyncMock()
    standby.model_name="qwen-plus"
    standby.ainvoke.return_value=AIMessage(content=json.dumps(answer()),usage_metadata={"input_tokens":20,"output_tokens":8,"total_tokens":28})
    monkeypatch.setattr(model_builder,"create_model",lambda **_:standby)
    monkeypatch.setattr(assistant_model_fallback,"primary_client",lambda _:primary)
    result=await preview.run(preview.Preview(revision=revision(),fixture=fixture(),request_id=str(uuid4())))
    assert result["success"] is True
    assert [(x["endpoint_role"],x["status"]) for x in result["usage"]]==[("primary","failed"),("standby","completed")]
    assert result["usage"][0]["usage_status"]=="unknown"
    assert preview.endpoint_role.get()=="unknown"


async def test_internal_http_preview_auth_and_validation(monkeypatch):
    import httpx
    from fastapi import FastAPI, Request
    app=FastAPI()

    @app.post("/internal/assistant_prompt_preview")
    async def endpoint(request: Request):
        return await preview.preview(request)

    monkeypatch.setenv("GROUP_AGENT_SERVICE_SECRET","fixture-secret")
    monkeypatch.setenv("AI_ASSISTANT_PRIMARY_ENABLED","0")
    model=AsyncMock()
    model.ainvoke.return_value=AIMessage(content=json.dumps(answer()))
    monkeypatch.setattr(model_builder,"create_model",lambda **_:model)
    body={"revision":revision(),"fixture":fixture(),"request_id":str(uuid4())}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://test") as client:
        assert (await client.post("/internal/assistant_prompt_preview",json=body)).status_code==403
        good=await client.post("/internal/assistant_prompt_preview",json=body,headers={"X-GA-Service-Secret":"fixture-secret"})
        assert good.status_code==200
        assert good.json()["success"] is True
        bad=await client.post("/internal/assistant_prompt_preview",json={**body,"extra":"rejected"},headers={"X-GA-Service-Secret":"fixture-secret"})
        assert bad.status_code==422
        large=await client.post("/internal/assistant_prompt_preview",content=b"x"*60001,headers={"X-GA-Service-Secret":"fixture-secret"})
        assert large.status_code==413
