"""Formatting is not a credential; preserve safe prose through real guards."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from apps.group_agent_api.agent_factory.cafe_output import afinalize_host_output, finalize_host_output
from apps.group_agent_api.agent_factory.cafe_agent import create_cafe_agent
from apps.group_agent_api.agent_factory.host_context import active_host_context
from apps.group_agent_api.agent_factory.venue_greeting import validate_greeting_reply
from tests.test_group_agent_host_da001 import _host
from tests.test_group_agent_host_da002 import ChatModel
from tests.test_group_agent_host_da003 import Reviewer


@pytest.mark.parametrize("text", ["坐会儿就好，  不用急。", "不聊项目也很好。  \n\n想安静一点，就慢慢坐。", "两个名字：\n  1. 旧歌未央\n  2. 留声小馆", "清爽一些，少甜一些，，就像一个轻松的下午。"])
async def test_formatting_survives_sync_async_semantic_review(tmp_path: Path, text: str) -> None:
    model = MagicMock()
    response = AIMessage(content=json.dumps({"verdict": "pass", "reply": text, "option_indices": []}, ensure_ascii=False))
    model.invoke.return_value = response
    model.ainvoke = AsyncMock(return_value=response)
    sync = finalize_host_output(text, host=_host(tmp_path), messages=[], model=model)
    asynchronous = await afinalize_host_output(text, host=_host(tmp_path), messages=[], model=model)
    assert sync == asynchronous
    assert sync.reply == text and not sync.violations and sync.review_passed
    sent = json.loads(model.invoke.call_args.args[0][-1].content)
    assert sent["reply"] == text


@pytest.mark.parametrize("text", ["邮箱 founder@example.test", "电话 13812345678", "微信号:synthetic_handle", "api_key=sk-TEST_SYNTHETIC_SECRET", "\x00欢迎"])
def test_real_sensitive_content_is_still_removed_before_review(tmp_path: Path, text: str) -> None:
    result = finalize_host_output(text, host=_host(tmp_path), messages=[])
    assert "host_sensitive_text" in result.violations
    assert text not in result.reply


def test_greeting_accepts_harmless_spacing(tmp_path: Path) -> None:
    payload = {"result_kind": "greeting", "reply": "欢迎，  坐会儿吧。", "suggested_replies": []}
    assert validate_greeting_reply(payload, host=_host(tmp_path)) == payload


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_malformed_review_gets_one_format_retry(tmp_path: Path, asynchronous: bool) -> None:
    text = "只是坐坐也很好。"
    responses = [AIMessage(content='{"verdict":"pass"}'), AIMessage(content=json.dumps({"verdict":"pass", "reply":text, "option_indices":[]}))]
    model = MagicMock()
    model.invoke.side_effect = responses
    model.ainvoke = AsyncMock(side_effect=responses)
    result = await afinalize_host_output(text, host=_host(tmp_path), messages=[], model=model) if asynchronous else finalize_host_output(text, host=_host(tmp_path), messages=[], model=model)
    calls = model.ainvoke.call_args_list if asynchronous else model.invoke.call_args_list
    assert len(calls) == 2 and result.review_passed and result.reply == text
    assert "格式" in calls[1].args[0][0].content


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_persistent_invalid_review_is_bounded_and_fails_closed(tmp_path: Path, asynchronous: bool) -> None:
    model = MagicMock()
    invalid = AIMessage(content='{"verdict":"pass"}')
    model.invoke.return_value = invalid
    model.ainvoke = AsyncMock(return_value=invalid)
    result = await afinalize_host_output("原文", host=_host(tmp_path), messages=[], model=model) if asynchronous else finalize_host_output("原文", host=_host(tmp_path), messages=[], model=model)
    assert (model.ainvoke.await_count if asynchronous else model.invoke.call_count) == 2
    assert not result.review_passed and "host_review_unavailable" in result.violations


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_format_retry_cannot_allow_sensitive_rewrite(tmp_path: Path, asynchronous: bool) -> None:
    model = MagicMock()
    invalid = AIMessage(content=json.dumps({"verdict":"fail", "reply":"邮箱 founder@example.test", "option_indices":[]}))
    model.invoke.return_value = invalid
    model.ainvoke = AsyncMock(return_value=invalid)
    result = await afinalize_host_output("聊聊咖啡", host=_host(tmp_path), messages=[], model=model) if asynchronous else finalize_host_output("聊聊咖啡", host=_host(tmp_path), messages=[], model=model)
    assert "founder@example.test" not in result.reply and not result.review_passed


async def test_formatted_chat_commits_exact_validated_reply(tmp_path: Path) -> None:
    text = "两个名字：  \n  1. 旧歌未央\n  2. 留声小馆"
    graph = create_cafe_agent(model=ChatModel(output=AIMessage(content=text)), checkpointer=InMemorySaver(), quality_model=Reviewer(captured=[]))
    config = {"configurable": {"thread_id": "formatting-regression"}}
    with active_host_context(_host(tmp_path)):
        result = await graph.ainvoke({"messages": [HumanMessage(content="取两个名字就好")]}, config)
    assert result["messages"][-1].content == text
    checkpoint = await graph.aget_state(config)
    assert checkpoint.values["messages"][-1].content == text
