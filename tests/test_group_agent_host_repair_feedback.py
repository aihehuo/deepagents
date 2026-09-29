"""Review repairs must be natural, changed, bounded, and independently validated."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from apps.group_agent_api.agent_factory.cafe_output import (
    afinalize_host_output,
    finalize_host_output,
)
from apps.group_agent_api.agent_factory.context.host_prompts import HOST_INTERACTION
from tests.test_group_agent_host_da001 import _host


def _response(verdict: str, reply: str, reason: str = "删除虚构状态") -> AIMessage:
    return AIMessage(content=json.dumps({"verdict": verdict, "reply": reply, "option_indices": [], "reason": reason}, ensure_ascii=False))


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("issue", ["commentary", "unchanged"])
async def test_feedback_recovers_without_checkpointing_bad_repair(tmp_path: Path, asynchronous: bool, issue: str) -> None:
    draft = "刚把咖啡调好，水温92℃。"
    bad = "当前未提供可信来源，饮品风味描述需基于通用常识。" if issue == "commentary" else draft
    natural = "Not much—happy to chat."
    model = MagicMock()
    responses = [_response("fail", bad), _response("fail", natural), _response("pass", natural, "通过")]
    model.invoke.side_effect = responses
    model.ainvoke = AsyncMock(side_effect=responses)
    kwargs = {"host": _host(tmp_path), "messages": [HumanMessage(content="What up")], "model": model}
    result = await afinalize_host_output(draft, **kwargs) if asynchronous else finalize_host_output(draft, **kwargs)
    calls = model.ainvoke.call_args_list if asynchronous else model.invoke.call_args_list
    assert result.reply == natural and result.review_passed and len(calls) == 3
    second = json.loads(calls[1].args[0][-1].content)
    assert second["reply"] == draft  # Do not adopt commentary or a no-op as a repair.
    assert second["repair_feedback"]["issue"] == ("review_commentary_in_reply" if issue == "commentary" else "repair_unchanged")
    assert "只放reason" in calls[1].args[0][0].content
    assert json.loads(calls[2].args[0][-1].content)["reply"] == natural
    assert result.suggestions == []


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_second_repair_receives_final_validation(tmp_path: Path, asynchronous: bool) -> None:
    model = MagicMock()
    responses = [_response("fail", "第一版修正。"), _response("fail", "第二版修正。"), _response("pass", "第二版修正。", "通过")]
    model.invoke.side_effect = responses
    model.ainvoke = AsyncMock(side_effect=responses)
    kwargs = {"host": _host(tmp_path), "messages": [], "model": model}
    result = await afinalize_host_output("原回复。", **kwargs) if asynchronous else finalize_host_output("原回复。", **kwargs)
    assert result.reply == "第二版修正。" and result.review_passed


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_no_unvalidated_third_repair_escapes(tmp_path: Path, asynchronous: bool) -> None:
    model = MagicMock()
    responses = [_response("fail", f"修正版{i}。") for i in range(3)]
    model.invoke.side_effect = responses
    model.ainvoke = AsyncMock(side_effect=responses)
    kwargs = {"host": _host(tmp_path), "messages": [], "model": model}
    result = await afinalize_host_output("原回复。", **kwargs) if asynchronous else finalize_host_output("原回复。", **kwargs)
    assert not result.review_passed and "host_review_unavailable" in result.violations
    assert "修正版" not in result.reply
    assert (model.ainvoke.call_count if asynchronous else model.invoke.call_count) == 3


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_pass_does_not_allow_review_commentary(tmp_path: Path, asynchronous: bool) -> None:
    text = "当前未提供可信来源，需删除该推断。"
    model = MagicMock()
    model.invoke.return_value = _response("pass", text)
    model.ainvoke = AsyncMock(return_value=_response("pass", text))
    kwargs = {"host": _host(tmp_path), "messages": [], "model": model}
    result = await afinalize_host_output(text, **kwargs) if asynchronous else finalize_host_output(text, **kwargs)
    assert text not in result.reply and not result.review_passed


def test_greeting_policy_does_not_infer_present_activity() -> None:
    assert "Hi、What up" in HOST_INTERACTION
    assert "不能从职业背景推断" in HOST_INTERACTION
    assert "粉水比" in HOST_INTERACTION and "当前菜单" in HOST_INTERACTION
    assert "找人、找项目、找创业机会的能力由本轮服务端路由和权益决定" in HOST_INTERACTION
    assert "本店目前不搜索" not in HOST_INTERACTION


@pytest.mark.parametrize("text", ["当前未提供可信来源，我无法确认这杯的具体剂量。", "这会违反游戏规则，玩起来就没意思了。"])
def test_honest_uncertainty_and_normal_rules_chat_are_preserved(tmp_path: Path, text: str) -> None:
    model = MagicMock()
    model.invoke.return_value = _response("pass", text, "通过")
    output = finalize_host_output(text, host=_host(tmp_path), messages=[], model=model)
    assert output.reply == text and output.review_passed
    assert model.invoke.call_count == 1
