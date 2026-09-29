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
from apps.group_agent_api.agent_factory.context.host_prompts import (
    HOST_IDENTITY,
    HOST_INTERACTION,
)
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
async def test_final_safe_repair_avoids_a_fourth_model_call(tmp_path: Path, asynchronous: bool) -> None:
    model = MagicMock()
    responses = [_response("fail", f"修正版{i}。") for i in range(3)]
    model.invoke.side_effect = responses
    model.ainvoke = AsyncMock(side_effect=responses)
    kwargs = {"host": _host(tmp_path), "messages": [], "model": model}
    result = await afinalize_host_output("原回复。", **kwargs) if asynchronous else finalize_host_output("原回复。", **kwargs)
    assert result.reply == "修正版2。" and result.review_passed
    assert "host_review_final_repair" in result.violations
    assert "host_review_unavailable" not in result.violations
    assert (model.ainvoke.call_count if asynchronous else model.invoke.call_count) == 3


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_final_noop_failure_preserves_verified_repair(
    tmp_path: Path, asynchronous: bool
) -> None:
    """A stale final fail must not erase an already reviewed safe rewrite."""
    draft = "建议草稿。"
    repaired = "先明确目标顾客和消费场景，再对比周边门店与价格带。"
    model = MagicMock()
    responses = [
        _response("fail", draft, "需要实际修正"),
        _response("fail", repaired, "已删除未经证实的细节"),
        _response("fail", repaired + "  \n", "重复上轮已删除的问题"),
    ]
    model.invoke.side_effect = responses
    model.ainvoke = AsyncMock(side_effect=responses)
    kwargs = {
        "host": _host(tmp_path),
        "messages": [HumanMessage(content="餐饮品牌定位应该怎么做？")],
        "model": model,
    }
    result = (
        await afinalize_host_output(draft, **kwargs)
        if asynchronous
        else finalize_host_output(draft, **kwargs)
    )
    assert result.reply == repaired
    assert result.review_passed
    assert "host_review_converged" in result.violations
    assert "host_review_unavailable" not in result.violations
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
    assert "市场调研、赛道选择或品牌定位" in HOST_IDENTITY
    assert "不得因“不是顾问”" in HOST_IDENTITY
    assert "开店大概需要多少投入" in HOST_IDENTITY
    assert "禁止只回复“不能确认”" in HOST_IDENTITY
    assert "不得用旧 doing/need/offer 重定向回答" in HOST_INTERACTION


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_reviewer_timeout_preserves_safe_cost_estimate(
    tmp_path: Path, asynchronous: bool
) -> None:
    text = (
        "按80平方米社区咖啡店粗估，前期投入约35万到60万元："
        "租押金、装修、设备、首批物料和三个月周转金都要算进去；"
        "嘉兴具体租金仍需按商圈和铺位核实。"
    )
    model = MagicMock()
    model.invoke.side_effect = TimeoutError("review timed out")
    model.ainvoke = AsyncMock(side_effect=TimeoutError("review timed out"))
    kwargs = {
        "host": _host(tmp_path),
        "messages": [HumanMessage(content="在嘉兴这样的城市大概需要多少投入？")],
        "model": model,
    }

    result = (
        await afinalize_host_output(text, **kwargs)
        if asynchronous
        else finalize_host_output(text, **kwargs)
    )

    assert result.reply == text
    assert not result.review_passed
    assert result.suggestions == []
    assert "host_review_degraded" in result.violations
    assert "host_review_unavailable" not in result.violations


def test_reviewer_timeout_never_restores_deterministically_blocked_text(
    tmp_path: Path,
) -> None:
    model = MagicMock()
    model.invoke.side_effect = TimeoutError("review timed out")

    result = finalize_host_output(
        "请联系 founder@example.test，api_key=sk-TEST_SYNTHETIC_SECRET",
        host=_host(tmp_path),
        messages=[],
        model=model,
    )

    assert "founder@example.test" not in result.reply
    assert "sk-TEST_SYNTHETIC_SECRET" not in result.reply
    assert "host_sensitive_text" in result.violations
    assert "host_review_degraded" in result.violations


@pytest.mark.parametrize("text", ["当前未提供可信来源，我无法确认这杯的具体剂量。", "这会违反游戏规则，玩起来就没意思了。"])
def test_honest_uncertainty_and_normal_rules_chat_are_preserved(tmp_path: Path, text: str) -> None:
    model = MagicMock()
    model.invoke.return_value = _response("pass", text, "通过")
    output = finalize_host_output(text, host=_host(tmp_path), messages=[], model=model)
    assert output.reply == text and output.review_passed
    assert model.invoke.call_count == 1
