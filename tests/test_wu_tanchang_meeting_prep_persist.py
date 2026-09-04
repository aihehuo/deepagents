"""Tests for post-turn meeting-prep auto-persist."""

# ruff: noqa: S101

from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from apps.wu_tanchang_api.agent_factory.meeting_prep_persist import (
    body_looks_complete,
    clear_thread_prep_saved,
    ensure_meeting_prep_persisted,
    extract_prep_body_from_messages,
    mark_thread_prep_saved,
    thread_prep_saved,
    turn_has_successful_save,
)


COMPLETE_PREP = """【会议准备 · 供面谈参考】

一、交谈要点
- 城市：成都
- 赛道：茶饮
- 阶段：已开店 / 瓶颈期
- 问题：位置偏、新客难拉，午市客流不稳
- 目标：优化单店模型与引流，把复购做扎实
- 计划：试过抖音与团购，效果一般，想找更稳的截流打法

二、初步建议
先拆菜单利润结构，再定周边客群截流动作；面谈时优先对齐「客群—菜单—位置」三角，而不是先谈装修档次。
补充：把现有团购单品换成高毛利引流款，再配一门店周边写字楼午市动作。

三、建议讨论话题
1. 菜单工程：哪些单品该砍、哪些该做招牌
2. 目标客群：写字楼午市 vs 社区晚餐，哪边更该先打透
3. 品牌叙事：门店话术如何支撑复购，而不是只拼折扣

四、可参考案例
茶饮店：老客复购强、外卖见顶时改做定点截流；对照你的写字楼半径，优先验证午市套餐与到店动线，而不是继续加投放预算。
"""
assert len(COMPLETE_PREP) >= 320


def test_extract_and_complete_detection() -> None:
    msgs = [
        HumanMessage(content="够了，出材料"),
        AIMessage(content=f"好的。\n\n{COMPLETE_PREP}\n\n要不要约谈？"),
    ]
    body = extract_prep_body_from_messages(msgs)
    assert body is not None
    assert body.startswith("【会议准备")
    assert body_looks_complete(body)


def test_turn_has_successful_save() -> None:
    assert not turn_has_successful_save([AIMessage(content=COMPLETE_PREP)])
    assert turn_has_successful_save(
        [
            ToolMessage(
                content="meeting_prep_saved",
                name="save_meeting_prep",
                tool_call_id="1",
            )
        ]
    )


def test_ensure_skips_when_already_saved() -> None:
    msgs = [
        AIMessage(content=COMPLETE_PREP),
        ToolMessage(
            content="meeting_prep_saved",
            name="save_meeting_prep",
            tool_call_id="1",
        ),
    ]
    with patch(
        "apps.wu_tanchang_api.agent_factory.agent.save_meeting_prep"
    ) as mock_tool:
        out = ensure_meeting_prep_persisted(
            new_messages=msgs,
            config={"configurable": {"thread_id": "t1"}, "metadata": {}},
        )
        assert out is None
        mock_tool.invoke.assert_not_called()


def test_ensure_auto_invokes_save_when_model_skipped_tool() -> None:
    clear_thread_prep_saved("t-auto")
    msgs = [AIMessage(content=f"材料如下：\n\n{COMPLETE_PREP}")]
    mock_tool = MagicMock()
    mock_tool.invoke.return_value = "meeting_prep_saved"
    config = {
        "configurable": {"thread_id": "t-auto"},
        "metadata": {"user_id": "9", "callback_url": "http://example/wu_tanchang_callbacks/x"},
    }
    with (
        patch(
            "apps.wu_tanchang_api.agent_factory.agent.save_meeting_prep",
            mock_tool,
        ),
        patch(
            "apps.wu_tanchang_api.agent_factory.meeting_prep_persist.meeting_prep_policy_for_run",
            return_value=(("一、", "二、", "三、", "四、"), ()),
        ),
    ):
        out = ensure_meeting_prep_persisted(new_messages=msgs, config=config)
    assert out == "meeting_prep_saved"
    mock_tool.invoke.assert_called_once()
    args = mock_tool.invoke.call_args
    assert "【会议准备" in args.args[0]["body"]
    assert thread_prep_saved("t-auto")
    clear_thread_prep_saved("t-auto")


def test_ensure_skips_incomplete_body() -> None:
    msgs = [AIMessage(content="【会议准备】\n\n一、交谈要点\n只有一节")]
    with patch(
        "apps.wu_tanchang_api.agent_factory.agent.save_meeting_prep"
    ) as mock_tool:
        out = ensure_meeting_prep_persisted(
            new_messages=msgs,
            config={"configurable": {"thread_id": "t2"}, "metadata": {}},
        )
        assert out is None
        mock_tool.invoke.assert_not_called()


def test_thread_prep_flag() -> None:
    clear_thread_prep_saved("flag-1")
    assert not thread_prep_saved("flag-1")
    mark_thread_prep_saved("flag-1")
    assert thread_prep_saved("flag-1")
    clear_thread_prep_saved("flag-1")
