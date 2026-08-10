import json

from apps.group_agent_api.agent_factory.suggested_replies import (
    extract_suggested_replies,
)


def test_extracts_model_authored_suggestions_and_hides_marker() -> None:
    reply, suggestions = extract_suggested_replies(
        '可以，先选一个方向。\n<suggested_replies>['
        '{"label":"看漏斗","reply":"帮我看最近 7 天的转化漏斗"},'
        '{"label":"看掉队样本","reply":"先分析只聊一句就离开的样本"}'
        "]</suggested_replies>"
    )

    assert reply == "可以，先选一个方向。"
    assert suggestions == [
        {
            "id": "suggested_1",
            "label": "看漏斗",
            "reply": "帮我看最近 7 天的转化漏斗",
        },
        {
            "id": "suggested_2",
            "label": "看掉队样本",
            "reply": "先分析只聊一句就离开的样本",
        },
    ]


def test_invalid_or_single_suggestion_fails_closed() -> None:
    reply, suggestions = extract_suggested_replies(
        '正文<suggested_replies>[{"label":"只有一个","reply":"不显示"}]</suggested_replies>'
    )
    assert reply == "正文"
    assert suggestions == []


def test_caps_at_four() -> None:
    items = [
        {"label": f"选项 {index}", "reply": f"回复 {index}"}
        for index in range(1, 7)
    ]
    reply, suggestions = extract_suggested_replies(
        f"正文<suggested_replies>{json.dumps(items)}</suggested_replies>"
    )
    assert reply == "正文"
    assert len(suggestions) == 4
