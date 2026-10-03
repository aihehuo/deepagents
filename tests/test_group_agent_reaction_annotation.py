"""Separate-model reaction annotation, including an opt-in real Qwen test."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from apps.group_agent_api.agent_factory.cafe_agent import create_cafe_agent
from apps.group_agent_api.agent_factory.cafe_reaction_annotation import (
    _candidates,
    aannotate_reactions,
    annotate_reactions,
)
from apps.group_agent_api.agent_factory.cafe_reaction_locator import locate_reaction_quote
from apps.group_agent_api.agent_factory.host_context import active_host_context
from tests.test_group_agent_host_da001 import _host
from tests.test_group_agent_host_da002 import ChatModel

_REPLY = (
    "我推开那扇没挂牌的门时，其实有点迟疑。不是怕贵，是怕一进门就要笑得得体。"
    "屋里只亮一盏灯，光晕刚好罩住灶台。我坐在靠窗的位置，终于不必急着解释自己。"
    "那一刻我意识到，今天来这里吃饭就够了，不需要把这顿饭变成一次社交任务。"
)
_QUOTE = "今天来这里吃饭就够了，不需要把这顿饭变成一次社交任务。"
_LONG_REPLY = (
    "我推开那扇没挂牌的门时，第一反应是放慢脚步。门里没有响亮的欢迎词，只有砂锅轻轻咕嘟的声音，"
    "让我觉得不必马上解释自己是谁、来这里要做什么。\n\n"
    "我在窗边坐下，手里还握着手机。桌上只有一张写着今日汤品的小纸条，没有催促我下单的提示，"
    "也没有让人立刻加入聊天的话题。我开始注意到，这种留白本身就是一种招待。\n\n"
    "汤端上来时，我先闻到一点桂花的味道。旁边的人谈着各自今天遇到的小事，我听了一会儿，"
    "才发现自己已经把手机放到一边，也不再想着应该说出什么有趣的话。\n\n"
    "这一刻最打动我的，并不是一项特别的服务，而是我可以安静地吃饭，也可以在想说话时加入大家。"
    "今天来这里吃饭就够了，不需要把这顿饭变成一次社交任务。\n\n"
    "离开时，我会记住那盏灯和桌上的热气。下次再推开门，我希望自己仍能按自己的节奏坐下来，"
    "而不是为了完成一套固定的体验流程。"
)
_MARKDOWN_REPLY = (
    "### 从创业者角度看这个餐饮产品\n"
    "**核心价值：替独居者减少每天晚餐前的选择负担。**\n\n"
    "| 模块 | 用户得到什么 | 如何验证 |\n"
    "|------|--------------|----------|\n"
    "| 餐食 | 一份热饭<br>不用自己备菜 | 看一周内愿意再次购买的人数 |\n\n"
    "- 不要先租大店，而是先用小规模试卖验证需求。\n"
    "- 菜单不必丰富，先把一道主菜的口味和出餐速度做稳定。\n"
    "我们交付的不只是食物，还包括更省心的晚餐决定。"
)


class AnnotationModel:
    """Record the isolated input and return bounded synthetic JSON responses."""

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = iter(outputs)
        self.calls: list[tuple[list[Any], dict[str, Any]]] = []

    def invoke(self, messages: list[Any], **kwargs: Any) -> AIMessage:
        self.calls.append((messages, kwargs))
        return AIMessage(content=next(self.outputs))

    async def ainvoke(self, messages: list[Any], **kwargs: Any) -> AIMessage:
        return self.invoke(messages, **kwargs)


def test_separate_annotation_receives_only_final_reply() -> None:
    chosen = next(item for item in _candidates(_REPLY) if item["quote"] == _QUOTE)
    model = AnnotationModel([json.dumps({"ids": [chosen["id"]]}, ensure_ascii=False)])
    result = annotate_reactions(_REPLY, model)
    assert result.status == "annotated"
    assert result.spans == [{"id": "span_1", "quote": _QUOTE}]
    messages, kwargs = model.calls[0]
    assert len(messages) == 2
    input_payload = json.loads(messages[1].content)
    assert input_payload["reply"] == _REPLY
    assert {item["id"]: item["quote"] for item in input_payload["candidates"]}[chosen["id"]] == _QUOTE
    assert kwargs["tools"] == [] and kwargs["tool_choice"] == "none"


def test_unknown_id_does_not_trigger_another_model_call() -> None:
    model = AnnotationModel([
        json.dumps({"ids": ["not_a_candidate"]}, ensure_ascii=False),
    ])
    result = annotate_reactions(_REPLY, model)
    assert result.status == "invalid_ids" and len(model.calls) == 1
    assert result.spans == []


def test_empty_and_sensitive_choices_do_not_invent_annotations() -> None:
    empty = AnnotationModel(['{"ids":[]}'])
    assert annotate_reactions(_REPLY, empty).status == "model_empty"
    assert len(empty.calls) == 1
    sensitive_reply = _REPLY + " 请联系 guest@example.test 获取详情。"
    assert all("guest@example.test" not in item["quote"] for item in _candidates(sensitive_reply))
    sensitive = AnnotationModel(['{"ids":["c999"]}'])
    assert annotate_reactions(sensitive_reply, sensitive).status == "invalid_ids"
    assert len(sensitive.calls) == 1
    short = AnnotationModel(['{"ids":[]}'])
    assert annotate_reactions("你好。", short).status == "skipped"
    assert not short.calls


def test_valid_model_ids_survive_invalid_or_duplicate_neighbors() -> None:
    chosen = next(item for item in _candidates(_REPLY) if item["quote"] == _QUOTE)
    overlapping = AnnotationModel([
        json.dumps({"ids": ["not_a_candidate", chosen["id"], chosen["id"]]}, ensure_ascii=False)
    ])
    result = annotate_reactions(_REPLY, overlapping)
    assert result.status == "annotated_partial"
    assert result.spans == [{"id": "span_1", "quote": _QUOTE}]
    assert len(overlapping.calls) == 1


def test_typography_difference_maps_to_unique_original_span() -> None:
    reply = "我们聊的是：**这种留白，本身就是一种招待！**\n\n后面再说别的。"
    model_quote = "这种留白本身就是一种招待。"
    location = locate_reaction_quote(reply, model_quote)
    assert location is not None
    assert reply[location[0]:location[1]] == "这种留白，本身就是一种招待"


def test_two_anchors_recover_short_omission_as_original_text() -> None:
    reply = "我觉得这种留白本身就是一种很体贴的招待。"
    model_quote = "这种留白就是一种很体贴的招待。"
    location = locate_reaction_quote(reply, model_quote)
    assert location is not None
    assert reply[location[0]:location[1]] == "这种留白本身就是一种很体贴的招待"


def test_markdown_table_reply_exposes_unique_prose_candidates() -> None:
    reply = (
        "### 这是一个餐饮产品\n"
        "**核心功能：帮独居者在三十分钟内减少决策负担。**\n"
        "| 模块 | 交付内容 |\n|------|---------|\n"
        "| **空间** | 一张桌子<br>六把椅子 |\n"
        "- 不追求复杂装修，先验证用户是否愿意再来。\n"
        "所以，这个产品要卖的是一顿省心的饭。"
    )
    candidates = _candidates(reply)
    assert any(item["quote"] == "核心功能：帮独居者在三十分钟内减少决策负担。" for item in candidates)
    assert any(item["quote"] == "不追求复杂装修，先验证用户是否愿意再来。" for item in candidates)
    assert all("<br>" not in item["quote"] for item in candidates)
    chosen = next(item for item in candidates if item["quote"].startswith("核心功能"))
    result = annotate_reactions(reply, AnnotationModel([json.dumps({"ids": [chosen["id"]]})]))
    assert result.status == "annotated"
    assert result.spans == [{"id": "span_1", "quote": chosen["quote"]}]


def test_long_reply_candidates_cover_the_end_as_well_as_the_start() -> None:
    reply = "\n".join(
        f"第{index}个建议：先用一周验证顾客是否愿意复购，再决定是否扩大门店。"
        for index in range(100)
    )
    candidates = _candidates(reply)
    assert len(candidates) == 48
    assert candidates[0]["quote"] in reply.splitlines()[0]
    assert candidates[-1]["quote"] in reply.splitlines()[-1]
    assert all(reply[item["start"]:].startswith(item["quote"]) for item in candidates)


@pytest.mark.parametrize(
    "reply,model_quote",
    [
        ("我不想把吃饭变成一项必须完成的任务。", "我想把吃饭变成一项必须完成的任务。"),
        ("这种留白本身就是一种招待。\n\n这种留白本身就是一种招待。", "这种留白就是一种招待。"),
        ("这种留白是我愿意留下来的原因。\n\n我希望下次还能来。", "这种留白是我愿意留下来的原因我希望下次还能来。"),
        ("这种留白和" + "很多没有关联的内容" * 5 + "是一种招待。", "这种留白是一种招待。"),
        ("在上海，这种留白本身就是一种招待。", "在深圳，这种留白就是一种招待。"),
    ],
)
def test_anchor_match_rejects_negation_ambiguity_cross_paragraph_long_gap_and_place(
    reply: str, model_quote: str
) -> None:
    assert locate_reaction_quote(reply, model_quote) is None


def test_provider_failure_does_not_block_reply() -> None:

    class FailedModel:
        def invoke(self, messages: list[Any], **kwargs: Any) -> AIMessage:
            raise RuntimeError("synthetic provider outage")

    assert annotate_reactions(_REPLY, FailedModel()).status == "provider_error"


async def test_cafe_graph_adds_separate_model_spans_to_final_reply(tmp_path: Path) -> None:
    core = ChatModel(output=AIMessage(content=_REPLY), captured=[])
    chosen = next(item for item in _candidates(_REPLY) if item["quote"] == _QUOTE)
    annotation = AnnotationModel([json.dumps({"ids": [chosen["id"]]}, ensure_ascii=False)])
    graph = create_cafe_agent(
        model=core,
        reaction_model=annotation,
        checkpointer=InMemorySaver(),
    )
    with active_host_context(_host(tmp_path)):
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="请从顾客视角写一段体验。")]},
            {"configurable": {"thread_id": "reaction-separate-model"}},
        )
    answer = result["messages"][-1]
    assert answer.content == _REPLY
    assert answer.additional_kwargs["cafe_reaction_spans"] == [
        {"id": "span_1", "quote": _QUOTE}
    ]
    assert len(core.captured) == 1 and len(annotation.calls) == 1


@pytest.mark.real_llm
@pytest.mark.integration
async def test_real_qwen_selects_excerpts_from_synthetic_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Opt-in paid test: genuinely call DashScope, never use a model stub."""
    if os.environ.get("GROUP_AGENT_REAL_LLM_TEST") != "1":
        pytest.skip("set GROUP_AGENT_REAL_LLM_TEST=1 to call the real model")
    if not os.environ.get("DASHSCOPE_API_KEY"):
        pytest.fail("DASHSCOPE_API_KEY is required for the opt-in real-model test")
    from langchain_openai import ChatOpenAI

    from apps.group_agent_api.agent_factory.model_builder import create_model

    monkeypatch.setenv("GROUP_AGENT_PROVIDER", "qwen")
    monkeypatch.setenv("GROUP_AGENT_MODEL", "qwen-plus")
    monkeypatch.setenv("GROUP_AGENT_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setenv("GROUP_AGENT_MODEL_MODE", "real")
    model = create_model(timeout_s=20, max_retries=0)
    assert isinstance(model, ChatOpenAI)
    reply = _LONG_REPLY + "\n\n" + _MARKDOWN_REPLY
    result = await aannotate_reactions(reply, model.bind(max_tokens=500, temperature=0))
    assert result.status == "annotated", result.status
    assert 1 <= len(result.spans) <= 5
    assert all(reply.count(span["quote"]) == 1 for span in result.spans)
