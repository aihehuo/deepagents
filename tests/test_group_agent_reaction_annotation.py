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
    aannotate_reactions,
    annotate_reactions,
)
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
    model = AnnotationModel([json.dumps({"quotes": [_QUOTE]}, ensure_ascii=False)])
    result = annotate_reactions(_REPLY, model)
    assert result.status == "annotated"
    assert result.spans == [{"id": "span_1", "quote": _QUOTE}]
    messages, kwargs = model.calls[0]
    assert len(messages) == 2
    assert json.loads(messages[1].content) == {"reply": _REPLY}
    assert kwargs["tools"] == [] and kwargs["tool_choice"] == "none"


def test_invalid_quote_retries_once_without_changing_reply() -> None:
    model = AnnotationModel([
        json.dumps({"quotes": ["这不是原文中的片段。"]}, ensure_ascii=False),
        json.dumps({"quotes": [_QUOTE]}, ensure_ascii=False),
    ])
    result = annotate_reactions(_REPLY, model)
    assert result.status == "annotated" and len(model.calls) == 2
    assert result.spans[0]["quote"] == _QUOTE


def test_empty_and_sensitive_choices_do_not_invent_annotations() -> None:
    empty = AnnotationModel(['{"quotes":[]}'])
    assert annotate_reactions(_REPLY, empty).status == "model_empty"
    assert len(empty.calls) == 1
    sensitive_reply = _REPLY + " 请联系 guest@example.test 获取详情。"
    sensitive = AnnotationModel([
        json.dumps({"quotes": ["请联系 guest@example.test 获取详情。"]}, ensure_ascii=False)
    ] * 2)
    assert annotate_reactions(sensitive_reply, sensitive).status == "invalid_after_retry"
    short = AnnotationModel(['{"quotes":[]}'])
    assert annotate_reactions("你好。", short).status == "skipped"
    assert not short.calls


def test_overlapping_quotes_and_provider_failure_never_block_reply() -> None:
    overlapping = AnnotationModel([
        json.dumps({"quotes": [_QUOTE, "今天来这里吃饭就够了"]}, ensure_ascii=False)
    ] * 2)
    assert annotate_reactions(_REPLY, overlapping).status == "invalid_after_retry"

    class FailedModel:
        def invoke(self, messages: list[Any], **kwargs: Any) -> AIMessage:
            raise RuntimeError("synthetic provider outage")

    assert annotate_reactions(_REPLY, FailedModel()).status == "provider_error"


async def test_cafe_graph_adds_separate_model_spans_to_final_reply(tmp_path: Path) -> None:
    core = ChatModel(output=AIMessage(content=_REPLY), captured=[])
    annotation = AnnotationModel([json.dumps({"quotes": [_QUOTE]}, ensure_ascii=False)])
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
async def test_real_qwen_selects_excerpts_from_synthetic_narrative(monkeypatch: pytest.MonkeyPatch) -> None:
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
    result = await aannotate_reactions(_LONG_REPLY, model.bind(max_tokens=500, temperature=0))
    assert result.status == "annotated", result.status
    assert 1 <= len(result.spans) <= 5
    assert all(_LONG_REPLY.count(span["quote"]) == 1 for span in result.spans)
