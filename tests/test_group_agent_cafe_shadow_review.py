"""Shadow semantic review is sampled, redacted, and never authoritative."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from apps.group_agent_api.agent_factory.cafe_agent import create_cafe_agent
from apps.group_agent_api.agent_factory.cafe_output import HostOutput
from apps.group_agent_api.agent_factory.cafe_shadow_review import (
    schedule_cafe_shadow_review,
)
from apps.group_agent_api.agent_factory.host_context import active_host_context
from tests.test_group_agent_host_da001 import _host
from tests.test_group_agent_host_da002 import ChatModel


def _enable(monkeypatch) -> None:
    monkeypatch.setenv("GROUP_AGENT_CAFE_SHADOW_REVIEW_ENABLED", "1")
    monkeypatch.setenv("GROUP_AGENT_CAFE_SHADOW_REVIEW_SAMPLE_RATE", "1")
    monkeypatch.setenv("GROUP_AGENT_CAFE_SHADOW_REVIEW_MAX_INFLIGHT", "2")


def test_shadow_review_returns_before_provider_and_redacts_context(
    tmp_path: Path, monkeypatch
) -> None:
    _enable(monkeypatch)
    started = threading.Event()
    release = threading.Event()
    captured: list[object] = []
    model = MagicMock()

    def invoke(messages, **_kwargs):
        captured.extend(messages)
        started.set()
        assert release.wait(timeout=2)
        return AIMessage(
            content=json.dumps(
                {
                    "verdict": "pass",
                    "reply": "安全正文。",
                    "option_indices": [],
                    "reason": "通过",
                },
                ensure_ascii=False,
            )
        )

    model.invoke.side_effect = invoke
    before = time.monotonic()
    scheduled = schedule_cafe_shadow_review(
        model=model,
        output=HostOutput(reply="安全正文。"),
        host=_host(tmp_path),
        messages=[
            HumanMessage(
                content="邮箱 founder@example.test，电话 13812345678，token=abcdefgh12345678"
            )
        ],
    )
    elapsed = time.monotonic() - before

    assert scheduled and elapsed < 0.2 and started.wait(timeout=1)
    payload = captured[-1].content
    assert "founder@example.test" not in payload
    assert "13812345678" not in payload
    assert "abcdefgh12345678" not in payload
    release.set()


def test_disabled_shadow_never_calls_model(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("GROUP_AGENT_CAFE_SHADOW_REVIEW_ENABLED", "0")
    model = MagicMock()

    scheduled = schedule_cafe_shadow_review(
        model=model,
        output=HostOutput(reply="直接展示。"),
        host=_host(tmp_path),
        messages=[],
    )

    assert not scheduled
    assert model.invoke.call_count == 0


def test_shadow_exception_has_no_authoritative_return_channel(
    tmp_path: Path, monkeypatch
) -> None:
    _enable(monkeypatch)
    called = threading.Event()
    model = MagicMock()

    def invoke(*_args, **_kwargs):
        called.set()
        raise TimeoutError("observer unavailable")

    model.invoke.side_effect = invoke
    output = HostOutput(reply="这条回答已经确定。", suggestions=[])

    assert schedule_cafe_shadow_review(
        model=model,
        output=output,
        host=_host(tmp_path),
        messages=[],
    )
    assert called.wait(timeout=1)
    assert output.reply == "这条回答已经确定。"
    assert output.suggestions == []


@pytest.mark.asyncio
async def test_graph_checkpoint_does_not_wait_for_shadow_review(
    tmp_path: Path, monkeypatch
) -> None:
    _enable(monkeypatch)
    started = threading.Event()
    release = threading.Event()
    reviewer = MagicMock()

    def invoke(*_args, **_kwargs):
        started.set()
        assert release.wait(timeout=2)
        return AIMessage(
            content=json.dumps(
                {
                    "verdict": "pass",
                    "reply": "已经确定的正文。",
                    "option_indices": [],
                },
                ensure_ascii=False,
            )
        )

    reviewer.invoke.side_effect = invoke
    graph = create_cafe_agent(
        model=ChatModel(output=AIMessage(content="已经确定的正文。")),
        checkpointer=InMemorySaver(),
        shadow_review_model=reviewer,
    )
    config = {"configurable": {"thread_id": "shadow-does-not-gate"}}
    try:
        with active_host_context(_host(tmp_path)):
            result = await graph.ainvoke(
                {"messages": [HumanMessage(content="正常聊聊")]},
                config,
            )
        assert result["messages"][-1].content == "已经确定的正文。"
        checkpoint = await graph.aget_state(config)
        assert checkpoint.values["messages"][-1].content == "已经确定的正文。"
        assert started.wait(timeout=1)
    finally:
        release.set()
