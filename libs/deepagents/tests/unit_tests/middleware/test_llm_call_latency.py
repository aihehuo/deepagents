"""Unit tests for LlmCallLatencyMiddleware and LlmCallSpan."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage

from deepagents.middleware.llm_call_latency import (
    LlmCallLatencyMiddleware,
    LlmCallSpan,
    allm_call_span,
    llm_call_span,
)


class _EmitRecorder:
    def __init__(self) -> None:
        self.events: list[str] = []

    def __call__(self, msg: str) -> None:
        self.events.append(msg)

    def actions(self) -> list[str]:
        out: list[str] = []
        for event in self.events:
            for part in event.split():
                if part.startswith("action="):
                    out.append(part.split("=", 1)[1])
                    break
        return out

    def field(self, event: str, key: str) -> str | None:
        prefix = f"{key}="
        for part in event.split():
            if part.startswith(prefix):
                return part[len(prefix) :]
        return None


def _make_request(*, thread_id: str = "wt::t1", model_name: str = "deepseek-v4-flash") -> ModelRequest:
    model = MagicMock()
    model.model_name = model_name
    runtime = MagicMock()
    runtime.config = {"configurable": {"thread_id": thread_id}}
    request = MagicMock(spec=ModelRequest)
    request.model = model
    request.messages = [HumanMessage(content="hi")]
    request.tools = []
    request.runtime = runtime
    return request


def test_span_emits_start_and_end() -> None:
    emit = _EmitRecorder()
    span = LlmCallSpan(
        call_seq=1,
        emit=emit,
        slow_thresholds_s=(),
        agent_label="front",
        thread_id="tid-1",
        model="m1",
        message_count=2,
        tools_count=1,
    )
    span.start()
    time.sleep(0.01)
    latency = span.end(input_tokens=10, output_tokens=5)

    assert emit.actions() == ["llm_call_start", "llm_call_end"]
    assert latency >= 0
    end_event = emit.events[-1]
    assert emit.field(end_event, "latency_ms") is not None
    assert emit.field(end_event, "input_tokens") == "10"
    assert emit.field(end_event, "output_tokens") == "5"
    assert emit.field(end_event, "agent") == "front"
    assert emit.field(end_event, "thread_id") == "tid-1"


def test_span_emits_error() -> None:
    emit = _EmitRecorder()
    span = LlmCallSpan(call_seq=2, emit=emit, slow_thresholds_s=(), agent_label="kb_analyst")
    span.start()
    latency = span.error(TimeoutError("provider hung"))

    assert emit.actions() == ["llm_call_start", "llm_call_error"]
    assert latency >= 0
    err = emit.events[-1]
    assert emit.field(err, "error_type") == "TimeoutError"
    assert "provider" in (emit.field(err, "error_message") or "")


def test_span_emits_slow_while_in_flight() -> None:
    emit = _EmitRecorder()
    span = LlmCallSpan(
        call_seq=3,
        emit=emit,
        slow_thresholds_s=(0.05,),
        agent_label="front",
        thread_id="tid-slow",
    )
    span.start()
    time.sleep(0.12)
    span.end()

    actions = emit.actions()
    assert actions[0] == "llm_call_start"
    assert "llm_call_slow" in actions
    assert actions[-1] == "llm_call_end"
    slow = next(e for e in emit.events if "action=llm_call_slow" in e)
    assert emit.field(slow, "elapsed_ms") is not None
    assert int(emit.field(slow, "elapsed_ms") or "0") >= 40


def test_middleware_wrap_model_call_success() -> None:
    emit = _EmitRecorder()
    mw = LlmCallLatencyMiddleware(
        emit=emit,
        slow_thresholds_s=(),
        agent_label="front",
    )
    request = _make_request()
    response = ModelResponse(
        result=[
            AIMessage(
                content="ok",
                usage_metadata={
                    "input_tokens": 3,
                    "output_tokens": 2,
                    "total_tokens": 5,
                },
            )
        ]
    )

    def handler(_req: ModelRequest) -> ModelResponse:
        return response

    out = mw.wrap_model_call(request, handler)
    assert out is response
    assert emit.actions() == ["llm_call_start", "llm_call_end"]
    assert emit.field(emit.events[0], "thread_id") == "wt::t1"
    assert emit.field(emit.events[0], "model") == "deepseek-v4-flash"
    assert emit.field(emit.events[-1], "input_tokens") == "3"
    assert emit.field(emit.events[-1], "output_tokens") == "2"


def test_middleware_wrap_model_call_error() -> None:
    emit = _EmitRecorder()
    mw = LlmCallLatencyMiddleware(emit=emit, slow_thresholds_s=(), agent_label="owner")
    request = _make_request(thread_id="t-err")

    def handler(_req: ModelRequest) -> ModelResponse:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        mw.wrap_model_call(request, handler)

    assert emit.actions() == ["llm_call_start", "llm_call_error"]
    assert emit.field(emit.events[-1], "error_type") == "RuntimeError"


@pytest.mark.asyncio
async def test_middleware_awrap_model_call_success() -> None:
    emit = _EmitRecorder()
    mw = LlmCallLatencyMiddleware(emit=emit, slow_thresholds_s=(), agent_label="front")
    request = _make_request()
    response = ModelResponse(result=[AIMessage(content="async-ok")])

    async def handler(_req: ModelRequest) -> ModelResponse:
        return response

    out = await mw.awrap_model_call(request, handler)
    assert out is response
    assert emit.actions() == ["llm_call_start", "llm_call_end"]


def test_llm_call_span_context_manager() -> None:
    emit = _EmitRecorder()
    with llm_call_span(emit=emit, agent_label="extract_intake", slow_thresholds_s=()):
        pass
    assert emit.actions() == ["llm_call_start", "llm_call_end"]


def test_llm_call_span_context_manager_error() -> None:
    emit = _EmitRecorder()
    with pytest.raises(ValueError, match="x"):
        with llm_call_span(emit=emit, agent_label="extract_intake", slow_thresholds_s=()):
            raise ValueError("x")
    assert emit.actions() == ["llm_call_start", "llm_call_error"]


@pytest.mark.asyncio
async def test_allm_call_span_context_manager() -> None:
    emit = _EmitRecorder()
    async with allm_call_span(
        emit=emit,
        agent_label="extract_intake",
        slow_thresholds_s=(),
        call_seq=1,
    ) as span:
        span.end(input_tokens=1, output_tokens=1)
    assert emit.actions() == ["llm_call_start", "llm_call_end"]
    # Second end from context manager else is a no-op
    assert emit.actions().count("llm_call_end") == 1
