"""Map in-flight LLM spans to user-visible thinking status.

``span_event_factory`` is called on the agent thread at LLM start so the
returned handler can be invoked later from ``threading.Timer`` (slow warnings)
without losing the request-scoped callback sink.
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable
from typing import Any

ProgressSink = Callable[[str], None]
SpanEventFn = Callable[[str, dict[str, Any]], None]

_progress_sink: contextvars.ContextVar[ProgressSink | None] = contextvars.ContextVar(
    "wu_user_progress_sink", default=None
)


def set_progress_sink(sink: ProgressSink | None) -> contextvars.Token:
    """Bind a status callback for the current agent-run thread."""
    return _progress_sink.set(sink)


def reset_progress_sink(token: contextvars.Token) -> None:
    """Restore the previous progress sink."""
    _progress_sink.reset(token)


def llm_span_to_user_status(action: str, fields: dict[str, Any]) -> str | None:
    """Translate latency middleware events into short chat-status lines."""
    agent = str(fields.get("agent") or "")
    if action == "llm_call_start":
        if agent == "kb_analyst":
            return "正在检索与分析案例…"
        if agent == "front":
            return "正在思考怎么回答…"
        if agent == "owner":
            return "正在处理你的问题…"
        return "正在思考…"
    if action == "llm_call_slow":
        try:
            elapsed = int(float(fields.get("threshold_s") or 0))
        except (TypeError, ValueError):
            elapsed = 0
        if agent == "kb_analyst":
            if elapsed >= 120:
                return "案例检索比平时更久，我还在分析…"
            if elapsed >= 60:
                return "正在深入比对案例，请再稍等…"
            return "正在检索案例，通常需要一点时间…"
        if elapsed >= 120:
            return "这一步还在生成，请再等一会儿…"
        if elapsed >= 60:
            return "回答需要多一点时间，我还在组织…"
        return "还在思考，马上继续…"
    return None


def span_event_factory() -> SpanEventFn | None:
    """Capture the current sink so slow-call timers keep the same callback."""
    sink = _progress_sink.get()
    if sink is None:
        return None

    def handler(action: str, fields: dict[str, Any]) -> None:
        text = llm_span_to_user_status(action, fields)
        if text:
            sink(text)

    return handler
