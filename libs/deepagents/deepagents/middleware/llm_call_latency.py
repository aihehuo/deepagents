"""Per-LLM-call latency observability middleware and helpers.

Emits structured one-line events so operators can see:
- every model call start (even if it never returns)
- slow in-flight warnings (default 30s / 60s / 120s)
- end latency or error latency

Events are always written to the uvicorn.error logger. An optional ``emit``
callback (e.g. UC18Observer.info) can mirror them into a UC log file.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)

_logger = logging.getLogger("uvicorn.error")

DEFAULT_SLOW_THRESHOLDS_S: tuple[float, ...] = (30.0, 60.0, 120.0)

EmitFn = Callable[[str], None]


def _safe_dual_emit(emit: EmitFn | None, msg: str) -> None:
    """Log to uvicorn and optionally to a UC observer. Never raises."""
    try:
        _logger.info("%s", msg)
    except Exception:  # noqa: BLE001
        pass
    if emit is None:
        return
    try:
        emit(msg)
    except Exception:  # noqa: BLE001
        pass


def _format_fields(**fields: Any) -> str:
    parts: list[str] = []
    for key, value in fields.items():
        if value is None or value == "":
            continue
        text = str(value).replace("\r", " ").replace("\n", " ")
        parts.append(f"{key}={text}")
    return " ".join(parts)


def model_label(model: Any) -> str:
    """Best-effort human-readable model name from a LangChain chat model."""
    if model is None:
        return ""
    for attr in ("model_name", "model"):
        value = getattr(model, attr, None)
        if value:
            return str(value)
    return type(model).__name__


def _thread_id_from_request(request: ModelRequest) -> str:
    runtime = getattr(request, "runtime", None)
    config = getattr(runtime, "config", None) if runtime is not None else None
    if not isinstance(config, dict):
        return ""
    configurable = config.get("configurable") or {}
    if not isinstance(configurable, dict):
        return ""
    tid = configurable.get("thread_id")
    return str(tid) if tid else ""


def _message_count(request: ModelRequest) -> int:
    messages = getattr(request, "messages", None)
    return len(messages) if messages is not None else 0


def _tools_count(request: ModelRequest) -> int:
    tools = getattr(request, "tools", None)
    return len(tools) if tools is not None else 0


def _extract_tokens_from_response(response: Any) -> tuple[int, int]:
    """Best-effort token extraction from a ModelResponse or raw AIMessage."""
    input_tokens = 0
    output_tokens = 0

    messages = None
    if hasattr(response, "messages"):
        messages = response.messages
    elif hasattr(response, "result"):
        result = response.result
        if isinstance(result, list):
            messages = result
        elif hasattr(result, "messages"):
            messages = result.messages

    candidates: list[Any] = []
    if messages:
        candidates.extend(messages)
    else:
        candidates.append(response)

    for message in candidates:
        usage_metadata = getattr(message, "usage_metadata", None)
        if usage_metadata:
            if isinstance(usage_metadata, dict):
                msg_in = (
                    usage_metadata.get("input_tokens")
                    or usage_metadata.get("prompt_tokens")
                    or 0
                )
                msg_out = (
                    usage_metadata.get("output_tokens")
                    or usage_metadata.get("completion_tokens")
                    or 0
                )
            else:
                msg_in = (
                    getattr(usage_metadata, "input_tokens", None)
                    or getattr(usage_metadata, "prompt_tokens", None)
                    or 0
                )
                msg_out = (
                    getattr(usage_metadata, "output_tokens", None)
                    or getattr(usage_metadata, "completion_tokens", None)
                    or 0
                )
            if msg_in or msg_out:
                input_tokens += int(msg_in or 0)
                output_tokens += int(msg_out or 0)
                continue

        response_metadata = getattr(message, "response_metadata", None)
        if response_metadata and isinstance(response_metadata, dict):
            msg_in = (
                response_metadata.get("input_tokens")
                or response_metadata.get("prompt_tokens")
                or 0
            )
            msg_out = (
                response_metadata.get("output_tokens")
                or response_metadata.get("completion_tokens")
                or 0
            )
            if msg_in or msg_out:
                input_tokens += int(msg_in or 0)
                output_tokens += int(msg_out or 0)

    return input_tokens, output_tokens


class LlmCallSpan:
    """Tracks one LLM invocation with start / slow / end / error events."""

    def __init__(
        self,
        *,
        call_id: str | None = None,
        call_seq: int = 0,
        emit: EmitFn | None = None,
        slow_thresholds_s: Sequence[float] = DEFAULT_SLOW_THRESHOLDS_S,
        agent_label: str = "",
        thread_id: str = "",
        model: str = "",
        message_count: int | None = None,
        tools_count: int | None = None,
    ) -> None:
        self.call_id = call_id or uuid.uuid4().hex[:12]
        self.call_seq = call_seq
        self._emit = emit
        self._slow_thresholds_s = tuple(
            sorted({float(t) for t in slow_thresholds_s if float(t) > 0})
        )
        self.agent_label = agent_label
        self.thread_id = thread_id
        self.model = model
        self.message_count = message_count
        self.tools_count = tools_count
        self._started_at = 0.0
        self._done = False
        self._lock = threading.Lock()
        self._timers: list[threading.Timer] = []

    def _base_fields(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "call_seq": self.call_seq,
            "agent": self.agent_label or None,
            "thread_id": self.thread_id or None,
            "model": self.model or None,
            "message_count": self.message_count,
            "tools_count": self.tools_count,
        }

    def start(self) -> None:
        """Emit llm_call_start and schedule slow-call timers."""
        self._started_at = time.perf_counter()
        _safe_dual_emit(
            self._emit,
            _format_fields(action="llm_call_start", **self._base_fields()),
        )
        for threshold in self._slow_thresholds_s:
            timer = threading.Timer(threshold, self._emit_slow, args=(threshold,))
            timer.daemon = True
            self._timers.append(timer)
            timer.start()

    def _emit_slow(self, threshold_s: float) -> None:
        with self._lock:
            if self._done:
                return
            elapsed_ms = int((time.perf_counter() - self._started_at) * 1000)
        _safe_dual_emit(
            self._emit,
            _format_fields(
                action="llm_call_slow",
                **self._base_fields(),
                threshold_s=int(threshold_s)
                if threshold_s == int(threshold_s)
                else threshold_s,
                elapsed_ms=elapsed_ms,
            ),
        )

    def _cancel_timers(self) -> None:
        for timer in self._timers:
            timer.cancel()
        self._timers.clear()

    def _finish_locked(self) -> int:
        self._done = True
        self._cancel_timers()
        return int((time.perf_counter() - self._started_at) * 1000)

    def end(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> int:
        """Emit llm_call_end and return latency_ms."""
        with self._lock:
            if self._done:
                return 0
            latency_ms = self._finish_locked()
        _safe_dual_emit(
            self._emit,
            _format_fields(
                action="llm_call_end",
                **self._base_fields(),
                latency_ms=latency_ms,
                input_tokens=input_tokens or None,
                output_tokens=output_tokens or None,
            ),
        )
        return latency_ms

    def error(self, exc: BaseException) -> int:
        """Emit llm_call_error and return latency_ms."""
        with self._lock:
            if self._done:
                return 0
            latency_ms = self._finish_locked()
        message = str(exc).replace("\r", " ").replace("\n", " ")[:200]
        _safe_dual_emit(
            self._emit,
            _format_fields(
                action="llm_call_error",
                **self._base_fields(),
                latency_ms=latency_ms,
                error_type=type(exc).__name__,
                error_message=message,
            ),
        )
        return latency_ms

    def close(self) -> None:
        """Cancel timers without emitting end (for abandoned spans)."""
        with self._lock:
            self._done = True
            self._cancel_timers()


@contextmanager
def llm_call_span(
    *,
    emit: EmitFn | None = None,
    slow_thresholds_s: Sequence[float] = DEFAULT_SLOW_THRESHOLDS_S,
    agent_label: str = "",
    thread_id: str = "",
    model: str = "",
    message_count: int | None = None,
    tools_count: int | None = None,
    call_seq: int = 0,
) -> Iterator[LlmCallSpan]:
    """Sync context manager that records one LLM call."""
    span = LlmCallSpan(
        call_seq=call_seq,
        emit=emit,
        slow_thresholds_s=slow_thresholds_s,
        agent_label=agent_label,
        thread_id=thread_id,
        model=model,
        message_count=message_count,
        tools_count=tools_count,
    )
    span.start()
    try:
        yield span
    except Exception as exc:
        span.error(exc)
        raise
    else:
        span.end()


@asynccontextmanager
async def allm_call_span(
    *,
    emit: EmitFn | None = None,
    slow_thresholds_s: Sequence[float] = DEFAULT_SLOW_THRESHOLDS_S,
    agent_label: str = "",
    thread_id: str = "",
    model: str = "",
    message_count: int | None = None,
    tools_count: int | None = None,
    call_seq: int = 0,
) -> AsyncIterator[LlmCallSpan]:
    """Async context manager that records one LLM call."""
    span = LlmCallSpan(
        call_seq=call_seq,
        emit=emit,
        slow_thresholds_s=slow_thresholds_s,
        agent_label=agent_label,
        thread_id=thread_id,
        model=model,
        message_count=message_count,
        tools_count=tools_count,
    )
    span.start()
    try:
        yield span
    except Exception as exc:
        span.error(exc)
        raise
    else:
        # Caller may already have called span.end() with tokens; no-op if done.
        span.end()


class LlmCallLatencyMiddleware(AgentMiddleware):
    """Wrap every model call with start / slow / end / error latency events."""

    state_schema = AgentState

    def __init__(
        self,
        *,
        emit: EmitFn | None = None,
        slow_thresholds_s: Sequence[float] = DEFAULT_SLOW_THRESHOLDS_S,
        agent_label: str = "",
    ) -> None:
        """Initialize middleware.

        Args:
            emit: Optional extra sink (e.g. ``UC18Observer.info``).
            slow_thresholds_s: Elapsed seconds at which to emit ``llm_call_slow``.
            agent_label: Short label for front/owner/subagent identity in logs.
        """
        self._emit = emit
        self._slow_thresholds_s = tuple(slow_thresholds_s)
        self._agent_label = agent_label
        self._seq_lock = threading.Lock()
        self._call_seq = 0

    def _next_seq(self) -> int:
        with self._seq_lock:
            self._call_seq += 1
            return self._call_seq

    def _new_span(self, request: ModelRequest) -> LlmCallSpan:
        return LlmCallSpan(
            call_seq=self._next_seq(),
            emit=self._emit,
            slow_thresholds_s=self._slow_thresholds_s,
            agent_label=self._agent_label,
            thread_id=_thread_id_from_request(request),
            model=model_label(getattr(request, "model", None)),
            message_count=_message_count(request),
            tools_count=_tools_count(request),
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Time a synchronous model call."""
        span = self._new_span(request)
        span.start()
        try:
            response = handler(request)
        except Exception as exc:
            span.error(exc)
            raise
        input_tokens, output_tokens = _extract_tokens_from_response(response)
        span.end(input_tokens=input_tokens, output_tokens=output_tokens)
        return response

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Time an asynchronous model call."""
        span = self._new_span(request)
        span.start()
        try:
            response = await handler(request)
        except Exception as exc:
            span.error(exc)
            raise
        input_tokens, output_tokens = _extract_tokens_from_response(response)
        span.end(input_tokens=input_tokens, output_tokens=output_tokens)
        return response
