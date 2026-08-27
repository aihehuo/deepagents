"""Shared LLM call tracers for real-LLM tests (main agent + sub-agents)."""

from __future__ import annotations

import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler


def message_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def classify_llm_role(messages: list[Any]) -> str:
    """Classify judge / rewrite / quality / main dialogue from system prompts."""
    blob = ""
    for msg in messages or []:
        typ = getattr(msg, "type", None) or msg.__class__.__name__
        if "System" in str(typ) or typ == "system":
            blob += message_text(getattr(msg, "content", None))
    # Order matters: prompts share phrases like「可匹配的最低充分」.
    if "改口步骤" in blob:
        return "REWRITE_MAIN_AGENT"
    if "推荐文案事实对照" in blob or "事实对照」裁判" in blob:
        return "JUDGE_SUBAGENT"
    if "画像质量裁判" in blob:
        return "PROFILE_QUALITY"
    if "邀请" in blob and ("润色" in blob or "脚手架" in blob):
        return "INVITE_POLISH"
    if "群内智能体" in blob or "search_candidates" in blob or "save_group_profile" in blob:
        return "MAIN_AGENT"
    return "MAIN_AGENT"


def _flatten_callback_messages(messages: Any) -> list[Any]:
    """LangChain callbacks pass List[List[BaseMessage]]; normalize to one turn."""
    if not messages:
        return []
    if isinstance(messages, list) and messages and isinstance(messages[0], list):
        return list(messages[0])
    return list(messages)


def _render_messages(messages: list[Any]) -> str:
    rendered: list[str] = []
    for i, msg in enumerate(messages or []):
        typ = getattr(msg, "type", None) or msg.__class__.__name__
        rendered.append(
            f"--- messages[{i}] {typ} ---\n"
            f"{message_text(getattr(msg, 'content', None))}"
        )
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            rendered.append(f"--- messages[{i}] tool_calls ---\n{tool_calls!r}")
    return "\n\n".join(rendered)


def _response_from_llm_result(response: Any) -> tuple[str, list[Any]]:
    outbound = ""
    tool_calls: list[Any] = []
    try:
        for gen_list in getattr(response, "generations", []) or []:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                if msg is None:
                    text = getattr(gen, "text", None)
                    if text:
                        outbound += str(text)
                    continue
                outbound += message_text(getattr(msg, "content", None))
                tool_calls.extend(list(getattr(msg, "tool_calls", None) or []))
    except Exception:  # noqa: BLE001
        outbound = str(response)
    return outbound, tool_calls


class LlmTranscriptHandler(BaseCallbackHandler):
    """Callback tracer that works with real ChatOpenAI (agent + sub-agents).

    Prefer this for brain-as-SUT / create_deep_agent paths — wrapping the model
    object breaks ``resolve_model`` / BaseChatModel checks.
    """

    raise_error = True

    def __init__(self, *, print_live: bool = True) -> None:
        super().__init__()
        self.print_live = print_live
        self.calls: list[dict[str, Any]] = []
        self._pending: dict[str, dict[str, Any]] = {}

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        del serialized, kwargs
        flat = _flatten_callback_messages(messages)
        role = classify_llm_role(flat)
        inbound = _render_messages(flat)
        call_no = len(self.calls) + 1
        key = str(run_id)
        self._pending[key] = {
            "n": call_no,
            "role": role,
            "request": inbound,
            "started": time.perf_counter(),
        }
        # Reserve a slot so call numbers stay stable if ends reorder.
        self.calls.append(
            {
                "n": call_no,
                "role": role,
                "elapsed_ms": None,
                "request": inbound,
                "response": None,
                "tool_calls": [],
                "pending": True,
            }
        )
        if self.print_live:
            print("\n" + "#" * 72)
            print(f"LLM CALL #{call_no} · role={role} · BEGIN")
            print("#" * 72)
            print(inbound)
            print("-" * 72)
            print(f"LLM CALL #{call_no} · invoking provider…")

    def on_chat_model_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        del kwargs
        key = str(run_id)
        pending = self._pending.pop(key, None)
        outbound, tool_calls = _response_from_llm_result(response)
        if pending is None:
            call_no = len(self.calls) + 1
            role = "LLM"
            elapsed_ms = 0
            inbound = ""
            self.calls.append(
                {
                    "n": call_no,
                    "role": role,
                    "elapsed_ms": elapsed_ms,
                    "request": inbound,
                    "response": outbound,
                    "tool_calls": tool_calls,
                    "pending": False,
                }
            )
        else:
            call_no = int(pending["n"])
            role = str(pending["role"])
            elapsed_ms = int((time.perf_counter() - float(pending["started"])) * 1000)
            inbound = str(pending["request"])
            for row in self.calls:
                if row.get("n") == call_no and row.get("pending"):
                    row["elapsed_ms"] = elapsed_ms
                    row["response"] = outbound
                    row["tool_calls"] = tool_calls
                    row["pending"] = False
                    break
        if self.print_live:
            print("-" * 72)
            print(f"LLM CALL #{call_no} · role={role} · RESPONSE ({elapsed_ms} ms)")
            print("-" * 72)
            print(outbound)
            if tool_calls:
                print(f"--- tool_calls ---\n{tool_calls!r}")
            print("#" * 72)
            print(f"LLM CALL #{call_no} · role={role} · END")
            print("#" * 72 + "\n")

    def on_chat_model_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        del kwargs
        key = str(run_id)
        pending = self._pending.pop(key, None)
        if pending is None:
            return
        call_no = int(pending["n"])
        role = str(pending["role"])
        elapsed_ms = int((time.perf_counter() - float(pending["started"])) * 1000)
        for row in self.calls:
            if row.get("n") == call_no and row.get("pending"):
                row["elapsed_ms"] = elapsed_ms
                row["response"] = f"ERROR: {type(error).__name__}: {error}"
                row["pending"] = False
                break
        if self.print_live:
            print("-" * 72)
            print(f"LLM CALL #{call_no} · role={role} · ERROR ({elapsed_ms} ms)")
            print("-" * 72)
            print(f"{type(error).__name__}: {error}")
            print("#" * 72 + "\n")


class TracingChatModel:
    """Wrap ``invoke`` / ``ainvoke`` for unit tests that pass a model into gates.

    Do **not** pass this into ``create_deep_agent`` — use ``LlmTranscriptHandler``
    on a real ChatOpenAI instead.
    """

    def __init__(
        self,
        inner: Any,
        *,
        calls: list[dict[str, Any]] | None = None,
        print_live: bool = True,
    ) -> None:
        self._inner = inner
        self.calls: list[dict[str, Any]] = calls if calls is not None else []
        self.print_live = print_live

    def _record_begin(self, messages: list[Any]) -> tuple[int, str, str, float]:
        call_no = len(self.calls) + 1
        role = classify_llm_role(messages)
        inbound = _render_messages(messages)
        if self.print_live:
            print("\n" + "#" * 72)
            print(f"LLM CALL #{call_no} · role={role} · BEGIN")
            print("#" * 72)
            print(inbound)
            print("-" * 72)
            print(f"LLM CALL #{call_no} · invoking provider…")
        return call_no, role, inbound, time.perf_counter()

    def _record_end(
        self,
        *,
        call_no: int,
        role: str,
        inbound: str,
        started: float,
        result: Any,
    ) -> Any:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        outbound = message_text(getattr(result, "content", None))
        tool_calls = list(getattr(result, "tool_calls", None) or [])
        if self.print_live:
            print("-" * 72)
            print(f"LLM CALL #{call_no} · role={role} · RESPONSE ({elapsed_ms} ms)")
            print("-" * 72)
            print(outbound)
            if tool_calls:
                print(f"--- tool_calls ---\n{tool_calls!r}")
            print("#" * 72)
            print(f"LLM CALL #{call_no} · role={role} · END")
            print("#" * 72 + "\n")
        self.calls.append(
            {
                "n": call_no,
                "role": role,
                "elapsed_ms": elapsed_ms,
                "request": inbound,
                "response": outbound,
                "tool_calls": tool_calls,
            }
        )
        return result

    def invoke(self, messages: list[Any], **kwargs: Any) -> Any:
        call_no, role, inbound, started = self._record_begin(list(messages or []))
        kw = dict(kwargs)
        cfg = dict(kw.get("config") or {})
        # Prevent double-logging when the inner ChatOpenAI also has callbacks.
        cfg["callbacks"] = []
        kw["config"] = cfg
        result = self._inner.invoke(messages, **kw)
        return self._record_end(
            call_no=call_no,
            role=role,
            inbound=inbound,
            started=started,
            result=result,
        )

    async def ainvoke(self, messages: list[Any], **kwargs: Any) -> Any:
        call_no, role, inbound, started = self._record_begin(list(messages or []))
        kw = dict(kwargs)
        cfg = dict(kw.get("config") or {})
        cfg["callbacks"] = []
        kw["config"] = cfg
        result = await self._inner.ainvoke(messages, **kw)
        return self._record_end(
            call_no=call_no,
            role=role,
            inbound=inbound,
            started=started,
            result=result,
        )

    def bind_tools(self, *args: Any, **kwargs: Any) -> TracingChatModel:
        return TracingChatModel(
            self._inner.bind_tools(*args, **kwargs),
            calls=self.calls,
            print_live=self.print_live,
        )

    def with_config(self, *args: Any, **kwargs: Any) -> TracingChatModel:
        return TracingChatModel(
            self._inner.with_config(*args, **kwargs),
            calls=self.calls,
            print_live=self.print_live,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def attach_invoke_trace(
    model: Any,
    *,
    calls: list[dict[str, Any]],
    print_live: bool = True,
) -> Any:
    """Patch ChatOpenAI ``_generate`` / ``_agenerate`` (leaf LLM entrypoints).

    LangGraph / bind_tools eventually call these; wrapping only invoke/ainvoke
    is not enough. Do not also wrap invoke — that would double-count.
    """
    from types import MethodType

    from langchain_core.messages import AIMessage, BaseMessage

    cls = model.__class__
    cls_generate = cls._generate
    cls_agenerate = cls._agenerate

    def _begin(messages: list[Any]) -> tuple[int, str, str, float]:
        flat = list(messages or [])
        call_no = len(calls) + 1
        role = classify_llm_role(flat)
        inbound = _render_messages(flat)
        if print_live:
            print("\n" + "#" * 72)
            print(f"LLM CALL #{call_no} · role={role} · BEGIN")
            print("#" * 72)
            print(inbound)
            print("-" * 72)
            print(f"LLM CALL #{call_no} · invoking provider…")
        return call_no, role, inbound, time.perf_counter()

    def _end_from_chat_result(
        call_no: int, role: str, inbound: str, started: float, result: Any
    ) -> Any:
        message = None
        try:
            gens = getattr(result, "generations", None) or []
            if gens and gens[0]:
                message = getattr(gens[0][0], "message", None)
        except Exception:  # noqa: BLE001
            message = None
        if message is None:
            message = AIMessage(content=str(result))
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        outbound = message_text(getattr(message, "content", None))
        tool_calls = list(getattr(message, "tool_calls", None) or [])
        if print_live:
            print("-" * 72)
            print(f"LLM CALL #{call_no} · role={role} · RESPONSE ({elapsed_ms} ms)")
            print("-" * 72)
            print(outbound)
            if tool_calls:
                print(f"--- tool_calls ---\n{tool_calls!r}")
            print("#" * 72)
            print(f"LLM CALL #{call_no} · role={role} · END")
            print("#" * 72 + "\n")
        calls.append(
            {
                "n": call_no,
                "role": role,
                "elapsed_ms": elapsed_ms,
                "request": inbound,
                "response": outbound,
                "tool_calls": tool_calls,
            }
        )
        return result

    def _generate(
        self: Any,
        messages: list[BaseMessage],
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        call_no, role, inbound, started = _begin(list(messages or []))
        result = cls_generate(
            self, messages, stop=stop, run_manager=run_manager, **kwargs
        )
        return _end_from_chat_result(call_no, role, inbound, started, result)

    async def _agenerate(
        self: Any,
        messages: list[BaseMessage],
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        call_no, role, inbound, started = _begin(list(messages or []))
        result = await cls_agenerate(
            self, messages, stop=stop, run_manager=run_manager, **kwargs
        )
        return _end_from_chat_result(call_no, role, inbound, started, result)

    object.__setattr__(model, "_generate", MethodType(_generate, model))
    object.__setattr__(model, "_agenerate", MethodType(_agenerate, model))
    return model


def format_call_inventory(calls: list[dict[str, Any]]) -> str:
    lines = ["=== LLM call inventory ==="]
    for c in calls:
        preview = (c.get("response") or "").replace("\n", " ")[:100]
        tools = c.get("tool_calls") or []
        tool_names = []
        for tc in tools:
            if isinstance(tc, dict):
                tool_names.append(str(tc.get("name") or "?"))
            else:
                tool_names.append(str(getattr(tc, "name", "?") or "?"))
        extra = f" tools={tool_names}" if tool_names else ""
        elapsed = c.get("elapsed_ms")
        elapsed_s = f"{elapsed}ms" if elapsed is not None else "pending"
        lines.append(
            f"  #{c['n']} {c['role']} {elapsed_s}{extra} preview={preview!r}"
        )
    lines.append("=== end inventory ===")
    return "\n".join(lines)
