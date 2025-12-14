from __future__ import annotations

import asyncio
import json
import logging
import os
import traceback
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field
except Exception as e:  # noqa: BLE001
    raise RuntimeError(
        "FastAPI is required to run the Business Co-Founder API. "
        "Install it with: `pip install fastapi uvicorn`."
    ) from e

from apps.business_cofounder_api.agent_factory import create_business_cofounder_agent


def _thread_id(*, user_id: str, conversation_id: str) -> str:
    return f"bc::{user_id}::{conversation_id}"

# Use uvicorn's configured logger so output reliably shows up in the terminal.
_logger = logging.getLogger("uvicorn.error")
if _logger.level == logging.NOTSET:
    _logger.setLevel(logging.INFO)


def _env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v in {"1", "true", "TRUE", "yes", "YES", "on", "ON"}


def _truncate(text: str, limit: int) -> str:
    if limit <= 0:
        return text
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... (truncated, {len(text)} chars total)"


def _log_chat_io(*, user_id: str, conversation_id: str, thread_id: str, user_message: str, reply: str) -> None:
    if not _env_flag("BC_API_LOG_CHAT_IO", default=False):
        return
    limit = int(os.environ.get("BC_API_LOG_TRUNCATE_CHARS", "2000"))
    _logger.info(
        "chat_io user_id=%s conversation_id=%s thread_id=%s\nUSER:\n%s\n\nASSISTANT:\n%s",
        user_id,
        conversation_id,
        thread_id,
        _truncate(user_message, limit),
        _truncate(reply, limit),
    )


def _log_debug_state(*, result: dict[str, Any], thread_id: str) -> None:
    """Optional debug logging of milestones/todos/tool calls to diagnose stalled workflows."""
    if not _env_flag("BC_API_LOG_STATE", default=False):
        return

    milestones = {
        "business_idea_complete": bool(result.get("business_idea_complete")),
        "persona_clarified": bool(result.get("persona_clarified")),
        "painpoint_enhanced": bool(result.get("painpoint_enhanced")),
        "pitch_created": bool(result.get("pitch_created")),
        "pricing_optimized": bool(result.get("pricing_optimized")),
    }
    todos = result.get("todos") if isinstance(result.get("todos"), list) else []
    in_progress = None
    for t in todos:
        if isinstance(t, dict) and t.get("status") == "in_progress":
            in_progress = (t.get("content") or "")[:120]
            break

    messages = result.get("messages", [])
    last_ai_tool_calls = None
    for m in reversed(messages):
        if getattr(m, "type", None) == "ai":
            tcs = getattr(m, "tool_calls", None)
            if tcs:
                last_ai_tool_calls = [
                    {"name": tc.get("name"), "args": tc.get("args")}
                    for tc in tcs
                    if isinstance(tc, dict)
                ][:5]
            break

    _logger.info(
        "chat_state thread_id=%s milestones=%s current_todo=%s last_ai_tool_calls=%s",
        thread_id,
        milestones,
        in_progress,
        last_ai_tool_calls,
    )


def _extract_state_values_from_checkpoint(checkpoint: Any) -> dict[str, Any]:
    """Best-effort extraction of LangGraph 'values' from a checkpoint object."""
    if isinstance(checkpoint, dict):
        for k in ("channel_values", "state", "values"):
            v = checkpoint.get(k)
            if isinstance(v, dict):
                return v
        return checkpoint
    return {}

def _extract_text_chunks_from_ai_message(message: Any) -> list[str]:
    """Best-effort extraction of streamed text chunks from an AI message/chunk.

    Different providers expose different shapes:
    - Anthropic: message.content_blocks -> [{"type":"text","text":"..."}]
    - OpenAI-compatible: message.content may be a string OR a list of blocks/dicts
    """
    chunks: list[str] = []

    content_blocks = getattr(message, "content_blocks", None)
    if isinstance(content_blocks, list) and content_blocks:
        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "text":
                continue
            text = block.get("text", "")
            if isinstance(text, str) and text:
                chunks.append(text)
        if chunks:
            return chunks

    content = getattr(message, "content", None)
    if isinstance(content, str) and content:
        return [content]
    if isinstance(content, list) and content:
        for item in content:
            if isinstance(item, str) and item:
                chunks.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text:
                    chunks.append(text)
        if chunks:
            return chunks

    text_attr = getattr(message, "text", None)
    if isinstance(text_attr, str) and text_attr:
        return [text_attr]

    return []


def _extract_text_chunks_from_ai_message(message: Any) -> list[str]:
    """Best-effort extraction of streamed text chunks from an AI message/chunk.

    Different providers expose different shapes:
    - Anthropic: message.content_blocks -> [{"type":"text","text":"..."}]
    - OpenAI-compatible: message.content may be a string OR a list of blocks/dicts
    """
    chunks: list[str] = []

    # Prefer content_blocks (Anthropic-style)
    content_blocks = getattr(message, "content_blocks", None)
    if isinstance(content_blocks, list) and content_blocks:
        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "text":
                continue
            text = block.get("text", "")
            if isinstance(text, str) and text:
                chunks.append(text)
        if chunks:
            return chunks

    # Fallback: content could be str or list
    content = getattr(message, "content", None)
    if isinstance(content, str) and content:
        return [content]
    if isinstance(content, list) and content:
        for item in content:
            if isinstance(item, str) and item:
                chunks.append(item)
                continue
            if isinstance(item, dict):
                # Common patterns: {"type":"text","text":"..."} or {"text":"..."}
                text = item.get("text")
                if isinstance(text, str) and text:
                    chunks.append(text)
        if chunks:
            return chunks

    # Last resort: some message types may expose .text
    text_attr = getattr(message, "text", None)
    if isinstance(text_attr, str) and text_attr:
        return [text_attr]

    return []


def _summarize_state_values(values: dict[str, Any]) -> dict[str, Any]:
    """Return a small JSON-serializable summary for debugging."""
    milestones = {
        "business_idea_complete": bool(values.get("business_idea_complete")),
        "persona_clarified": bool(values.get("persona_clarified")),
        "painpoint_enhanced": bool(values.get("painpoint_enhanced")),
        "pitch_created": bool(values.get("pitch_created")),
        "pricing_optimized": bool(values.get("pricing_optimized")),
    }
    todos = values.get("todos") if isinstance(values.get("todos"), list) else []
    msg_count = 0
    msgs = values.get("messages")
    if isinstance(msgs, list):
        msg_count = len(msgs)

    written_files: list[str] = []
    if isinstance(msgs, list):
        for m in reversed(msgs[-200:]):  # scan recent messages only
            if not isinstance(m, ToolMessage):
                continue
            if getattr(m, "name", "") != "write_file":
                continue
            content = getattr(m, "content", "")
            if isinstance(content, str) and content.startswith("Updated file "):
                written_files.append(content.replace("Updated file ", "", 1).strip())

    # Keep deterministic order, unique
    written_files = sorted({p for p in written_files if p})

    return {
        "milestones": milestones,
        "todos": todos,
        "message_count": msg_count,
        "written_files": written_files,
    }


class ChatRequest(BaseModel):
    user_id: str = Field(..., description="Upstream server-provided user id")
    message: str = Field(..., description="User message")
    conversation_id: str = Field("default", description="Conversation id (defaults to 'default')")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Optional metadata from upstream")


class ChatResponse(BaseModel):
    user_id: str
    conversation_id: str
    thread_id: str
    reply: str


class ResetRequest(BaseModel):
    user_id: str
    conversation_id: str = "default"


class ResetResponse(BaseModel):
    user_id: str
    conversation_id: str
    thread_id: str
    ok: bool


@dataclass
class _AppState:
    agent: Any
    checkpoints_path: str
    # Ensure the same thread_id is processed serially (avoid checkpoint races).
    thread_locks: dict[str, asyncio.Lock]


app = FastAPI(title="Business Co-Founder Agent API", version="0.1.0")
_state: _AppState | None = None


@app.on_event("startup")
async def _startup() -> None:
    global _state
    agent, checkpoints_path = create_business_cofounder_agent(agent_id="business_cofounder_agent")
    _state = _AppState(agent=agent, checkpoints_path=str(checkpoints_path), thread_locks={})


@app.get("/health")
async def health() -> dict[str, str]:
    assert _state is not None
    return {"status": "ok", "checkpoints_path": _state.checkpoints_path}


@app.get("/state")
async def get_state(user_id: str, conversation_id: str = "default") -> dict[str, Any]:
    """Debug endpoint: return current milestone flags + todo list from checkpoint state.

    Disabled by default. Enable with:
      BC_API_ENABLE_STATE_ENDPOINT=1
    """
    if not _env_flag("BC_API_ENABLE_STATE_ENDPOINT", default=False):
        raise HTTPException(status_code=404, detail="Not found")

    assert _state is not None
    tid = _thread_id(user_id=user_id, conversation_id=conversation_id)

    lock = _state.thread_locks.get(tid)
    if lock is None:
        lock = asyncio.Lock()
        _state.thread_locks[tid] = lock

    async with lock:
        checkpointer = getattr(_state.agent, "checkpointer", None)
        if checkpointer is None or not hasattr(checkpointer, "get_tuple"):
            raise HTTPException(status_code=500, detail="Agent checkpointer not available")

        try:
            ckpt_tuple = await asyncio.to_thread(
                checkpointer.get_tuple, {"configurable": {"thread_id": tid}}
            )
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Failed to read checkpoint: {e!s}") from e

    if ckpt_tuple is None:
        return {
            "thread_id": tid,
            "checkpoints_path": _state.checkpoints_path,
            "milestones": {},
            "todos": [],
            "message_count": 0,
        }

    checkpoint = ckpt_tuple.checkpoint if hasattr(ckpt_tuple, "checkpoint") else ckpt_tuple[1]
    values = _extract_state_values_from_checkpoint(checkpoint)
    summary = _summarize_state_values(values)

    # Also compute the canonical todo list derived from milestone flags.
    # This helps debug cases where `todos` in the checkpoint is stale (e.g. if an LLM rewrote it,
    # or if milestones were updated at the end of a run and todos will only be refreshed on the next run).
    try:
        from deepagents.middleware.business_idea_development import BusinessIdeaDevelopmentMiddleware

        canonical_todos = BusinessIdeaDevelopmentMiddleware()._generate_todos_from_state(values)  # noqa: SLF001
    except Exception:  # noqa: BLE001
        canonical_todos = []

    return {
        "thread_id": tid,
        "checkpoints_path": _state.checkpoints_path,
        **summary,
        "todos_canonical": canonical_todos,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    assert _state is not None
    tid = _thread_id(user_id=req.user_id, conversation_id=req.conversation_id)

    lock = _state.thread_locks.get(tid)
    if lock is None:
        lock = asyncio.Lock()
        _state.thread_locks[tid] = lock

    async with lock:
        try:
            result = await _state.agent.ainvoke(
                {"messages": [HumanMessage(content=req.message)]},
                {
                    "configurable": {"thread_id": tid},
                    "metadata": {"user_id": req.user_id, **(req.metadata or {})},
                },
            )
        except Exception as e:  # noqa: BLE001
            # Print a full traceback to server logs for local debugging.
            _logger.exception(
                "POST /chat failed user_id=%s conversation_id=%s thread_id=%s error_type=%s error_message=%s",
                req.user_id,
                req.conversation_id,
                tid,
                type(e).__name__,
                str(e),
            )

            # Optionally include traceback in HTTP response (useful for local dev; avoid enabling in prod).
            detail: dict[str, Any] = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "thread_id": tid,
            }
            if _env_flag("BC_API_RETURN_TRACEBACK", default=False):
                detail["traceback"] = traceback.format_exc()

            # Internal API: return a helpful message for debugging.
            raise HTTPException(
                status_code=502,
                detail=detail,
            ) from e

    messages = result.get("messages", [])
    ai_messages = [m for m in messages if getattr(m, "type", None) == "ai"]
    reply = str(ai_messages[-1].content) if ai_messages else ""

    _log_chat_io(
        user_id=req.user_id,
        conversation_id=req.conversation_id,
        thread_id=tid,
        user_message=req.message,
        reply=reply,
    )
    _log_debug_state(result=result, thread_id=tid)

    return ChatResponse(
        user_id=req.user_id,
        conversation_id=req.conversation_id,
        thread_id=tid,
        reply=reply,
    )


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """Stream the assistant response as Server-Sent Events (SSE).

    This avoids client/proxy timeouts for long generations (e.g. HTML via coder subagent).

    SSE event format:
    - data: {"type":"delta","text":"..."}  (many)
    - data: {"type":"final","text":"..."}  (once)
    - data: {"type":"error","detail":{...}} (once, if error)
    """
    assert _state is not None
    tid = _thread_id(user_id=req.user_id, conversation_id=req.conversation_id)

    lock = _state.thread_locks.get(tid)
    if lock is None:
        lock = asyncio.Lock()
        _state.thread_locks[tid] = lock

    async def _gen():
        final_parts: list[str] = []
        delta_count = 0
        seen_types: dict[str, int] = {}
        last_written_html_path: str | None = None
        async with lock:
            try:
                async for chunk in _state.agent.astream(
                    {"messages": [HumanMessage(content=req.message)]},
                    config={
                        "configurable": {"thread_id": tid},
                        "metadata": {"user_id": req.user_id, **(req.metadata or {})},
                    },
                    stream_mode=["messages", "updates"],
                    subgraphs=True,
                    durability="exit",
                ):
                    # With subgraphs=True and multiple stream modes, chunks are:
                    # (namespace, stream_mode, data)
                    if not isinstance(chunk, tuple) or len(chunk) != 3:
                        continue

                    _namespace, current_stream_mode, data = chunk

                    if current_stream_mode != "messages":
                        continue
                    # Messages stream returns (message, metadata)
                    if not isinstance(data, tuple) or len(data) != 2:
                        continue
                    message, _metadata = data

                    cls = type(message).__name__
                    seen_types[cls] = seen_types.get(cls, 0) + 1

                    # Track HTML file writes so we can fallback if the model writes a file but doesn't return text.
                    if isinstance(message, ToolMessage):
                        tool_name = getattr(message, "name", "") or ""
                        tool_content = getattr(message, "content", "") or ""
                        if tool_name == "write_file" and isinstance(tool_content, str):
                            # Filesystem tool returns: "Updated file <path>"
                            prefix = "Updated file "
                            if tool_content.startswith(prefix):
                                p = tool_content[len(prefix) :].strip()
                                if p.lower().endswith(".html"):
                                    last_written_html_path = p
                        continue

                    # Ignore human echoes
                    if isinstance(message, HumanMessage):
                        continue

                    # Stream assistant output: handle both full messages and streaming chunks
                    if not isinstance(message, (AIMessage, AIMessageChunk)):
                        # Some providers may not use these exact classes; fall back on type=="ai" when present.
                        if getattr(message, "type", None) != "ai":
                            continue

                    for text in _extract_text_chunks_from_ai_message(message):
                        final_parts.append(text)
                        delta_count += 1
                        payload = {"type": "delta", "text": text}
                        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

                final_text = "".join(final_parts).strip()

                # Fallback: if no text was streamed but an HTML file was written, read it and return its contents.
                if not final_text and last_written_html_path:
                    try:
                        from pathlib import Path

                        pth = Path(last_written_html_path)
                        if pth.exists() and pth.is_file() and pth.stat().st_size <= 2 * 1024 * 1024:
                            final_text = pth.read_text(encoding="utf-8", errors="replace").strip()
                    except Exception:  # noqa: BLE001
                        pass

                if _env_flag("BC_API_STREAM_DEBUG", default=False):
                    _logger.info(
                        "chat_stream_debug thread_id=%s delta_count=%s seen_message_types=%s last_written_html=%s final_len=%s",
                        tid,
                        delta_count,
                        seen_types,
                        last_written_html_path,
                        len(final_text),
                    )
                _log_chat_io(
                    user_id=req.user_id,
                    conversation_id=req.conversation_id,
                    thread_id=tid,
                    user_message=req.message,
                    reply=final_text,
                )
                yield f"data: {json.dumps({'type':'final','text':final_text}, ensure_ascii=False)}\n\n"
            except Exception as e:  # noqa: BLE001
                _logger.exception(
                    "POST /chat/stream failed user_id=%s conversation_id=%s thread_id=%s error_type=%s error_message=%s",
                    req.user_id,
                    req.conversation_id,
                    tid,
                    type(e).__name__,
                    str(e),
                )
                detail: dict[str, Any] = {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "thread_id": tid,
                }
                if _env_flag("BC_API_RETURN_TRACEBACK", default=False):
                    detail["traceback"] = traceback.format_exc()
                yield f"data: {json.dumps({'type':'error','detail':detail}, ensure_ascii=False)}\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream; charset=utf-8")


@app.post("/reset", response_model=ResetResponse)
async def reset(req: ResetRequest) -> ResetResponse:
    """Reset a user's conversation by deleting the thread from the checkpointer."""
    assert _state is not None
    tid = _thread_id(user_id=req.user_id, conversation_id=req.conversation_id)

    # The checkpointer is held by the agent graph; we access it via .checkpointer (best-effort).
    checkpointer = getattr(_state.agent, "checkpointer", None)
    if checkpointer is not None and hasattr(checkpointer, "delete_thread"):
        checkpointer.delete_thread(tid)

    # Also drop the in-process lock (fresh start)
    _state.thread_locks.pop(tid, None)

    return ResetResponse(user_id=req.user_id, conversation_id=req.conversation_id, thread_id=tid, ok=True)


