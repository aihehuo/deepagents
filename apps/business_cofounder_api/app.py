from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import traceback
from concurrent.futures import ThreadPoolExecutor
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


def _resolve_write_path(virtual_path: str, docs_dir: str | None = None) -> str:
    """Resolve a virtual filesystem path to the actual write path in docs_dir.
    
    DocsOnlyWriteBackend maps all writes to docs_dir, taking only the filename
    from the virtual path. This function mimics that behavior for display purposes.
    
    Args:
        virtual_path: Virtual path from the agent (e.g., "/path/to/file.md")
        docs_dir: Docs directory where files are actually written
        
    Returns:
        Actual write path in docs_dir, or original path if docs_dir is not available
    """
    if not virtual_path or not docs_dir:
        return virtual_path
    
    try:
        from pathlib import Path
        docs = Path(docs_dir).expanduser().resolve()
        
        # DocsOnlyWriteBackend._map_write_path extracts just the filename
        filename = Path(virtual_path).name or "output.txt"
        actual_path = (docs / filename).resolve()
        
        return str(actual_path)
    except Exception:  # noqa: BLE001
        return virtual_path


def _resolve_read_path(virtual_path: str, backend_root_dir: str | None = None) -> str:
    """Resolve a virtual filesystem path to actual read path.
    
    For reads, the path is resolved relative to the backend's root directory.
    Virtual paths start with "/" and are resolved relative to root_dir.
    
    Args:
        virtual_path: Virtual path (e.g., "/" or "/path/to/file")
        backend_root_dir: Root directory of the backend
        
    Returns:
        Resolved absolute path, or original path if resolution fails
    """
    if not virtual_path:
        return virtual_path
    
    # If path is already absolute (contains directory separators and looks like absolute path)
    # Check if it looks like an absolute path that shouldn't be resolved
    path_obj = None
    try:
        from pathlib import Path
        path_obj = Path(virtual_path)
        # If it's already an absolute path and exists or looks like a real absolute path
        if path_obj.is_absolute() and len(path_obj.parts) > 2:
            # Check if it starts with common absolute path prefixes
            if str(path_obj).startswith(("/Users/", "/home/", "/tmp/", "/var/", "/opt/", "/usr/")):
                return str(path_obj.resolve())
    except Exception:  # noqa: BLE001
        pass
    
    # If no backend_root_dir, return as-is
    if not backend_root_dir:
        return virtual_path
    
    try:
        from pathlib import Path
        root = Path(backend_root_dir).resolve()
        
        # Virtual paths start with "/" - remove it and resolve relative to root_dir
        if virtual_path == "/":
            return str(root)
        
        # Remove leading slash and resolve
        relative_path = virtual_path.lstrip("/")
        if not relative_path:
            return str(root)
        
        resolved = (root / relative_path).resolve()
        return str(resolved)
    except Exception:  # noqa: BLE001
        return virtual_path


def _format_tool_call_progress(tool_name: str, tool_args: dict[str, Any] | None = None, docs_dir: str | None = None, backend_root_dir: str | None = None) -> str:
    """Format a progress message for a tool call, including relevant parameters.
    
    Note: File paths shown are virtual filesystem paths (relative to agent's working directory),
    not absolute local filesystem paths.
    
    Args:
        tool_name: Name of the tool being called
        tool_args: Dictionary of tool call arguments
        
    Returns:
        Formatted progress message string
    """
    if not tool_args:
        return f"Calling {tool_name}..."
    
    # Extract relevant parameters based on tool name
    if tool_name == "read_file":
        file_path = tool_args.get("file_path", "")
        offset = tool_args.get("offset")
        limit = tool_args.get("limit")
        if file_path:
            # For reads, resolve relative to backend root
            actual_path = _resolve_read_path(file_path, backend_root_dir)
            parts = [f"Reading {actual_path}"]
            if offset is not None or limit is not None:
                offset_str = str(offset) if offset is not None else "0"
                limit_str = f", limit={limit}" if limit is not None else ""
                parts.append(f" (offset={offset_str}{limit_str})")
            return "".join(parts)
    
    elif tool_name == "write_file":
        file_path = tool_args.get("file_path", "")
        if file_path:
            # For writes, show actual path in docs_dir (DocsOnlyWriteBackend maps all writes there)
            actual_path = _resolve_write_path(file_path, docs_dir)
            return f"Writing {actual_path}"
    
    elif tool_name == "edit_file":
        file_path = tool_args.get("file_path", "")
        if file_path:
            # For edits, show actual path in docs_dir (DocsOnlyWriteBackend maps all edits there)
            actual_path = _resolve_write_path(file_path, docs_dir)
            return f"Editing {actual_path}"
    
    elif tool_name == "ls" or tool_name == "list_files":
        path = tool_args.get("path", "")
        if path:
            # For directory listing, resolve relative to backend root
            actual_path = _resolve_read_path(path, backend_root_dir)
            return f"Listing files in {actual_path}"
    
    elif tool_name == "glob":
        pattern = tool_args.get("pattern", "")
        path = tool_args.get("path", "")
        if pattern and path:
            # For glob, resolve relative to backend root
            actual_path = _resolve_read_path(path, backend_root_dir)
            return f"Globbing '{pattern}' in {actual_path}"
        elif pattern:
            return f"Globbing '{pattern}'"
        elif path:
            # For glob, resolve relative to backend root
            actual_path = _resolve_read_path(path, backend_root_dir)
            return f"Globbing in {actual_path}"
    
    elif tool_name == "grep":
        pattern = tool_args.get("pattern", "")
        path = tool_args.get("path", "")
        if pattern and path:
            # For grep, resolve relative to backend root
            actual_path = _resolve_read_path(path, backend_root_dir)
            return f"Searching for '{pattern[:30]}...' in {actual_path}"
        elif pattern:
            return f"Searching for '{pattern[:30]}...'"
        elif path:
            # For grep, resolve relative to backend root
            actual_path = _resolve_read_path(path, backend_root_dir)
            return f"Searching in {actual_path}"
    
    elif tool_name == "execute" or tool_name == "shell":
        command = tool_args.get("command", "")
        if command:
            # Truncate long commands
            cmd_preview = command[:50] + "..." if len(command) > 50 else command
            return f"Executing: {cmd_preview}"
    
    elif tool_name == "task":
        subagent_type = tool_args.get("subagent_type", "")
        if subagent_type:
            return f"Delegating to {subagent_type} subagent"
    
    # Default: just tool name
    return f"Calling {tool_name}..."


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
    # Docs directory where agent writes files (DocsOnlyWriteBackend constraint)
    docs_dir: str | None = None


app = FastAPI(title="Business Co-Founder Agent API", version="0.1.0")
_state: _AppState | None = None

# Async networking in CPython uses a threadpool for DNS resolution (loop.getaddrinfo -> run_in_executor).
# Some production hosts have extremely low thread limits; to avoid "can't start new thread" under load,
# we install a tiny, fixed-size default executor and warm it up at startup.
_ASYNCIO_DEFAULT_EXECUTOR: ThreadPoolExecutor | None = None


def _patch_openai_no_thread() -> None:
    """Patch OpenAI python SDK to avoid asyncio.to_thread in ultra-restricted environments.

    Some production environments have extremely low thread limits and crash with:
      RuntimeError: can't start new thread

    The OpenAI SDK's async path calls asyncio.to_thread() for small sync helpers (e.g. platform detection).
    If thread creation is disallowed, that fails. This patch replaces that helper with a direct call.

    Enable with: BC_API_OPENAI_NO_THREAD=1
    """
    if not _env_flag("BC_API_OPENAI_NO_THREAD", default=False):
        return
    try:
        import openai._utils._sync as _openai_sync  # type: ignore
    except Exception:
        return

    async def _to_thread_noop(func, /, *args, **kwargs):  # type: ignore[no-untyped-def]
        return func(*args, **kwargs)

    try:
        _openai_sync.to_thread = _to_thread_noop  # type: ignore[attr-defined]
        _logger.info("Applied BC_API_OPENAI_NO_THREAD patch (openai._utils._sync.to_thread).")
    except Exception:
        return


async def _configure_asyncio_default_executor() -> None:
    global _ASYNCIO_DEFAULT_EXECUTOR
    if _ASYNCIO_DEFAULT_EXECUTOR is not None:
        return
    max_workers = int(os.environ.get("BC_API_ASYNCIO_EXECUTOR_WORKERS", "1"))
    if max_workers < 1:
        max_workers = 1
    _ASYNCIO_DEFAULT_EXECUTOR = ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="bc-asyncio",
    )
    loop = asyncio.get_running_loop()
    loop.set_default_executor(_ASYNCIO_DEFAULT_EXECUTOR)
    # Warm up: forces the executor to start at least one worker thread now,
    # so later DNS lookups don't try (and fail) to spawn a new thread.
    try:
        await loop.run_in_executor(None, lambda: None)
        _logger.info("Configured asyncio default executor (workers=%s).", max_workers)
    except Exception as e:  # noqa: BLE001
        _logger.warning(
            "Failed to warm up asyncio default executor (workers=%s): %s: %s. "
            "Async DNS/networking may fail with 'can't start new thread'.",
            max_workers,
            type(e).__name__,
            str(e),
        )


@app.on_event("startup")
async def _startup() -> None:
    global _state
    await _configure_asyncio_default_executor()
    _patch_openai_no_thread()
    agent, checkpoints_path = create_business_cofounder_agent(agent_id="business_cofounder_agent")
    
    # Extract docs_dir from agent configuration
    # The backend is wrapped in DocsOnlyWriteBackend which constrains all writes to docs_dir.
    from pathlib import Path
    docs_dir = str(Path.home() / ".deepagents" / "business_cofounder_api" / "docs")
    
    _state = _AppState(
        agent=agent,
        checkpoints_path=str(checkpoints_path),
        thread_locks={},
        docs_dir=docs_dir,
    )


@app.get("/health")
async def health() -> dict[str, str]:
    _logger.info("GET /health - received request")
    assert _state is not None
    return {"status": "ok", "checkpoints_path": _state.checkpoints_path}


@app.get("/state")
async def get_state(user_id: str, conversation_id: str = "default") -> dict[str, Any]:
    """Debug endpoint: return current milestone flags + todo list from checkpoint state.

    Disabled by default. Enable with:
      BC_API_ENABLE_STATE_ENDPOINT=1
    """
    _logger.info("GET /state - received request (user_id=%s, conversation_id=%s)", user_id, conversation_id)
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
    _logger.info(
        "POST /chat - received request (user_id=%s, conversation_id=%s, message_len=%d)",
        req.user_id,
        req.conversation_id,
        len(req.message),
    )
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
    - data: {"type":"delta","text":"..."}  (many) - text chunks from assistant
    - data: {"type":"progress","message":"..."}  (many) - progress updates during execution
    - data: {"type":"final","text":"..."}  (once) - final complete response
    - data: {"type":"error","detail":{...}} (once, if error)
    
    Progress updates are sent when:
    - Tool calls are being prepared or executed
    - Nodes in the agent graph are being processed
    - Tool execution completes
    """
    _logger.info(
        "POST /chat/stream - received request (user_id=%s, conversation_id=%s, message_len=%d)",
        req.user_id,
        req.conversation_id,
        len(req.message),
    )
    assert _state is not None
    tid = _thread_id(user_id=req.user_id, conversation_id=req.conversation_id)

    lock = _state.thread_locks.get(tid)
    if lock is None:
        lock = asyncio.Lock()
        _state.thread_locks[tid] = lock

    async def _gen():
        import time
        final_parts: list[str] = []
        delta_count = 0
        seen_types: dict[str, int] = {}
        last_written_html_path: str | None = None
        last_progress_update: float = 0.0
        # Track tool calls by ID to match with ToolMessages
        tool_call_args_cache: dict[str, dict[str, Any]] = {}
        # Track model call stats - start timing from the beginning of the request
        request_start_time = time.time()
        model_call_start_time: float | None = None
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

                    # Handle UPDATES stream mode - provides progress information
                    if current_stream_mode == "updates":
                        # Updates stream structure: dict where keys are node names or special markers
                        # Example: {"node_name": {"tool_calls": [...]}} or {"__interrupt__": [...]}
                        if isinstance(data, dict):
                            # Skip special markers like "__interrupt__"
                            for key, update_data in data.items():
                                if key.startswith("__") and key.endswith("__"):
                                    continue  # Skip special markers
                                
                                node_name = key
                                print(f"node_name: {node_name}")
                                # update_data is the actual update content for this node
                                if not isinstance(update_data, dict):
                                    continue
                                # Send progress update (throttle to avoid spam)
                                import time
                                now = time.time()
                                if now - last_progress_update > 0:  # Max once per 0.5 seconds
                                    # Special handling for "tools" node - contains ToolMessages with results
                                    if node_name == "tools":
                                        messages = update_data.get("messages", [])
                                        for msg in messages:
                                            # Check if it's a ToolMessage (tool execution result)
                                            if isinstance(msg, ToolMessage) or (isinstance(msg, dict) and msg.get("type") == "tool"):
                                                tool_name = msg.get("name", "") if isinstance(msg, dict) else getattr(msg, "name", "")
                                                tool_call_id = msg.get("tool_call_id", "") if isinstance(msg, dict) else getattr(msg, "tool_call_id", "")
                                                
                                                # Look up cached tool args using tool_call_id
                                                cached_tool_info = tool_call_args_cache.get(tool_call_id, {}) if tool_call_id else {}
                                                cached_args = cached_tool_info.get("args", {})
                                                
                                                if tool_name:
                                                    # Format progress message with file path from cached args
                                                    from pathlib import Path
                                                    docs_dir = _state.docs_dir if _state else None
                                                    backend_root_dir = str(Path.cwd()) if _state else None
                                                    progress_msg = _format_tool_call_progress(tool_name, cached_args, docs_dir, backend_root_dir)
                                                    payload = {"type": "progress", "message": progress_msg}
                                                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                                                    last_progress_update = now
                                        continue
                                    
                                    # For "model" node, tool_calls are in the AIMessage within messages
                                    if node_name == "model":
                                        # Track start time for this model call if not already set
                                        # Note: The model node appears AFTER completion in the updates stream,
                                        # so we use request_start_time as an approximation
                                        # (this includes middleware processing time, not just model call time)
                                        if model_call_start_time is None:
                                            model_call_start_time = request_start_time
                                        
                                        messages = update_data.get("messages", [])
                                        for msg in messages:
                                            # Check if it's an AIMessage with tool_calls
                                            if isinstance(msg, dict):
                                                msg_type = msg.get("type", "")
                                                if msg_type == "ai":
                                                    # Extract token usage and stats - try multiple locations
                                                    response_metadata = msg.get("response_metadata", {}) or {}
                                                    usage_metadata = msg.get("usage_metadata") or response_metadata.get("usage_metadata") or {}
                                                    
                                                    # Extract token counts from various possible locations
                                                    input_tokens = 0
                                                    output_tokens = 0
                                                    
                                                    # Try usage_metadata dict
                                                    if isinstance(usage_metadata, dict):
                                                        input_tokens = usage_metadata.get("input_tokens") or usage_metadata.get("prompt_tokens") or 0
                                                        output_tokens = usage_metadata.get("output_tokens") or usage_metadata.get("completion_tokens") or 0
                                                    
                                                    # Try response_metadata directly
                                                    if isinstance(response_metadata, dict):
                                                        if not input_tokens:
                                                            input_tokens = response_metadata.get("input_tokens") or response_metadata.get("prompt_tokens") or 0
                                                        if not output_tokens:
                                                            output_tokens = response_metadata.get("output_tokens") or response_metadata.get("completion_tokens") or 0
                                                    
                                                    # Try top-level message fields
                                                    if not input_tokens:
                                                        input_tokens = msg.get("input_tokens") or msg.get("prompt_tokens") or 0
                                                    if not output_tokens:
                                                        output_tokens = msg.get("output_tokens") or msg.get("completion_tokens") or 0
                                                    
                                                    # Calculate processing time
                                                    # Use request_start_time as the baseline since model node appears after completion
                                                    processing_time = time.time() - request_start_time
                                                    # Reset model_call_start_time for potential next model call in same request
                                                    model_call_start_time = None
                                                    
                                                    # Print stats (even if zero, for debugging)
                                                    _logger.info(
                                                        "[LLM Call Stats] input_tokens=%d, output_tokens=%d, processing_time=%.2fs, response_metadata_keys=%s",
                                                        input_tokens,
                                                        output_tokens,
                                                        processing_time,
                                                        list(response_metadata.keys()) if isinstance(response_metadata, dict) else [],
                                                    )
                                                    
                                                    tool_calls = msg.get("tool_calls", [])
                                                    if tool_calls:
                                                        for tc in tool_calls[:1]:  # Just first tool call
                                                            if isinstance(tc, dict):
                                                                tool_name = tc.get("name", "")
                                                                tool_call_id = tc.get("id", "") or tc.get("tool_call_id", "")
                                                                # Try multiple ways to get args
                                                                tool_args = tc.get("args", {}) or tc.get("arguments", {})
                                                                # Handle case where args might be nested under "function"
                                                                if not tool_args and "function" in tc:
                                                                    func_data = tc.get("function", {})
                                                                    if isinstance(func_data, dict):
                                                                        args_str = func_data.get("arguments", "")
                                                                        if isinstance(args_str, str):
                                                                            try:
                                                                                tool_args = json.loads(args_str)
                                                                            except Exception:
                                                                                tool_args = {}
                                                                        else:
                                                                            tool_args = func_data.get("arguments", {})
                                                                # If args is a string (JSON), parse it
                                                                elif isinstance(tool_args, str):
                                                                    try:
                                                                        tool_args = json.loads(tool_args)
                                                                    except Exception:
                                                                        tool_args = {}
                                                                
                                                                # Cache tool call args by ID for later use with ToolMessages
                                                                if tool_call_id and tool_name:
                                                                    tool_call_args_cache[tool_call_id] = {"name": tool_name, "args": tool_args}
                                                                
                                                                if tool_name:
                                                                    from pathlib import Path
                                                                    docs_dir = _state.docs_dir if _state else None
                                                                    backend_root_dir = str(Path.cwd()) if _state else None
                                                                    progress_msg = _format_tool_call_progress(tool_name, tool_args, docs_dir, backend_root_dir)
                                                                    payload = {"type": "progress", "message": progress_msg}
                                                                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                                                                    last_progress_update = now
                                                        break  # Only process first message with tool_calls
                                            # Handle AIMessage objects (not dicts)
                                            elif hasattr(msg, "tool_calls") and msg.tool_calls:
                                                import time
                                                # Extract token usage and stats from AIMessage object
                                                # Use request_start_time as the baseline since model node appears after completion
                                                processing_time = time.time() - request_start_time
                                                # Reset model_call_start_time for potential next model call in same request
                                                model_call_start_time = None
                                                
                                                # Try to get usage_metadata from the message
                                                input_tokens = 0
                                                output_tokens = 0
                                                
                                                usage_metadata = getattr(msg, "usage_metadata", None)
                                                if usage_metadata:
                                                    if isinstance(usage_metadata, dict):
                                                        input_tokens = usage_metadata.get("input_tokens") or usage_metadata.get("prompt_tokens") or 0
                                                        output_tokens = usage_metadata.get("output_tokens") or usage_metadata.get("completion_tokens") or 0
                                                    else:
                                                        # Try as object with attributes
                                                        input_tokens = getattr(usage_metadata, "input_tokens", None) or getattr(usage_metadata, "prompt_tokens", None) or 0
                                                        output_tokens = getattr(usage_metadata, "output_tokens", None) or getattr(usage_metadata, "completion_tokens", None) or 0
                                                
                                                # Try response_metadata if usage_metadata didn't work
                                                if not input_tokens and not output_tokens:
                                                    response_metadata = getattr(msg, "response_metadata", None)
                                                    if response_metadata:
                                                        if isinstance(response_metadata, dict):
                                                            input_tokens = response_metadata.get("input_tokens") or response_metadata.get("prompt_tokens") or 0
                                                            output_tokens = response_metadata.get("output_tokens") or response_metadata.get("completion_tokens") or 0
                                                        else:
                                                            input_tokens = getattr(response_metadata, "input_tokens", None) or getattr(response_metadata, "prompt_tokens", None) or 0
                                                            output_tokens = getattr(response_metadata, "output_tokens", None) or getattr(response_metadata, "completion_tokens", None) or 0
                                                
                                                # Print stats (with debug info)
                                                _logger.info(
                                                    "[LLM Call Stats] input_tokens=%d, output_tokens=%d, processing_time=%.2fs, has_usage_metadata=%s, has_response_metadata=%s",
                                                    input_tokens,
                                                    output_tokens,
                                                    processing_time,
                                                    usage_metadata is not None,
                                                    hasattr(msg, "response_metadata"),
                                                )
                                                
                                                for tc in msg.tool_calls[:1]:
                                                    if isinstance(tc, dict):
                                                        tool_name = tc.get("name", "")
                                                        tool_call_id = tc.get("id", "") or tc.get("tool_call_id", "")
                                                        tool_args = tc.get("args", {}) or tc.get("arguments", {})
                                                        if isinstance(tool_args, str):
                                                            try:
                                                                tool_args = json.loads(tool_args)
                                                            except Exception:
                                                                tool_args = {}
                                                        
                                                        # Cache tool call args
                                                        if tool_call_id and tool_name:
                                                            tool_call_args_cache[tool_call_id] = {"name": tool_name, "args": tool_args}
                                                        
                                                        if tool_name:
                                                            from pathlib import Path
                                                            docs_dir = _state.docs_dir if _state else None
                                                            backend_root_dir = str(Path.cwd()) if _state else None
                                                            progress_msg = _format_tool_call_progress(tool_name, tool_args, docs_dir, backend_root_dir)
                                                            payload = {"type": "progress", "message": progress_msg}
                                                            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                                                            last_progress_update = now
                                                break
                                    else:
                                        # For other nodes, try to extract tool call info from the update data
                                        tool_calls = update_data.get("tool_calls", [])
                                        if tool_calls:
                                            for tc in tool_calls[:1]:  # Just first tool call
                                                if isinstance(tc, dict):
                                                    tool_name = tc.get("name", "")
                                                    # Try multiple ways to get args - different providers structure this differently
                                                    tool_args = tc.get("args", {}) or tc.get("arguments", {})
                                                    # Handle case where args might be nested under "function"
                                                    if not tool_args and "function" in tc:
                                                        func_data = tc.get("function", {})
                                                        if isinstance(func_data, dict):
                                                            args_str = func_data.get("arguments", "")
                                                            if isinstance(args_str, str):
                                                                try:
                                                                    tool_args = json.loads(args_str)
                                                                except Exception:
                                                                    tool_args = {}
                                                            else:
                                                                tool_args = func_data.get("arguments", {})
                                                    # If args is a string (JSON), parse it
                                                    elif isinstance(tool_args, str):
                                                        try:
                                                            tool_args = json.loads(tool_args)
                                                        except Exception:
                                                            tool_args = {}
                                                    
                                                    if tool_name:
                                                        from pathlib import Path
                                                        docs_dir = _state.docs_dir if _state else None
                                                        backend_root_dir = str(Path.cwd()) if _state else None
                                                        progress_msg = _format_tool_call_progress(tool_name, tool_args, docs_dir, backend_root_dir)
                                                        payload = {"type": "progress", "message": progress_msg}
                                                        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                                                        last_progress_update = now
                                        else:
                                            # Generic node execution (no tool calls, just node processing)
                                            progress_msg = f"Processing {node_name}..."
                                            payload = {"type": "progress", "message": progress_msg}
                                            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                                            last_progress_update = now
                        continue

                    # Handle MESSAGES stream mode
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
                        tool_call_id = getattr(message, "tool_call_id", "") or ""
                        
                        # Try to get tool args from cache using tool_call_id
                        cached_tool_info = tool_call_args_cache.get(tool_call_id, {}) if tool_call_id else {}
                        cached_args = cached_tool_info.get("args", {})
                        if cached_tool_info.get("name"):
                            tool_name = cached_tool_info["name"]  # Use cached name if available
                        
                        # Send progress update when tool execution completes
                        if tool_name:
                            # Try to extract file path from tool content or cached args
                            file_path = None
                            
                            # First, try to get file_path from cached args (most reliable)
                            if cached_args:
                                file_path = cached_args.get("file_path", "") or cached_args.get("path", "")
                            
                            # Fallback: try to extract from tool content
                            if not file_path:
                                if tool_name == "write_file" and isinstance(tool_content, str):
                                    # Filesystem tool returns: "Updated file <path>"
                                    prefix = "Updated file "
                                    if tool_content.startswith(prefix):
                                        file_path = tool_content[len(prefix) :].strip()
                                        if file_path.lower().endswith(".html"):
                                            last_written_html_path = file_path
                                elif tool_name == "read_file" and isinstance(tool_content, str):
                                    # Try to extract file path from read_file content
                                    # read_file content might contain file path info, or we can look for patterns
                                    # For now, try to find file path in the content if it's a short error message
                                    if len(tool_content) < 200:
                                        # Look for common patterns that might indicate file path
                                        # Try to find absolute paths in the content
                                        path_match = re.search(r'/(?:[^/\s]+/)*[^/\s]+', tool_content)
                                        if path_match:
                                            file_path = path_match.group(0)
                            
                            # Format completion message with file path if available
                            if file_path:
                                progress_msg = f"Completed {tool_name}: {file_path}"
                            else:
                                progress_msg = f"Completed {tool_name}"
                            payload = {"type": "progress", "message": progress_msg}
                            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                        continue

                    # Ignore human echoes
                    if isinstance(message, HumanMessage):
                        continue

                    # Stream assistant output: handle both full messages and streaming chunks
                    if not isinstance(message, (AIMessage, AIMessageChunk)):
                        # Some providers may not use these exact classes; fall back on type=="ai" when present.
                        if getattr(message, "type", None) != "ai":
                            continue

                    # Extract token usage from AIMessage if available (messages stream has more complete metadata)
                    if isinstance(message, (AIMessage, AIMessageChunk)) or getattr(message, "type", None) == "ai":
                        # Try to extract token usage - this is often more complete in messages stream
                        usage_metadata = getattr(message, "usage_metadata", None)
                        response_metadata = getattr(message, "response_metadata", None)
                        
                        input_tokens = 0
                        output_tokens = 0
                        
                        # Try usage_metadata first
                        if usage_metadata:
                            if isinstance(usage_metadata, dict):
                                input_tokens = usage_metadata.get("input_tokens") or usage_metadata.get("prompt_tokens") or 0
                                output_tokens = usage_metadata.get("output_tokens") or usage_metadata.get("completion_tokens") or 0
                            else:
                                input_tokens = getattr(usage_metadata, "input_tokens", None) or getattr(usage_metadata, "prompt_tokens", None) or 0
                                output_tokens = getattr(usage_metadata, "output_tokens", None) or getattr(usage_metadata, "completion_tokens", None) or 0
                        
                        # Try response_metadata if usage_metadata didn't work
                        if (not input_tokens and not output_tokens) and response_metadata:
                            if isinstance(response_metadata, dict):
                                input_tokens = response_metadata.get("input_tokens") or response_metadata.get("prompt_tokens") or 0
                                output_tokens = response_metadata.get("output_tokens") or response_metadata.get("completion_tokens") or 0
                            else:
                                input_tokens = getattr(response_metadata, "input_tokens", None) or getattr(response_metadata, "prompt_tokens", None) or 0
                                output_tokens = getattr(response_metadata, "output_tokens", None) or getattr(response_metadata, "completion_tokens", None) or 0
                        
                        # Log token usage if found
                        if input_tokens or output_tokens:
                            _logger.info(
                                "[LLM Call Stats from messages stream] input_tokens=%d, output_tokens=%d",
                                input_tokens,
                                output_tokens,
                            )
                    
                    # Check for tool calls in AI messages and send progress updates
                    tool_calls = getattr(message, "tool_calls", None)
                    if tool_calls:
                        for tc in tool_calls[:3]:  # Limit to first 3 tool calls
                            if isinstance(tc, dict):
                                tool_name = tc.get("name", "")
                                tool_call_id = tc.get("id", "") or tc.get("tool_call_id", "")
                                # Try multiple ways to get args - different providers structure this differently
                                tool_args = tc.get("args", {}) or tc.get("arguments", {})
                                # Handle case where args might be nested under "function"
                                if not tool_args and "function" in tc:
                                    func_data = tc.get("function", {})
                                    if isinstance(func_data, dict):
                                        args_str = func_data.get("arguments", "")
                                        if isinstance(args_str, str):
                                            try:
                                                tool_args = json.loads(args_str)
                                            except Exception:
                                                tool_args = {}
                                        else:
                                            tool_args = func_data.get("arguments", {})
                                # If args is a string (JSON), parse it
                                elif isinstance(tool_args, str):
                                    try:
                                        tool_args = json.loads(tool_args)
                                    except Exception:
                                        tool_args = {}
                                
                                # Cache tool call args by ID for later use with ToolMessages
                                if tool_call_id and tool_name:
                                    tool_call_args_cache[tool_call_id] = {"name": tool_name, "args": tool_args}
                                
                                # Debug logging if enabled
                                if _env_flag("BC_API_STREAM_DEBUG", default=False):
                                    _logger.debug(
                                        "Tool call in AI message: name=%s, id=%s, args=%s, tc_keys=%s",
                                        tool_name,
                                        tool_call_id,
                                        tool_args,
                                        list(tc.keys()) if isinstance(tc, dict) else [],
                                    )
                                
                                if tool_name:
                                    from pathlib import Path
                                    docs_dir = _state.docs_dir if _state else None
                                    backend_root_dir = str(Path.cwd()) if _state else None
                                    progress_msg = _format_tool_call_progress(tool_name, tool_args, docs_dir, backend_root_dir)
                                    payload = {"type": "progress", "message": progress_msg}
                                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

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
    _logger.info(
        "POST /reset - received request (user_id=%s, conversation_id=%s)",
        req.user_id,
        req.conversation_id,
    )
    assert _state is not None
    tid = _thread_id(user_id=req.user_id, conversation_id=req.conversation_id)

    # The checkpointer is held by the agent graph; we access it via .checkpointer (best-effort).
    checkpointer = getattr(_state.agent, "checkpointer", None)
    if checkpointer is not None and hasattr(checkpointer, "delete_thread"):
        checkpointer.delete_thread(tid)

    # Also drop the in-process lock (fresh start)
    _state.thread_locks.pop(tid, None)

    return ResetResponse(user_id=req.user_id, conversation_id=req.conversation_id, thread_id=tid, ok=True)


