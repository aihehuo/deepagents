"""Business Co-Founder API Server - Refactored to use background threads.

This refactored version matches BPGenerationAgent's threading pattern:
- Graph execution runs in background threads using sync invoke/stream
- FastAPI endpoints remain async but delegate heavy work to threads
- Avoids "can't start new thread" errors by not using async LLM calls
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import threading
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


# Version information for deployment verification
# Can be set via environment variable BC_API_VERSION at build/deploy time
# Falls back to git commit hash if available, or "dev" if not
def _get_version() -> str:
    """Get version string for deployment verification."""
    # First check environment variable (set at build/deploy time)
    env_version = os.environ.get("BC_API_VERSION")
    if env_version:
        return env_version
    
    # Try to get git commit hash
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        )
        if result.returncode == 0 and result.stdout.strip():
            commit_hash = result.stdout.strip()
            # Also try to get commit date
            try:
                date_result = subprocess.run(
                    ["git", "log", "-1", "--format=%ai", commit_hash],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                )
                if date_result.returncode == 0 and date_result.stdout.strip():
                    date_str = date_result.stdout.strip().split()[0]  # Just the date part
                    return f"{commit_hash} ({date_str})"
            except Exception:  # noqa: BLE001
                pass
            return commit_hash
    except Exception:  # noqa: BLE001
        pass
    
    return "dev"


_API_VERSION = _get_version()


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
    # Thread-level locks to ensure serialized access per thread_id
    thread_locks: dict[str, threading.Lock]
    # Global lock to serialize all sync graph execution (no thread pool executor)
    # Since thread creation is blocked in this environment, we run sync code directly
    # and use this lock to ensure only one graph execution happens at a time
    sync_execution_lock: threading.Lock


app = FastAPI(title="Business Co-Founder Agent API (Refactored)", version="0.2.0")
_state: _AppState | None = None


def _run_chat_sync(
    agent: Any,
    inputs: dict[str, Any],
    config: dict[str, Any],
    result_queue: queue.Queue,
    error_queue: queue.Queue,
) -> None:
    """Run agent.invoke synchronously in a background thread."""
    try:
        result = agent.invoke(inputs, config)
        result_queue.put(result)
    except Exception as e:  # noqa: BLE001
        error_queue.put((type(e).__name__, str(e), traceback.format_exc()))
        # Put None as error marker so queue.get() doesn't block forever
        result_queue.put(None)


def _run_chat_sync_with_lock(
    agent: Any,
    inputs: dict[str, Any],
    config: dict[str, Any],
    result_queue: queue.Queue,
    error_queue: queue.Queue,
    per_thread_lock: threading.Lock,
) -> None:
    """Run agent.invoke synchronously with locks for serialization.
    
    Uses both:
    - per_thread_lock: serializes requests for the same thread_id
    - _state.sync_execution_lock: serializes all graph executions globally (since we can't use thread pool)
    """
    thread_id = config.get("configurable", {}).get("thread_id", "unknown")
    _logger.info("[_run_chat_sync_with_lock] Starting sync execution for thread_id=%s", thread_id)
    
    # First acquire per-thread lock, then global lock
    # This ensures thread_id serialization, and global serialization to avoid thread creation
    try:
        _logger.info("[_run_chat_sync_with_lock] Acquiring locks for thread_id=%s", thread_id)
        with per_thread_lock, _state.sync_execution_lock:
            _logger.info("[_run_chat_sync_with_lock] Locks acquired, calling agent.invoke for thread_id=%s", thread_id)
            try:
                result = agent.invoke(inputs, config)
                _logger.info(
                    "[_run_chat_sync_with_lock] agent.invoke completed for thread_id=%s, result keys=%s",
                    thread_id,
                    list(result.keys()) if isinstance(result, dict) else type(result).__name__,
                )
                result_queue.put(result)
                _logger.info("[_run_chat_sync_with_lock] Result put in queue for thread_id=%s", thread_id)
            except Exception as e:  # noqa: BLE001
                _logger.exception(
                    "[_run_chat_sync_with_lock] Exception during agent.invoke for thread_id=%s: %s: %s",
                    thread_id,
                    type(e).__name__,
                    str(e),
                )
                error_queue.put((type(e).__name__, str(e), traceback.format_exc()))
                # Put None as error marker so queue.get() doesn't block forever
                result_queue.put(None)
                _logger.info("[_run_chat_sync_with_lock] Error put in queues for thread_id=%s", thread_id)
        _logger.info("[_run_chat_sync_with_lock] Locks released, function completed for thread_id=%s", thread_id)
    except Exception as e:  # noqa: BLE001
        _logger.exception(
            "[_run_chat_sync_with_lock] Exception acquiring locks for thread_id=%s: %s: %s",
            thread_id,
            type(e).__name__,
            str(e),
        )
        error_queue.put((type(e).__name__, str(e), traceback.format_exc()))
        result_queue.put(None)


def _run_chat_stream_sync(
    agent: Any,
    inputs: dict[str, Any],
    config: dict[str, Any],
    chunk_queue: queue.Queue,
    error_queue: queue.Queue,
) -> None:
    """Run agent.stream synchronously in a background thread, putting chunks into queue."""
    try:
        for chunk in agent.stream(
            inputs,
            config,
            stream_mode=["messages", "updates"],
            subgraphs=True,
            durability="exit",
        ):
            chunk_queue.put(chunk)
        # Signal completion
        chunk_queue.put(None)
    except Exception as e:  # noqa: BLE001
        error_queue.put((type(e).__name__, str(e), traceback.format_exc()))
        chunk_queue.put(None)  # Signal completion even on error


def _run_chat_stream_sync_with_lock(
    agent: Any,
    inputs: dict[str, Any],
    config: dict[str, Any],
    chunk_queue: queue.Queue,
    error_queue: queue.Queue,
    per_thread_lock: threading.Lock,
) -> None:
    """Run agent.stream synchronously with locks for serialization.
    
    Uses both:
    - per_thread_lock: serializes requests for the same thread_id
    - _state.sync_execution_lock: serializes all graph executions globally (since we can't use thread pool)
    """
    with per_thread_lock, _state.sync_execution_lock:
        _run_chat_stream_sync(agent, inputs, config, chunk_queue, error_queue)


def _patch_langgraph_checkpoint_executor() -> None:
    """Patch LangGraph's checkpoint executor to avoid thread creation.
    
    LangGraph uses ThreadPoolExecutor internally for checkpoint operations,
    which fails in environments where thread creation is blocked.
    This patch makes ThreadPoolExecutor.submit() run synchronously instead of creating threads.
    """
    if not _env_flag("BC_API_DISABLE_CHECKPOINT_EXECUTOR", default=True):
        return
    
    try:
        from concurrent.futures import ThreadPoolExecutor
        from concurrent.futures._base import Future
        
        # Store original submit method
        _original_threadpool_submit = ThreadPoolExecutor.submit
        
        def _synchronous_submit(self, fn, /, *args, **kwargs):
            """Patched submit that runs synchronously instead of creating threads."""
            # Create a Future object to match the expected interface
            future = Future()
            try:
                # Run the function directly in the current thread
                result = fn(*args, **kwargs)
                future.set_result(result)
            except Exception as e:  # noqa: BLE001
                future.set_exception(e)
            return future
        
        # Patch ThreadPoolExecutor.submit to run synchronously
        ThreadPoolExecutor.submit = _synchronous_submit
        _logger.info(
            "Patched ThreadPoolExecutor.submit() to run synchronously "
            "(no thread creation for checkpoint operations)"
        )
        
        # Also patch get_executor_for_config to return a synchronous executor
        try:
            from langchain_core.runnables.config import get_executor_for_config
            
            class SynchronousExecutor:
                """A synchronous executor that runs tasks directly (no thread pool)."""
                def submit(self, fn, /, *args, **kwargs):
                    future = Future()
                    try:
                        result = fn(*args, **kwargs)
                        future.set_result(result)
                    except Exception as e:  # noqa: BLE001
                        future.set_exception(e)
                    return future
            
            _synchronous_executor = SynchronousExecutor()
            
            def _patched_get_executor(config):
                return _synchronous_executor
            
            import langchain_core.runnables.config as config_module
            config_module.get_executor_for_config = _patched_get_executor
            _logger.info("Also patched get_executor_for_config() to return synchronous executor")
        except Exception as e:  # noqa: BLE001
            _logger.warning("Failed to patch get_executor_for_config (non-fatal): %s", e)
            
    except Exception as e:  # noqa: BLE001
        _logger.warning("Failed to patch ThreadPoolExecutor (non-fatal): %s", e)


@app.on_event("startup")
def _startup() -> None:
    global _state

    # Print version information prominently for deployment verification
    _logger.info("=" * 80)
    _logger.info("Business Co-Founder API Server - Starting")
    _logger.info("=" * 80)
    _logger.info("VERSION: %s", _API_VERSION)
    _logger.info("=" * 80)

    # Patch LangGraph's checkpoint executor to avoid thread creation
    _patch_langgraph_checkpoint_executor()

    agent, checkpoints_path = create_business_cofounder_agent(agent_id="business_cofounder_agent")
    
    # Since thread creation is blocked in this environment (even at startup),
    # we cannot use ThreadPoolExecutor. Instead, we run sync graph execution directly
    # and use a global lock to serialize all executions (only one at a time).
    # This blocks the async event loop, but ensures no thread creation errors.
    _logger.warning(
        "Running in no-thread mode: sync graph execution will run directly "
        "(blocks event loop, but avoids 'can't start new thread' errors). "
        "Requests will be serialized."
    )
    
    _state = _AppState(
        agent=agent,
        checkpoints_path=str(checkpoints_path),
        thread_locks={},
        sync_execution_lock=threading.Lock(),
    )
    _logger.info("Initialized agent (no-thread mode: sync execution with serialization)")
    _logger.info("=" * 80)
    _logger.info("Server ready - VERSION: %s", _API_VERSION)
    _logger.info("=" * 80)


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint with version information for deployment verification."""
    assert _state is not None
    return {
        "status": "ok",
        "checkpoints_path": _state.checkpoints_path,
        "version": _API_VERSION,
    }


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
        lock = threading.Lock()
        _state.thread_locks[tid] = lock

    with lock:
        checkpointer = getattr(_state.agent, "checkpointer", None)
        if checkpointer is None or not hasattr(checkpointer, "get_tuple"):
            raise HTTPException(status_code=500, detail="Agent checkpointer not available")

        try:
            ckpt_tuple = checkpointer.get_tuple({"configurable": {"thread_id": tid}})
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
    """Synchronous chat endpoint using background thread with sync invoke.

    Matches BPGenerationAgent pattern: heavy work runs in background thread using sync graph.invoke.
    """
    assert _state is not None
    tid = _thread_id(user_id=req.user_id, conversation_id=req.conversation_id)

    lock = _state.thread_locks.get(tid)
    if lock is None:
        lock = threading.Lock()
        _state.thread_locks[tid] = lock

    result_queue: queue.Queue = queue.Queue()
    error_queue: queue.Queue = queue.Queue()

    inputs = {"messages": [HumanMessage(content=req.message)]}
    config = {
        "configurable": {"thread_id": tid},
        "metadata": {"user_id": req.user_id, **(req.metadata or {})},
    }

    # Run sync invoke directly (no thread pool executor since thread creation is blocked)
    # Use asyncio.to_thread to run the sync code without blocking the event loop
    # The sync_execution_lock serializes all graph executions globally
    timeout_s = float(os.environ.get("BC_API_INVOKE_TIMEOUT_S", "300.0"))

    try:
        # Execute sync code in asyncio's default thread pool (uses existing threads if available)
        # If that also fails, we'll fall back to direct execution
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    _run_chat_sync_with_lock,
                    _state.agent,
                    inputs,
                    config,
                    result_queue,
                    error_queue,
                    lock,
                ),
                timeout=timeout_s,
            )
            # Get result from queue
            try:
                result = await asyncio.to_thread(result_queue.get, timeout=1.0)
            except queue.Empty:
                raise HTTPException(
                    status_code=502,
                    detail={"error_type": "EmptyResult", "error_message": "No result from agent (queue timeout)", "thread_id": tid},
                ) from None
        except RuntimeError as e:
            if "can't start new thread" in str(e):
                # Fallback: run directly in current thread (blocks event loop, but works)
                _logger.warning(
                    "[chat] Thread creation blocked for thread_id=%s, running sync code directly (blocks event loop)",
                    tid,
                )
                _logger.info("[chat] Calling _run_chat_sync_with_lock directly for thread_id=%s", tid)
                try:
                    _run_chat_sync_with_lock(_state.agent, inputs, config, result_queue, error_queue, lock)
                    _logger.info("[chat] _run_chat_sync_with_lock completed for thread_id=%s, checking queue", tid)
                    
                    # Check for errors first
                    if not error_queue.empty():
                        exc_type, exc_msg, exc_tb = error_queue.get()
                        _logger.error(
                            "[chat] Error found in queue for thread_id=%s: %s: %s",
                            tid,
                            exc_type,
                            exc_msg,
                        )
                        detail: dict[str, Any] = {
                            "error_type": exc_type,
                            "error_message": exc_msg,
                            "thread_id": tid,
                        }
                        if _env_flag("BC_API_RETURN_TRACEBACK", default=False):
                            detail["traceback"] = exc_tb
                        raise HTTPException(status_code=502, detail=detail) from None
                    
                    # Get result from queue
                    try:
                        result = result_queue.get(timeout=2.0)
                        _logger.info(
                            "[chat] Got result from queue for thread_id=%s, result is None=%s",
                            tid,
                            result is None,
                        )
                        if result is None:
                            raise HTTPException(
                                status_code=502,
                                detail={
                                    "error_type": "EmptyResult",
                                    "error_message": "No result from agent (None returned)",
                                    "thread_id": tid,
                                },
                            ) from None
                    except queue.Empty:
                        _logger.error("[chat] Queue timeout for thread_id=%s", tid)
                        raise HTTPException(
                            status_code=502,
                            detail={
                                "error_type": "EmptyResult",
                                "error_message": "No result from agent (queue timeout)",
                                "thread_id": tid,
                            },
                        ) from None
                except HTTPException:
                    raise
                except Exception as e2:  # noqa: BLE001
                    _logger.exception(
                        "[chat] Unexpected exception in fallback path for thread_id=%s: %s: %s",
                        tid,
                        type(e2).__name__,
                        str(e2),
                    )
                    raise HTTPException(
                        status_code=502,
                        detail={
                            "error_type": type(e2).__name__,
                            "error_message": str(e2),
                            "thread_id": tid,
                        },
                    ) from e2
            else:
                raise
    except asyncio.TimeoutError:
        _logger.error(
            "POST /chat timeout user_id=%s conversation_id=%s thread_id=%s timeout_s=%s",
            req.user_id,
            req.conversation_id,
            tid,
            timeout_s,
        )
        raise HTTPException(
            status_code=504,
            detail={"error_type": "TimeoutError", "error_message": f"Request timed out after {timeout_s}s", "thread_id": tid},
        ) from None

    # Check if result is None (error marker) or if error_queue has an error
    if result is None or not error_queue.empty():
        if not error_queue.empty():
            exc_type, exc_msg, exc_tb = error_queue.get()
        else:
            exc_type = "UnknownError"
            exc_msg = "No result from agent"
            exc_tb = ""
        _logger.exception(
            "POST /chat failed user_id=%s conversation_id=%s thread_id=%s error_type=%s error_message=%s",
            req.user_id,
            req.conversation_id,
            tid,
            exc_type,
            exc_msg,
        )
        detail: dict[str, Any] = {
            "error_type": exc_type,
            "error_message": exc_msg,
            "thread_id": tid,
        }
        if _env_flag("BC_API_RETURN_TRACEBACK", default=False):
            detail["traceback"] = exc_tb
        raise HTTPException(status_code=502, detail=detail)

    # Extract reply from result
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

    Uses background thread with sync graph.stream, bridging to async SSE via queue.

    SSE event format:
    - data: {"type":"delta","text":"..."}  (many)
    - data: {"type":"final","text":"..."}  (once)
    - data: {"type":"error","detail":{...}} (once, if error)
    """
    assert _state is not None
    tid = _thread_id(user_id=req.user_id, conversation_id=req.conversation_id)

    lock = _state.thread_locks.get(tid)
    if lock is None:
        lock = threading.Lock()
        _state.thread_locks[tid] = lock

    async def _gen():
        chunk_queue: queue.Queue = queue.Queue(maxsize=100)  # Backpressure if consumer is slow
        error_queue: queue.Queue = queue.Queue()

        inputs = {"messages": [HumanMessage(content=req.message)]}
        config = {
            "configurable": {"thread_id": tid},
            "metadata": {"user_id": req.user_id, **(req.metadata or {})},
        }

        # Start sync stream (no thread pool executor since thread creation is blocked)
        # Use asyncio.to_thread to run in background, with fallback to direct execution
        async def _start_stream():
            try:
                await asyncio.to_thread(
                    _run_chat_stream_sync_with_lock,
                    _state.agent,
                    inputs,
                    config,
                    chunk_queue,
                    error_queue,
                    lock,
                )
            except RuntimeError as e:
                if "can't start new thread" in str(e):
                    # Fallback: run directly (blocks event loop, but works)
                    _logger.warning("Thread creation blocked, running stream sync code directly (blocks event loop)")
                    _run_chat_stream_sync_with_lock(_state.agent, inputs, config, chunk_queue, error_queue, lock)
                else:
                    raise
        
        # Start stream execution (non-blocking)
        asyncio.create_task(_start_stream())

        # Bridge sync queue to async SSE generator
        # Note: lock is held by the executor thread during graph execution
        final_parts: list[str] = []
        delta_count = 0
        seen_types: dict[str, int] = {}
        last_written_html_path: str | None = None
        last_progress_update: float = 0.0
        done = False

        try:
            while not done:
                try:
                    # Poll queue with timeout - use asyncio.to_thread to avoid blocking event loop
                    chunk = await asyncio.to_thread(chunk_queue.get, timeout=0.5)
                except queue.Empty:
                    # Queue is empty, check for errors
                    if not error_queue.empty():
                        exc_type, exc_msg, exc_tb = error_queue.get()
                        detail: dict[str, Any] = {
                            "error_type": exc_type,
                            "error_message": exc_msg,
                            "thread_id": tid,
                        }
                        if _env_flag("BC_API_RETURN_TRACEBACK", default=False):
                            detail["traceback"] = exc_tb
                        yield f"data: {json.dumps({'type':'error','detail':detail}, ensure_ascii=False)}\n\n"
                        return
                    # Keep polling - executor thread is still running
                    continue

                # None marker signals completion
                if chunk is None:
                    done = True
                    break

                # Process chunk (same logic as before)
                if not isinstance(chunk, tuple) or len(chunk) != 3:
                    continue

                _namespace, current_stream_mode, data = chunk

                # Handle UPDATES stream mode - provides progress information
                if current_stream_mode == "updates":
                    # Updates stream contains node execution and tool call information
                    if isinstance(data, dict):
                        # Extract node name and status
                        node_name = data.get("node", "")
                        if node_name:
                            # Send progress update (throttle to avoid spam)
                            import time
                            now = time.time()
                            if now - last_progress_update > 0.5:  # Max once per 0.5 seconds
                                # Try to extract tool call info
                                tool_calls = data.get("tool_calls", [])
                                if tool_calls:
                                    for tc in tool_calls[:1]:  # Just first tool call
                                        tool_name = tc.get("name", "") if isinstance(tc, dict) else ""
                                        if tool_name:
                                            progress_msg = f"Calling {tool_name}..."
                                            payload = {"type": "progress", "message": progress_msg}
                                            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                                            last_progress_update = now
                                else:
                                    # Generic node execution
                                    progress_msg = f"Processing {node_name}..."
                                    payload = {"type": "progress", "message": progress_msg}
                                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                                    last_progress_update = now
                    continue

                # Handle MESSAGES stream mode
                if current_stream_mode != "messages":
                    continue

                if not isinstance(data, tuple) or len(data) != 2:
                    continue

                message, _metadata = data

                cls = type(message).__name__
                seen_types[cls] = seen_types.get(cls, 0) + 1

                # Track HTML file writes
                if isinstance(message, ToolMessage):
                    tool_name = getattr(message, "name", "") or ""
                    tool_content = getattr(message, "content", "") or ""
                    
                    # Send progress update when tool execution completes
                    if tool_name:
                        progress_msg = f"Completed {tool_name}"
                        payload = {"type": "progress", "message": progress_msg}
                        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    
                    if tool_name == "write_file" and isinstance(tool_content, str):
                        prefix = "Updated file "
                        if tool_content.startswith(prefix):
                            p = tool_content[len(prefix) :].strip()
                            if p.lower().endswith(".html"):
                                last_written_html_path = p
                    continue

                # Ignore human echoes
                if isinstance(message, HumanMessage):
                    continue

                # Stream assistant output
                if not isinstance(message, (AIMessage, AIMessageChunk)):
                    if getattr(message, "type", None) != "ai":
                        continue

                # Check for tool calls in AI messages and send progress updates
                tool_calls = getattr(message, "tool_calls", None)
                if tool_calls:
                    for tc in tool_calls[:3]:  # Limit to first 3 tool calls
                        if isinstance(tc, dict):
                            tool_name = tc.get("name", "")
                            if tool_name:
                                progress_msg = f"Preparing to call {tool_name}..."
                                payload = {"type": "progress", "message": progress_msg}
                                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

                for text in _extract_text_chunks_from_ai_message(message):
                    final_parts.append(text)
                    delta_count += 1
                    payload = {"type": "delta", "text": text}
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            final_text = "".join(final_parts).strip()

            # Fallback: if no text was streamed but an HTML file was written, read it
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

    checkpointer = getattr(_state.agent, "checkpointer", None)
    if checkpointer is not None and hasattr(checkpointer, "delete_thread"):
        checkpointer.delete_thread(tid)

    # Also drop the in-process lock (fresh start)
    _state.thread_locks.pop(tid, None)

    return ResetResponse(user_id=req.user_id, conversation_id=req.conversation_id, thread_id=tid, ok=True)

