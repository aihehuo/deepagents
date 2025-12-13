from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage

try:
    from fastapi import FastAPI, HTTPException
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
            # Internal API: return a helpful message for debugging.
            raise HTTPException(
                status_code=502,
                detail={
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "thread_id": tid,
                },
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

    return ChatResponse(
        user_id=req.user_id,
        conversation_id=req.conversation_id,
        thread_id=tid,
        reply=reply,
    )


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


