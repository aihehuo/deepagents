"""Chat endpoint for conversation."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import HTTPException
from langchain_core.messages import AIMessage, HumanMessage

from apps.wu_tanchang_api.app.models import ChatRequest, ChatResponse
from apps.wu_tanchang_api.app.state import AppState
from apps.wu_tanchang_api.app.utils import thread_id

_logger = logging.getLogger("uvicorn.error")


def _resolve_agent(state: AppState, agent_name: str) -> tuple[str, Any]:
    """Resolve agent name to (name, agent)."""
    name = agent_name or state.default_agent
    if name not in state.agents:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "unknown_agent",
                "message": f"Unknown agent: {name}",
                "available": list(state.agents.keys()),
            },
        )
    return name, state.agents[name]


async def chat(req: ChatRequest, state: AppState) -> ChatResponse:
    """Handle a conversation turn."""
    agent_name, agent = _resolve_agent(state, req.agent_name)
    tid = thread_id(agent_name=agent_name, user_id=req.user_id, conversation_id=req.conversation_id)
    lock = state.thread_locks.setdefault(tid, asyncio.Lock())

    async with lock:
        try:
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content=req.message)]},
                {
                    "configurable": {"thread_id": tid},
                    "metadata": {"user_id": req.user_id, **(req.metadata or {})},
                },
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=502,
                detail={"error_type": type(exc).__name__, "error_message": str(exc), "thread_id": tid},
            ) from exc

    # Extract the last AI message with content
    messages = result.get("messages", [])
    reply = ""
    for msg in reversed(messages):
        is_ai = isinstance(msg, AIMessage) or getattr(msg, "type", None) == "ai"
        if is_ai and msg.content:
            reply = str(msg.content)
            break

    return ChatResponse(
        user_id=req.user_id,
        conversation_id=req.conversation_id,
        thread_id=tid,
        reply=reply,
    )
