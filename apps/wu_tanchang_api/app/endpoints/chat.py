"""Chat endpoint for conversation."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import HTTPException
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from apps.wu_tanchang_api.app.models import ChatRequest, ChatResponse
from apps.wu_tanchang_api.app.state import AppState
from apps.wu_tanchang_api.app.utils import thread_id

_logger = logging.getLogger("uvicorn.error")

# The guide message shown once material has been delivered
_GUIDE_MESSAGE = """这份材料我已经准备好了。建议你预约吴探长一对一深聊，他会基于这份材料针对你的情况给出具体方案。

你可以直接联系吴探长预约时间，或者告诉我你需要什么帮助来安排这次深聊？"""


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


async def _has_delivered_material(agent: Any, tid: str) -> bool:
    """Check if the agent has already delivered material for this thread.

    Looks for a `mark_material_delivered` tool call in the checkpoint history.
    """
    checkpointer = getattr(agent, "checkpointer", None)
    if checkpointer is None:
        return False
    checkpoint = await checkpointer.aget({"configurable": {"thread_id": tid}})
    if checkpoint is None:
        return False
    messages = checkpoint.get("channel_values", {}).get("messages", [])
    for msg in messages:
        # Check for ToolMessage with matching name
        if isinstance(msg, ToolMessage) and getattr(msg, "name", None) == "mark_material_delivered":
            return True
        # Check for tool_calls on AIMessage
        if isinstance(msg, AIMessage):
            for tc in (msg.tool_calls or []):
                if tc.get("name") == "mark_material_delivered":
                    return True
    return False


async def chat(req: ChatRequest, state: AppState) -> ChatResponse:
    """Handle a conversation turn."""
    agent_name, agent = _resolve_agent(state, req.agent_name)
    tid = thread_id(agent_name=agent_name, user_id=req.user_id, conversation_id=req.conversation_id)

    # Check if material has already been delivered
    if await _has_delivered_material(agent, tid):
        return ChatResponse(
            user_id=req.user_id,
            conversation_id=req.conversation_id,
            thread_id=tid,
            reply=_GUIDE_MESSAGE,
        )

    lock = state.thread_locks.setdefault(tid, asyncio.Lock())
    async with lock:
        try:
            # Record existing message count to isolate this round's new messages
            checkpoint = await agent.checkpointer.aget({"configurable": {"thread_id": tid}}) if hasattr(agent, "checkpointer") else None
            msg_count_before = len(checkpoint.get("channel_values", {}).get("messages", [])) if checkpoint else 0

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

    # Collect all AI message content from this round (not just the last one)
    # This ensures material content is included when mark_material_delivered is called
    messages = result.get("messages", [])
    new_messages = messages[msg_count_before:]
    parts: list[str] = []
    for msg in new_messages:
        is_ai = isinstance(msg, AIMessage) or getattr(msg, "type", None) == "ai"
        if is_ai and msg.content:
            content = str(msg.content).strip()
            if content:
                parts.append(content)
    reply = "\n\n".join(parts)

    return ChatResponse(
        user_id=req.user_id,
        conversation_id=req.conversation_id,
        thread_id=tid,
        reply=reply,
    )
