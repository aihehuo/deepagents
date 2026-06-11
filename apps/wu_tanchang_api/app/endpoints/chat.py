"""Chat endpoints for synchronous and streaming conversations."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

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

    if not state.try_start_agent_run(tid, "chat"):
        raise HTTPException(
            status_code=409,
            detail={"error": "stream_in_progress", "message": "Agent run already in progress for this conversation", "thread_id": tid},
        )

    lock = state.thread_locks.setdefault(tid, asyncio.Lock())
    try:
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
    finally:
        state.finish_agent_run(tid, "chat")

    # Collect all AI message content from this round (not just the last one).
    # The agent should write material text into AIMessage.content before
    # calling mark_material_delivered. We collect all AI messages to
    # capture both material content and the guide message.
    messages = result.get("messages", [])
    new_messages = messages[msg_count_before:]
    parts: list[str] = []
    for msg in new_messages:
        if not isinstance(msg, AIMessage):
            continue
        content = str(msg.content).strip() if msg.content else ""
        if not content:
            continue
        # Skip system artifacts emitted by the agent
        if content.startswith("Updated todo list"):
            continue
        parts.append(content)
    reply = "\n\n".join(parts)

    return ChatResponse(
        user_id=req.user_id,
        conversation_id=req.conversation_id,
        thread_id=tid,
        reply=reply,
    )


async def chat_stream(req: ChatRequest, state: AppState) -> StreamingResponse:
    """Stream the assistant response as Server-Sent Events (SSE).

    SSE event format:
    - data: {"type":"delta","text":"..."}  (many) — text chunks from assistant
    - data: {"type":"progress","message":"..."}  (many) — progress updates during execution
    - data: {"type":"final","text":"..."}  (once) — final complete response
    - data: {"type":"error","detail":{...}} (once, if error)
    """
    agent_name, agent = _resolve_agent(state, req.agent_name)
    tid = thread_id(agent_name=agent_name, user_id=req.user_id, conversation_id=req.conversation_id)

    # Check if material has already been delivered
    if await _has_delivered_material(agent, tid):
        async def _early_gen() -> None:
            yield f"data: {json.dumps({'type':'final','text':_GUIDE_MESSAGE}, ensure_ascii=False)}\n\n"
        return StreamingResponse(_early_gen(), media_type="text/event-stream; charset=utf-8")

    async def _gen() -> None:
        final_parts: list[str] = []
        if not state.try_start_agent_run(tid, "chat_stream"):
            detail = {"error": "stream_in_progress", "message": "Agent run already in progress for this conversation", "thread_id": tid}
            yield f"data: {json.dumps({'type':'error','detail':detail}, ensure_ascii=False)}\n\n"
            return
        lock = state.thread_locks.setdefault(tid, asyncio.Lock())
        try:
            async with lock:
                try:
                    # Record existing message count to isolate this round's new messages
                    checkpoint = await agent.checkpointer.aget({"configurable": {"thread_id": tid}}) if hasattr(agent, "checkpointer") else None
                    msg_count_before = len(checkpoint.get("channel_values", {}).get("messages", [])) if checkpoint else 0

                    async for chunk in agent.astream(
                        {"messages": [HumanMessage(content=req.message)]},
                        config={
                            "configurable": {"thread_id": tid},
                            "metadata": {"user_id": req.user_id, **(req.metadata or {})},
                        },
                        stream_mode=["messages", "updates"],
                        subgraphs=True,
                    ):
                        if not isinstance(chunk, tuple) or len(chunk) != 3:
                            continue
                        _, current_stream_mode, data = chunk

                        # Handle "updates" mode — progress info
                        if current_stream_mode == "updates" and isinstance(data, dict):
                            for key, update_data in data.items():
                                if key.startswith("__") and key.endswith("__"):
                                    continue
                                if not isinstance(update_data, dict):
                                    continue
                                # Detect sub-agent / tool calls in progress
                                msgs = update_data.get("messages", [])
                                for msg in msgs:
                                    if isinstance(msg, ToolMessage) or (isinstance(msg, dict) and msg.get("type") == "tool"):
                                        tool_name = msg.get("name", "") if isinstance(msg, dict) else getattr(msg, "name", "")
                                        if tool_name:
                                            progress_msg = f"正在调用知识库: {tool_name}..."
                                            payload = {"type": "progress", "message": progress_msg}
                                            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

                        # Handle "messages" mode — text deltas
                        if current_stream_mode == "messages" and isinstance(data, tuple) and len(data) == 2:
                            msg_chunk, _metadata = data
                            if isinstance(msg_chunk, AIMessageChunk):
                                if msg_chunk.content:
                                    chunk_text = str(msg_chunk.content)
                                    final_parts.append(chunk_text)
                                    payload = {"type": "delta", "text": chunk_text}
                                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

                except Exception as exc:  # noqa: BLE001
                    _logger.error("[ChatStream] Error: %s: %s", type(exc).__name__, str(exc))
                    detail = {"error_type": type(exc).__name__, "error_message": str(exc), "thread_id": tid}
                    yield f"data: {json.dumps({'type':'error','detail':detail}, ensure_ascii=False)}\n\n"
                    return
        finally:
            state.finish_agent_run(tid, "chat_stream")

        # Build final reply from all collected text
        reply = "".join(final_parts).strip()

        # Fallback: if no text was streamed, extract from checkpoint messages
        if not reply:
            messages = checkpoint.get("channel_values", {}).get("messages", []) if checkpoint else []
            new_messages = messages[msg_count_before:]
            parts: list[str] = []
            for msg in new_messages:
                if not isinstance(msg, AIMessage):
                    continue
                content = str(msg.content).strip() if msg.content else ""
                if not content:
                    continue
                if content.startswith("Updated todo list"):
                    continue
                parts.append(content)
            reply = "\n\n".join(parts)

        yield f"data: {json.dumps({'type':'final','text':reply}, ensure_ascii=False)}\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream; charset=utf-8")
