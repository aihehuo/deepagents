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
from apps.wu_tanchang_api.app.utils import get_progress_status, thread_id

_logger = logging.getLogger("uvicorn.error")

# The guide message shown once material has been delivered
_GUIDE_MESSAGE = """这份材料我已经准备好了。建议你预约吴探长一对一深聊，他会基于这份材料针对你的情况给出具体方案。

你可以直接联系吴探长预约时间，或者告诉我你需要什么帮助来安排这次深聊？"""


async def resolve_dynamic_agent(
    state: AppState,
    user_id: str,
    metadata: dict[str, Any],
    agent_name_override: str = "",
) -> tuple[str, Any]:
    """Dynamically route and resolve the agent based on user_id and calendar_id.

    If user_id and calendar_id match, it routes to:
      - workspace_{calendar_id}_owner
      - Fallback 1: workspace_owner_default
      - Fallback 2: workspace_owner
    Otherwise, it routes to:
      - workspace_{calendar_id}
      - Fallback 1: workspace_default
      - Fallback 2: workspace
    """
    from pathlib import Path
    from apps.wu_tanchang_api.config import WuAgentConfig
    from apps.wu_tanchang_api.agent_factory.agent import create_agent
    import re

    # 1. Extract IDs
    effective_user = metadata.get("user_a_id") or metadata.get("user_id") or user_id
    effective_calendar = metadata.get("user_b_id") or metadata.get("calendar_id")

    # Whitelist check for calendar_id / user_b_id (S3) to prevent path traversal
    if effective_calendar is not None:
        if not re.fullmatch(r"^[A-Za-z0-9_\-]{1,32}$", str(effective_calendar)):
            raise HTTPException(
                status_code=400,
                detail="Invalid calendar_id or user_b_id format. Must match ^[A-Za-z0-9_\\-]{1,32}$",
            )

    # 2. Determine owner mode
    is_owner = (effective_calendar is not None) and (
        str(effective_user) == str(effective_calendar)
    )

    # 3. Determine agent profile name configuration
    config_name = agent_name_override or ("owner" if is_owner else "default")

    # 4. Resolve target and fallback workspaces
    backend_path = Path(state.backend_root)
    if is_owner:
        target_ws = f"workspace_{effective_calendar}_owner"
        fallbacks = ["workspace_owner_default", "workspace_owner"]
    else:
        if effective_calendar:
            target_ws = f"workspace_{effective_calendar}"
        else:
            target_ws = "workspace_default"
        fallbacks = ["workspace_default", "workspace"]

    # Check existence
    resolved_workspace = target_ws
    if not (backend_path / resolved_workspace).exists():
        found_fallback = False
        for fallback in fallbacks:
            if (backend_path / fallback).exists():
                resolved_workspace = fallback
                found_fallback = True
                break
        if not found_fallback:
            resolved_workspace = fallbacks[-1]

    # 5. Check cache or compile on-the-fly
    is_base_workspace = False
    if config_name in state.agents:
        base_cfg = state.agent_configs.get(config_name)
        if base_cfg and base_cfg.workspace == resolved_workspace:
            is_base_workspace = True
        elif not base_cfg:
            is_base_workspace = True

    if is_base_workspace:
        return config_name, state.agents[config_name]

    cache_key = f"{config_name}::{resolved_workspace}"

    if cache_key in state.agents:
        return config_name, state.agents[cache_key]

    async with state.get_compilation_lock(cache_key):
        # Check again under lock
        if cache_key in state.agents:
            return config_name, state.agents[cache_key]

        # Enforce P5 cache size limit (pop oldest entries in state.agents when size >= 50)
        while len(state.agents) >= 50:
            evict_key = None
            for key in state.agents:
                if "::" in key:
                    evict_key = key
                    break
            if evict_key is None:
                evict_key = next(iter(state.agents))
            state.agents.pop(evict_key, None)
            state.agent_configs.pop(evict_key, None)
            state.compilation_locks.pop(evict_key, None)

        # Resolve model configuration
        base_cfg = state.agent_configs.get(config_name)
        if base_cfg:
            provider = base_cfg.provider
            model = base_cfg.model
            max_tokens = base_cfg.max_tokens
        else:
            from apps.wu_tanchang_api.config import get_selected_provider

            provider = get_selected_provider()
            model = "deepseek-v4-flash"
            max_tokens = 4000 if is_owner else 800

        agent_cfg = WuAgentConfig(
            name=config_name,
            provider=provider,
            model=model,
            max_tokens=max_tokens,
            workspace=resolved_workspace,
        )

        agent, _ckpt = await asyncio.to_thread(
            create_agent,
            backend_root=backend_path,
            agent_config=agent_cfg,
        )

        state.agents[cache_key] = agent
        state.agent_configs[cache_key] = agent_cfg

        return config_name, agent


async def _resolve_agent(state: AppState, agent_name: str) -> tuple[str, Any]:
    """Resolve agent name to (name, agent) for backward compatibility."""
    return await resolve_dynamic_agent(state, "", {}, agent_name)


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
        if (
            isinstance(msg, ToolMessage)
            and getattr(msg, "name", None) == "mark_material_delivered"
        ):
            return True
        # Check for tool_calls on AIMessage
        if isinstance(msg, AIMessage):
            for tc in msg.tool_calls or []:
                if tc.get("name") == "mark_material_delivered":
                    return True
    return False


async def chat(req: ChatRequest, state: AppState) -> ChatResponse:
    """Handle a conversation turn."""
    agent_name, agent = await resolve_dynamic_agent(
        state, req.user_id, req.metadata or {}, req.agent_name
    )
    tid = thread_id(
        agent_name=agent_name, user_id=req.user_id, conversation_id=req.conversation_id
    )

    if not state.try_start_agent_run(tid, "chat"):
        raise HTTPException(
            status_code=409,
            detail={
                "error": "stream_in_progress",
                "message": "Agent run already in progress for this conversation",
                "thread_id": tid,
            },
        )

    lock = state.thread_locks.setdefault(tid, asyncio.Lock())
    try:
        async with lock:
            try:
                # Record existing message count to isolate this round's new messages
                checkpoint = (
                    await agent.checkpointer.aget({"configurable": {"thread_id": tid}})
                    if hasattr(agent, "checkpointer")
                    else None
                )
                msg_count_before = (
                    len(checkpoint.get("channel_values", {}).get("messages", []))
                    if checkpoint
                    else 0
                )

                from apps.wu_tanchang_api.agent_factory.agent import (
                    register_active_agent,
                    unregister_active_agent,
                )

                register_active_agent(tid, agent)
                try:
                    result = await agent.ainvoke(
                        {"messages": [HumanMessage(content=req.message)]},
                        {
                            "configurable": {"thread_id": tid},
                            "metadata": {
                                "user_id": req.user_id,
                                **(req.metadata or {}),
                            },
                        },
                    )
                finally:
                    unregister_active_agent(tid)
                    if hasattr(agent, "checkpointer") and hasattr(agent.checkpointer, "flush"):
                        agent.checkpointer.flush()
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(
                    status_code=502,
                    detail={
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "thread_id": tid,
                    },
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
    agent_name, agent = await resolve_dynamic_agent(
        state, req.user_id, req.metadata or {}, req.agent_name
    )
    tid = thread_id(
        agent_name=agent_name, user_id=req.user_id, conversation_id=req.conversation_id
    )

    async def _gen() -> None:
        final_parts: list[str] = []
        if not state.try_start_agent_run(tid, "chat_stream"):
            detail = {
                "error": "stream_in_progress",
                "message": "Agent run already in progress for this conversation",
                "thread_id": tid,
            }
            yield f"data: {json.dumps({'type': 'error', 'detail': detail}, ensure_ascii=False)}\n\n"
            return
        from apps.wu_tanchang_api.agent_factory.agent import register_active_agent

        register_active_agent(tid, agent)
        lock = state.thread_locks.setdefault(tid, asyncio.Lock())
        try:
            async with lock:
                try:
                    # Record existing message count to isolate this round's new messages
                    checkpoint = (
                        await agent.checkpointer.aget(
                            {"configurable": {"thread_id": tid}}
                        )
                        if hasattr(agent, "checkpointer")
                        else None
                    )
                    msg_count_before = (
                        len(checkpoint.get("channel_values", {}).get("messages", []))
                        if checkpoint
                        else 0
                    )

                    last_status: str | None = None

                    async for chunk in agent.astream(
                        {"messages": [HumanMessage(content=req.message)]},
                        config={
                            "configurable": {"thread_id": tid},
                            "metadata": {
                                "user_id": req.user_id,
                                **(req.metadata or {}),
                            },
                        },
                        stream_mode=["messages", "updates"],
                        subgraphs=True,
                    ):
                        if not isinstance(chunk, tuple) or len(chunk) != 3:
                            continue
                        subgraph_path, current_stream_mode, data = chunk

                        # Skip subagent raw text chunks to avoid polluting the main chat bubble
                        if current_stream_mode == "messages" and subgraph_path:
                            continue

                        # Handle "updates" mode — progress info (including subagents)
                        if current_stream_mode == "updates":
                            status = get_progress_status(
                                subgraph_path, current_stream_mode, data
                            )
                            if status and status != last_status:
                                last_status = status
                                payload = {"type": "progress", "message": status}
                                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

                        # Handle "messages" mode — text deltas (only from the main agent)
                        if (
                            current_stream_mode == "messages"
                            and not subgraph_path
                            and isinstance(data, tuple)
                            and len(data) == 2
                        ):
                            msg_chunk, _metadata = data
                            if isinstance(msg_chunk, AIMessageChunk):
                                if msg_chunk.content:
                                    chunk_text = str(msg_chunk.content)
                                    final_parts.append(chunk_text)
                                    payload = {"type": "delta", "text": chunk_text}
                                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

                except Exception as exc:  # noqa: BLE001
                    _logger.error(
                        "[ChatStream] Error: %s: %s", type(exc).__name__, str(exc)
                    )
                    detail = {
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "thread_id": tid,
                    }
                    yield f"data: {json.dumps({'type': 'error', 'detail': detail}, ensure_ascii=False)}\n\n"
                    return
        finally:
            from apps.wu_tanchang_api.agent_factory.agent import unregister_active_agent

            unregister_active_agent(tid)
            state.finish_agent_run(tid, "chat_stream")
            if hasattr(agent, "checkpointer") and hasattr(agent.checkpointer, "flush"):
                agent.checkpointer.flush()

        # Build final reply from all collected text
        reply = "".join(final_parts).strip()

        # Fallback: if no text was streamed, extract from checkpoint messages
        if not reply:
            messages = (
                checkpoint.get("channel_values", {}).get("messages", [])
                if checkpoint
                else []
            )
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

        yield f"data: {json.dumps({'type': 'final', 'text': reply}, ensure_ascii=False)}\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream; charset=utf-8")
