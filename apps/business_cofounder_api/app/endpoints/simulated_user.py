"""Simulated user agent endpoints for testing and simulation."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from apps.business_cofounder_api.app.models import SimulatedUserChatRequest, SimulatedUserChatResponse
from apps.business_cofounder_api.app.state import AppState

# Use uvicorn's configured logger so output reliably shows up in the terminal.
_logger = logging.getLogger("uvicorn.error")


def _get_user_agent_thread_id(simulation_id: str) -> str:
    """Generate thread ID for user agent from simulation_id."""
    return f"sim_user::{simulation_id}"


async def simulated_user_chat(req: SimulatedUserChatRequest, state: AppState) -> SimulatedUserChatResponse:
    """Chat endpoint for simulated user agent.
    
    This endpoint invokes the user agent synchronously and returns the complete
    LLM-generated message in the response.
    
    Args:
        req: Request with simulation_id and message
        state: Application state
        
    Returns:
        SimulatedUserChatResponse with the agent's reply
    """
    if state.user_agent is None:
        _logger.error("POST /simulated_user/chat - user agent not initialized")
        tid = _get_user_agent_thread_id(req.simulation_id)
        return SimulatedUserChatResponse(
            success=False,
            session_id=tid,
            thread_id=tid,
            reply="Error: User agent not available",
        )
    
    _logger.info(
        "POST /simulated_user/chat - received request (simulation_id=%s, message_len=%d)",
        req.simulation_id,
        len(req.message),
    )
    
    tid = _get_user_agent_thread_id(req.simulation_id)
    
    # Use shared thread_locks (thread IDs are unique across all agents)
    lock = state.thread_locks.get(tid)
    if lock is None:
        lock = asyncio.Lock()
        state.thread_locks[tid] = lock
    
    async with lock:
        try:
            # Invoke user agent synchronously
            result = await state.user_agent.ainvoke(
                {"messages": [HumanMessage(content=req.message)]},
                {
                    "configurable": {"thread_id": tid},
                    "metadata": {"user_id": req.user_id, "simulation_id": req.simulation_id, **(req.metadata or {})},
                },
            )
            
            # Extract reply from result
            messages = result.get("messages", [])
            ai_messages = [m for m in messages if isinstance(m, AIMessage) or (hasattr(m, "type") and getattr(m, "type") == "ai")]
            
            if not ai_messages:
                _logger.warning("POST /simulated_user/chat - no AI message in result (simulation_id=%s)", req.simulation_id)
                return SimulatedUserChatResponse(
                    success=False,
                    session_id=tid,
                    thread_id=tid,
                    reply="Error: No response from user agent",
                )
            
            reply = str(ai_messages[-1].content)
            
            _logger.info(
                "POST /simulated_user/chat - completed (simulation_id=%s, reply_len=%d)",
                req.simulation_id,
                len(reply),
            )
            
            return SimulatedUserChatResponse(
                success=True,
                session_id=tid,
                thread_id=tid,
                reply=reply,
            )
            
        except Exception as e:  # noqa: BLE001
            _logger.error(
                "POST /simulated_user/chat - error (simulation_id=%s): %s: %s",
                req.simulation_id,
                type(e).__name__,
                str(e),
                exc_info=True,
            )
            return SimulatedUserChatResponse(
                success=False,
                session_id=tid,
                thread_id=tid,
                reply=f"Error: {type(e).__name__}: {str(e)}",
            )
