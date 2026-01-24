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
    
    This endpoint handles both initialization (first message with assignment) 
    and follow-up conversation messages. It auto-detects which case it is by
    checking if there are previous messages in the thread.
    
    Args:
        req: Request with simulation_id and message
        state: Application state
        
    Returns:
        SimulatedUserChatResponse with user agent's reply
    """
    if state.user_agent is None:
        _logger.error("POST /simulated_user/chat - user agent not initialized")
        return SimulatedUserChatResponse(
            simulation_id=req.simulation_id,
            user_id=req.user_id,
            thread_id=_get_user_agent_thread_id(req.simulation_id),
            reply="Error: User agent not available",
            is_initialization=False,
            success=False,
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
            # Check if this is the first message (initialization)
            is_initialization = False
            try:
                current_state = await state.user_agent.aget_state(
                    {"configurable": {"thread_id": tid}}
                )
                if current_state and hasattr(current_state, "values"):
                    messages = current_state.values.get("messages", [])
                    # If no messages or only system messages, this is initialization
                    human_messages = [m for m in messages if hasattr(m, "type") and getattr(m, "type") == "human"]
                    is_initialization = len(human_messages) == 0
                else:
                    is_initialization = True
            except Exception as e:  # noqa: BLE001
                # If we can't get state, assume it's initialization
                _logger.debug("Could not get state for thread %s, assuming initialization: %s", tid, str(e))
                is_initialization = True
            
            # Prepare the message
            if is_initialization:
                # First message - add prompt to generate rough startup idea
                initialization_prompt = f"""Based on this assignment/context:

{req.message}

Share whatever rough idea or thought comes to mind. Don't worry about making it detailed or complete - just share what you're thinking in your own words. It's okay if it's vague, incomplete, or you're not sure if it makes sense. Express it naturally, as someone with no startup experience would."""
                message_content = initialization_prompt
                _logger.info("POST /simulated_user/chat - treating as initialization (simulation_id=%s)", req.simulation_id)
            else:
                # Follow-up message - send directly
                message_content = req.message
                _logger.info("POST /simulated_user/chat - treating as follow-up (simulation_id=%s)", req.simulation_id)
            
            # Invoke user agent
            result = await state.user_agent.ainvoke(
                {"messages": [HumanMessage(content=message_content)]},
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
                    simulation_id=req.simulation_id,
                    user_id=req.user_id,
                    thread_id=tid,
                    reply="Error: No response from user agent",
                    is_initialization=is_initialization,
                    success=False,
                )
            
            reply = str(ai_messages[-1].content)
            
            _logger.info(
                "POST /simulated_user/chat - completed (simulation_id=%s, is_initialization=%s, reply_len=%d)",
                req.simulation_id,
                is_initialization,
                len(reply),
            )
            
            return SimulatedUserChatResponse(
                simulation_id=req.simulation_id,
                user_id=req.user_id,
                thread_id=tid,
                reply=reply,
                is_initialization=is_initialization,
                success=True,
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
                simulation_id=req.simulation_id,
                user_id=req.user_id,
                thread_id=tid,
                reply=f"Error: {type(e).__name__}: {str(e)}",
                is_initialization=False,
                success=False,
            )
