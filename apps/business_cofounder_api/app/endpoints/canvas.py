"""Canvas endpoint for retrieving expert analysis data."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import HTTPException

from apps.business_cofounder_api.app.models import CanvasResponse, KanbanRequest
from apps.business_cofounder_api.app.state import AppState
from apps.business_cofounder_api.app.utils import thread_id

# Use uvicorn's configured logger so output reliably shows up in the terminal.
_logger = logging.getLogger("uvicorn.error")


async def get_canvas(req: KanbanRequest, state: AppState) -> CanvasResponse:
    """Get current canvas data and expert guidance.
    
    Returns:
    - canvas: Domain-agnostic JSON structure from expert analysis
    - expert_guidance: Strategic guidance for the conversation
    - Conversation round information and sync status
    
    The canvas structure varies by expertise type and is defined by the expert's prompt.
    The backend treats it as an opaque JSON blob - the frontend is responsible for
    interpreting and displaying it.
    
    This endpoint is only available in dual-agent mode.
    
    Args:
        req: Request with user_id and conversation_id
        state: Application state
        
    Returns:
        CanvasResponse with canvas data and expert guidance
    """
    _logger.info(
        "POST /canvas - received request (user_id=%s, conversation_id=%s)",
        req.user_id,
        req.conversation_id,
    )
    
    # Check if dual-agent mode is enabled
    if not state.use_dual_agent:
        raise HTTPException(
            status_code=501,
            detail={
                "error": "Canvas endpoint not available",
                "message": "This endpoint is only available in dual-agent mode. Set BC_API_USE_DUAL_AGENT=1 to enable.",
            },
        )
    
    tid = thread_id(user_id=req.user_id, conversation_id=req.conversation_id)
    
    try:
        # Get current state from checkpointer
        config = {"configurable": {"thread_id": tid}}
        
        # Use facilitator agent's checkpointer
        checkpointer = state.facilitator_agent.checkpointer if state.facilitator_agent else state.agent.checkpointer
        checkpoint = await checkpointer.aget(config)
        
        if checkpoint is None:
            # No conversation history yet
            return CanvasResponse(
                user_id=req.user_id,
                conversation_id=req.conversation_id,
                thread_id=tid,
                canvas=None,
                expert_guidance=None,
                current_round=0,
                last_sync_round=0,
                analysis_timestamp=None,
            )
        
        # Extract state from checkpoint
        state_values = checkpoint.get("channel_values", {})
        
        # Get canvas and guidance data
        canvas = state_values.get("canvas")
        expert_guidance = state_values.get("expert_guidance")
        current_round = state_values.get("conversation_round", 0)
        last_sync = state_values.get("last_expert_sync", 0)
        timestamp = state_values.get("analysis_timestamp")
        
        _logger.info(
            "[Canvas] Retrieved for thread %s: round=%d, last_sync=%d, has_canvas=%s",
            tid,
            current_round,
            last_sync,
            canvas is not None,
        )
        
        return CanvasResponse(
            user_id=req.user_id,
            conversation_id=req.conversation_id,
            thread_id=tid,
            canvas=canvas,
            expert_guidance=expert_guidance,
            current_round=current_round,
            last_sync_round=last_sync,
            analysis_timestamp=timestamp,
        )
        
    except Exception as e:  # noqa: BLE001
        _logger.error(
            "POST /canvas failed: user_id=%s conversation_id=%s error=%s",
            req.user_id,
            req.conversation_id,
            str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to retrieve canvas",
                "message": str(e),
                "thread_id": tid,
            },
        ) from e
