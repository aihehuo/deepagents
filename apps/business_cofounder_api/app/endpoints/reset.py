"""Reset endpoint for clearing conversation state."""

from __future__ import annotations

import logging

from apps.business_cofounder_api.app.models import ResetRequest, ResetResponse
from apps.business_cofounder_api.app.state import AppState
from apps.business_cofounder_api.app.utils import thread_id

# Use uvicorn's configured logger so output reliably shows up in the terminal.
_logger = logging.getLogger("uvicorn.error")


async def reset(req: ResetRequest, state: AppState) -> ResetResponse:
    """Reset a user's conversation by deleting the thread from the checkpointer.
    
    Args:
        req: Request with user_id and conversation_id
        state: Application state
        
    Returns:
        ResetResponse indicating success
    """
    _logger.info(
        "POST /reset - received request (user_id=%s, conversation_id=%s)",
        req.user_id,
        req.conversation_id,
    )
    tid = thread_id(user_id=req.user_id, conversation_id=req.conversation_id)

    # The checkpointer is held by the agent graph; we access it via .checkpointer (best-effort).
    checkpointer = getattr(state.agent, "checkpointer", None)
    if checkpointer is not None and hasattr(checkpointer, "delete_thread"):
        checkpointer.delete_thread(tid)

    # Also drop the in-process lock (fresh start)
    state.thread_locks.pop(tid, None)

    return ResetResponse(user_id=req.user_id, conversation_id=req.conversation_id, thread_id=tid, ok=True)
