"""State debug endpoint."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import HTTPException

from apps.business_cofounder_api.app.state import AppState
from apps.business_cofounder_api.app.utils import env_flag, extract_state_values_from_checkpoint, summarize_state_values, thread_id

# Use uvicorn's configured logger so output reliably shows up in the terminal.
_logger = logging.getLogger("uvicorn.error")


async def get_state(user_id: str, conversation_id: str, state: AppState) -> dict[str, Any]:
    """Debug endpoint: return current milestone flags + todo list from checkpoint state.

    Disabled by default. Enable with:
      BC_API_ENABLE_STATE_ENDPOINT=1
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        state: Application state
        
    Returns:
        Dictionary with state summary
    """
    _logger.info("GET /state - received request (user_id=%s, conversation_id=%s)", user_id, conversation_id)
    if not env_flag("BC_API_ENABLE_STATE_ENDPOINT", default=False):
        raise HTTPException(status_code=404, detail="Not found")

    tid = thread_id(user_id=user_id, conversation_id=conversation_id)

    lock = state.thread_locks.get(tid)
    if lock is None:
        lock = asyncio.Lock()
        state.thread_locks[tid] = lock

    async with lock:
        checkpointer = getattr(state.agent, "checkpointer", None)
        if checkpointer is None or not hasattr(checkpointer, "get_tuple"):
            raise HTTPException(status_code=500, detail="Agent checkpointer not available")

        try:
            ckpt_tuple = await asyncio.to_thread(
                checkpointer.get_tuple, {"configurable": {"thread_id": tid}}
            )
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Failed to read checkpoint: {e!s}") from e

    if ckpt_tuple is None:
        return {
            "thread_id": tid,
            "checkpoints_path": state.checkpoints_path,
            "milestones": {},
            "todos": [],
            "message_count": 0,
        }

    checkpoint = ckpt_tuple.checkpoint if hasattr(ckpt_tuple, "checkpoint") else ckpt_tuple[1]
    values = extract_state_values_from_checkpoint(checkpoint)
    summary = summarize_state_values(values)

    # Also compute the canonical todo list derived from milestone flags.
    # This helps debug cases where `todos` in the checkpoint is stale (e.g. if an LLM rewrote it,
    # or if milestones were updated at the end of a run and todos will only be refreshed on the next run).
    try:
        from deepagents.middleware.business_idea_development import BusinessIdeaDevelopmentMiddleware

        canonical_todos = BusinessIdeaDevelopmentMiddleware()._generate_todos_from_state(values)  # noqa: SLF001
    except Exception:  # noqa: BLE001
        canonical_todos = []

    return {
        "thread_id": tid,
        "checkpoints_path": state.checkpoints_path,
        **summary,
        "todos_canonical": canonical_todos,
    }
