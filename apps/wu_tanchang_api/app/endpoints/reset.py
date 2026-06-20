"""Reset conversation endpoint."""

from __future__ import annotations

import logging

from fastapi import HTTPException

from apps.wu_tanchang_api.app.models import ResetRequest, ResetResponse
from apps.wu_tanchang_api.app.state import AppState
from apps.wu_tanchang_api.app.utils import thread_id

_logger = logging.getLogger("uvicorn.error")


async def reset(req: ResetRequest, state: AppState) -> ResetResponse:
    """Delete thread checkpoint and release lock."""
    agent_name = req.agent_name or state.default_agent
    import re
    is_valid = (
        agent_name in {"default", "owner"}
        or agent_name in state.agents
        or re.fullmatch(r"^[A-Za-z0-9_\-]{1,64}$", agent_name)
    )
    if not is_valid:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "unknown_agent",
                "message": f"Unknown agent: {agent_name}",
                "available": list(state.agents.keys()),
            },
        )
    tid = thread_id(
        agent_name=agent_name, user_id=req.user_id, conversation_id=req.conversation_id
    )

    deleted_any = False
    # Iterate over all cached agent instances to delete the thread from all in-memory savers.
    # This prevents out-of-sync saver instances from restoring the deleted checkpoint.
    for name, agent in list(state.agents.items()):
        checkpointer = getattr(agent, "checkpointer", None)
        if checkpointer is not None and hasattr(checkpointer, "delete_thread"):
            try:
                checkpointer.delete_thread(tid)
                deleted_any = True
                _logger.info("Deleted thread %s from checkpointer of agent: %s", tid, name)
            except Exception as e:
                _logger.warning("Failed to delete thread from agent %s: %s", name, e)

    state.thread_locks.pop(tid, None)
    _logger.info("Reset thread %s (agent=%s, deleted_any=%s)", tid, agent_name, deleted_any)
    return ResetResponse(
        user_id=req.user_id, conversation_id=req.conversation_id, thread_id=tid, ok=True
    )
