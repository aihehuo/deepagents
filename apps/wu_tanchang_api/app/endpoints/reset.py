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
    if agent_name not in state.agents:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "unknown_agent",
                "message": f"Unknown agent: {agent_name}",
                "available": list(state.agents.keys()),
            },
        )
    tid = thread_id(agent_name=agent_name, user_id=req.user_id, conversation_id=req.conversation_id)
    intake_agent = state.agents[agent_name]
    checkpointer = getattr(intake_agent, "checkpointer", None)
    if checkpointer is not None and hasattr(checkpointer, "delete_thread"):
        checkpointer.delete_thread(tid)
    state.thread_locks.pop(tid, None)
    _logger.info("Reset thread %s (agent=%s)", tid, agent_name)
    return ResetResponse(user_id=req.user_id, conversation_id=req.conversation_id, thread_id=tid, ok=True)
