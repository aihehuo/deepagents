"""Utilities for group_agent_api."""

from __future__ import annotations

from typing import Any


def thread_id(*, user_id: str, group_id: str, conversation_id: str) -> str:
    """LangGraph thread id: session isolation per user × group × conversation."""
    return f"ga::{user_id}::{group_id}::{conversation_id}"


def get_agent_checkpointer(agent: Any) -> Any | None:
    bound = agent
    while hasattr(bound, "bound"):
        bound = bound.bound
    return getattr(bound, "checkpointer", None)


async def aget_agent_state(agent: Any, config: dict[str, Any]) -> Any:
    bound = agent
    while hasattr(bound, "bound"):
        bound = bound.bound
    if hasattr(bound, "aget_state"):
        return await bound.aget_state(config)
    return None
