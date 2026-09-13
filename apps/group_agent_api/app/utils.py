"""Utilities for group_agent_api."""

from __future__ import annotations

from typing import Any


def thread_id(
    *,
    user_id: str,
    group_id: str,
    conversation_id: str,
    episode_id: str | None = None,
    execution_mode: str | None = None,
) -> str:
    """LangGraph thread id: isolate per user × group × conversation × episode.

    Including episode_id prevents「开新一轮」 from reusing prior-episode agent
    memory (which caused stale doing/need/offer after direction changes).
    """
    if execution_mode is not None and (execution_mode not in {"cafe_host", "group_agent"} or group_id != "global"):
        raise ValueError("schema_invalid: invalid mode thread scope")
    mode = f"::{execution_mode}" if execution_mode is not None else ""
    base = f"ga::{user_id}::{group_id}{mode}::{conversation_id}"
    ep = (episode_id or "").strip()
    if ep:
        return f"{base}::{ep}"
    return base


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
