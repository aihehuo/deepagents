"""Utilities for Wu Tanchang API."""

from __future__ import annotations


def thread_id(*, agent_name: str = "default", user_id: str, conversation_id: str) -> str:
    """Build LangGraph thread id with agent name prefix."""
    return f"wt::{agent_name}::{user_id}::{conversation_id}"
