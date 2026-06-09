"""Utilities for Wu Tanchang API."""

from __future__ import annotations

import logging
import os

_logger = logging.getLogger("uvicorn.error")


def thread_id(*, agent_name: str = "default", user_id: str, conversation_id: str) -> str:
    """Build LangGraph thread id with agent name prefix."""
    return f"wt::{agent_name}::{user_id}::{conversation_id}"


def env_flag(name: str, default: bool = False) -> bool:
    """Parse boolean environment variable."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}
