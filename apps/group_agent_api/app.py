"""Backward-compatible entrypoint for uvicorn."""

from apps.group_agent_api.app import app, _state

__all__ = ["app", "_state"]
