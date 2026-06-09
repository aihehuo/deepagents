"""Backward-compatible entrypoint for uvicorn."""

from apps.wu_tanchang_api.app import app, _state

__all__ = ["app", "_state"]
