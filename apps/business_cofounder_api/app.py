"""Backward-compatible wrapper for the refactored app package.

This file maintains backward compatibility by re-exporting the FastAPI app
instance and global state from the new app package structure.
"""

from apps.business_cofounder_api.app import app, _state

__all__ = ["app", "_state"]
