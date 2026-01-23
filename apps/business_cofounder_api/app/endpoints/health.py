"""Health check endpoint."""

from __future__ import annotations

import logging

from apps.business_cofounder_api.app.state import AppState

# Use uvicorn's configured logger so output reliably shows up in the terminal.
_logger = logging.getLogger("uvicorn.error")


async def health(state: AppState) -> dict[str, str]:
    """Health check endpoint.
    
    Args:
        state: Application state
        
    Returns:
        Health status and checkpoints path
    """
    _logger.info("GET /health - received request")
    return {"status": "ok", "checkpoints_path": state.checkpoints_path}
