"""Health check endpoint."""

from __future__ import annotations

from apps.wu_tanchang_api.app.state import AppState


async def health(state: AppState) -> dict[str, str]:
    """Return service health status."""
    return {
        "status": "ok",
        "service": "wu_tanchang_api",
        "checkpoints_path": state.checkpoints_path,
        "backend_root": state.backend_root,
    }
