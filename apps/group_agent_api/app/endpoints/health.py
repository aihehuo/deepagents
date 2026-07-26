"""Health check."""

from __future__ import annotations

from apps.group_agent_api.app.state import AppState


async def health(state: AppState) -> dict[str, str]:
    return {
        "status": "ok",
        "service": "group_agent_api",
        "checkpoints_path": state.checkpoints_path,
        "base_dir": str(state.base_dir),
    }
