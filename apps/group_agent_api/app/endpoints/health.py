"""Health / readiness checks (REQ-032)."""

from __future__ import annotations

from fastapi import Response

from apps.group_agent_api.app.state import AppState
from apps.group_agent_api.execution.broker import broker_ready
from apps.group_agent_api.execution.config import durable_queue_enabled


async def health(state: AppState) -> dict[str, str]:
    """Liveness: process up; does not depend on Redis."""
    return {
        "status": "ok",
        "service": "group_agent_api",
        "checkpoints_path": state.checkpoints_path,
        "base_dir": str(state.base_dir),
        "durable_queue": "1" if durable_queue_enabled() else "0",
    }


async def readiness(state: AppState, response: Response) -> dict[str, object]:
    """Readiness: when durable mode on, Redis ledger + broker must be reachable."""
    body: dict[str, object] = {
        "status": "ready",
        "service": "group_agent_api",
        "durable_queue": durable_queue_enabled(),
    }
    if not durable_queue_enabled():
        return body

    redis_ok = False
    if state.durable_store is not None:
        try:
            redis_ok = bool(state.durable_store.ping())
        except Exception:  # noqa: BLE001
            redis_ok = False
    broker_info = {"broker": "unavailable"}
    if state.durable_config is not None:
        broker_info = broker_ready(state.durable_config)

    body["execution_store"] = "ok" if redis_ok else "unavailable"
    body["broker"] = broker_info.get("broker")
    # Never leak Redis URL / keys.
    if not redis_ok or broker_info.get("broker") != "ok":
        response.status_code = 503
        body["status"] = "not_ready"
    return body
