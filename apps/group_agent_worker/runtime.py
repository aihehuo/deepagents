"""Worker runtime: shared AppState bootstrap for Celery processes (REQ-032)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from apps.group_agent_api.agent_factory import create_agent
from apps.group_agent_api.agent_factory.agent import default_runtime_dir
from apps.group_agent_api.app.state import AppState
from apps.group_agent_api.execution.backpressure import BackpressureController
from apps.group_agent_api.execution.config import DurableQueueConfig, load_durable_queue_config
from apps.group_agent_api.execution.redis_store import ExecutionStore

_logger = logging.getLogger(__name__)

_RUNTIME: dict[str, Any] = {}


def get_worker_runtime() -> dict[str, Any]:
    """Initialize agent/runtime once per worker process."""
    if _RUNTIME.get("ready"):
        return _RUNTIME

    cfg = load_durable_queue_config(require_enabled=True)
    assert cfg is not None
    store = ExecutionStore.from_config(cfg)
    if not store.ping():
        raise RuntimeError("worker redis ping failed")

    runtime = Path(os.environ.get("GROUP_AGENT_RUNTIME_DIR", str(default_runtime_dir())))
    runtime.mkdir(parents=True, exist_ok=True)
    agent, admin_agent, ckpt_path = create_agent(base_dir=runtime)

    polish_model = None
    quality_model = None
    try:
        from apps.group_agent_api.agent_factory.integrations.config import llm_polish_enabled
        from apps.group_agent_api.agent_factory.model_builder import create_model

        if llm_polish_enabled():
            polish_model = create_model(log_prefix="[GroupAgentWorkerPolish]")
        quality_model = polish_model or create_model(log_prefix="[GroupAgentWorkerQuality]")
    except Exception as exc:  # noqa: BLE001
        _logger.warning("worker polish/quality model unavailable error_type=%s", type(exc).__name__)

    state = AppState(
        agent=agent,
        admin_agent=admin_agent,
        base_dir=runtime,
        checkpoints_path=str(ckpt_path),
        polish_model=polish_model,
        quality_model=quality_model,
        durable_config=cfg,
        durable_store=store,
        backpressure=BackpressureController(store._r, cfg),  # noqa: SLF001
    )
    _RUNTIME.update(
        {
            "ready": True,
            "config": cfg,
            "store": store,
            "state": state,
            "backpressure": state.backpressure,
        }
    )
    _logger.info(
        "worker_runtime_ready instance=%s prefix=%s",
        cfg.worker_instance_id,
        cfg.redis_prefix,
    )
    return _RUNTIME


def worker_readiness() -> dict[str, object]:
    """Safe readiness payload (no Redis URL / keys)."""
    try:
        rt = get_worker_runtime()
        cfg: DurableQueueConfig = rt["config"]
        store: ExecutionStore = rt["store"]
        ok = store.ping()
        return {
            "status": "ready" if ok else "not_ready",
            "execution_store": "ok" if ok else "unavailable",
            "queue": cfg.celery_queue,
            "instance": cfg.worker_instance_id,
        }
    except Exception:  # noqa: BLE001
        return {"status": "not_ready", "execution_store": "unavailable"}
