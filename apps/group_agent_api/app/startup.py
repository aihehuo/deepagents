"""Startup for group_agent_api."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from apps.group_agent_api.agent_factory import create_agent
from apps.group_agent_api.agent_factory.agent import APP_NAME, default_runtime_dir
from apps.group_agent_api.agent_factory.integrations.config import (
    assert_startup_security,
    integration_mode,
)
from apps.group_agent_api.app.state import AppState
from deepagents.observability import UCObserver

_logger = logging.getLogger("uvicorn.error")


def _load_dotenv_if_present() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


async def startup(state_ref: dict[str, AppState | None]) -> None:
    _load_dotenv_if_present()
    assert_startup_security()
    runtime = Path(os.environ.get("GROUP_AGENT_RUNTIME_DIR", str(default_runtime_dir())))
    runtime.mkdir(parents=True, exist_ok=True)
    UCObserver.set_log_dir(runtime / "logs")

    agent, admin_agent, ckpt_path = create_agent(base_dir=runtime)
    polish_model = None
    quality_model = None
    try:
        from apps.group_agent_api.agent_factory.integrations.config import (
            llm_polish_enabled,
        )
        from apps.group_agent_api.agent_factory.model_builder import create_model

        if llm_polish_enabled():
            polish_model = create_model(log_prefix="[GroupAgentPolish]")
        # Match-ready judge: reuse polish model when present, else dedicated instance.
        quality_model = polish_model or create_model(log_prefix="[GroupAgentQuality]")
    except Exception as exc:  # noqa: BLE001
        _logger.warning("polish/quality model unavailable: %s", exc)

    durable_config = None
    durable_store = None
    backpressure = None
    from apps.group_agent_api.execution.config import (
        durable_queue_enabled,
        load_durable_queue_config,
    )

    if durable_queue_enabled():
        durable_config = load_durable_queue_config(require_enabled=True)
        assert durable_config is not None
        from apps.group_agent_api.execution.backpressure import BackpressureController
        from apps.group_agent_api.execution.redis_store import ExecutionStore

        durable_store = ExecutionStore.from_config(durable_config)
        if not durable_store.ping():
            raise RuntimeError("durable Redis ping failed at startup")
        backpressure = BackpressureController(durable_store._r, durable_config)  # noqa: SLF001
        _logger.info(
            "durable_queue enabled prefix=%s queue=%s",
            durable_config.redis_prefix,
            durable_config.celery_queue,
        )

    state_ref["state"] = AppState(
        agent=agent,
        admin_agent=admin_agent,
        base_dir=runtime,
        checkpoints_path=str(ckpt_path),
        polish_model=polish_model,
        quality_model=quality_model,
        durable_config=durable_config,
        durable_store=durable_store,
        backpressure=backpressure,
    )
    _logger.info(
        "%s ready runtime=%s integration=%s durable=%s",
        APP_NAME,
        runtime,
        integration_mode(),
        "1" if durable_queue_enabled() else "0",
    )
