"""Startup handler for Wu Tanchang API."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from apps.wu_tanchang_api.agent_factory import create_agent
from apps.wu_tanchang_api.agent_factory.utils import (
    default_runtime_dir,
    default_workspace_path,
    ensure_runtime_workspace,
)
from apps.wu_tanchang_api.app.state import AppState
from apps.wu_tanchang_api.config import get_selected_provider, load_agent_registry, load_env_file

_logger = logging.getLogger("uvicorn.error")


async def startup(state_ref: dict[str, AppState | None]) -> None:
    """Initialize agents and application state."""
    load_env_file()
    workspace_src = Path(os.environ.get("WU_API_WORKSPACE", str(default_workspace_path())))
    runtime_dir = default_runtime_dir()
    backend_root = ensure_runtime_workspace(workspace_src=workspace_src, runtime_dir=runtime_dir)

    registry = load_agent_registry()

    if registry is not None and registry.agents:
        # Multi-agent mode
        agents: dict[str, object] = {}
        agent_configs: dict[str, object] = {}

        for name, agent_cfg in registry.agents.items():
            _logger.info(
                "  Agent '%s': provider=%s model=%s workspace=%s",
                name,
                agent_cfg.provider,
                agent_cfg.intake_model,
                agent_cfg.workspace or "(default)",
            )
            agent, _ckpt = create_agent(
                backend_root=backend_root,
                agent_config=agent_cfg,
            )
            agents[name] = agent
            agent_configs[name] = agent_cfg

        state_ref["state"] = AppState(
            agents=agents,
            agent_configs=agent_configs,
            default_agent=registry.default_name,
            checkpoints_path=str(runtime_dir / "checkpoints.pkl"),
            thread_locks={},
            backend_root=str(backend_root),
        )
        _logger.info("Wu Tanchang API ready with %d agent profile(s)", len(agents))
        for name in agents:
            _logger.info("  - %s", name)
    else:
        # Legacy single-agent mode
        provider = get_selected_provider()
        agent, _ckpt = create_agent(
            backend_root=backend_root,
            provider=provider,
        )

        state_ref["state"] = AppState(
            agents={"default": agent},
            agent_configs={},
            default_agent="default",
            checkpoints_path=str(runtime_dir / "checkpoints.pkl"),
            thread_locks={},
            backend_root=str(backend_root),
        )
        _logger.info("Wu Tanchang API ready (workspace=%s)", workspace_src)
