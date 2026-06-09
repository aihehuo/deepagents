"""Application state for Wu Tanchang API."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from apps.wu_tanchang_api.config import WuAgentConfig


@dataclass
class AppState:
    """Runtime state for the API.

    ``agents`` and ``agent_configs`` are dicts keyed by agent name.
    Each agent has its own workspace and persona.
    """

    agents: dict[str, Any] = field(default_factory=dict)
    agent_configs: dict[str, WuAgentConfig] = field(default_factory=dict)
    default_agent: str = "default"
    checkpoints_path: str = ""
    thread_locks: dict[str, asyncio.Lock] = field(default_factory=dict)
    backend_root: str = ""
    expertise_dir: str = ""

    @property
    def agent(self) -> Any:
        """Backward-compat: returns default agent."""
        return self.agents.get(self.default_agent)
