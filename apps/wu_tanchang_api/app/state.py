"""Application state for Wu Tanchang API."""

from __future__ import annotations

import asyncio
import threading
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
    active_callback_threads: dict[str, threading.Thread] = field(default_factory=dict)
    active_callback_threads_lock: threading.Lock = field(default_factory=threading.Lock)
    active_agent_runs: dict[str, str] = field(default_factory=dict)
    active_agent_runs_lock: threading.Lock = field(default_factory=threading.Lock)
    compilation_lock: threading.Lock = field(default_factory=threading.Lock)
    backend_root: str = ""

    @property
    def agent(self) -> Any:
        """Backward-compat: returns default agent."""
        return self.agents.get(self.default_agent)

    def try_start_agent_run(self, thread_id: str, owner: str) -> bool:
        """Reserve a thread id for one active agent execution across all endpoints."""
        with self.active_agent_runs_lock:
            if thread_id in self.active_agent_runs:
                return False
            self.active_agent_runs[thread_id] = owner
            return True

    def finish_agent_run(self, thread_id: str, owner: str) -> None:
        """Release an active agent execution reservation."""
        with self.active_agent_runs_lock:
            if self.active_agent_runs.get(thread_id) == owner:
                self.active_agent_runs.pop(thread_id, None)
