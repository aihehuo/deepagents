"""Application state and background task lifecycle manager for group_agent_api."""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from apps.group_agent_api.agent_factory.integrations.config import async_max_active

_logger = logging.getLogger("uvicorn.error")


@dataclass
class AppState:
    agent: Any
    base_dir: Path
    checkpoints_path: str = ""
    polish_model: Any | None = None  # optional LLM for invite polish
    thread_locks: dict[str, asyncio.Lock] = field(default_factory=dict)
    active_agent_runs: dict[str, str] = field(default_factory=dict)
    active_tasks: dict[str, asyncio.Task[Any]] = field(default_factory=dict)
    active_agent_runs_lock: threading.Lock = field(default_factory=threading.Lock)

    def try_start_agent_run(self, thread_id: str, owner: str) -> bool:
        """Atomically enforce global max active limit AND per-conversation lock."""
        with self.active_agent_runs_lock:
            if len(self.active_agent_runs) >= async_max_active():
                _logger.warning("Global async_max_active limit reached (%d)", len(self.active_agent_runs))
                return False
            if thread_id in self.active_agent_runs:
                _logger.warning("Per-conversation run in progress for thread_id=%s", thread_id)
                return False
            self.active_agent_runs[thread_id] = owner
            return True

    def finish_agent_run(self, thread_id: str, owner: str) -> None:
        with self.active_agent_runs_lock:
            if self.active_agent_runs.get(thread_id) == owner:
                self.active_agent_runs.pop(thread_id, None)

    def register_task(self, thread_id: str, task: asyncio.Task[Any]) -> None:
        """Register background task reference with auto-cleanup done callback."""
        with self.active_agent_runs_lock:
            self.active_tasks[thread_id] = task
        task.add_done_callback(lambda _: self._on_task_done(thread_id, task))

    def _on_task_done(self, thread_id: str, task: asyncio.Task[Any]) -> None:
        with self.active_agent_runs_lock:
            if self.active_tasks.get(thread_id) == task:
                self.active_tasks.pop(thread_id, None)

    async def shutdown(self) -> None:
        """FastAPI shutdown lifecycle handler: cancel and await in-flight tasks cleanly."""
        with self.active_agent_runs_lock:
            tasks = list(self.active_tasks.values())

        if not tasks:
            return

        _logger.info("Shutdown initiated: cancelling %d in-flight background tasks", len(tasks))
        for task in tasks:
            if not task.done():
                task.cancel()

        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=5.0,
            )
        except asyncio.TimeoutError:
            _logger.warning("Timed out waiting for background tasks to cancel during shutdown")

        with self.active_agent_runs_lock:
            self.active_tasks.clear()
            self.active_agent_runs.clear()
