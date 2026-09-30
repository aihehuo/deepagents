"""Application state and background task lifecycle manager for group_agent_api."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

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
    admin_agent: Any | None = None  # ops-brain persona (source=group_agent_admin_debug)
    polish_model: Any | None = None  # optional LLM for invite polish
    quality_model: Any | None = None  # optional LLM for profile match-ready (REQ-029)
    planner_model: Any | None = None  # bounded model for intent/query planning
    cafe_shadow_review_model: Any | None = None  # observer only; never gates replies
    thread_locks: dict[str, asyncio.Lock] = field(default_factory=dict)
    active_agent_runs: dict[str, str] = field(default_factory=dict)
    active_tasks: dict[str, asyncio.Task[Any]] = field(default_factory=dict)
    active_agent_runs_lock: threading.Lock = field(default_factory=threading.Lock)
    # REQ-032 durable queue (None when GROUP_AGENT_DURABLE_QUEUE_ENABLED=0)
    durable_config: Any | None = None
    durable_store: Any | None = None
    backpressure: Any | None = None
    greeting_model: Any | None = None  # bare, tool-free venue greeting model
    cafe_agent: Any | None = None  # policy-2 graph, no legacy/business/builtin tools
    search_only_agent: Any | None = field(default=None, kw_only=True)
    write_only_agent: Any | None = field(default=None, kw_only=True)
    no_tool_agent: Any | None = field(default=None, kw_only=True)
    anonymous_agent: Any | None = field(default=None, kw_only=True)

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

    def try_start_anonymous_run(self, request_id: str, *, max_active: int) -> bool:
        """Atomically enforce both anonymous and application concurrency limits.

        Args:
            request_id: Server-generated identifier for a stateless request.
            max_active: Maximum simultaneous anonymous requests.

        Returns:
            Whether the anonymous request reserved a slot.
        """
        with self.active_agent_runs_lock:
            anonymous = sum(owner == "anonymous" for owner in self.active_agent_runs.values())
            if anonymous >= max_active or len(self.active_agent_runs) >= async_max_active():
                return False
            if request_id in self.active_agent_runs:
                return False
            self.active_agent_runs[request_id] = "anonymous"
            return True

    def register_task(self, thread_id: str, task: asyncio.Task[Any]) -> None:
        """Register background task reference with auto-cleanup done callback."""
        with self.active_agent_runs_lock:
            self.active_tasks[thread_id] = task
        task.add_done_callback(lambda _: self._on_task_done(thread_id, task))

    def member_agent_for_capability(
        self,
        *,
        can_search: bool,
        can_write_profile: bool,
    ) -> CompiledStateGraph | None:
        """Select business tools using trusted session permissions.

        Args:
            can_search: Whether this session can search members.
            can_write_profile: Whether this session can save its profile.

        Returns:
            Configured graph with exactly the requested permissions.

        Raises:
            RuntimeError: The requested permission graph is unavailable.
        """
        if can_search and can_write_profile:
            return self.agent
        if can_search:
            selected = self.search_only_agent
        elif can_write_profile:
            selected = self.write_only_agent
        else:
            selected = self.no_tool_agent
        if selected is None:
            msg = "Member permission graph is not initialized"
            raise RuntimeError(msg)
        return selected

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
        except TimeoutError:
            _logger.warning("Timed out waiting for background tasks to cancel during shutdown")

        with self.active_agent_runs_lock:
            self.active_tasks.clear()
            self.active_agent_runs.clear()
