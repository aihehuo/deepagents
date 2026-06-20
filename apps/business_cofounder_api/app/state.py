"""Application state management."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any


@dataclass
class AppState:
    """Application state for the Business Co-Founder API."""
    
    agent: Any  # primary agent (legacy single-agent mode)
    fallback_agent: Any  # fallback agent
    checkpoints_path: str
    # Ensure the same thread_id is processed serially (avoid checkpoint races).
    thread_locks: dict[str, asyncio.Lock]
    # Backend root directory (base_dir) where agent can write files
    # With virtual_mode=True, all paths are resolved relative to this root
    backend_root: str | None = None
    # Docs directory (kept for backward compatibility with existing code)
    docs_dir: str | None = None
    # Dual-agent architecture (optional, for facilitator-expert split)
    facilitator_agent: Any | None = None  # frontend conversation agent
    expert_agent: Any | None = None  # expert analysis agent
    expert_checkpoints_path: str | None = None
    facilitator_checkpoints_path: str | None = None
    expertise_dir: str | None = None  # Directory containing expertise templates
    # Flag to enable dual-agent mode
    use_dual_agent: bool = False
    # Simulated user agent (for testing/simulation)
    user_agent: Any | None = None  # simulated user agent
    user_agent_checkpoints_path: str | None = None
