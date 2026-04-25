"""Startup and shutdown handlers for the FastAPI application."""

from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from apps.business_cofounder_api.agent_factory import (
    create_business_cofounder_agent,
    create_expert_agent,
    create_facilitator_agent,
    create_user_agent,
)

from apps.business_cofounder_api.app.state import AppState
from apps.business_cofounder_api.app.utils import env_flag

# Use uvicorn's configured logger so output reliably shows up in the terminal.
_logger = logging.getLogger("uvicorn.error")

# Global asyncio executor (set during startup)
_ASYNCIO_DEFAULT_EXECUTOR: ThreadPoolExecutor | None = None


def patch_openai_no_thread() -> None:
    """Patch OpenAI python SDK to avoid asyncio.to_thread in ultra-restricted environments.

    Some production environments have extremely low thread limits and crash with:
      RuntimeError: can't start new thread

    The OpenAI SDK's async path calls asyncio.to_thread() for small sync helpers (e.g. platform detection).
    If thread creation is disallowed, that fails. This patch replaces that helper with a direct call.

    Enable with: BC_API_OPENAI_NO_THREAD=1
    """
    if not env_flag("BC_API_OPENAI_NO_THREAD", default=False):
        return
    try:
        import openai._utils._sync as _openai_sync  # type: ignore
    except Exception:
        return

    async def _to_thread_noop(func, /, *args, **kwargs):  # type: ignore[no-untyped-def]
        return func(*args, **kwargs)

    try:
        _openai_sync.to_thread = _to_thread_noop  # type: ignore[attr-defined]
        _logger.info("Applied BC_API_OPENAI_NO_THREAD patch (openai._utils._sync.to_thread).")
    except Exception:
        return


async def configure_asyncio_default_executor() -> None:
    """Configure asyncio default executor for DNS/networking operations."""
    global _ASYNCIO_DEFAULT_EXECUTOR
    if _ASYNCIO_DEFAULT_EXECUTOR is not None:
        return
    max_workers = int(os.environ.get("BC_API_ASYNCIO_EXECUTOR_WORKERS", "1"))
    if max_workers < 1:
        max_workers = 1
    _ASYNCIO_DEFAULT_EXECUTOR = ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="bc-asyncio",
    )
    loop = asyncio.get_running_loop()
    loop.set_default_executor(_ASYNCIO_DEFAULT_EXECUTOR)
    # Warm up: forces the executor to start at least one worker thread now,
    # so later DNS lookups don't try (and fail) to spawn a new thread.
    try:
        await loop.run_in_executor(None, lambda: None)
        _logger.info("Configured asyncio default executor (workers=%s).", max_workers)
    except Exception as e:  # noqa: BLE001
        _logger.warning(
            "Failed to warm up asyncio default executor (workers=%s): %s: %s. "
            "Async DNS/networking may fail with 'can't start new thread'.",
            max_workers,
            type(e).__name__,
            str(e),
        )


async def startup(state_ref: dict[str, AppState | None]) -> None:
    """FastAPI startup event handler - initializes agents and application state.

    Args:
        state_ref: Dictionary with 'state' key to store the initialized AppState
    """
    await configure_asyncio_default_executor()
    patch_openai_no_thread()

    # Check if dual-agent mode is enabled
    use_dual_agent = env_flag("BC_API_USE_DUAL_AGENT", default=True)

    _logger.info("=" * 80)
    _logger.info("API Startup - Agent Configuration")
    _logger.info("=" * 80)
    _logger.info("  Dual-Agent Mode: %s", "ENABLED" if use_dual_agent else "DISABLED")

    # Extract backend_root from agent configuration
    # With virtual_mode=True, all file operations are sandboxed to backend_root
    backend_root = str(Path.home() / ".deepagents" / "business_cofounder_api")
    docs_dir = str(Path.home() / ".deepagents" / "business_cofounder_api" / "docs")

    if use_dual_agent:
        _logger.info("  Initializing DUAL-AGENT architecture...")
        _logger.info("  - Frontend: Facilitator Agent (natural conversation)")
        _logger.info("  - Expert: Analyzer Agent (methodology & analysis)")

        # Create facilitator agent (frontend)
        facilitator_agent, facilitator_checkpoints = create_facilitator_agent(
            agent_id="facilitator",
            provider="deepseek",
            sync_interval=5,
        )
        _logger.info("  ✓ Facilitator Agent initialized")
        _logger.info("    Checkpoints: %s", facilitator_checkpoints)

        # Create expert agent (analyzer)
        # Get default expertise type from env (can be overridden per conversation)
        assigned_expertise = os.getenv("DEFAULT_EXPERTISE_TYPE", "pitch_expert")
        print(f"DEFAULT_EXPERTISE_TYPE: {assigned_expertise}")

        expert_agent, expert_checkpoints = create_expert_agent(
            agent_id="expert_analyzer",
            provider="deepseek",
            expertise_type=assigned_expertise,
        )
        _logger.info("  ✓ Expert Agent initialized")
        _logger.info("    Checkpoints: %s", expert_checkpoints)
        _logger.info("    Assigned expertise: %s", assigned_expertise)

        # Use facilitator as primary agent for backward compatibility
        primary_agent = facilitator_agent
        checkpoints_path = facilitator_checkpoints

        # Create fallback (use facilitator as fallback too, or could create a fallback expert)
        fallback_agent = facilitator_agent

        # Set expertise directory
        expertise_dir_path = str(Path(backend_root) / "expertise")

        # Create simulated user agent (for testing/simulation)
        user_agent, user_agent_checkpoints = create_user_agent(
            agent_id="simulated_user",
            provider="deepseek",
        )
        _logger.info("  ✓ Simulated User Agent initialized")
        _logger.info("    Checkpoints: %s", user_agent_checkpoints)

        state_ref["state"] = AppState(
            agent=primary_agent,
            fallback_agent=fallback_agent,
            checkpoints_path=str(checkpoints_path),
            thread_locks={},
            backend_root=backend_root,
            docs_dir=docs_dir,
            facilitator_agent=facilitator_agent,
            expert_agent=expert_agent,
            expert_checkpoints_path=str(expert_checkpoints),
            facilitator_checkpoints_path=str(facilitator_checkpoints),
            expertise_dir=expertise_dir_path,
            use_dual_agent=True,
            user_agent=user_agent,
            user_agent_checkpoints_path=str(user_agent_checkpoints),
        )
        _logger.info("=" * 80)
        _logger.info("Dual-Agent Architecture: READY")
        _logger.info("=" * 80)
    else:
        _logger.info("  Initializing SINGLE-AGENT architecture (legacy mode)...")

        # Legacy single-agent mode
        primary_agent, checkpoints_path = create_business_cofounder_agent(
            agent_id="business_cofounder_agent",
            provider="deepseek"
        )
        fallback_agent, _ = create_business_cofounder_agent(
            agent_id="business_cofounder_agent",
            provider="deepseek"
        )

        # Create simulated user agent (for testing/simulation)
        user_agent, user_agent_checkpoints = create_user_agent(
            agent_id="simulated_user",
            provider="deepseek",
        )
        _logger.info("  ✓ Simulated User Agent initialized")
        _logger.info("    Checkpoints: %s", user_agent_checkpoints)

        state_ref["state"] = AppState(
            agent=primary_agent,
            fallback_agent=fallback_agent,
            checkpoints_path=str(checkpoints_path),
            thread_locks={},
            backend_root=backend_root,
            docs_dir=docs_dir,
            use_dual_agent=False,
            user_agent=user_agent,
            user_agent_checkpoints_path=str(user_agent_checkpoints),
        )
        _logger.info("  ✓ Single Agent initialized")
        _logger.info("=" * 80)
        _logger.info("Single-Agent Architecture: READY")
        _logger.info("=" * 80)
