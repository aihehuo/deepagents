"""Code agent creation - standalone agent for coding tasks."""

from __future__ import annotations

import logging
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.accountant import AccountantMiddleware
from deepagents.middleware.language import LanguageDetectionMiddleware

from apps.business_cofounder_api.agent_factory.memory import ApiMemoryMiddleware
from apps.business_cofounder_api.agent_factory.model_builder import create_model
from apps.business_cofounder_api.checkpointer import DiskBackedInMemorySaver

# Use uvicorn's configured logger so output reliably shows up in the terminal.
_logger = logging.getLogger("uvicorn.error")


# Code Agent System Prompt
CODE_AGENT_SYSTEM_PROMPT = """You are a coding-focused agent.

You specialize in:
- Writing and editing code in existing repositories
- Creating clean, correct HTML/CSS/JS when asked (HTML is treated as code)
- Making minimal, well-scoped changes and explaining them clearly

Operating rules:
- Prefer making concrete file edits using available file tools.
- If multiple files are involved, be explicit about which ones you changed and why.
- If you are unsure, ask for the missing detail *once* with the smallest set of clarifying questions.
"""


def create_code_agent(
    *,
    agent_id: str,
    provider: str = "qwen",
) -> tuple[object, Path]:
    """Create a standalone code agent for coding tasks.

    This agent specializes in:
    - Writing and editing code
    - Creating HTML/CSS/JS
    - Making well-scoped changes
    - Debugging and implementation work

    Args:
        agent_id: Identifier for the agent
        provider: Model provider to use ("qwen" or "deepseek", default: "qwen")

    Returns:
        (agent_graph, checkpoints_path)
    """
    # Create model with CODER_SUBAGENT_MODEL suffix
    # Note: Temperature defaults to 0.2 for coder if MODEL_API_TEMPERATURE is not set
    # This is handled by the model config parsing in create_model
    model = create_model(
        provider=provider,
        model_name_suffix="CODER_SUBAGENT_MODEL",
        log_prefix="[CodeAgent]",
        set_max_input_tokens=True,
    )
    
    # Override temperature to 0.2 for code generation (more deterministic)
    # This matches the default behavior in build_coder_subagent_from_env
    import os
    if not os.environ.get("MODEL_API_TEMPERATURE"):
        # Only override if temperature wasn't explicitly set via env var
        if hasattr(model, "temperature"):
            model.temperature = 0.2
        elif hasattr(model, "model_kwargs") and isinstance(model.model_kwargs, dict):
            model.model_kwargs["temperature"] = 0.2

    # Base directory
    base_dir = Path.home() / ".deepagents" / "business_cofounder_api"
    checkpoints_path = base_dir / "code_agent_checkpoints.pkl"

    base_dir.mkdir(parents=True, exist_ok=True)

    # Middleware configuration for code agent
    middleware = [
        AccountantMiddleware(max_tool_calls=50),  # Higher limit for complex coding tasks
        LanguageDetectionMiddleware(),
        ApiMemoryMiddleware(base_dir=base_dir),  # Support memory for user preferences
    ]

    _logger.info("[CodeAgent] Middleware configuration:")
    _logger.info("  - AccountantMiddleware (max_tool_calls=50)")
    _logger.info("  - LanguageDetectionMiddleware")
    _logger.info("  - ApiMemoryMiddleware")

    # Use virtual_mode=True for security
    backend = FilesystemBackend(root_dir=str(base_dir), virtual_mode=True)

    checkpointer = DiskBackedInMemorySaver(file_path=checkpoints_path)

    # Code-focused system prompt
    system_prompt = CODE_AGENT_SYSTEM_PROMPT

    agent = create_deep_agent(
        model=model,
        backend=backend,
        checkpointer=checkpointer,
        subagents=[],  # No subagents
        middleware=middleware,
        system_prompt=system_prompt,
    )

    _logger.info("[CodeAgent] Agent created successfully")
    _logger.info("  Checkpoints path: %s", checkpoints_path)

    return agent, checkpoints_path
