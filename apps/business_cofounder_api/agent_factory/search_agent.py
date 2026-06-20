"""Search agent creation - standalone agent for AI He Huo search tasks."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.accountant import AccountantMiddleware
from deepagents.middleware.aihehuo import AihehuoMiddleware
from deepagents.middleware.datetime import DateTimeMiddleware
from deepagents.middleware.language import LanguageDetectionMiddleware

from apps.business_cofounder_api.agent_factory.memory import ApiMemoryMiddleware
from apps.business_cofounder_api.agent_factory.model_builder import create_model
from apps.business_cofounder_api.checkpointer import DiskBackedInMemorySaver

# Use uvicorn's configured logger so output reliably shows up in the terminal.
_logger = logging.getLogger("uvicorn.error")


# Search Agent System Prompt
SEARCH_AGENT_SYSTEM_PROMPT = """You are an AI He Huo (爱合伙) search specialist agent.

You specialize in:
- Searching for co-founders, business partners, and team members on the AI He Huo platform
- Finding investors interested in specific industries, technologies, or business models
- Discovering domain experts with relevant experience
- Exploring related business ideas and projects
- Matching entrepreneurs with potential partners based on business needs

Operating rules:
- Use aihehuo_search_members to find people (co-founders, investors, experts)
- Use aihehuo_search_ideas to find related business ideas and projects
- Create multiple targeted searches for different roles/needs
- Use natural language queries (full sentences, not just keywords)
- Query must be longer than 5 characters for member searches
- Use the investor parameter when searching specifically for investors
- Synthesize results and provide clear recommendations
- Be specific and targeted in your search queries

**Report Writing Requirements:**
- When writing reports or summaries, use Chinese (中文) as the language
- For each candidate you recommend, you MUST include their profile page link/URL if it's available in the search results
- Profile links are essential - always extract and include them from the search response data
"""


def create_search_agent(
    *,
    agent_id: str,
    provider: str = "qwen",
) -> tuple[object, Path]:
    """Create a standalone search agent for AI He Huo platform searches.

    This agent specializes in:
    - Finding co-founders, partners, and team members
    - Finding investors
    - Discovering domain experts
    - Exploring related business ideas

    Args:
        agent_id: Identifier for the agent
        provider: Model provider to use ("qwen" or "deepseek", default: "qwen")

    Returns:
        (agent_graph, checkpoints_path)
    """
    # Create model with AIHEHUO_SUBAGENT_MODEL suffix
    # Note: Temperature defaults to 0.7 for search agent if MODEL_API_TEMPERATURE is not set
    model = create_model(
        provider=provider,
        model_name_suffix="AIHEHUO_SUBAGENT_MODEL",
        log_prefix="[SearchAgent]",
        set_max_input_tokens=True,
    )
    
    # Override temperature to 0.7 for search agent (more creative)
    # This matches the default behavior in build_aihehuo_subagent_from_env
    if not os.environ.get("MODEL_API_TEMPERATURE"):
        # Only override if temperature wasn't explicitly set via env var
        if hasattr(model, "temperature"):
            model.temperature = 0.7
        elif hasattr(model, "model_kwargs") and isinstance(model.model_kwargs, dict):
            model.model_kwargs["temperature"] = 0.7

    # Base directory
    base_dir = Path.home() / ".deepagents" / "business_cofounder_api"
    checkpoints_path = base_dir / "search_agent_checkpoints.pkl"

    base_dir.mkdir(parents=True, exist_ok=True)

    # Middleware configuration for search agent
    # Include AihehuoMiddleware for search capabilities and DateTimeMiddleware for timestamps
    middleware = [
        AccountantMiddleware(max_tool_calls=50),  # Higher limit for multiple searches
        LanguageDetectionMiddleware(),
        DateTimeMiddleware(),  # Provides get_current_datetime tool for accurate timestamps
        ApiMemoryMiddleware(base_dir=base_dir),  # Support memory for user preferences
        AihehuoMiddleware(),  # Provides aihehuo_search_members and aihehuo_search_ideas tools
    ]

    _logger.info("[SearchAgent] Middleware configuration:")
    _logger.info("  - AccountantMiddleware (max_tool_calls=50)")
    _logger.info("  - LanguageDetectionMiddleware")
    _logger.info("  - DateTimeMiddleware")
    _logger.info("  - ApiMemoryMiddleware")
    _logger.info("  - AihehuoMiddleware")
    
    # Check for AI He Huo API key
    aihehuo_api_key = os.environ.get("AIHEHUO_API_KEY")
    if not aihehuo_api_key:
        _logger.warning("[SearchAgent] AIHEHUO_API_KEY not set - search functionality will be limited")

    # Use virtual_mode=True for security
    backend = FilesystemBackend(root_dir=str(base_dir), virtual_mode=True)

    checkpointer = DiskBackedInMemorySaver(file_path=checkpoints_path)

    # Search-focused system prompt
    system_prompt = SEARCH_AGENT_SYSTEM_PROMPT

    agent = create_deep_agent(
        model=model,
        backend=backend,
        checkpointer=checkpointer,
        subagents=[],  # No subagents
        middleware=middleware,
        system_prompt=system_prompt,
    )

    _logger.info("[SearchAgent] Agent created successfully")
    _logger.info("  Checkpoints path: %s", checkpoints_path)

    return agent, checkpoints_path
