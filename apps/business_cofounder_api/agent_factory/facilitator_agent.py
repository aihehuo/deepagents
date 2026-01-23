"""Facilitator agent creation for dual-agent architecture."""

from __future__ import annotations

import logging
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.accountant import AccountantMiddleware
from deepagents.middleware.expert_guidance import ExpertGuidanceMiddleware
from deepagents.middleware.language import LanguageDetectionMiddleware
from deepagents.middleware.prompt_logging import PromptLoggingMiddleware

from apps.business_cofounder_api.agent_factory.memory import ApiMemoryMiddleware
from apps.business_cofounder_api.agent_factory.model_builder import create_model
from apps.business_cofounder_api.checkpointer import DiskBackedInMemorySaver

# Use uvicorn's configured logger so output reliably shows up in the terminal.
_logger = logging.getLogger("uvicorn.error")


# Facilitator Agent System Prompt
FACILITATOR_AGENT_SYSTEM_PROMPT = """You are a business co-founder conversation partner.

## Your Role

You are helping an entrepreneur explore and develop their startup idea through natural, insightful conversation.

**Your Mission:**
- Have genuine, thoughtful conversations about their business
- Ask insightful questions to understand their vision, challenges, and goals
- Help them articulate their ideas clearly
- Provide initial thoughts, reflections, and perspectives
- Build rapport and create a safe space for exploration

**Your Style:**
- Be conversational and natural, not procedural or rigid
- Ask one or a few questions at a time - don't overwhelm
- Listen actively and build on what they share
- Show curiosity and genuine interest
- Adapt to their communication style and pace
- Don't force them through a structured process

**Your Approach:**
- Follow their lead while gently guiding toward clarity
- When they share something, acknowledge it before moving forward
- If something is unclear, ask for clarification
- If they seem stuck, offer prompts or examples
- If they're excited, explore that energy
- If they're uncertain, help them think it through

**Important Principles:**
- Focus on understanding deeply, not executing workflows
- Let the conversation flow naturally based on their needs
- Quality of dialogue over quantity of information
- Build on insights incrementally rather than rushing
- **Keep responses concise and casual** - aim for 2-4 sentences typically, like a natural conversation
- Avoid lengthy explanations or walls of text - be brief and conversational

**Backend Support:**
You have a backend analysis system that periodically reviews our conversations and provides strategic guidance on what to focus on next. This guidance appears below and should inform (but not dictate) your conversation approach.

## Memory

You have access to long-term memory to remember:
- User preferences and communication style
- Previous conversations and context
- Business ideas and their evolution

Use memory to provide continuity across conversations and personalize your interactions.
"""


def create_facilitator_agent(
    *,
    agent_id: str,
    provider: str = "qwen",
    sync_interval: int = 10,
) -> tuple[object, Path]:
    """Create a lightweight facilitator agent for natural business conversations.

    This agent focuses on conversational facilitation rather than structured workflows.
    It's designed to work with a backend analysis agent that provides periodic guidance.

    The facilitator agent has:
    - Minimal middleware (only essential conversation features)
    - Natural conversation prompt (no rigid workflows)
    - Backend guidance integration (receives strategic direction)
    - Memory support (remembers user preferences and context)

    Args:
        agent_id: Identifier for the agent
        provider: Model provider to use ("qwen" or "deepseek", default: "qwen")
        sync_interval: Number of conversation rounds between backend syncs (default: 10)

    Returns:
        (agent_graph, checkpoints_path)
    """
    # Create model
    model = create_model(
        provider=provider,
        model_name_suffix="MAIN_AGENT_MODEL",
        log_prefix="[FacilitatorAgent]",
        set_max_input_tokens=False,  # Facilitator doesn't need summarization
    )
    
    # Limit response length for casual chat - override max_tokens to keep responses concise
    # Default max_tokens is 20000, but for casual conversation we want much shorter responses
    facilitator_max_tokens = 800  # ~200-300 words, suitable for casual chat
    model.max_tokens = facilitator_max_tokens
    _logger.info("[FacilitatorAgent] Response length limited to %d tokens for casual chat", facilitator_max_tokens)

    # Base directory
    base_dir = Path.home() / ".deepagents" / "business_cofounder_api"
    checkpoints_path = base_dir / "facilitator_checkpoints.pkl"

    base_dir.mkdir(parents=True, exist_ok=True)

    # Minimal middleware configuration
    # Note: PromptLoggingMiddleware should be last to capture final prompt state
    middleware = [
        AccountantMiddleware(max_tool_calls=25),
        LanguageDetectionMiddleware(),
        ApiMemoryMiddleware(base_dir=base_dir),
        ExpertGuidanceMiddleware(sync_interval=sync_interval),
        PromptLoggingMiddleware(),  # Add last to log final prompt stack before LLM call
    ]

    _logger.info("[FacilitatorAgent] Middleware configuration:")
    _logger.info("  - AccountantMiddleware (max_tool_calls=25)")
    _logger.info("  - LanguageDetectionMiddleware")
    _logger.info("  - ApiMemoryMiddleware")
    _logger.info("  - ExpertGuidanceMiddleware (sync_interval=%d)", sync_interval)
    _logger.info("  - PromptLoggingMiddleware (logs complete prompt stack before LLM call)")

    # Use virtual_mode=True for security
    backend = FilesystemBackend(root_dir=str(base_dir), virtual_mode=True)

    checkpointer = DiskBackedInMemorySaver(file_path=checkpoints_path)

    # Simple, conversational system prompt
    system_prompt = FACILITATOR_AGENT_SYSTEM_PROMPT

    agent = create_deep_agent(
        model=model,
        backend=backend,
        checkpointer=checkpointer,
        subagents=[],  # No subagents for facilitator
        middleware=middleware,
        system_prompt=system_prompt,
    )

    _logger.info("[FacilitatorAgent] Agent created successfully")
    _logger.info("  Checkpoints path: %s", checkpoints_path)

    return agent, checkpoints_path
