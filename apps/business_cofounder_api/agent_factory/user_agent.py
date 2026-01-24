"""Simulated user agent creation for testing and simulation."""

from __future__ import annotations

import logging
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.accountant import AccountantMiddleware
from deepagents.middleware.language import LanguageDetectionMiddleware
from deepagents.middleware.prompt_logging import PromptLoggingMiddleware

from apps.business_cofounder_api.agent_factory.memory import ApiMemoryMiddleware
from apps.business_cofounder_api.agent_factory.model_builder import create_model
from apps.business_cofounder_api.checkpointer import DiskBackedInMemorySaver

# Use uvicorn's configured logger so output reliably shows up in the terminal.
_logger = logging.getLogger("uvicorn.error")


# Simulated User Agent System Prompt
SIMULATED_USER_AGENT_SYSTEM_PROMPT = """You are someone with almost zero startup experience who has a rough idea they want to explore.

## Your Background

You have NO experience with startups, business planning, or entrepreneurship. You don't know business terminology, frameworks, or how startups work. You're just someone with a vague thought or idea that might become something.

## Your Role

You are talking to a business advisor who is helping you figure out your idea. You need their help because you don't know what you're doing or what's important.

**Your Behavior:**
- Express ideas in simple, everyday language - avoid business jargon
- Show uncertainty and hesitation - you're not confident about your ideas
- Ask questions when you don't understand something
- Be honest about what you don't know
- Share rough, incomplete thoughts - your ideas are half-formed
- Express confusion about next steps or what matters
- Be open to guidance and structure from the advisor

**Your Communication Style:**
- Be conversational and natural
- Keep responses concise (2-4 sentences typically)
- Match the language of the advisor/facilitator
- Use phrases like "I'm not sure if this makes sense..." or "I was thinking maybe..."
- Show uncertainty: "I don't really know..." or "Maybe something like..."
- Express incomplete thinking: "I guess..." or "I'm not entirely clear on..."

**When Receiving an Assignment:**
- If you receive an assignment or general context, share whatever rough, vague idea comes to mind
- Don't try to make it detailed or concrete - you don't know how to do that
- Your idea will be incomplete, missing key parts, or unclear
- Express uncertainty about whether it's even a good idea
- Use simple language to describe what you're thinking

**During Conversation:**
- Respond naturally to questions, but often with uncertainty
- Share incomplete thoughts and half-formed ideas
- Ask for help understanding what the advisor is asking
- Express confusion when you don't know how to answer
- Show how you're learning and thinking through things with the advisor's help
- Admit when you don't know something or haven't thought about it

**Important:**
- You are NOT an experienced entrepreneur - you're a beginner
- Your ideas should be vague, uncertain, and incomplete
- You need guidance and structure - you can't provide detailed business plans
- Use everyday language, not business terms
- Show natural uncertainty and hesitation
- Match the language used by the advisor/facilitator
"""


def create_user_agent(
    *,
    agent_id: str,
    provider: str = "qwen",
) -> tuple[object, Path]:
    """Create a simulated user agent for testing and simulation.
    
    This agent acts as someone with zero startup experience who can:
    - Share rough, vague ideas from general assignments (not detailed or concrete)
    - Engage in natural conversations with uncertainty and hesitation
    - Express incomplete thoughts and ask for guidance
    - Respond authentically as a beginner would
    
    The user agent has:
    - Minimal middleware (only essential conversation features)
    - Zero-experience persona prompt (vague, uncertain, needs guidance)
    - Memory support (remembers conversation context)
    - Separate checkpoint file for thread isolation
    
    Args:
        agent_id: Identifier for the agent
        provider: Model provider to use ("qwen" or "deepseek", default: "qwen")
    
    Returns:
        (agent_graph, checkpoints_path)
    """
    # Create model
    model = create_model(
        provider=provider,
        model_name_suffix="MAIN_AGENT_MODEL",
        log_prefix="[UserAgent]",
        set_max_input_tokens=False,  # User agent doesn't need summarization
    )
    
    # Limit response length for casual chat - override max_tokens to keep responses concise
    # Default max_tokens is 20000, but for casual conversation we want much shorter responses
    user_agent_max_tokens = 800  # ~200-300 words, suitable for casual chat
    model.max_tokens = user_agent_max_tokens
    _logger.info("[UserAgent] Response length limited to %d tokens for casual chat", user_agent_max_tokens)

    # Base directory
    base_dir = Path.home() / ".deepagents" / "business_cofounder_api"
    checkpoints_path = base_dir / "user_agent_checkpoints.pkl"

    base_dir.mkdir(parents=True, exist_ok=True)

    # Minimal middleware configuration
    # Note: PromptLoggingMiddleware should be last to capture final prompt state
    middleware = [
        AccountantMiddleware(max_tool_calls=25),
        LanguageDetectionMiddleware(),
        ApiMemoryMiddleware(base_dir=base_dir),
        PromptLoggingMiddleware(),  # Add last to log final prompt stack before LLM call
    ]

    _logger.info("[UserAgent] Middleware configuration:")
    _logger.info("  - AccountantMiddleware (max_tool_calls=25)")
    _logger.info("  - LanguageDetectionMiddleware")
    _logger.info("  - ApiMemoryMiddleware")
    _logger.info("  - PromptLoggingMiddleware (logs complete prompt stack before LLM call)")

    # Use virtual_mode=True for security
    backend = FilesystemBackend(root_dir=str(base_dir), virtual_mode=True)

    checkpointer = DiskBackedInMemorySaver(file_path=checkpoints_path)

    # Simple, conversational system prompt
    system_prompt = SIMULATED_USER_AGENT_SYSTEM_PROMPT

    agent = create_deep_agent(
        model=model,
        backend=backend,
        checkpointer=checkpointer,
        subagents=[],  # No subagents for user agent
        middleware=middleware,
        system_prompt=system_prompt,
    )

    _logger.info("[UserAgent] Agent created successfully")
    _logger.info("  Checkpoints path: %s", checkpoints_path)

    return agent, checkpoints_path
