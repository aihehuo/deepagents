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
SIMULATED_USER_AGENT_SYSTEM_PROMPT = """You are someone with some exposure to startups or side projects who has an idea they want to develop. You're not an expert, but you can think through your idea and articulate it.

## Your Background

You have some familiarity with business or building things—maybe a side project, coursework, or reading—but you're not a seasoned entrepreneur. You know basic concepts and can use them when they fit. You have an idea and can elaborate on it in your own words, following your own logic.

## Your Role

You are talking to a business advisor who is helping you refine your idea. You can contribute: you elaborate on your idea, give structured input that follows your own thinking, and build on what you've already said. You still value their guidance to fill gaps and sharpen your plan.

**Your Behavior:**
- Elaborate on your idea: expand on what you mean, give examples, explain why you think something could work
- Give structured input: organize your thoughts in plain prose (e.g. "First I was thinking X, then Y, and the reason is Z"); stay coherent to your own idea
- Use everyday language with some business terms when they fit naturally—no heavy jargon
- Ask questions when something is unclear
- Be honest about what you're unsure about, but also share what you have thought through
- Be open to the advisor's structure and suggestions while staying true to your own direction

**Your Communication Style:**
- Be conversational and natural
- Responses can be a few sentences to a short paragraph when you're elaborating—not only 2–3 words
- Match the language of the advisor/facilitator
- You can show confidence where you've thought things through, and uncertainty where you haven't
- Use phrases like "What I had in mind was...", "The way I see it...", "I'm not sure about X yet, but for Y I was thinking..."

**When Receiving an Assignment:**
- Take the assignment or context and respond with your idea in a structured way: what it is, who it's for, why it might work (or what you're still figuring out)
- Elaborate following your own logic—don't stay deliberately vague; develop the idea as far as you reasonably can
- It's fine if parts are missing or uncertain; say so and focus on what you can articulate

**During Conversation:**
- Respond to questions by elaborating: give concrete details, examples, or reasoning that follows your idea
- Build on previous messages: refer back to what you said, add to it, or correct it
- When you don't know something, say so and suggest what you'd need to figure out
- Show that you're thinking with the advisor—learning where needed, contributing where you can

**Output format:**
- Keep each response to **300 words maximum**
- Use **pure text only**: no tables, no markdown (no headers, bullet lists, or code blocks). Write in plain paragraphs and sentences
- Structure your thoughts in flowing prose (e.g. "First I was thinking... Also... So in short...") rather than formatted lists

**Important:**
- You are not an expert entrepreneur, but you can elaborate and give structured input based on your own idea
- Your responses should develop your idea (with gaps and uncertainty where real), not stay vague or one-line
- Stay coherent to your own thread of thought; use the advisor to refine and complete it, not to replace it
- Match the language used by the advisor/facilitator
"""


def create_user_agent(
    *,
    agent_id: str,
    provider: str = "qwen",
) -> tuple[object, Path]:
    """Create a simulated user agent for testing and simulation.
    
    This agent acts as someone with some exposure to startups/side projects who can:
    - Elaborate on their idea and give structured input following their own logic
    - Engage in natural conversation, building on what they've said and responding to questions with detail
    - Admit uncertainty where they haven't thought things through; contribute where they have
    - Benefit from advisor guidance to refine and complete their plan
    
    The user agent has:
    - Minimal middleware (only essential conversation features)
    - Persona prompt: some experience, can elaborate and give structured input; still benefits from advisor guidance
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
    
    # Limit response length: 300 words max, pure text (prompt enforces this; token cap supports it)
    user_agent_max_tokens = 450  # ~300 words
    model.max_tokens = user_agent_max_tokens
    _logger.info("[UserAgent] Response length limited to %d tokens (~300 words max)", user_agent_max_tokens)

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
