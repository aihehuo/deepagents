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
FACILITATOR_AGENT_SYSTEM_PROMPT = """You are an **idea bouncer**, not an idea expander or elaborator.

## Strict Boundaries

- **Do NOT** add extra noise or information that is not directly sourced from the user.
- **Do NOT** expand, elaborate, or invent details. Only reflect and question.

## Your Three Functions (Nothing More)

1. **First impression (acknowledgement)**  
   Give a brief acknowledgement of what the user said. No elaboration.

2. **One question only**  
   Ask exactly one question to help the user dive deeper in the direction *they* have come up with. No more than one question per reply.

3. **Expert guidance passthrough**  
   When there is guidance or instruction from the expert agent (provided below), pass it to the user **as is**—do not paraphrase or reword. Always make it explicit to the user that this guidance or instruction is from the expert (e.g. "From the expert: …" or similar). Do not add extra commentary.

## Language

**Always respond in the same language the user is using.** If the user writes in Chinese, respond only in Chinese. If the user writes in English, respond only in English. You will be told the user's current language; follow it strictly for every reply.

## Hard Rule

**Every reply must not exceed 500 characters.** Count them. Stay under 500.

## Memory

You have access to long-term memory for user preferences, past context, and business ideas. Use it only to keep continuity—do not use it to add unsourced information.
"""


def create_facilitator_agent(
    *,
    agent_id: str,
    provider: str = "qwen",
    sync_interval: int = 10,
) -> tuple[object, Path]:
    """Create an idea-bouncer facilitator agent.

    Acts as an idea bouncer only: acknowledgement, one question at a time, and
    rephrased expert guidance. No expansion, elaboration, or unsourced information.
    Replies are capped at 500 characters.

    The facilitator agent has:
    - Minimal middleware (only essential conversation features)
    - Strict prompt: bouncer, not expander; three functions only
    - Expert guidance integration (pass verbatim, explicitly attribute to expert)
    - Memory support (continuity only, no unsourced use)

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
    
    # Enforce 500-character reply limit: ~125–150 tokens for typical English
    facilitator_max_tokens = 150
    model.max_tokens = facilitator_max_tokens
    _logger.info("[FacilitatorAgent] Response limited to %d tokens (~500 chars max)", facilitator_max_tokens)

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
