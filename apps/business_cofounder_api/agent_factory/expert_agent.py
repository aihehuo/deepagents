"""Expert agent creation for dual-agent architecture."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.accountant import AccountantMiddleware
from deepagents.middleware.aihehuo import AihehuoMiddleware
from deepagents.middleware.artifacts import ArtifactsMiddleware
from deepagents.middleware.asset_upload import AssetUploadMiddleware
from deepagents.middleware.business_idea_development import BusinessIdeaDevelopmentMiddleware
from deepagents.middleware.business_idea_tracker import BusinessIdeaTrackerMiddleware
from deepagents.middleware.language import LanguageDetectionMiddleware
from deepagents_cli.skills.middleware import SkillsMiddleware

from apps.business_cofounder_api.agent_factory.middleware_builder import VirtualPathSkillsMiddleware
from apps.business_cofounder_api.agent_factory.model_builder import create_model
from apps.business_cofounder_api.agent_factory.utils import (
    copy_default_expertise_if_missing,
    copy_example_skills_if_missing,
)
from apps.business_cofounder_api.checkpointer import DiskBackedInMemorySaver

# Use uvicorn's configured logger so output reliably shows up in the terminal.
_logger = logging.getLogger("uvicorn.error")


# Generic Expert Agent System Prompt (Base)
# Expertise-specific content will be loaded from expertise files
GENERIC_EXPERT_AGENT_PROMPT = """You are an expert analysis agent in a dual-agent mentoring system.

## Your Role

You work alongside a facilitator agent who conducts natural conversations with users.
Your job is to:
1. Analyze conversations from an expert perspective
2. Extract meaningful insights and key points
3. Track progress and identify patterns
4. Provide strategic guidance to the facilitator

## How You Operate

- You receive conversation history periodically (typically every 10 rounds)
- You analyze it through the lens of your specific expertise
- You output structured data (canvas) and strategic guidance
- Your guidance is injected into the facilitator's system prompt

## Output Requirements

You must produce a JSON object with:

1. **expert_guidance** (string, 2-4 sentences):
   Strategic direction for the facilitator agent.
   This guidance will be injected into the facilitator's system prompt to guide upcoming conversations.
   IMPORTANT: the guidance statement MUST be phrase from second-person perspective, talking directly to the facilitator.
   Example: "You should focus on understanding the user's business idea through natural conversation.
   Ask thoughtful questions to help them articulate their vision, challenges, and goals."
   Example: "You should focus on understanding the user's business idea through natural conversation.
   !!NOT from a third person perspective!!
   WRONG EXAMPLE: "The facilitator should focus on understanding the user's business idea through natural conversation.

2. **canvas** (JSON object):
   Structured assessment of current state.
   The canvas structure is defined by your specific expertise template.

3. **partner_query** (optional, string in Chinese, 10-20 words):
   When analyzing business ideas, if the idea is sufficiently developed and you have enough information
   about the business needs, generate a partner search query. This should be a natural Chinese sentence/phrase
   (10-20 words) describing the ideal partner based on the business idea, target market, and specific needs.
   The query will be used to search for potential partners on the AI He Huo platform.
   Only include this if you have enough information to create a meaningful search query.
   Example: "寻找有AI技术背景的创业者，希望合作开发教育科技产品"

**Output Format:**
```json
{{
  "expert_guidance": "Your 2-4 sentence strategic guidance here...",
  "canvas": {{
    ... structure defined by your expertise ...
  }},
  "partner_query": "Optional: 10-20 word Chinese sentence describing ideal partner (only if business idea is sufficiently developed)"
}}
```

## Important Principles

- **Operate asynchronously**: The facilitator doesn't wait for you
- **Be strategic, not tactical**: Focus on direction, not specific words
- **Build on progress**: Acknowledge what's been achieved and what's next
- **Quality over quantity**: Fewer great insights better than many mediocre ones
- **Guide, don't dictate**: Provide direction while respecting natural flow
- **Match user's language**: Always respond in the same language the user is using. If the user writes in Chinese, respond in Chinese. If they write in English, respond in English. This applies to ALL output including expert_guidance, canvas content, and summaries.

{expertise_content}
"""


def create_expert_agent(
    *,
    agent_id: str,
    provider: str = "qwen",
    expertise_type: str = "business_cofounder",
) -> tuple[object, Path]:
    """Create an expert analysis agent with specified expertise.

    This agent handles:
    - Conversation analysis and insight extraction
    - Progress tracking using domain-specific methodology
    - Canvas generation (structured progress dashboard)
    - Strategic guidance generation for facilitator agent

    The expert agent has:
    - Full middleware stack (all analysis and methodology features)
    - Domain-specific skills (loaded based on expertise type)
    - Pluggable expertise templates
    - Search and upload capabilities for research and deliverables

    Args:
        agent_id: Identifier for the agent
        provider: Model provider to use ("qwen" or "deepseek", default: "qwen")
        expertise_type: Type of expertise to load (default: "business_cofounder")

    Returns:
        (agent_graph, checkpoints_path)
    """
    # Model configuration (can use a different/more powerful model)
    # Check for EXPERT_AGENT_MODEL first, fall back to MAIN_AGENT_MODEL
    expert_model_suffix = os.getenv("EXPERT_AGENT_MODEL_SUFFIX", "MAIN_AGENT_MODEL")

    # Create model
    model = create_model(
        provider=provider,
        model_name_suffix=expert_model_suffix,
        log_prefix="[ExpertAgent]",
        set_max_input_tokens=True,
    )

    # Base directory
    base_dir = Path.home() / ".deepagents" / "business_cofounder_api"
    skills_dir = base_dir / "skills"
    docs_dir = base_dir / "docs"
    expertise_dir = base_dir / "expertise"
    checkpoints_path = base_dir / "expert_checkpoints.pkl"

    base_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    copy_example_skills_if_missing(dest_skills_dir=skills_dir)
    copy_default_expertise_if_missing(dest_expertise_dir=expertise_dir)
    
    # Load expertise
    from apps.business_cofounder_api.expertise_loader import load_expertise
    
    try:
        expertise = load_expertise(expertise_type, expertise_dir)
        _logger.info("[ExpertAgent] Loaded expertise: %s", expertise["name"])
        _logger.info("  Description: %s", expertise["description"])
    except (FileNotFoundError, ValueError) as e:
        _logger.error("[ExpertAgent] Failed to load expertise '%s': %s", expertise_type, str(e))
        raise RuntimeError(f"Cannot create expert agent without valid expertise: {str(e)}") from e

    # Use virtual_mode=True for security
    backend_fs = FilesystemBackend(root_dir=str(base_dir), virtual_mode=True)

    checkpointer = DiskBackedInMemorySaver(file_path=checkpoints_path)

    # Create SkillsMiddleware with path conversion wrapper
    base_skills_middleware = SkillsMiddleware(
        skills_dir=skills_dir,
        assistant_id=agent_id,
        project_skills_dir=None,
    )

    virtual_skills_middleware = VirtualPathSkillsMiddleware(
        base_skills_middleware, skills_dir
    )

    # Full middleware stack for backend analysis
    middleware = [
        AccountantMiddleware(
            max_tool_calls=50
        ),  # Higher limit for backend analysis
        LanguageDetectionMiddleware(),
        BusinessIdeaTrackerMiddleware(),
        BusinessIdeaDevelopmentMiddleware(strict_todo_sync=True),
        virtual_skills_middleware,
        AihehuoMiddleware(),  # For market/co-founder research
        AssetUploadMiddleware(
            backend_root=str(base_dir), docs_dir=str(docs_dir)
        ),  # For deliverables
        ArtifactsMiddleware(),  # Track generated artifacts
    ]

    _logger.info("[ExpertAgent] Middleware configuration:")
    _logger.info("  - AccountantMiddleware (max_tool_calls=50)")
    _logger.info("  - LanguageDetectionMiddleware")
    _logger.info("  - BusinessIdeaTrackerMiddleware")
    _logger.info("  - BusinessIdeaDevelopmentMiddleware")
    _logger.info("  - VirtualPathSkillsMiddleware")
    _logger.info("  - AihehuoMiddleware")
    _logger.info("  - AssetUploadMiddleware")
    _logger.info("  - ArtifactsMiddleware")

    # Build system prompt with loaded expertise
    system_prompt = GENERIC_EXPERT_AGENT_PROMPT.format(
        expertise_content=expertise["system_prompt"]
    )

    agent = create_deep_agent(
        model=model,
        backend=backend_fs,
        checkpointer=checkpointer,
        subagents=[],  # No subagents - code and search are now separate standalone agents
        middleware=middleware,
        system_prompt=system_prompt,
    )

    _logger.info("[ExpertAgent] Agent created successfully")
    _logger.info("  Checkpoints path: %s", checkpoints_path)
    _logger.info("  Skills directory: %s", skills_dir)
    _logger.info("  Docs directory: %s", docs_dir)

    return agent, checkpoints_path
