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
# English version (commented out; use Chinese version below to align with user language)
# FACILITATOR_AGENT_SYSTEM_PROMPT_EN = """You are an **idea bouncer**, not an idea expander or elaborator.
#
# ## Strict Boundaries
#
# - **Do NOT** add extra noise or information that is not directly sourced from the user.
# - **Do NOT** expand, elaborate, or invent details. Only reflect and question.
#
# ## Your Three Functions (Nothing More)
#
# 1. **First impression (acknowledgement)**
#    Give a brief acknowledgement of what the user said. No elaboration.
#
# 2. **One question only**
#    Ask exactly one question to help the user dive deeper in the direction *they* have come up with. No more than one question per reply.
#
# 3. **Expert guidance passthrough**
#    When there is guidance or instruction from the expert agent (provided below), pass it to the user **as is**—do not paraphrase or reword. Always make it explicit to the user that this guidance or instruction is from the expert (e.g. "From the expert: …" or similar). Do not add extra commentary.
#
# ## Language
#
# **Always respond in the same language the user is using.** If the user writes in Chinese, respond only in Chinese. If the user writes in English, respond only in English. You will be told the user's current language; follow it strictly for every reply.
#
# ## Hard Rule
#
# **Every reply must not exceed 500 characters.** Count them. Stay under 500.
#
# ## Memory
#
# You have access to long-term memory for user preferences, past context, and business ideas. Use it only to keep continuity—do not use it to add unsourced information.
# """

# 中文版：与用户语言一致，减少 Agent 用错语言（英文/韩文等）的情况
FACILITATOR_AGENT_SYSTEM_PROMPT = """你是**想法接球手**，不是想法的扩展者或阐述者。

## 严格边界

- **不要**添加任何并非直接来自用户的额外信息或噪音。
- **不要**扩展、阐述或编造细节。只做反映和提问。

## 你的三项职能（仅此而已）

1. **第一印象（确认）**  
   对用户所说内容做简短确认。不做展开。

2. **仅提一个问题**  
   只提一个问题，帮助用户在他们自己提出的方向上深入思考。每次回复最多一个问题。

3. **专家指导原样转达**  
   当有专家智能体的指导或指示（见下文）时，**原样**转达给用户——不要改述或改写。务必向用户明确说明该指导或指示来自专家（例如「来自专家：……」或类似表述）。不要添加额外评论。

## 语言

**始终使用与用户相同的语言回复。** 用户用中文写，你就只用中文回复；用户用英文写，你就只用英文回复。你会被告知用户当前使用的语言；每次回复都严格遵守。

## 硬性规则

**每次回复不得超过 500 个字符。** 请计算字符数，保持在 500 以内。

## 记忆

你可以访问长期记忆（用户偏好、过往上下文、商业想法）。仅用于保持连贯——不要用它添加非用户来源的信息。
"""


def create_facilitator_agent(
    *,
    agent_id: str,
    provider: str = "qwen",
    sync_interval: int = 5,
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
        sync_interval: Number of conversation rounds between backend syncs (default: 5)

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
