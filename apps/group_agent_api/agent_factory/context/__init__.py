"""mod.brain.context — YAML-switchable system / turn prompt fragments."""

from __future__ import annotations

from apps.group_agent_api.agent_factory.context.ids import (
    ALL_CONTEXT_IDS,
    CTX_FORCE_SAVE_PROMPT,
    CTX_SYSTEM_ADVISOR_TONE,
    CTX_SYSTEM_NETWORK_DONTS,
    CTX_SYSTEM_PERSIST_RULES,
    CTX_SYSTEM_ROLE_AND_GOAL,
    CTX_SYSTEM_SEARCH_TOOL,
    CTX_SYSTEM_SUGGESTED_REPLIES,
    CTX_SYSTEM_WECHAT_STYLE,
    CTX_TURN_KNOWN_PROFILE,
    CTX_TURN_PRIOR_RECOMMENDATION,
    CTX_TURN_REFERRAL,
    MODULE_ID,
    SYSTEM_FRAGMENT_IDS,
    TURN_FRAGMENT_IDS,
)
from apps.group_agent_api.agent_factory.context.module import (
    FORCE_SAVE_PROMPT_TEXT,
    build_system_prompt,
    context_module_enabled,
    enabled_context_ids,
    force_save_prompt,
    is_context_enabled,
)
from apps.group_agent_api.agent_factory.context.turns import (
    known_profile_system_message,
    prior_recommendation_system_content,
    referral_context_system_message,
)

__all__ = [
    "MODULE_ID",
    "ALL_CONTEXT_IDS",
    "SYSTEM_FRAGMENT_IDS",
    "TURN_FRAGMENT_IDS",
    "CTX_SYSTEM_ROLE_AND_GOAL",
    "CTX_SYSTEM_ADVISOR_TONE",
    "CTX_SYSTEM_SEARCH_TOOL",
    "CTX_SYSTEM_WECHAT_STYLE",
    "CTX_SYSTEM_PERSIST_RULES",
    "CTX_SYSTEM_NETWORK_DONTS",
    "CTX_SYSTEM_SUGGESTED_REPLIES",
    "CTX_TURN_KNOWN_PROFILE",
    "CTX_TURN_PRIOR_RECOMMENDATION",
    "CTX_TURN_REFERRAL",
    "CTX_FORCE_SAVE_PROMPT",
    "FORCE_SAVE_PROMPT_TEXT",
    "context_module_enabled",
    "is_context_enabled",
    "enabled_context_ids",
    "build_system_prompt",
    "force_save_prompt",
    "known_profile_system_message",
    "referral_context_system_message",
    "prior_recommendation_system_content",
]
