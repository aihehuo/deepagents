"""Module / context fragment identities for mod.brain.context."""

MODULE_ID = "mod.brain.context"

# System prompt fragments (TSD-14 §4.2 / inventory).
CTX_SYSTEM_ROLE_AND_GOAL = "ctx.system.role_and_goal"
CTX_SYSTEM_ADVISOR_TONE = "ctx.system.advisor_tone"
CTX_SYSTEM_SEARCH_TOOL = "ctx.system.search_tool"
CTX_SYSTEM_WECHAT_STYLE = "ctx.system.wechat_style"
CTX_SYSTEM_PERSIST_RULES = "ctx.system.persist_rules"
CTX_SYSTEM_NETWORK_DONTS = "ctx.system.network_donts"
CTX_SYSTEM_SUGGESTED_REPLIES = "ctx.system.suggested_replies"

# Turn-level injects.
CTX_TURN_KNOWN_PROFILE = "ctx.turn.known_profile"
CTX_TURN_PRIOR_RECOMMENDATION = "ctx.turn.prior_recommendation"
CTX_TURN_REFERRAL = "ctx.turn.referral"

# Force-save HumanMessage fragment (not part of system_prompt).
CTX_FORCE_SAVE_PROMPT = "ctx.force_save_prompt"

SYSTEM_FRAGMENT_IDS: tuple[str, ...] = (
    CTX_SYSTEM_ROLE_AND_GOAL,
    CTX_SYSTEM_ADVISOR_TONE,
    CTX_SYSTEM_SEARCH_TOOL,
    CTX_SYSTEM_WECHAT_STYLE,
    CTX_SYSTEM_PERSIST_RULES,
    CTX_SYSTEM_NETWORK_DONTS,
    CTX_SYSTEM_SUGGESTED_REPLIES,
)

TURN_FRAGMENT_IDS: tuple[str, ...] = (
    CTX_TURN_KNOWN_PROFILE,
    CTX_TURN_PRIOR_RECOMMENDATION,
    CTX_TURN_REFERRAL,
)

ALL_CONTEXT_IDS: tuple[str, ...] = (
    *SYSTEM_FRAGMENT_IDS,
    *TURN_FRAGMENT_IDS,
    CTX_FORCE_SAVE_PROMPT,
)
