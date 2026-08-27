"""YAML-gated context Module (mod.brain.context).

Default on ≡ today's SYSTEM_PROMPT + turn injects. Off fragment → omit that
guidance (observable shorter / different prompt or no SystemMessage inject).
"""

from __future__ import annotations

from typing import Iterable

from apps.group_agent_api.agent_factory.context.fragments import (
    FORCE_SAVE_PROMPT_TEXT,
    PROMPT_PIECES,
)
from apps.group_agent_api.agent_factory.context.ids import (
    ALL_CONTEXT_IDS,
    CTX_FORCE_SAVE_PROMPT,
    MODULE_ID,
    SYSTEM_FRAGMENT_IDS,
)

__all__ = [
    "MODULE_ID",
    "ALL_CONTEXT_IDS",
    "SYSTEM_FRAGMENT_IDS",
    "context_module_enabled",
    "is_context_enabled",
    "enabled_context_ids",
    "build_system_prompt",
    "force_save_prompt",
    "FORCE_SAVE_PROMPT_TEXT",
]


def context_module_enabled(*, enabled: bool | None = None) -> bool:
    """Master switch ``mod.brain.context`` (default true when YAML omits key)."""
    if enabled is not None:
        return bool(enabled)
    from apps.group_agent_api.agent_factory.module_config import load_modules_config

    cfg = load_modules_config()
    if MODULE_ID not in cfg.modules:
        # Fail-open to today's behavior when Module row not yet declared.
        return True
    return cfg.is_module_enabled(MODULE_ID)


def is_context_enabled(
    context_id: str,
    *,
    enabled_ids: Iterable[str] | None = None,
    module_enabled: bool | None = None,
) -> bool:
    """True when master Module is on and this ctx.* bit is on.

    Missing YAML keys for known ids default to **on** (preset current).
    """
    cid = str(context_id or "").strip()
    if not cid:
        return False

    if enabled_ids is not None:
        return cid in {str(x).strip() for x in enabled_ids if str(x).strip()}

    if not context_module_enabled(enabled=module_enabled):
        return False

    from apps.group_agent_api.agent_factory.module_config import load_modules_config

    cfg = load_modules_config()
    if hasattr(cfg, "is_context_enabled"):
        return cfg.is_context_enabled(cid)
    # Fallback if older ModulesConfig without helper.
    if cid not in cfg.context:
        return True
    return bool(cfg.context.get(cid, False))


def enabled_context_ids(
    *,
    enabled_ids: Iterable[str] | None = None,
    module_enabled: bool | None = None,
) -> frozenset[str]:
    if enabled_ids is not None:
        return frozenset(str(x).strip() for x in enabled_ids if str(x).strip())
    if not context_module_enabled(enabled=module_enabled):
        return frozenset()
    return frozenset(cid for cid in ALL_CONTEXT_IDS if is_context_enabled(cid))


def build_system_prompt(
    enabled_ids: Iterable[str] | None = None,
    *,
    module_enabled: bool | None = None,
) -> str:
    """Assemble member system_prompt from enabled ctx.system.* fragments.

    ``enabled_ids`` overrides YAML (tests). When None, reads live modules.yaml.
    """
    if enabled_ids is not None:
        on = frozenset(str(x).strip() for x in enabled_ids if str(x).strip())
    elif not context_module_enabled(enabled=module_enabled):
        on = frozenset()
    else:
        on = frozenset(
            cid
            for cid in SYSTEM_FRAGMENT_IDS
            if is_context_enabled(cid, module_enabled=module_enabled)
        )

    parts: list[str] = []
    for cid, text in PROMPT_PIECES:
        if cid in on:
            parts.append(text.rstrip())
    return "\n\n".join(parts)


def force_save_prompt(
    *,
    enabled: bool | None = None,
    enabled_ids: Iterable[str] | None = None,
) -> str:
    """Return FORCE_SAVE HumanMessage text, or empty when fragment off."""
    if enabled is not None:
        on = bool(enabled)
    else:
        on = is_context_enabled(CTX_FORCE_SAVE_PROMPT, enabled_ids=enabled_ids)
    return FORCE_SAVE_PROMPT_TEXT if on else ""
