"""YAML-gated multi-level search Module (mod.brain.search_relax).

Off → today's single-shot behavior (relax_level forced to 0, no soft drop).
On  → model may issue additional ``search_candidates`` calls with
      ``relax_level`` / optional ``pool``; each level is a real tool call (D-B03).

``mod.brain.profile_pool`` is noted as the next hook only — ``pool`` is accepted
and logged, but ``agent_profiles`` is not preferred until that Module lands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from apps.group_agent_api.agent_factory.search_relax.ids import (
    DEFAULT_POOL,
    MODULE_ID,
    PROFILE_POOL,
    PROFILE_POOL_MODULE_ID,
    PROTOCOL_NAME,
)
from apps.group_agent_api.agent_factory.search_relax.strategy import (
    STRATEGY_DOC,
    apply_relax,
)

__all__ = [
    "MODULE_ID",
    "PROFILE_POOL_MODULE_ID",
    "PROTOCOL_NAME",
    "STRATEGY_DOC",
    "DEFAULT_POOL",
    "PROFILE_POOL",
    "ResolvedSearchRelax",
    "search_relax_enabled",
    "search_relax_max_levels",
    "profile_pool_enabled",
    "resolve_search_relax",
    "search_relax_system_addon",
]


@dataclass(frozen=True)
class ResolvedSearchRelax:
    """Outcome of resolving tool args against the Module switch."""

    enabled: bool
    args: RelaxedSearchArgs
    profile_pool_hook: bool  # True only when pool requested but Module not mounted
    clamped: bool
    raw_level: int


def search_relax_enabled(*, enabled: bool | None = None) -> bool:
    if enabled is not None:
        return bool(enabled)
    from apps.group_agent_api.agent_factory.module_config import load_modules_config

    return load_modules_config().is_module_enabled(MODULE_ID)


def search_relax_max_levels(*, max_levels: int | None = None) -> int:
    """Max inclusive levels counting from 0 (e.g. 2 → L0 and L1)."""
    if max_levels is not None:
        return max(1, min(5, int(max_levels)))
    from apps.group_agent_api.agent_factory.module_config import load_modules_config

    return load_modules_config().search_relax_max_levels()


def profile_pool_enabled(*, enabled: bool | None = None) -> bool:
    """Next-hook flag. Always False until ``mod.brain.profile_pool`` is mounted."""
    if enabled is not None:
        return bool(enabled)
    from apps.group_agent_api.agent_factory.module_config import load_modules_config

    return load_modules_config().is_module_enabled(PROFILE_POOL_MODULE_ID)


def resolve_search_relax(
    *,
    query: str,
    rank_query: str,
    relax_level: int | str | None = 0,
    pool: str | None = "",
    constraints: dict[str, Any] | list[Any] | None = None,
    enabled: bool | None = None,
    max_levels: int | None = None,
) -> ResolvedSearchRelax:
    """Map tool args → effective search args.

    Off path: force L0 identity (≡ today's single search).
    On path: clamp level to ``[0, max_levels-1]`` and apply strategy.
    Never schedules another search — D-B03.
    """
    raw_level = _parse_level(relax_level)
    on = search_relax_enabled(enabled=enabled)
    cap = search_relax_max_levels(max_levels=max_levels)
    requested_pool = str(pool or "").strip()

    if not on:
        args = apply_relax(
            level=0,
            query=query,
            rank_query=rank_query,
            constraints=constraints,
            pool=DEFAULT_POOL,
        )
        return ResolvedSearchRelax(
            enabled=False,
            args=args,
            profile_pool_hook=False,
            clamped=raw_level != 0,
            raw_level=raw_level,
        )

    effective = min(max(0, raw_level), cap - 1)
    clamped = effective != raw_level

    # profile_pool next hook: accept pool for audit, but do not prefer agent_profiles
    # until that Module is on. Unknown pools fall back to all_reachable.
    pool_hook = False
    resolved_pool = requested_pool or DEFAULT_POOL
    if resolved_pool == PROFILE_POOL and not profile_pool_enabled():
        pool_hook = True
        resolved_pool = DEFAULT_POOL
    elif resolved_pool not in {DEFAULT_POOL, PROFILE_POOL}:
        resolved_pool = DEFAULT_POOL

    args = apply_relax(
        level=effective,
        query=query,
        rank_query=rank_query,
        constraints=constraints,
        pool=resolved_pool,
    )
    return ResolvedSearchRelax(
        enabled=True,
        args=args,
        profile_pool_hook=pool_hook,
        clamped=clamped,
        raw_level=raw_level,
    )


def search_relax_system_addon(
    *,
    enabled: bool | None = None,
    max_levels: int | None = None,
) -> str:
    """Extra system-prompt lines when the Module is on; empty when off."""
    if not search_relax_enabled(enabled=enabled):
        return ""
    cap = search_relax_max_levels(max_levels=max_levels)
    top = max(0, cap - 1)
    return (
        "\n## 搜人多级放宽（mod.brain.search_relax 已开启）\n"
        f"- 策略：{STRATEGY_DOC}\n"
        "- 首次调用 `search_candidates` 时用 `relax_level=0`（硬约束）。\n"
        f"- 若工具返回 `status=empty`，可**再调用一次**工具并提高 `relax_level`"
        f"（最高 {top}）；每一级必须是真实 tool call，禁止等待系统代搜（D-B03）。\n"
        "- 可选参数 `pool` 预留；`profile_pool` Module 未挂载前按全池处理。\n"
        "- 仍 empty 则如实说没找到，禁止编造候选人。\n"
        "- 本模块开启时，允许同一轮内按级别多次搜人；不要口头承诺「稍后系统再搜」。\n"
    )


def _parse_level(raw: int | str | None) -> int:
    if raw is None or raw == "":
        return 0
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0
