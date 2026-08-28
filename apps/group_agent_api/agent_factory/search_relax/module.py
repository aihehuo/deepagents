"""YAML-gated multi-level search Module (mod.brain.search_relax).

Off → today's single-shot behavior (relax_level forced to 0, no soft drop).
On  → model may issue additional ``search_candidates`` calls with
      ``relax_level`` / optional ``pool``; each level is a real tool call (D-B03).

``mod.brain.profile_pool`` (independent YAML Module) controls pool default:

Pool precedence (documented):
1. Model-supplied known pool (``agent_profiles`` | ``all_reachable``) wins
   over the Module default — except ``agent_profiles`` while Module is **off**
   falls back to ``all_reachable`` and sets ``profile_pool_hook`` (audit).
2. Omitted / empty pool →
   - relax_level=0: Module on: ``agent_profiles``; off: ``all_reachable``.
   - relax_level>=1: ``all_reachable`` (relaxation expands pool).
3. Unknown pool token → ``all_reachable``.
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
    RelaxedSearchArgs,
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
    "search_relax_enabled_for_user",
    "search_relax_max_levels",
    "profile_pool_enabled",
    "profile_pool_enabled_for_user",
    "resolve_pool",
    "resolve_search_relax",
    "search_relax_system_addon",
]


@dataclass(frozen=True)
class ResolvedSearchRelax:
    """Outcome of resolving tool args against the Module switch."""

    enabled: bool
    args: RelaxedSearchArgs
    profile_pool_hook: bool  # True when agent_profiles requested but Module off
    clamped: bool
    raw_level: int
    profile_pool_enabled: bool = False
    pool_source: str = "default"  # model | module_default | fallback_module_off | unknown


def search_relax_enabled(
    *,
    enabled: bool | None = None,
    user_id: int | str | None = None,
) -> bool:
    if enabled is not None:
        return bool(enabled)
    from apps.group_agent_api.agent_factory.module_config import (
        search_relax_enabled_for_user,
    )

    return search_relax_enabled_for_user(user_id)


def search_relax_enabled_for_user(user_id: int | str | None = None) -> bool:
    from apps.group_agent_api.agent_factory.module_config import (
        search_relax_enabled_for_user as _module_search_relax_enabled_for_user,
    )

    return _module_search_relax_enabled_for_user(user_id)


def search_relax_max_levels(*, max_levels: int | None = None) -> int:
    """Max inclusive levels counting from 0 (e.g. 2 → L0 and L1)."""
    if max_levels is not None:
        return max(1, min(5, int(max_levels)))
    from apps.group_agent_api.agent_factory.module_config import load_modules_config

    return load_modules_config().search_relax_max_levels()


def profile_pool_enabled(
    *,
    enabled: bool | None = None,
    user_id: int | str | None = None,
) -> bool:
    """True when ``mod.brain.profile_pool`` is on (YAML / override / canary user)."""
    if enabled is not None:
        return bool(enabled)
    from apps.group_agent_api.agent_factory.module_config import (
        profile_pool_enabled_for_user,
    )

    return profile_pool_enabled_for_user(user_id)


def profile_pool_enabled_for_user(user_id: int | str | None = None) -> bool:
    from apps.group_agent_api.agent_factory.module_config import (
        profile_pool_enabled_for_user as _module_profile_pool_enabled_for_user,
    )

    return _module_profile_pool_enabled_for_user(user_id)


def resolve_pool(
    pool: str | None = "",
    *,
    relax_level: int | str | None = 0,
    profile_pool: bool | None = None,
    user_id: int | str | None = None,
) -> tuple[str, bool, str]:
    """Resolve effective pool + hook flag + source label.

    See module docstring for precedence vs model-supplied pool.
    """
    pp_on = profile_pool_enabled(enabled=profile_pool, user_id=user_id)
    level = _parse_level(relax_level)
    requested = str(pool or "").strip()

    if not requested:
        if level >= 1:
            return DEFAULT_POOL, False, "relaxed_expansion"
        if pp_on:
            return PROFILE_POOL, False, "module_default"
        return DEFAULT_POOL, False, "default"

    if requested == PROFILE_POOL:
        if pp_on:
            return PROFILE_POOL, False, "model"
        return DEFAULT_POOL, True, "fallback_module_off"

    if requested == DEFAULT_POOL:
        return DEFAULT_POOL, False, "model"

    return DEFAULT_POOL, False, "unknown"


def resolve_search_relax(
    *,
    query: str,
    rank_query: str,
    relax_level: int | str | None = 0,
    pool: str | None = "",
    constraints: dict[str, Any] | list[Any] | None = None,
    enabled: bool | None = None,
    max_levels: int | None = None,
    profile_pool: bool | None = None,
    user_id: int | str | None = None,
) -> ResolvedSearchRelax:
    """Map tool args → effective search args.

    Off path: force L0 identity (≡ today's single search) but still resolve pool
    via ``mod.brain.profile_pool``.
    On path: clamp level to ``[0, max_levels-1]`` and apply strategy.
    Never schedules another search — D-B03.
    """
    raw_level = _parse_level(relax_level)
    on = search_relax_enabled(enabled=enabled, user_id=user_id)
    cap = search_relax_max_levels(max_levels=max_levels)
    pp_on = profile_pool_enabled(enabled=profile_pool, user_id=user_id)

    if not on:
        resolved_pool, pool_hook, pool_source = resolve_pool(
            pool, relax_level=0, profile_pool=pp_on, user_id=user_id
        )
        args = apply_relax(
            level=0,
            query=query,
            rank_query=rank_query,
            constraints=constraints,
            pool=resolved_pool,
        )
        return ResolvedSearchRelax(
            enabled=False,
            args=args,
            profile_pool_hook=pool_hook,
            clamped=raw_level != 0,
            raw_level=raw_level,
            profile_pool_enabled=pp_on,
            pool_source=pool_source,
        )

    effective = min(max(0, raw_level), cap - 1)
    clamped = effective != raw_level
    resolved_pool, pool_hook, pool_source = resolve_pool(
        pool, relax_level=effective, profile_pool=pp_on, user_id=user_id
    )

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
        profile_pool_enabled=pp_on,
        pool_source=pool_source,
    )


def search_relax_system_addon(
    *,
    enabled: bool | None = None,
    max_levels: int | None = None,
    profile_pool: bool | None = None,
    user_id: int | str | None = None,
) -> str:
    """Extra system-prompt lines when the Module is on; empty when off."""
    if not search_relax_enabled(enabled=enabled, user_id=user_id):
        return ""
    cap = search_relax_max_levels(max_levels=max_levels)
    top = max(0, cap - 1)
    pp_on = profile_pool_enabled(enabled=profile_pool, user_id=user_id)
    if pp_on:
        pool_line = (
            "- `pool`：`mod.brain.profile_pool` 已开；省略时默认 `agent_profiles`（L1 放宽时自动扩展为 `all_reachable`）。"
            " 模型显式传入 `all_reachable` / `agent_profiles` 优先于默认。\n"
        )
    else:
        pool_line = (
            "- `pool`：`mod.brain.profile_pool` 关闭时按 `all_reachable`；"
            " 若传入 `agent_profiles` 会回退全池并记审计 hook。\n"
        )
    return (
        "\n## 搜人多级放宽（mod.brain.search_relax 已开启）\n"
        f"- 策略：{STRATEGY_DOC}\n"
        "- L0: 优先搜索群内有画像成员 (pool=agent_profiles) 并保留硬性与偏好条件。\n"
        "- L1: 若初次未找到或召回较少，放宽偏好条件并将搜索范围扩展至全平台可达人脉 (pool=all_reachable)。\n"
        "- 首次调用 `search_candidates` 时用 `relax_level=0`（硬约束）。\n"
        f"- 若工具返回 `status=empty`，可**再调用一次**工具并提高 `relax_level`"
        f"（最高 {top}）；每一级必须是真实 tool call，禁止等待系统代搜（D-B03）。\n"
        f"{pool_line}"
        "- `constraints` 可显式传入；省略时从已落库画像 `match_constraints` 加载。"
        " L1 会丢掉 strength=soft，保留 hard（城市/行业/必须）。\n"
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
