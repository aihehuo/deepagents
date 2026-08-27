"""mod.brain.search_relax — multi-level search as a YAML Module (TSD-14 §8.12).

Off = today's single-shot search. On = model may call ``search_candidates``
again with ``relax_level`` / optional ``pool``. Orchestrator never auto-searches
(D-B03). ``mod.brain.profile_pool`` gates default ``pool=agent_profiles``.
"""

from __future__ import annotations

from apps.group_agent_api.agent_factory.search_relax.ids import (
    DEFAULT_POOL,
    MODULE_ID,
    PROFILE_POOL,
    PROFILE_POOL_MODULE_ID,
    PROTOCOL_NAME,
)
from apps.group_agent_api.agent_factory.search_relax.module import (
    ResolvedSearchRelax,
    profile_pool_enabled,
    resolve_pool,
    resolve_search_relax,
    search_relax_enabled,
    search_relax_max_levels,
    search_relax_system_addon,
)
from apps.group_agent_api.agent_factory.search_relax.strategy import (
    STRATEGY_DOC,
    RelaxedSearchArgs,
    apply_relax,
    drop_soft_constraints,
    shorten_rank_query,
)

__all__ = [
    "DEFAULT_POOL",
    "MODULE_ID",
    "PROFILE_POOL",
    "PROFILE_POOL_MODULE_ID",
    "PROTOCOL_NAME",
    "STRATEGY_DOC",
    "RelaxedSearchArgs",
    "ResolvedSearchRelax",
    "apply_relax",
    "drop_soft_constraints",
    "profile_pool_enabled",
    "resolve_pool",
    "resolve_search_relax",
    "search_relax_enabled",
    "search_relax_max_levels",
    "search_relax_system_addon",
    "shorten_rank_query",
]
