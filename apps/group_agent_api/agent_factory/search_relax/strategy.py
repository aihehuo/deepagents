"""Multi-level search strategy (TSD-14 §8.12).

D-B03: every level is a model-driven ``search_candidates`` tool call.
This module only transforms args for the level the model requested — it never
triggers a follow-up search from the orchestrator.

Documented ladder (default ``max_levels=2`` → L0 + L1)::

    L0 hard  — keep query / rank_query / constraints as supplied;
               pool defaults to ``all_reachable``.
    L1 soft  — drop soft facets: strip constraints with strength=soft;
               shorten rank_query to first clause (≤40 chars).
               P0 hard (city/industry) must remain if the model still sends them.
    L2+      — reserved; clamped by ``search_relax.max_levels``.

``pool=agent_profiles`` is preferred when ``mod.brain.profile_pool`` is on
(see ``resolve_pool`` precedence in ``search_relax.module``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from apps.group_agent_api.agent_factory.search_relax.ids import DEFAULT_POOL

# Human-readable strategy for prompts / docs (keep in sync with apply_relax).
STRATEGY_DOC = (
    "L0 hard → keep query/rank_query/constraints; "
    "L1 → drop soft facets (soft constraints + shorten rank_query); "
    "each level requires a real search_candidates tool call (D-B03)."
)

_CLAUSE_SPLIT = re.compile(r"[；;。\n]|当前卡点|比如")


@dataclass(frozen=True)
class RelaxedSearchArgs:
    query: str
    rank_query: str
    constraints: dict[str, Any] | list[Any] | None
    pool: str
    relax_level: int
    dropped_soft: bool
    strategy_note: str


def apply_relax(
    *,
    level: int,
    query: str,
    rank_query: str,
    constraints: dict[str, Any] | list[Any] | None = None,
    pool: str | None = None,
) -> RelaxedSearchArgs:
    """Apply in-tool transforms for the requested ``relax_level``.

    Level 0 is identity. Level ≥1 drops soft facets. Does not invent a new
    search — caller must already be inside a tool invocation.
    """
    q = str(query or "").strip()
    rq = str(rank_query or "").strip() or q
    resolved_pool = (pool or "").strip() or DEFAULT_POOL
    lvl = max(0, int(level))

    if lvl <= 0:
        return RelaxedSearchArgs(
            query=q,
            rank_query=rq,
            constraints=_copy_constraints(constraints),
            pool=resolved_pool,
            relax_level=0,
            dropped_soft=False,
            strategy_note="L0_hard",
        )

    cleaned, dropped = drop_soft_constraints(constraints)
    short_rq = shorten_rank_query(rq)
    return RelaxedSearchArgs(
        query=q,
        rank_query=short_rq,
        constraints=cleaned,
        pool=resolved_pool,
        relax_level=lvl,
        dropped_soft=dropped or (short_rq != rq),
        strategy_note="L1_drop_soft_facets",
    )


def drop_soft_constraints(
    constraints: dict[str, Any] | list[Any] | None,
) -> tuple[dict[str, Any] | list[Any] | None, bool]:
    """Remove soft-strength facets; keep hard / unspecified."""
    if constraints is None:
        return None, False
    if isinstance(constraints, list):
        kept: list[Any] = []
        dropped = False
        for item in constraints:
            if isinstance(item, dict):
                strength = str(item.get("strength") or "").strip().lower()
                if strength == "soft":
                    dropped = True
                    continue
            kept.append(item)
        return kept, dropped
    if isinstance(constraints, dict):
        # Hand envelope: {"version": "ga-constraint-v1", "items": [...]}
        items = constraints.get("items")
        if isinstance(items, list):
            kept_items, dropped = drop_soft_constraints(items)
            out_env = dict(constraints)
            out_env["items"] = kept_items if isinstance(kept_items, list) else []
            return out_env, dropped
        soft = constraints.get("soft")
        hard = constraints.get("hard")
        if soft is not None or hard is not None:
            out = {k: v for k, v in constraints.items() if k != "soft"}
            return out, soft is not None
        out_map: dict[str, Any] = {}
        dropped = False
        for key, value in constraints.items():
            if isinstance(value, dict) and str(value.get("strength") or "").lower() == "soft":
                dropped = True
                continue
            out_map[key] = value
        return out_map, dropped
    return constraints, False


def shorten_rank_query(rank_query: str, *, max_chars: int = 40) -> str:
    """Drop soft/long-tail facets from rank_query (first clause, capped)."""
    text = str(rank_query or "").strip()
    if not text:
        return ""
    parts = _CLAUSE_SPLIT.split(text, maxsplit=1)
    head = (parts[0] if parts else text).strip() or text
    if len(head) > max_chars:
        return head[:max_chars].rstrip()
    return head


def _copy_constraints(
    constraints: dict[str, Any] | list[Any] | None,
) -> dict[str, Any] | list[Any] | None:
    if constraints is None:
        return None
    if isinstance(constraints, list):
        return list(constraints)
    if isinstance(constraints, dict):
        return dict(constraints)
    return constraints
