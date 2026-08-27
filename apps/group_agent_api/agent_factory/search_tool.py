"""Model-callable search tool (hand.search_exec). Search is not an orchestrator step."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from apps.group_agent_api.agent_factory.integrations.group_bind import (
    align_match_to_trusted_group,
)
from apps.group_agent_api.agent_factory.integrations import match_backend
from apps.group_agent_api.agent_factory.revisit import excluded_ids_for_match
from apps.group_agent_api.agent_factory.search_relax import resolve_search_relax

_logger = logging.getLogger("uvicorn.error")

SEARCH_TOOL_NAME = "search_candidates"
_CONSTRAINT_VERSION = "ga-constraint-v1"


def _meta_flag(metadata: dict[str, Any], key: str, default: bool = True) -> bool:
    raw = metadata.get(key, default)
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in {"", "none"}:
        return default
    return text not in {"0", "false", "no", "off"}


@dataclass
class SearchTurnResult:
    called: bool = False
    query: str = ""
    rank_query: str = ""
    status: str = "skipped"
    reason: str = "model_did_not_search"
    candidates: list[dict[str, Any]] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)
    relax_level: int = 0
    pool: str = ""


def extract_search_this_turn(messages: list[Any], start: int) -> SearchTurnResult:
    """Last ``search_candidates`` tool call + result in this agent turn."""
    result = SearchTurnResult()
    start = max(0, int(start or 0))
    last_args: dict[str, Any] = {}
    last_payload: dict[str, Any] = {}
    called = False

    for message in messages[start:]:
        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls and hasattr(message, "additional_kwargs"):
            tool_calls = (message.additional_kwargs or {}).get("tool_calls")
        if not tool_calls:
            continue
        for tc in tool_calls:
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            if name != SEARCH_TOOL_NAME:
                continue
            called = True
            args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", None)
            if isinstance(args, dict):
                last_args = args

    for message in messages[start:]:
        if not isinstance(message, ToolMessage):
            continue
        name = str(getattr(message, "name", "") or "")
        if name and name != SEARCH_TOOL_NAME:
            continue
        content = str(getattr(message, "content", "") or "").strip()
        if not content:
            continue
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            if name == SEARCH_TOOL_NAME:
                last_payload = {"status": "error", "reason": "invalid_tool_result", "candidates": []}
                called = True
            continue
        if not isinstance(parsed, dict):
            continue
        if name == SEARCH_TOOL_NAME or parsed.get("status") in {
            "matched",
            "empty",
            "weak",
            "skipped",
            "rejected",
            "error",
        }:
            last_payload = parsed
            called = True

    if not called:
        return result

    result.called = True
    result.query = str(last_args.get("query") or last_payload.get("query") or "")
    result.rank_query = str(
        last_args.get("rank_query") or last_payload.get("rank_query") or result.query
    )
    result.status = str(last_payload.get("status") or "error")
    result.reason = str(last_payload.get("reason") or "")
    cands = last_payload.get("candidates") or []
    result.candidates = [c for c in cands if isinstance(c, dict)]
    result.payload = dict(last_payload)
    result.payload["query"] = result.query
    result.payload["rank_query"] = result.rank_query
    try:
        result.relax_level = int(
            last_payload.get("relax_level", last_args.get("relax_level", 0)) or 0
        )
    except (TypeError, ValueError):
        result.relax_level = 0
    result.pool = str(last_payload.get("pool") or last_args.get("pool") or "")
    return result


def _constraints_nonempty(raw: Any) -> bool:
    if raw is None or raw == "" or raw is False:
        return False
    if isinstance(raw, list):
        return len(raw) > 0
    if isinstance(raw, dict):
        items = raw.get("items")
        if isinstance(items, list):
            return len(items) > 0
        return len(raw) > 0
    return True


def _load_match_constraints_from_profile(
    metadata: dict[str, Any],
) -> list[Any] | None:
    """Load saved profile.match_constraints for this user×group (local store)."""
    user_id = str(metadata.get("user_id") or "").strip()
    group_id = str(metadata.get("group_id") or "").strip()
    base_dir_raw = metadata.get("base_dir")
    if not user_id or not group_id or not base_dir_raw:
        return None
    try:
        from apps.group_agent_api.agent_factory.profile_store import load_profile

        profile = load_profile(Path(str(base_dir_raw)), user_id, group_id)
    except Exception:  # noqa: BLE001 — missing/invalid path → no autoload
        return None
    if profile is None:
        return None
    mc = getattr(profile, "match_constraints", None) or []
    if not isinstance(mc, list) or not mc:
        return None
    return list(mc)


def resolve_search_constraints(
    tool_constraints: Any,
    metadata: dict[str, Any],
) -> tuple[dict[str, Any] | list[Any] | None, str]:
    """Prefer explicit tool arg; else autoload profile match_constraints."""
    if _constraints_nonempty(tool_constraints):
        return tool_constraints, "tool_arg"
    loaded = _load_match_constraints_from_profile(metadata)
    if loaded:
        return loaded, "profile"
    return None, "none"


def normalize_constraints_for_hand(
    constraints: dict[str, Any] | list[Any] | None,
) -> dict[str, Any] | None:
    """Wrap list MatchConstraintV1 items into ga-constraint-v1 envelope for hand."""
    if constraints is None:
        return None
    if isinstance(constraints, list):
        if not constraints:
            return None
        return {"version": _CONSTRAINT_VERSION, "items": list(constraints)}
    if isinstance(constraints, dict):
        if "items" in constraints:
            items = constraints.get("items") or []
            if not isinstance(items, list) or not items:
                return None
            out = dict(constraints)
            out.setdefault("version", _CONSTRAINT_VERSION)
            out["items"] = list(items)
            return out
        return constraints if constraints else None
    return None


@tool(parse_docstring=True)
def search_candidates(
    query: str,
    rank_query: str = "",
    relax_level: int = 0,
    pool: str = "",
    constraints: list[dict[str, Any]] | dict[str, Any] | None = None,
    *,
    config: RunnableConfig,
) -> str:
    """在当前群搜索可匹配的候选人。用户要求匹配、搜人、推荐或 @ 时必须调用。

    Args:
        query: 你根据本轮对话和画像组织的检索词（必填）。不要留空。
        rank_query: 可选的细排序文本；默认与 query 相同。
        relax_level: 放宽级别（0=硬约束）。仅当 mod.brain.search_relax 开启时生效；
            empty 后可再调本工具并提高级别。编排不会代搜。
        pool: 候选池。省略时：mod.brain.profile_pool 开 → agent_profiles，关 →
            all_reachable。模型显式传入优先；关模块时 agent_profiles 回退全池。
        constraints: 可选匹配约束（与 save_group_profile.match_constraints 同形）。
            省略时从本用户×群已落库画像自动加载。L1 会丢掉 soft、保留 hard。
    """
    metadata = config.get("metadata") or {}
    user_id = str(metadata.get("user_id") or "").strip()
    group_id = str(metadata.get("group_id") or "").strip()
    q = str(query or "").strip()
    rq = str(rank_query or "").strip() or q

    if not q:
        return json.dumps(
            {
                "status": "rejected",
                "reason": "empty_query",
                "query": "",
                "rank_query": "",
                "relax_level": 0,
                "pool": "",
                "candidates": [],
            },
            ensure_ascii=False,
        )
    if not user_id or not group_id:
        return json.dumps(
            {
                "status": "error",
                "reason": "missing_user_or_group",
                "query": q,
                "rank_query": rq,
                "relax_level": 0,
                "pool": "",
                "candidates": [],
            },
            ensure_ascii=False,
        )
    if not _meta_flag(metadata, "run_match", True):
        return json.dumps(
            {
                "status": "skipped",
                "reason": "run_match_disabled",
                "query": q,
                "rank_query": rq,
                "relax_level": 0,
                "pool": "",
                "candidates": [],
            },
            ensure_ascii=False,
        )

    resolved_constraints, constraints_source = resolve_search_constraints(
        constraints, metadata
    )
    resolved = resolve_search_relax(
        query=q,
        rank_query=rq,
        relax_level=relax_level,
        pool=pool,
        constraints=resolved_constraints,
    )
    effective = resolved.args
    hand_constraints = normalize_constraints_for_hand(effective.constraints)

    extra_meta = {
        k: v
        for k, v in metadata.items()
        if k not in {"user_token", "group_token", "base_dir"}
    }
    result = match_backend.run_match(
        query=effective.query,
        group_id=group_id,
        excluded_ids=excluded_ids_for_match(user_id, extra_meta),
        group_token=str(metadata.get("group_token") or "") or None,
        user_bearer=str(metadata.get("user_token") or "") or None,
        rank_query=effective.rank_query,
        constraints=hand_constraints,
        relax_level=effective.relax_level,
        pool=effective.pool,
    )
    aligned = align_match_to_trusted_group(result, trusted_group_id=group_id)
    payload = {
        "status": aligned.status,
        "reason": aligned.reason or "",
        "query": aligned.query or effective.query,
        "rank_query": effective.rank_query,
        "relax_level": effective.relax_level,
        "pool": effective.pool,
        "strategy": effective.strategy_note,
        "search_relax_enabled": resolved.enabled,
        "profile_pool_enabled": resolved.profile_pool_enabled,
        "pool_source": resolved.pool_source,
        "constraints_source": constraints_source,
        "dropped_soft": effective.dropped_soft,
        "candidates": list(aligned.candidates or []),
    }
    if resolved.profile_pool_hook:
        payload["profile_pool_hook"] = "noted_not_mounted"
    _logger.info(
        "action=search_candidates_tool user_id=%s group_id=%s status=%s "
        "query_len=%d relax_level=%s pool=%s pool_source=%s "
        "search_relax=%s profile_pool=%s strategy=%s "
        "constraints_source=%s dropped_soft=%s",
        user_id,
        group_id,
        payload["status"],
        len(effective.query),
        effective.relax_level,
        effective.pool,
        resolved.pool_source,
        resolved.enabled,
        resolved.profile_pool_enabled,
        effective.strategy_note,
        constraints_source,
        effective.dropped_soft,
    )
    return json.dumps(payload, ensure_ascii=False)
