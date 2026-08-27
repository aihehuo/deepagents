"""Model-callable search tool (hand.search_exec). Search is not an orchestrator step."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from apps.group_agent_api.agent_factory.integrations.group_bind import (
    align_match_to_trusted_group,
)
from apps.group_agent_api.agent_factory.integrations import match_backend
from apps.group_agent_api.agent_factory.revisit import excluded_ids_for_match

_logger = logging.getLogger("uvicorn.error")

SEARCH_TOOL_NAME = "search_candidates"


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
    return result


@tool(parse_docstring=True)
def search_candidates(
    query: str,
    rank_query: str = "",
    *,
    config: RunnableConfig,
) -> str:
    """在当前群搜索可匹配的候选人。用户要求匹配、搜人、推荐或 @ 时必须调用。

    Args:
        query: 你根据本轮对话和画像组织的检索词（必填）。不要留空。
        rank_query: 可选的细排序文本；默认与 query 相同。
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
                "candidates": [],
            },
            ensure_ascii=False,
        )

    extra_meta = {
        k: v
        for k, v in metadata.items()
        if k not in {"user_token", "group_token", "base_dir"}
    }
    result = match_backend.run_match(
        query=q,
        group_id=group_id,
        excluded_ids=excluded_ids_for_match(user_id, extra_meta),
        group_token=str(metadata.get("group_token") or "") or None,
        user_bearer=str(metadata.get("user_token") or "") or None,
        rank_query=rq,
    )
    aligned = align_match_to_trusted_group(result, trusted_group_id=group_id)
    payload = {
        "status": aligned.status,
        "reason": aligned.reason or "",
        "query": aligned.query or q,
        "rank_query": rq,
        "candidates": list(aligned.candidates or []),
    }
    _logger.info(
        "action=search_candidates_tool user_id=%s group_id=%s status=%s query_len=%d",
        user_id,
        group_id,
        payload["status"],
        len(q),
    )
    return json.dumps(payload, ensure_ascii=False)
