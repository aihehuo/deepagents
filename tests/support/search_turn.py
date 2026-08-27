"""Fake-agent helpers: simulate a model-called search_candidates turn.

Search is a model tool. Orchestrator does not run match. Tests that need a
match result must put a tool call + ToolMessage in the agent transcript.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from apps.group_agent_api.agent_factory.match_stub import MatchResult


def search_payload_from_match(
    result: MatchResult,
    *,
    query: str | None = None,
    rank_query: str | None = None,
) -> dict[str, Any]:
    q = query if query is not None else str(result.query or "")
    rq = rank_query if rank_query is not None else q
    return {
        "status": result.status,
        "reason": result.reason or "",
        "query": q,
        "rank_query": rq,
        "candidates": list(result.candidates or []),
    }


def search_tool_messages(
    *,
    status: str,
    reason: str = "",
    query: str = "test query",
    rank_query: str = "",
    candidates: list[dict[str, Any]] | None = None,
    tool_call_id: str = "search_candidates_1",
) -> list[Any]:
    """AIMessage tool_call + ToolMessage JSON, as the react loop would emit."""
    rq = rank_query or query
    payload = {
        "status": status,
        "reason": reason,
        "query": query,
        "rank_query": rq,
        "candidates": list(candidates or []),
    }
    return [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_candidates",
                    "args": {"query": query, "rank_query": rq},
                    "id": tool_call_id,
                }
            ],
        ),
        ToolMessage(
            content=json.dumps(payload, ensure_ascii=False),
            name="search_candidates",
            tool_call_id=tool_call_id,
        ),
    ]


def search_tool_messages_from_match(
    result: MatchResult,
    *,
    query: str | None = None,
    rank_query: str | None = None,
    tool_call_id: str = "search_candidates_1",
) -> list[Any]:
    payload = search_payload_from_match(result, query=query, rank_query=rank_query)
    return search_tool_messages(
        status=str(payload["status"]),
        reason=str(payload["reason"]),
        query=str(payload["query"]),
        rank_query=str(payload["rank_query"]),
        candidates=list(payload["candidates"]),
        tool_call_id=tool_call_id,
    )
