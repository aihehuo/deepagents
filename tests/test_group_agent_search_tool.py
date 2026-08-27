"""search_candidates is a model tool; orchestrator must not run match."""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from apps.group_agent_api.agent_factory.match_stub import MatchResult
from apps.group_agent_api.agent_factory.search_tool import (
    extract_search_this_turn,
    search_candidates,
)


def test_empty_query_is_rejected() -> None:
    raw = search_candidates.invoke(
        {"query": "  ", "rank_query": ""},
        config={"metadata": {"user_id": "u1", "group_id": "g1", "run_match": "true"}},
    )
    payload = json.loads(raw)
    assert payload["status"] == "rejected"
    assert payload["reason"] == "empty_query"


def test_run_match_disabled_skips_without_calling_backend(
    monkeypatch,
) -> None:
    def _boom(**_kwargs):
        raise AssertionError("run_match must not fire when disabled")

    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.search_tool.match_backend.run_match",
        _boom,
    )
    raw = search_candidates.invoke(
        {"query": "python", "rank_query": "python"},
        config={"metadata": {"user_id": "u1", "group_id": "g1", "run_match": "false"}},
    )
    payload = json.loads(raw)
    assert payload["status"] == "skipped"
    assert payload["reason"] == "run_match_disabled"


def test_extract_search_this_turn_reads_model_tool_args() -> None:
    messages = [
        HumanMessage(content="请匹配"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_candidates",
                    "args": {"query": "need python cofounder", "rank_query": "python"},
                    "id": "tc1",
                }
            ],
        ),
        ToolMessage(
            content=json.dumps(
                {
                    "status": "matched",
                    "reason": "ok",
                    "query": "need python cofounder",
                    "rank_query": "python",
                    "candidates": [{"user_id": "c1"}],
                },
                ensure_ascii=False,
            ),
            name="search_candidates",
            tool_call_id="tc1",
        ),
        AIMessage(content="找到一位。"),
    ]
    turn = extract_search_this_turn(messages, 0)
    assert turn.called is True
    assert turn.query == "need python cofounder"
    assert turn.rank_query == "python"
    assert turn.status == "matched"
    assert turn.candidates[0]["user_id"] == "c1"


def test_extract_search_this_turn_empty_when_model_did_not_call() -> None:
    messages = [
        HumanMessage(content="请匹配"),
        AIMessage(content="先聊聊你的需求。"),
    ]
    turn = extract_search_this_turn(messages, 0)
    assert turn.called is False
    assert turn.reason == "model_did_not_search"


def test_search_candidates_forwards_model_query(monkeypatch) -> None:
    captured: dict = {}

    def _fake_run_match(**kwargs):
        captured.update(kwargs)
        return MatchResult(
            status="empty",
            candidates=[],
            query=kwargs["query"],
            group_id=kwargs["group_id"],
            reason="empty_pool",
        )

    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.search_tool.match_backend.run_match",
        _fake_run_match,
    )
    raw = search_candidates.invoke(
        {"query": "need langchain python", "rank_query": "langgraph"},
        config={"metadata": {"user_id": "u105", "group_id": "g1", "run_match": "true"}},
    )
    payload = json.loads(raw)
    assert captured["query"] == "need langchain python"
    assert captured["rank_query"] == "langgraph"
    assert payload["status"] == "empty"
    assert payload["query"] == "need langchain python"
