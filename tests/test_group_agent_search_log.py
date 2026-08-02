# frozen_string_literal: true
"""Tests for SearchLogEntry generation in ChatResponse and async callback payloads."""

from __future__ import annotations

import pytest
from apps.group_agent_api.app.models import SearchLogEntry, ChatResponse, MatchResponse


def test_search_log_entry_model():
    entry = SearchLogEntry(
        search_id="search_12345",
        timestamp="2026-08-02T21:35:00Z",
        query="懂教育 教研 教培经验",
        rank_query="需要懂教育的合伙人",
        match_status="matched",
        match_reason="matched_3",
        candidate_count=3,
        candidate_names=["王*", "史*路", "王*君"]
    )
    assert entry.search_id == "search_12345"
    assert entry.query == "懂教育 教研 教培经验"
    assert entry.candidate_count == 3
    assert len(entry.candidate_names) == 3


def test_chat_response_includes_search_log():
    entry = SearchLogEntry(
        search_id="search_999",
        timestamp="2026-08-02T21:35:00Z",
        query="全栈 工程师",
        match_status="matched",
        candidate_count=1,
        candidate_names=["徐*"]
    )
    res = ChatResponse(
        user_id="101",
        group_id="1000",
        conversation_id="conv_1",
        thread_id="th_1",
        reply="已为您找到匹配人选",
        search_log=entry,
    )
    assert res.search_log is not None
    assert res.search_log.query == "全栈 工程师"
    assert res.search_log.candidate_count == 1
