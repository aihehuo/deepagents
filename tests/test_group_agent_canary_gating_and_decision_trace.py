"""Unit tests for user ID 1 / canary gating & decision point tracing."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from apps.group_agent_api.agent_factory.debug_trace import (
    debug_trace_enabled,
    end_decision_trace,
    get_current_decision_trace,
    record_decision_point,
    start_decision_trace,
    write_turn_trace,
)
from apps.group_agent_api.agent_factory.integrations.config import (
    is_canary_user,
    profile_pool_enabled_for_user,
    search_relax_enabled_for_user,
    v2_canary_enabled,
    v2_enabled_for_user,
    v2_user_allowlist,
)
from apps.group_agent_api.agent_factory.module_config import (
    load_modules_config,
    reload_modules_config,
    reset_modules_config_cache,
)
from apps.group_agent_api.agent_factory.search_relax import (
    resolve_pool,
    resolve_search_relax,
    search_relax_system_addon,
)
from apps.group_agent_api.agent_factory.search_tool import search_candidates


@pytest.fixture(autouse=True)
def _reset_config() -> Any:
    reset_modules_config_cache()
    yield
    reset_modules_config_cache()


def _write_yaml(path: Path, *, search_relax: bool = False, profile_pool: bool = False) -> Path:
    text = f"""version: 1
preset: current
modules:
  mod.brain.search_relax: {'true' if search_relax else 'false'}
  mod.brain.profile_pool: {'true' if profile_pool else 'false'}
  mod.brain.reply_grounding: true
"""
    path.write_text(text, encoding="utf-8")
    return path


def test_canary_user_checking_basic(monkeypatch: pytest.MonkeyPatch) -> None:
    # 1. Canary disabled
    monkeypatch.delenv("GROUP_AGENT_V2_CANARY", raising=False)
    monkeypatch.delenv("GROUP_AGENT_V2_CANARY_ENABLED", raising=False)
    monkeypatch.delenv("GROUP_AGENT_V2_USER_ALLOWLIST", raising=False)
    assert is_canary_user(1) is False
    assert is_canary_user("1") is False

    # 2. Canary enabled without explicit allowlist (default allows user 1)
    monkeypatch.setenv("GROUP_AGENT_V2_CANARY", "1")
    assert is_canary_user(1) is True
    assert is_canary_user("1") is True
    assert is_canary_user("  1  ") is True
    assert is_canary_user(2) is False
    assert is_canary_user("user_1") is False
    assert is_canary_user(None) is False
    assert is_canary_user(-1) is False

    # 3. Explicit allowlist
    monkeypatch.setenv("GROUP_AGENT_V2_USER_ALLOWLIST", "1,42,100")
    assert is_canary_user(1) is True
    assert is_canary_user(42) is True
    assert is_canary_user(100) is True
    assert is_canary_user(2) is False

    # 4. Force off override
    monkeypatch.setenv("GROUP_AGENT_V2_FORCE_OFF", "1")
    assert is_canary_user(1) is False
    assert is_canary_user(42) is False


def test_search_relax_and_profile_pool_canary_gating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    yaml_path = _write_yaml(tmp_path / "off.yaml", search_relax=False, profile_pool=False)
    reload_modules_config(yaml_path)

    # Globally off
    assert search_relax_enabled_for_user(None) is False
    assert search_relax_enabled_for_user(1) is False
    assert profile_pool_enabled_for_user(1) is False

    # Enable canary for user 1
    monkeypatch.setenv("GROUP_AGENT_V2_CANARY", "1")
    assert search_relax_enabled_for_user(1) is True
    assert search_relax_enabled_for_user("1") is True
    assert profile_pool_enabled_for_user(1) is True
    assert profile_pool_enabled_for_user("1") is True

    # User 2 is not enabled
    assert search_relax_enabled_for_user(2) is False
    assert profile_pool_enabled_for_user(2) is False

    # resolve_search_relax for user 1 (canary) at L0 defaults to agent_profiles, at L1 expands to all_reachable
    resolved_u1_l0 = resolve_search_relax(
        query="AI 架构师",
        rank_query="AI 架构师 上海",
        relax_level=0,
        pool="",
        user_id=1,
    )
    assert resolved_u1_l0.enabled is True
    assert resolved_u1_l0.profile_pool_enabled is True
    assert resolved_u1_l0.args.pool == "agent_profiles"
    assert resolved_u1_l0.args.relax_level == 0

    resolved_u1 = resolve_search_relax(
        query="AI 架构师",
        rank_query="AI 架构师 上海",
        relax_level=1,
        pool="",
        user_id=1,
    )
    assert resolved_u1.enabled is True
    assert resolved_u1.profile_pool_enabled is True
    assert resolved_u1.args.pool == "all_reachable"
    assert resolved_u1.pool_source == "relaxed_expansion"
    assert resolved_u1.args.relax_level == 1
    assert resolved_u1.args.strategy_note == "L1_drop_soft_and_expand_pool"

    # resolve_search_relax for user 2 gets enabled=False and pool all_reachable
    resolved_u2 = resolve_search_relax(
        query="AI 架构师",
        rank_query="AI 架构师 上海",
        relax_level=1,
        pool="",
        user_id=2,
    )
    assert resolved_u2.enabled is False
    assert resolved_u2.profile_pool_enabled is False
    assert resolved_u2.args.pool == "all_reachable"
    assert resolved_u2.args.relax_level == 0


def test_search_candidates_tool_with_user_canary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    yaml_path = _write_yaml(tmp_path / "off.yaml", search_relax=False, profile_pool=False)
    reload_modules_config(yaml_path)

    monkeypatch.setenv("GROUP_AGENT_V2_CANARY", "1")

    mock_match = MagicMock()
    mock_match.status = "matched"
    mock_match.reason = "found_candidates"
    mock_match.query = "AI 架构师"
    mock_match.group_id = "group_999"
    mock_match.candidates = [
        {
            "user_id": "10",
            "name": "Alice",
            "score": 0.88,
            "source_group_id": "group_999",
            "doing": {"value": "LLM 架构设计", "disclosure": "confirmed_public"},
        },
        {
            "user_id": "11",
            "name": "Bob",
            "score": 0.72,
            "source_group_id": "group_999",
            "doing": {"value": "Agent 落地", "disclosure": "confirmed_public"},
        },
    ]

    with patch(
        "apps.group_agent_api.agent_factory.search_tool.match_backend.run_match",
        return_value=mock_match,
    ) as run_mock:
        config = {
            "metadata": {
                "user_id": "1",
                "group_id": "group_999",
                "run_id": "run_test_canary_1",
            }
        }
        res_json = search_candidates.invoke(
            {
                "query": "AI 架构师",
                "relax_level": 1,
                "pool": "",
                "constraints": [
                    {"field": "city", "operator": "in", "values": ["上海"], "strength": "hard"},
                    {"field": "tags", "operator": "in", "values": ["LLM"], "strength": "soft"},
                ],
            },
            config=config,
        )
        data = json.loads(res_json)
        assert data["search_relax_enabled"] is True
        assert data["profile_pool_enabled"] is True
        assert data["relax_level"] == 1
        assert data["pool"] == "all_reachable"
        assert data["strategy"] == "L1_drop_soft_and_expand_pool"
        assert data["dropped_soft"] == 1
        assert len(data["candidates"]) == 2


def test_decision_point_recording_and_trace_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    monkeypatch.setenv("GROUP_AGENT_DEBUG_TRACE", "1")

    buf, token = start_decision_trace()
    try:
        record_decision_point(
            phase="intent_route",
            detail={"route": "search", "reply_mode": "recommendation", "secret_token": "secret123"},
            run_id="run_abc",
            thread_id="tid_xyz",
        )
        record_decision_point(
            phase="constraint_extraction",
            detail={
                "source": "tool_arg",
                "hard_constraints": [{"field": "city", "values": ["北京"]}],
                "soft_constraints": [],
            },
            run_id="run_abc",
            thread_id="tid_xyz",
        )
        record_decision_point(
            phase="tool_call_search_candidates",
            detail={
                "pool": "agent_profiles",
                "relax_level": 0,
                "candidates_count": 2,
                "match_score_distribution": {"min": 0.8, "max": 0.9, "avg": 0.85},
            },
            run_id="run_abc",
            thread_id="tid_xyz",
        )
        record_decision_point(
            phase="search_relaxation",
            detail={"trigger": "zero_or_weak_candidates", "relax_level": 1, "status": "empty"},
            run_id="run_abc",
            thread_id="tid_xyz",
        )
        record_decision_point(
            phase="reply_grounding",
            detail={
                "initial_draft": "推荐张三",
                "passed": True,
                "verdict": "pass",
                "rewrite_attempts": 0,
                "final_text": "推荐张三",
            },
            run_id="run_abc",
            thread_id="tid_xyz",
        )
        record_decision_point(
            phase="ingress_mouth",
            detail={"status": "delivered", "attempts": 1, "reply_mode": "recommendation"},
            run_id="run_abc",
            thread_id="tid_xyz",
        )

        current_trace = get_current_decision_trace()
        assert len(current_trace) == 6
        phases = [p["phase"] for p in current_trace]
        assert phases == [
            "intent_route",
            "constraint_extraction",
            "tool_call_search_candidates",
            "search_relaxation",
            "reply_grounding",
            "ingress_mouth",
        ]

        # Verify secret redaction
        assert current_trace[0]["detail"]["secret_token"] == "[redacted]"

        # Check log line format
        assert "DECISION_POINT run_id=run_abc turn=- phase=intent_route" in caplog.text

        # Write turn trace file
        trace_file = write_turn_trace(
            base_dir=tmp_path,
            run_id="run_abc",
            thread_id="tid_xyz",
            user_id="1",
            group_id="group_1",
            conversation_id="conv_1",
            episode_id="ep_1",
            user_message="找个架构师",
            messages=[],
            msg_count_before=0,
            reply="为您找到以下候选人",
            match_status="matched",
        )
        assert trace_file is not None
        assert Path(trace_file).is_file()

        content = json.loads(Path(trace_file).read_text(encoding="utf-8"))
        assert content["run_id"] == "run_abc"
        assert len(content["decision_points"]) == 6
        assert content["decision_points"][0]["phase"] == "intent_route"
        assert content["decision_points"][0]["detail"]["route"] == "search"
    finally:
        end_decision_trace(token)
