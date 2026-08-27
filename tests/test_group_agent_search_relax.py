"""Brief C · mod.brain.search_relax — off ≡ single-shot; on ≥2 real tool levels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from apps.group_agent_api.agent_factory.match_stub import MatchResult
from apps.group_agent_api.agent_factory.module_config import (
    load_modules_config,
    reload_modules_config,
    reset_modules_config_cache,
)
from apps.group_agent_api.agent_factory.search_relax import (
    MODULE_ID,
    STRATEGY_DOC,
    apply_relax,
    resolve_search_relax,
    search_relax_enabled,
    search_relax_max_levels,
    search_relax_system_addon,
)
from apps.group_agent_api.agent_factory.search_tool import search_candidates


def _write_modules_yaml(
    path: Path,
    *,
    search_relax: bool,
    max_levels: int = 2,
) -> Path:
    text = f"""version: 1
preset: current
modules:
  mod.brain.reply_grounding: true
  {MODULE_ID}: {'true' if search_relax else 'false'}
search_relax:
  max_levels: {max_levels}
"""
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _reset_modules_config() -> Any:
    reset_modules_config_cache()
    yield
    reset_modules_config_cache()


def test_default_yaml_search_relax_off() -> None:
    cfg = load_modules_config()
    assert cfg.is_module_enabled(MODULE_ID) is False
    assert cfg.search_relax_max_levels() == 2
    assert search_relax_enabled() is False
    assert search_relax_system_addon() == ""
    assert "L0 hard" in STRATEGY_DOC


def test_off_path_forces_l0_identity(tmp_path: Path) -> None:
    yaml_path = _write_modules_yaml(tmp_path / "off.yaml", search_relax=False)
    reload_modules_config(yaml_path)

    long_rq = "需要找懂 langchain 的合伙人；比如做过 RAG 落地的人"
    resolved = resolve_search_relax(
        query="langchain 合伙人",
        rank_query=long_rq,
        relax_level=1,
        pool="agent_profiles",
        constraints=[
            {"field": "city", "strength": "hard", "values": ["上海"]},
            {"field": "stack", "strength": "soft", "values": ["langchain"]},
        ],
    )
    assert resolved.enabled is False
    assert resolved.args.relax_level == 0
    assert resolved.args.rank_query == long_rq
    assert resolved.args.strategy_note == "L0_hard"
    assert resolved.clamped is True  # raw L1 ignored when off
    # Soft constraints untouched when module off
    soft = [c for c in (resolved.args.constraints or []) if c.get("strength") == "soft"]
    assert len(soft) == 1


def test_off_path_search_candidates_equiv_single_shot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    yaml_path = _write_modules_yaml(tmp_path / "off_tool.yaml", search_relax=False)
    reload_modules_config(yaml_path)

    captured: list[dict[str, Any]] = []

    def _fake_run_match(**kwargs):
        captured.append(dict(kwargs))
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
    long_rq = "需要找懂 langchain 的合伙人；比如做过 RAG 落地的人"
    raw = search_candidates.invoke(
        {
            "query": "langchain 合伙人",
            "rank_query": long_rq,
            "relax_level": 1,
            "pool": "agent_profiles",
        },
        config={"metadata": {"user_id": "u1", "group_id": "g1", "run_match": "true"}},
    )
    payload = json.loads(raw)
    assert payload["status"] == "empty"
    assert payload["relax_level"] == 0
    assert payload["search_relax_enabled"] is False
    assert payload["strategy"] == "L0_hard"
    assert payload["rank_query"] == long_rq
    assert captured[0]["relax_level"] == 0
    assert captured[0]["rank_query"] == long_rq


def test_on_path_l1_drops_soft_facets(tmp_path: Path) -> None:
    yaml_path = _write_modules_yaml(
        tmp_path / "on.yaml", search_relax=True, max_levels=2
    )
    reload_modules_config(yaml_path)
    assert search_relax_enabled() is True
    assert search_relax_max_levels() == 2
    addon = search_relax_system_addon()
    assert "mod.brain.search_relax" in addon
    assert "D-B03" in addon

    long_rq = "需要找懂 langchain 的合伙人；比如做过 RAG 落地的人"
    resolved = resolve_search_relax(
        query="langchain 合伙人",
        rank_query=long_rq,
        relax_level=1,
        constraints=[
            {"field": "city", "strength": "hard", "values": ["上海"]},
            {"field": "stack", "strength": "soft", "values": ["langchain"]},
        ],
    )
    assert resolved.enabled is True
    assert resolved.args.relax_level == 1
    assert resolved.args.strategy_note == "L1_drop_soft_facets"
    assert resolved.args.dropped_soft is True
    assert resolved.args.rank_query == "需要找懂 langchain 的合伙人"
    strengths = {c["field"]: c["strength"] for c in resolved.args.constraints}
    assert strengths["city"] == "hard"
    assert "stack" not in strengths


def test_on_path_two_real_tool_levels_fake_hand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D-B03: two levels = two explicit search_candidates invokes (no orchestrator)."""
    yaml_path = _write_modules_yaml(
        tmp_path / "on_levels.yaml", search_relax=True, max_levels=2
    )
    reload_modules_config(yaml_path)

    calls: list[dict[str, Any]] = []

    def _fake_run_match(**kwargs):
        calls.append(dict(kwargs))
        level = int(kwargs.get("relax_level") or 0)
        if level == 0:
            return MatchResult(
                status="empty",
                candidates=[],
                query=kwargs["query"],
                group_id=kwargs["group_id"],
                reason="empty_l0",
            )
        return MatchResult(
            status="matched",
            candidates=[
                {
                    "user_id": "c_relaxed",
                    "group_id": kwargs["group_id"],
                    "doing": {
                        "value": "做教育产品",
                        "disclosure": "confirmed_public",
                    },
                }
            ],
            query=kwargs["query"],
            group_id=kwargs["group_id"],
            reason="ok_l1",
        )

    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.search_tool.match_backend.run_match",
        _fake_run_match,
    )

    meta = {"metadata": {"user_id": "u2", "group_id": "g1", "run_match": "true"}}
    long_rq = "AI 教育需要产品合伙人；比如做过自适应学习引擎"

    # Level 0 — model tool call #1
    p0 = json.loads(
        search_candidates.invoke(
            {
                "query": "AI 教育 产品",
                "rank_query": long_rq,
                "relax_level": 0,
            },
            config=meta,
        )
    )
    assert p0["status"] == "empty"
    assert p0["relax_level"] == 0
    assert p0["strategy"] == "L0_hard"
    assert p0["search_relax_enabled"] is True

    # Level 1 — model tool call #2 (explicit; not orchestrator post-LLM)
    p1 = json.loads(
        search_candidates.invoke(
            {
                "query": "AI 教育 产品",
                "rank_query": long_rq,
                "relax_level": 1,
            },
            config=meta,
        )
    )
    assert p1["status"] == "matched"
    assert p1["relax_level"] == 1
    assert p1["strategy"] == "L1_drop_soft_facets"
    assert p1["candidates"][0]["user_id"] == "c_relaxed"
    assert p1["rank_query"] != long_rq  # shortened

    assert len(calls) == 2
    assert calls[0]["relax_level"] == 0
    assert calls[1]["relax_level"] == 1
    assert calls[1]["rank_query"] == apply_relax(
        level=1, query="AI 教育 产品", rank_query=long_rq
    ).rank_query


def test_profile_pool_hook_noted_not_mounted(tmp_path: Path) -> None:
    yaml_path = _write_modules_yaml(tmp_path / "pool.yaml", search_relax=True)
    reload_modules_config(yaml_path)
    resolved = resolve_search_relax(
        query="x",
        rank_query="x",
        relax_level=0,
        pool="agent_profiles",
    )
    assert resolved.profile_pool_hook is True
    assert resolved.args.pool == "all_reachable"
