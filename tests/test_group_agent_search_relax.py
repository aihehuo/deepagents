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
    PROFILE_POOL,
    PROFILE_POOL_MODULE_ID,
    STRATEGY_DOC,
    apply_relax,
    profile_pool_enabled,
    resolve_pool,
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
    profile_pool: bool = False,
) -> Path:
    text = f"""version: 1
preset: current
modules:
  mod.brain.reply_grounding: true
  {MODULE_ID}: {'true' if search_relax else 'false'}
  mod.brain.profile_pool: {'true' if profile_pool else 'false'}
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
    assert cfg.is_module_enabled(PROFILE_POOL_MODULE_ID) is False
    assert cfg.profile_pool_enabled() is False
    assert cfg.search_relax_max_levels() == 2
    assert search_relax_enabled() is False
    assert profile_pool_enabled() is False
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
    assert "L0: 优先搜索群内有画像成员" in addon
    assert "L1: 若初次未找到或召回较少" in addon

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
    assert resolved.args.strategy_note == "L1_drop_soft_and_expand_pool"
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
    assert p1["strategy"] == "L1_drop_soft_and_expand_pool"
    assert p1["candidates"][0]["user_id"] == "c_relaxed"
    assert p1["rank_query"] != long_rq  # shortened

    assert len(calls) == 2
    assert calls[0]["relax_level"] == 0
    assert calls[1]["relax_level"] == 1
    assert calls[1]["rank_query"] == apply_relax(
        level=1, query="AI 教育 产品", rank_query=long_rq
    ).rank_query


def test_profile_pool_hook_noted_not_mounted(tmp_path: Path) -> None:
    yaml_path = _write_modules_yaml(
        tmp_path / "pool.yaml", search_relax=True, profile_pool=False
    )
    reload_modules_config(yaml_path)
    resolved = resolve_search_relax(
        query="x",
        rank_query="x",
        relax_level=0,
        pool="agent_profiles",
    )
    assert resolved.profile_pool_hook is True
    assert resolved.profile_pool_enabled is False
    assert resolved.args.pool == "all_reachable"
    assert resolved.pool_source == "fallback_module_off"


def test_profile_pool_on_defaults_agent_profiles(tmp_path: Path) -> None:
    yaml_path = _write_modules_yaml(
        tmp_path / "pp_on.yaml", search_relax=True, profile_pool=True
    )
    reload_modules_config(yaml_path)
    assert profile_pool_enabled() is True

    omitted = resolve_search_relax(query="x", rank_query="x", relax_level=0, pool="")
    assert omitted.profile_pool_hook is False
    assert omitted.profile_pool_enabled is True
    assert omitted.args.pool == PROFILE_POOL
    assert omitted.pool_source == "module_default"

    explicit = resolve_search_relax(
        query="x", rank_query="x", relax_level=0, pool="agent_profiles"
    )
    assert explicit.args.pool == PROFILE_POOL
    assert explicit.pool_source == "model"
    assert explicit.profile_pool_hook is False

    relaxed = resolve_search_relax(query="x", rank_query="x", relax_level=1, pool="")
    assert relaxed.args.pool == "all_reachable"
    assert relaxed.pool_source == "relaxed_expansion"
    assert relaxed.args.strategy_note == "L1_drop_soft_and_expand_pool"

    relaxed_explicit = resolve_search_relax(
        query="x", rank_query="x", relax_level=1, pool="agent_profiles"
    )
    assert relaxed_explicit.args.pool == PROFILE_POOL
    assert relaxed_explicit.pool_source == "model"


def test_profile_pool_on_model_all_reachable_wins(tmp_path: Path) -> None:
    """Model-supplied known pool beats Module default."""
    yaml_path = _write_modules_yaml(
        tmp_path / "pp_override.yaml", search_relax=True, profile_pool=True
    )
    reload_modules_config(yaml_path)
    resolved = resolve_search_relax(
        query="x", rank_query="x", relax_level=0, pool="all_reachable"
    )
    assert resolved.args.pool == "all_reachable"
    assert resolved.pool_source == "model"
    assert resolved.profile_pool_hook is False


def test_profile_pool_off_omitted_is_all_reachable(tmp_path: Path) -> None:
    yaml_path = _write_modules_yaml(
        tmp_path / "pp_off.yaml", search_relax=False, profile_pool=False
    )
    reload_modules_config(yaml_path)
    resolved = resolve_search_relax(query="x", rank_query="x", pool="")
    assert resolved.args.pool == "all_reachable"
    assert resolved.profile_pool_hook is False
    assert resolved.pool_source == "default"


def test_profile_pool_on_with_search_relax_off(tmp_path: Path) -> None:
    """profile_pool is independent of search_relax — still defaults agent_profiles."""
    yaml_path = _write_modules_yaml(
        tmp_path / "pp_solo.yaml", search_relax=False, profile_pool=True
    )
    reload_modules_config(yaml_path)
    resolved = resolve_search_relax(
        query="x", rank_query="x", relax_level=1, pool=""
    )
    assert resolved.enabled is False
    assert resolved.args.relax_level == 0
    assert resolved.args.pool == PROFILE_POOL
    assert resolved.pool_source == "module_default"


def test_resolve_pool_helper_precedence() -> None:
    pool, hook, src = resolve_pool("", relax_level=0, profile_pool=True)
    assert (pool, hook, src) == (PROFILE_POOL, False, "module_default")
    pool, hook, src = resolve_pool("", relax_level=1, profile_pool=True)
    assert (pool, hook, src) == ("all_reachable", False, "relaxed_expansion")
    pool, hook, src = resolve_pool("all_reachable", relax_level=1, profile_pool=True)
    assert (pool, hook, src) == ("all_reachable", False, "model")
    pool, hook, src = resolve_pool("agent_profiles", relax_level=1, profile_pool=True)
    assert (pool, hook, src) == (PROFILE_POOL, False, "model")
    pool, hook, src = resolve_pool("agent_profiles", relax_level=1, profile_pool=False)
    assert (pool, hook, src) == ("all_reachable", True, "fallback_module_off")
    pool, hook, src = resolve_pool("weird", relax_level=1, profile_pool=True)
    assert (pool, hook, src) == ("all_reachable", False, "unknown")


def test_profile_pool_on_passed_to_hand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    yaml_path = _write_modules_yaml(
        tmp_path / "pp_hand.yaml", search_relax=True, profile_pool=True
    )
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
    raw = search_candidates.invoke(
        {"query": "教育 合伙人", "rank_query": "教育 合伙人", "relax_level": 0},
        config={"metadata": {"user_id": "u_pp", "group_id": "g1", "run_match": "true"}},
    )
    payload = json.loads(raw)
    assert payload["pool"] == PROFILE_POOL
    assert payload["profile_pool_enabled"] is True
    assert payload["pool_source"] == "module_default"
    assert "profile_pool_hook" not in payload
    assert captured[0]["pool"] == PROFILE_POOL


def test_profile_pool_http_posts_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """HTTP hand POSTs pool to new_api (wired, not audit-only stub)."""
    import logging

    from apps.group_agent_api.agent_factory.integrations import match_backend

    yaml_path = _write_modules_yaml(
        tmp_path / "pp_http.yaml", search_relax=True, profile_pool=True
    )
    reload_modules_config(yaml_path)

    captured: list[dict[str, Any]] = []

    def _fake_fetch(**kwargs):
        captured.append(dict(kwargs))
        return MatchResult(
            status="empty",
            candidates=[],
            query=kwargs.get("query") or "",
            group_id="g1",
            reason="http_empty",
        )

    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.integrations.match_backend.fetch_group_agent_match",
        _fake_fetch,
    )
    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        result = match_backend.run_match(
            query="x",
            group_id="g1",
            force_mode="http",
            pool=PROFILE_POOL,
            relax_level=0,
        )
    assert result.status == "empty"
    assert captured[0]["pool"] == PROFILE_POOL
    assert any(
        "match_backend_pool" in r.message and "http_hand_posts_pool" in r.message
        for r in caplog.records
    )


_HARD_SOFT = [
    {
        "field": "city",
        "operator": "in",
        "values": ["上海"],
        "strength": "hard",
    },
    {
        "field": "industry",
        "operator": "in",
        "values": ["教育"],
        "strength": "hard",
    },
    {
        "field": "experience_tags",
        "operator": "any",
        "values": ["langchain"],
        "strength": "soft",
    },
]


def _constraint_fields(hand_constraints: dict[str, Any] | None) -> set[str]:
    assert hand_constraints is not None
    items = hand_constraints.get("items") or []
    return {str(c.get("field")) for c in items if isinstance(c, dict)}


def test_live_path_l0_hard_and_soft_reach_hand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Brief D: L0 keeps hard+soft through search_candidates → run_match."""
    yaml_path = _write_modules_yaml(tmp_path / "d_l0.yaml", search_relax=True)
    reload_modules_config(yaml_path)
    captured: list[dict[str, Any]] = []

    def _fake_run_match(**kwargs):
        captured.append(dict(kwargs))
        return MatchResult(
            status="empty",
            candidates=[],
            query=kwargs["query"],
            group_id=kwargs["group_id"],
            reason="empty_l0",
        )

    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.search_tool.match_backend.run_match",
        _fake_run_match,
    )
    long_rq = "需要找懂 langchain 的合伙人；比如做过 RAG 落地的人"
    raw = search_candidates.invoke(
        {
            "query": "教育 合伙人",
            "rank_query": long_rq,
            "relax_level": 0,
            "constraints": list(_HARD_SOFT),
        },
        config={"metadata": {"user_id": "u_d0", "group_id": "g1", "run_match": "true"}},
    )
    payload = json.loads(raw)
    assert payload["relax_level"] == 0
    assert payload["strategy"] == "L0_hard"
    assert payload["dropped_soft"] is False
    assert payload["constraints_source"] == "tool_arg"
    assert payload["rank_query"] == long_rq
    fields = _constraint_fields(captured[0]["constraints"])
    assert fields == {"city", "industry", "experience_tags"}
    strengths = {
        c["field"]: c["strength"]
        for c in captured[0]["constraints"]["items"]
    }
    assert strengths["city"] == "hard"
    assert strengths["experience_tags"] == "soft"


def test_live_path_l1_drops_soft_keeps_hard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Brief D: L1 + Module on → soft dropped, hard kept, rank_query shortens."""
    yaml_path = _write_modules_yaml(tmp_path / "d_l1.yaml", search_relax=True)
    reload_modules_config(yaml_path)
    captured: list[dict[str, Any]] = []

    def _fake_run_match(**kwargs):
        captured.append(dict(kwargs))
        return MatchResult(
            status="empty",
            candidates=[],
            query=kwargs["query"],
            group_id=kwargs["group_id"],
            reason="empty_l1",
        )

    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.search_tool.match_backend.run_match",
        _fake_run_match,
    )
    long_rq = "需要找懂 langchain 的合伙人；比如做过 RAG 落地的人"
    raw = search_candidates.invoke(
        {
            "query": "教育 合伙人",
            "rank_query": long_rq,
            "relax_level": 1,
            "constraints": list(_HARD_SOFT),
        },
        config={"metadata": {"user_id": "u_d1", "group_id": "g1", "run_match": "true"}},
    )
    payload = json.loads(raw)
    assert payload["relax_level"] == 1
    assert payload["strategy"] == "L1_drop_soft_and_expand_pool"
    assert payload["dropped_soft"] is True
    assert payload["rank_query"] == "需要找懂 langchain 的合伙人"
    assert payload["rank_query"] != long_rq
    fields = _constraint_fields(captured[0]["constraints"])
    assert fields == {"city", "industry"}
    assert "experience_tags" not in fields


def test_live_path_module_off_does_not_drop_soft(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Brief D: Module off → soft kept even if model passes relax_level=1."""
    yaml_path = _write_modules_yaml(tmp_path / "d_off.yaml", search_relax=False)
    reload_modules_config(yaml_path)
    captured: list[dict[str, Any]] = []

    def _fake_run_match(**kwargs):
        captured.append(dict(kwargs))
        return MatchResult(
            status="empty",
            candidates=[],
            query=kwargs["query"],
            group_id=kwargs["group_id"],
            reason="empty_off",
        )

    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.search_tool.match_backend.run_match",
        _fake_run_match,
    )
    long_rq = "需要找懂 langchain 的合伙人；比如做过 RAG 落地的人"
    raw = search_candidates.invoke(
        {
            "query": "教育 合伙人",
            "rank_query": long_rq,
            "relax_level": 1,
            "constraints": list(_HARD_SOFT),
        },
        config={"metadata": {"user_id": "u_doff", "group_id": "g1", "run_match": "true"}},
    )
    payload = json.loads(raw)
    assert payload["search_relax_enabled"] is False
    assert payload["relax_level"] == 0
    assert payload["dropped_soft"] is False
    assert payload["rank_query"] == long_rq
    fields = _constraint_fields(captured[0]["constraints"])
    assert fields == {"city", "industry", "experience_tags"}


def test_live_path_profile_autoload_constraints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Brief D: omit constraints → load match_constraints from saved profile."""
    from apps.group_agent_api.agent_factory.profile_schema import profile_from_flat
    from apps.group_agent_api.agent_factory.profile_store import save_profile

    yaml_path = _write_modules_yaml(tmp_path / "d_auto.yaml", search_relax=True)
    reload_modules_config(yaml_path)

    base_dir = tmp_path / "runtime"
    profile = profile_from_flat(
        user_id="u_auto",
        group_id="g1",
        doing="做 AI 教育",
        need="找技术合伙人",
        offer="教研资源",
        match_constraints=list(_HARD_SOFT),
    )
    save_profile(base_dir, profile)

    captured: list[dict[str, Any]] = []

    def _fake_run_match(**kwargs):
        captured.append(dict(kwargs))
        return MatchResult(
            status="empty",
            candidates=[],
            query=kwargs["query"],
            group_id=kwargs["group_id"],
            reason="empty_auto",
        )

    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.search_tool.match_backend.run_match",
        _fake_run_match,
    )
    raw = search_candidates.invoke(
        {
            "query": "AI 教育 技术合伙人",
            "rank_query": "需要找懂 langchain 的合伙人；比如做过 RAG",
            "relax_level": 1,
            # constraints omitted → profile autoload
        },
        config={
            "metadata": {
                "user_id": "u_auto",
                "group_id": "g1",
                "run_match": "true",
                "base_dir": str(base_dir),
            }
        },
    )
    payload = json.loads(raw)
    assert payload["constraints_source"] == "profile"
    assert payload["relax_level"] == 1
    fields = _constraint_fields(captured[0]["constraints"])
    assert fields == {"city", "industry"}
    assert "experience_tags" not in fields


def test_profile_pool_on_l1_expands_pool_to_all_reachable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    yaml_path = _write_modules_yaml(
        tmp_path / "pp_l1_expand.yaml", search_relax=True, profile_pool=True
    )
    reload_modules_config(yaml_path)

    captured: list[dict[str, Any]] = []

    def _fake_run_match(**kwargs):
        captured.append(dict(kwargs))
        return MatchResult(
            status="empty",
            candidates=[],
            query=kwargs["query"],
            group_id=kwargs["group_id"],
            reason="empty",
        )

    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.search_tool.match_backend.run_match",
        _fake_run_match,
    )

    # L0 with omitted pool -> agent_profiles
    p0 = json.loads(
        search_candidates.invoke(
            {"query": "K12数学 教研", "relax_level": 0},
            config={"metadata": {"user_id": "u_k12", "group_id": "g1", "run_match": "true"}},
        )
    )
    assert p0["pool"] == "agent_profiles"
    assert p0["strategy"] == "L0_hard"
    assert captured[0]["pool"] == "agent_profiles"

    # L1 with omitted pool -> expands to all_reachable
    p1 = json.loads(
        search_candidates.invoke(
            {"query": "K12数学 教研", "relax_level": 1},
            config={"metadata": {"user_id": "u_k12", "group_id": "g1", "run_match": "true"}},
        )
    )
    assert p1["pool"] == "all_reachable"
    assert p1["strategy"] == "L1_drop_soft_and_expand_pool"
    assert captured[1]["pool"] == "all_reachable"


def test_system_prompt_teaches_hard_soft_constraints() -> None:
    from apps.group_agent_api.agent_factory.agent import SYSTEM_PROMPT

    assert "match_constraints" in SYSTEM_PROMPT
    assert "strength=hard" in SYSTEM_PROMPT
    assert "strength=soft" in SYSTEM_PROMPT
    assert "city" in SYSTEM_PROMPT
    assert "industry" in SYSTEM_PROMPT
