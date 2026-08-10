"""Admin-only conversion diagnosis tools for group-agent."""

from __future__ import annotations

import json
from typing import Any

from apps.group_agent_api.agent_factory import admin_ops_tools


ADMIN_CONFIG = {"metadata": {"source": "group_agent_admin_debug"}}


def test_conversion_tools_are_registered() -> None:
    names = {tool.name for tool in admin_ops_tools.ADMIN_OPS_TOOLS}
    assert {
        "admin_funnel_analysis",
        "admin_dropoff_samples",
        "admin_compare_conversations",
    }.issubset(names)


def test_funnel_tool_calls_read_only_micro_endpoint(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        captured.update(path=path, params=params)
        return {"ok": True, "stages": {"identity_ready_uv": 10}}

    monkeypatch.setattr(admin_ops_tools, "_get_json", fake_get)
    result = admin_ops_tools.admin_funnel_analysis.invoke(
        {"days": 99}, config=ADMIN_CONFIG
    )

    assert json.loads(result)["ok"] is True
    assert captured == {
        "path": "/group_agent/ops_funnel_analysis",
        "params": {"days": 30},
    }


def test_dropoff_tool_bounds_stage_and_limit(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        captured.update(path=path, params=params)
        return {"ok": True, "samples": []}

    monkeypatch.setattr(admin_ops_tools, "_get_json", fake_get)
    admin_ops_tools.admin_dropoff_samples.invoke(
        {"days": 0, "stage": "raw_transcript", "limit": 500},
        config=ADMIN_CONFIG,
    )

    assert captured == {
        "path": "/group_agent/ops_dropoff_samples",
        "params": {"days": 7, "stage": "f2_not_f3", "limit": 20},
    }


def test_conversion_tools_require_trusted_admin_source() -> None:
    result = admin_ops_tools.admin_compare_conversations.invoke(
        {"days": 7}, config={"metadata": {"admin_mode": True}}
    )

    assert json.loads(result)["error"] == "admin_mode_required"


def test_admin_prompt_requires_evidence_before_diagnosis() -> None:
    prompt = admin_ops_tools.ADMIN_SYSTEM_PROMPT
    assert "数据事实" in prompt
    assert "样本推断" in prompt
    assert "不得把相关性说成因果" in prompt
    assert "F2 身份就绪 → F2.5 首次开口 → F3" in prompt
    assert "禁止使用 F1" in prompt
    assert "禁止写 F2→F1" in prompt
    assert "F2.5首次开口" in admin_ops_tools.ADMIN_TURN_REMINDER
    assert "禁止使用F1" in admin_ops_tools.ADMIN_TURN_REMINDER
