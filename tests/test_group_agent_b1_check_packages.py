"""Focused tests for Brief B1 YAML-gated check packages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from apps.group_agent_api.agent_factory.checks.action_claim import (
    CHECK_ID as ACTION_CLAIM_ID,
    apply_action_claim_guard,
)
from apps.group_agent_api.agent_factory.checks.finalize_templates import (
    CHECK_ID as FINALIZE_TEMPLATES_ID,
    finalize_templates_enabled,
)
from apps.group_agent_api.agent_factory.checks.invented_candidate import (
    CHECK_ID as INVENTED_CANDIDATE_ID,
    scrub_invented_candidate_if_enabled,
)
from apps.group_agent_api.agent_factory.content_quality import (
    finalize_user_visible_reply,
)
from apps.group_agent_api.agent_factory.module_config import (
    load_modules_config,
    reload_modules_config,
    reset_modules_config_cache,
)
from apps.group_agent_api.agent_factory.profile_schema import profile_from_flat


def _write_checks_yaml(path: Path, **checks: bool) -> Path:
    lines = [
        "version: 1",
        "preset: current",
        "checks:",
    ]
    for kid, on in checks.items():
        lines.append(f"  {kid}: {'true' if on else 'false'}")
    lines.append("modules:")
    lines.append("  mod.brain.reply_grounding: true")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _reset_modules_config() -> Any:
    reset_modules_config_cache()
    yield
    reset_modules_config_cache()


def test_default_yaml_enables_b1_checks() -> None:
    cfg = load_modules_config()
    assert cfg.is_check_enabled(ACTION_CLAIM_ID) is True
    assert cfg.is_check_enabled(INVENTED_CANDIDATE_ID) is True
    assert cfg.is_check_enabled(FINALIZE_TEMPLATES_ID) is True
    assert finalize_templates_enabled() is True


def test_action_claim_off_is_noop(tmp_path: Path) -> None:
    yaml_path = _write_checks_yaml(
        tmp_path / "off_action.yaml",
        **{ACTION_CLAIM_ID: False},
    )
    reload_modules_config(yaml_path)
    claim = "我已经帮您发送到群里并@了对方，请留意消息。"
    out, blocked = apply_action_claim_guard(claim)
    assert blocked is False
    assert out == claim


def test_action_claim_on_replaces(tmp_path: Path) -> None:
    yaml_path = _write_checks_yaml(
        tmp_path / "on_action.yaml",
        **{ACTION_CLAIM_ID: True},
    )
    reload_modules_config(yaml_path)
    claim = "我已经帮您发送到群里并@了对方，请留意消息。"
    out, blocked = apply_action_claim_guard(claim)
    assert blocked is True
    assert "我无法直接向群内发送消息或通知管理员" in out


def test_invented_candidate_off_keeps_text(tmp_path: Path) -> None:
    invented = "匹配到一位候选人，背景如下：曾主导某教培项目。"
    yaml_path = _write_checks_yaml(
        tmp_path / "off_invented.yaml",
        **{INVENTED_CANDIDATE_ID: False},
    )
    reload_modules_config(yaml_path)
    assert scrub_invented_candidate_if_enabled(invented) == invented


def test_invented_candidate_on_scrubs(tmp_path: Path) -> None:
    invented = "匹配到一位候选人，背景如下：曾主导某教培项目。"
    yaml_path = _write_checks_yaml(
        tmp_path / "on_invented.yaml",
        **{INVENTED_CANDIDATE_ID: True},
    )
    reload_modules_config(yaml_path)
    assert scrub_invented_candidate_if_enabled(invented) == ""


def test_finalize_templates_off_keeps_da_reply(tmp_path: Path) -> None:
    yaml_path = _write_checks_yaml(
        tmp_path / "off_finalize.yaml",
        **{
            FINALIZE_TEMPLATES_ID: False,
            INVENTED_CANDIDATE_ID: True,
        },
    )
    reload_modules_config(yaml_path)
    profile = profile_from_flat(
        user_id="u1",
        group_id="g1",
        doing="做AI教育产品",
        need="找教研合伙人",
        offer="有原型",
    )
    model_reply = "好的，我先按你说的方向帮你想想怎么描述需求。"
    out = finalize_user_visible_reply(
        original_reply=model_reply,
        profile=profile,
        profile_persisted=True,
        match_status="skipped",
        candidate_count=0,
        delivery_kind=None,
        invite_ok=None,
        network_unlocked=True,
    )
    assert out == model_reply
    assert "我理解并已更新画像" not in out
    assert "下一步" not in out


def test_finalize_templates_on_may_stack_confirmation(tmp_path: Path) -> None:
    yaml_path = _write_checks_yaml(
        tmp_path / "on_finalize.yaml",
        **{
            FINALIZE_TEMPLATES_ID: True,
            INVENTED_CANDIDATE_ID: True,
        },
    )
    reload_modules_config(yaml_path)
    profile = profile_from_flat(
        user_id="u1",
        group_id="g1",
        doing="做AI教育产品",
        need="找教研合伙人",
        offer="有原型",
    )
    # Short stub forces template path (not substantive custom reply).
    out = finalize_user_visible_reply(
        original_reply="好的。",
        profile=profile,
        profile_persisted=True,
        match_status="skipped",
        candidate_count=0,
        delivery_kind=None,
        invite_ok=None,
        network_unlocked=True,
        profile_saved_this_turn=True,
    )
    assert "我理解并已更新画像" in out


def test_smoke_reply_grounding_and_mouth_ingress_import() -> None:
    from apps.group_agent_api.agent_factory.brain_repair import (
        apply_mouth_repair,
        format_ingress_deny,
    )
    from apps.group_agent_api.agent_factory.checks.reply_grounding import (
        apply_reply_grounding_gate,
        check_reply_grounding,
    )

    assert callable(apply_reply_grounding_gate)
    assert callable(check_reply_grounding)
    assert callable(apply_mouth_repair)
    assert callable(format_ingress_deny)
