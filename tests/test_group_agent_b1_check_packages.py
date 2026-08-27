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


def _write_checks_yaml(
    path: Path,
    *,
    reply_grounding: bool = True,
    **checks: bool,
) -> Path:
    lines = [
        "version: 1",
        "preset: current",
        "checks:",
    ]
    for kid, on in checks.items():
        lines.append(f"  {kid}: {'true' if on else 'false'}")
    lines.append("modules:")
    lines.append(
        f"  mod.brain.reply_grounding: {'true' if reply_grounding else 'false'}"
    )
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
    assert cfg.is_check_enabled("chk.capability_guard") is True
    assert cfg.is_check_enabled("chk.invite_llm_polish") is True
    assert cfg.is_check_enabled("chk.profile_quality_llm") is True
    assert cfg.is_check_enabled("chk.force_save_retry") is True
    assert cfg.is_check_enabled("chk.deterministic_profile_save") is True
    assert cfg.is_check_enabled("chk.match_v2_schema") is True
    assert cfg.brain_check_master_enabled() is True
    assert finalize_templates_enabled() is True


def test_action_claim_off_is_noop(tmp_path: Path) -> None:
    yaml_path = _write_checks_yaml(
        tmp_path / "off_action.yaml",
        reply_grounding=False,
        **{ACTION_CLAIM_ID: False},
    )
    reload_modules_config(yaml_path)
    claim = "我已经帮您发送到群里并@了对方，请留意消息。"
    out, blocked = apply_action_claim_guard(claim)
    assert blocked is False
    assert out == claim


def test_action_claim_on_replaces(tmp_path: Path) -> None:
    # RG off so silent replace is the active path (not deferred to L0).
    yaml_path = _write_checks_yaml(
        tmp_path / "on_action.yaml",
        reply_grounding=False,
        **{ACTION_CLAIM_ID: True},
    )
    reload_modules_config(yaml_path)
    claim = "我已经帮您发送到群里并@了对方，请留意消息。"
    out, blocked = apply_action_claim_guard(claim)
    assert blocked is True
    assert "我无法直接向群内发送消息或通知管理员" in out


def test_action_claim_skipped_when_reply_grounding_on(tmp_path: Path) -> None:
    """Brief F: prefer RG L0 over silent replace when both YAML bits are on."""
    yaml_path = _write_checks_yaml(
        tmp_path / "rg_defers_action.yaml",
        reply_grounding=True,
        **{ACTION_CLAIM_ID: True},
    )
    reload_modules_config(yaml_path)
    claim = "我已经帮您发送到群里并@了对方，请留意消息。"
    out, blocked = apply_action_claim_guard(claim)
    assert blocked is False
    assert out == claim
    # Explicit force still applies (tests / escape hatch).
    out2, blocked2 = apply_action_claim_guard(claim, enabled=True)
    assert blocked2 is True
    assert "我无法直接向群内发送消息或通知管理员" in out2


def test_capability_guard_off_passthrough(tmp_path: Path) -> None:
    from apps.group_agent_api.agent_factory.capability import CapabilityTier
    from apps.group_agent_api.agent_factory.guard import enforce_capability_guard

    yaml_path = _write_checks_yaml(
        tmp_path / "cap_off.yaml",
        **{"chk.capability_guard": False},
    )
    reload_modules_config(yaml_path)
    leak_reply = "群里有人可以推荐，@张三 值得认识"
    cands = [
        {
            "user_id": "c1",
            "source_group_id": "other_group",
            "doing": {"value": "x", "disclosure": "confirmed_public"},
        }
    ]
    guarded = enforce_capability_guard(
        tier=CapabilityTier.not_in_group,
        reply=leak_reply,
        candidates=cands,
        caller_group_id="g1",
        user_id="u1",
    )
    assert guarded.ok is True
    assert guarded.blocked is False
    assert guarded.candidates == cands
    assert guarded.reply == leak_reply


def test_capability_guard_on_blocks_not_in_group(tmp_path: Path) -> None:
    from apps.group_agent_api.agent_factory.capability import CapabilityTier
    from apps.group_agent_api.agent_factory.guard import enforce_capability_guard

    yaml_path = _write_checks_yaml(
        tmp_path / "cap_on.yaml",
        **{"chk.capability_guard": True},
    )
    reload_modules_config(yaml_path)
    leak_reply = "群里有人可以推荐，@张三 值得认识"
    guarded = enforce_capability_guard(
        tier=CapabilityTier.not_in_group,
        reply=leak_reply,
        candidates=[{"user_id": "c1"}],
        caller_group_id="g1",
        user_id="u1",
    )
    assert guarded.blocked is True
    assert guarded.candidates == []


def test_invite_llm_polish_yaml_off_without_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from apps.group_agent_api.agent_factory.integrations.config import (
        llm_polish_enabled,
    )

    monkeypatch.delenv("GROUP_AGENT_LLM_POLISH", raising=False)
    monkeypatch.setenv("GROUP_AGENT_MODEL_MODE", "real")
    monkeypatch.setenv("GROUP_AGENT_PROVIDER", "qwen")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test-not-empty")
    yaml_path = _write_checks_yaml(
        tmp_path / "polish_off.yaml",
        **{"chk.invite_llm_polish": False},
    )
    reload_modules_config(yaml_path)
    assert llm_polish_enabled() is False


def test_invite_llm_polish_env_overrides_yaml_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from apps.group_agent_api.agent_factory.integrations.config import (
        llm_polish_enabled,
    )

    monkeypatch.setenv("GROUP_AGENT_LLM_POLISH", "1")
    monkeypatch.setenv("GROUP_AGENT_MODEL_MODE", "real")
    monkeypatch.setenv("GROUP_AGENT_PROVIDER", "qwen")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test-not-empty")
    yaml_path = _write_checks_yaml(
        tmp_path / "polish_env.yaml",
        **{"chk.invite_llm_polish": False},
    )
    reload_modules_config(yaml_path)
    assert llm_polish_enabled() is True


def test_fixture_miss_honors_membership_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """search_relax real_llm synthetic ids must not force not_in_group."""
    from apps.group_agent_api.agent_factory.capability import CapabilityTier
    from apps.group_agent_api.agent_factory.integrations.membership_backend import (
        resolve_session_capability,
    )

    monkeypatch.setenv("GROUP_AGENT_INTEGRATION", "stub")
    monkeypatch.setenv("GROUP_AGENT_ENV", "test")
    monkeypatch.setenv("GROUP_AGENT_TEST_LEVEL", "L1")
    res = resolve_session_capability(
        membership_override="in_group",
        unionid=None,
        group_token=None,
        group_id="group_constraints_llm",
        user_id="u_constraints_llm",
    )
    assert res.tier is CapabilityTier.in_group
    assert res.reason == "stub_membership_fixture_miss"


def test_fixture_row_still_authoritative_over_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from apps.group_agent_api.agent_factory.capability import CapabilityTier
    from apps.group_agent_api.agent_factory.integrations.membership_backend import (
        resolve_session_capability,
    )

    monkeypatch.setenv("GROUP_AGENT_INTEGRATION", "stub")
    monkeypatch.setenv("GROUP_AGENT_ENV", "test")
    monkeypatch.setenv("GROUP_AGENT_TEST_LEVEL", "L1")
    # u105 is in_group in L1 fixture — override cannot demote.
    res = resolve_session_capability(
        membership_override="not_in_group",
        unionid=None,
        group_token=None,
        group_id="group_l1_alpha",
        user_id="u105",
    )
    assert res.tier is CapabilityTier.in_group
    assert res.reason == "fixture_authoritative_in_group"


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


def test_profile_quality_llm_off_skips_second_llm(tmp_path: Path) -> None:
    """Off = Layer1 rules only; length-ok profile ready without invoking model."""
    from apps.group_agent_api.agent_factory.checks.profile_quality import (
        CHECK_ID as PQ_ID,
    )
    from apps.group_agent_api.agent_factory.profile_quality import (
        assess_profile_match_ready,
    )

    yaml_path = _write_checks_yaml(
        tmp_path / "pq_off.yaml",
        **{PQ_ID: False},
    )
    reload_modules_config(yaml_path)

    class _MustNotCall:
        def invoke(self, _msgs):  # pragma: no cover
            raise AssertionError("Layer2 LLM must not run when chk off")

    profile = profile_from_flat(
        user_id="u1",
        group_id="g1",
        doing="在做智能宠物喂食器硬件与 App",
        need="需要联网固件与量产供应链对接",
        offer="有工厂资源和硬件设计经验",
    )
    q = assess_profile_match_ready(
        profile=profile,
        model=_MustNotCall(),
        base_dir=tmp_path,
    )
    assert q.ready is True
    assert q.source == "rules"
    assert "profile_quality_llm_skipped" in q.reasons


def test_profile_quality_llm_on_calls_model(tmp_path: Path) -> None:
    from apps.group_agent_api.agent_factory.checks.profile_quality import (
        CHECK_ID as PQ_ID,
    )
    from apps.group_agent_api.agent_factory.profile_quality import (
        assess_profile_match_ready,
    )

    yaml_path = _write_checks_yaml(
        tmp_path / "pq_on.yaml",
        **{PQ_ID: True},
    )
    reload_modules_config(yaml_path)

    class _ReadyModel:
        def invoke(self, _msgs):
            class _M:
                content = (
                    '{"ready":true,"score":80,"doing_ok":true,"need_ok":true,'
                    '"offer_ok":true,"reasons":[],"gaps":[]}'
                )

            return _M()

    profile = profile_from_flat(
        user_id="u1",
        group_id="g1",
        doing="在做智能宠物喂食器硬件与 App",
        need="需要联网固件与量产供应链对接",
        offer="有工厂资源和硬件设计经验",
    )
    q = assess_profile_match_ready(
        profile=profile,
        model=_ReadyModel(),
        base_dir=tmp_path,
    )
    assert q.ready is True
    assert q.source == "llm"


def test_soft_master_off_skips_profile_quality_not_capability(
    tmp_path: Path,
) -> None:
    from apps.group_agent_api.agent_factory.capability import CapabilityTier
    from apps.group_agent_api.agent_factory.checks.profile_quality import (
        CHECK_ID as PQ_ID,
        profile_quality_llm_enabled,
    )
    from apps.group_agent_api.agent_factory.guard import enforce_capability_guard

    lines = [
        "version: 1",
        "preset: current",
        "checks:",
        f"  {PQ_ID}: true",
        "  chk.capability_guard: true",
        "  chk.force_save_retry: true",
        "modules:",
        "  mod.brain.check: false",
        "  mod.brain.reply_grounding: false",
        "",
    ]
    yaml_path = tmp_path / "soft_master_off.yaml"
    yaml_path.write_text("\n".join(lines), encoding="utf-8")
    reload_modules_config(yaml_path)

    assert profile_quality_llm_enabled() is False
    assert load_modules_config().is_check_enabled("chk.force_save_retry") is True

    leak_reply = "群里有人可以推荐，@张三 值得认识"
    guarded = enforce_capability_guard(
        tier=CapabilityTier.not_in_group,
        reply=leak_reply,
        candidates=[{"user_id": "c1"}],
        caller_group_id="g1",
        user_id="u1",
    )
    assert guarded.blocked is True


def test_force_save_and_deterministic_default_on_when_missing(
    tmp_path: Path,
) -> None:
    from apps.group_agent_api.agent_factory.checks.deterministic_profile_save import (
        deterministic_profile_save_enabled,
    )
    from apps.group_agent_api.agent_factory.checks.force_save_retry import (
        force_save_retry_enabled,
    )
    from apps.group_agent_api.agent_factory.checks.match_v2_schema import (
        match_v2_schema_enabled,
    )

    yaml_path = _write_checks_yaml(tmp_path / "hard_missing.yaml")
    reload_modules_config(yaml_path)
    assert force_save_retry_enabled() is True
    assert deterministic_profile_save_enabled() is True
    assert match_v2_schema_enabled() is True


def test_force_save_retry_yaml_off(tmp_path: Path) -> None:
    from apps.group_agent_api.agent_factory.checks.force_save_retry import (
        CHECK_ID as FS_ID,
        force_save_retry_enabled,
    )

    yaml_path = _write_checks_yaml(
        tmp_path / "fs_off.yaml",
        **{FS_ID: False},
    )
    reload_modules_config(yaml_path)
    assert force_save_retry_enabled() is False


def test_deterministic_profile_save_yaml_off(tmp_path: Path) -> None:
    from apps.group_agent_api.agent_factory.checks.deterministic_profile_save import (
        CHECK_ID as DET_ID,
        deterministic_profile_save_enabled,
    )

    yaml_path = _write_checks_yaml(
        tmp_path / "det_off.yaml",
        **{DET_ID: False},
    )
    reload_modules_config(yaml_path)
    assert deterministic_profile_save_enabled() is False
