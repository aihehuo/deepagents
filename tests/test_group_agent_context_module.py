"""Tests for mod.brain.context — YAML-switchable prompt fragments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from apps.group_agent_api.agent_factory.agent import FORCE_SAVE_PROMPT, SYSTEM_PROMPT
from apps.group_agent_api.agent_factory.context import (
    ALL_CONTEXT_IDS,
    CTX_FORCE_SAVE_PROMPT,
    CTX_SYSTEM_NETWORK_DONTS,
    CTX_SYSTEM_ROLE_AND_GOAL,
    CTX_SYSTEM_SUGGESTED_REPLIES,
    CTX_TURN_KNOWN_PROFILE,
    CTX_TURN_PRIOR_RECOMMENDATION,
    CTX_TURN_REFERRAL,
    MODULE_ID,
    SYSTEM_FRAGMENT_IDS,
    build_system_prompt,
    context_module_enabled,
    force_save_prompt,
    is_context_enabled,
    known_profile_system_message,
    prior_recommendation_system_content,
    referral_context_system_message,
)
from apps.group_agent_api.agent_factory.module_config import (
    load_modules_config,
    reload_modules_config,
    reset_modules_config_cache,
)
from apps.group_agent_api.agent_factory.profile_schema import profile_from_flat
from apps.group_agent_api.agent_factory.profile_store import save_profile
from apps.group_agent_api.agent_factory.revisit import RevisitHint


def _write_context_yaml(
    path: Path,
    *,
    module_on: bool = True,
    **context_bits: bool,
) -> Path:
    lines = [
        "version: 1",
        "preset: current",
        "context:",
    ]
    for kid, on in context_bits.items():
        lines.append(f"  {kid}: {'true' if on else 'false'}")
    lines.append("modules:")
    lines.append(f"  {MODULE_ID}: {'true' if module_on else 'false'}")
    lines.append("  mod.brain.reply_grounding: true")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _reset_modules_config() -> Any:
    reset_modules_config_cache()
    yield
    reset_modules_config_cache()


def test_default_yaml_enables_context_module_and_fragments() -> None:
    cfg = load_modules_config()
    assert cfg.is_module_enabled(MODULE_ID) is True
    assert context_module_enabled() is True
    for cid in ALL_CONTEXT_IDS:
        assert is_context_enabled(cid) is True, cid
        assert cfg.is_context_enabled(cid) is True


def test_default_on_system_prompt_matches_legacy_constant() -> None:
    """All system fragments on ≡ agent.SYSTEM_PROMPT (preset current)."""
    assembled = build_system_prompt(SYSTEM_FRAGMENT_IDS)
    assert assembled == SYSTEM_PROMPT
    assert "具体的 doing / need / offer" in assembled
    assert "search_candidates" in assembled
    assert "suggested_replies" in assembled
    assert FORCE_SAVE_PROMPT == force_save_prompt(enabled=True)


def test_off_fragment_shortens_prompt(tmp_path: Path) -> None:
    yaml_path = _write_context_yaml(
        tmp_path / "no_network.yaml",
        **{cid: True for cid in SYSTEM_FRAGMENT_IDS},
    )
    # Flip network_donts off
    text = yaml_path.read_text(encoding="utf-8")
    text = text.replace(
        f"  {CTX_SYSTEM_NETWORK_DONTS}: true",
        f"  {CTX_SYSTEM_NETWORK_DONTS}: false",
    )
    yaml_path.write_text(text, encoding="utf-8")
    reload_modules_config(yaml_path)

    full = build_system_prompt(SYSTEM_FRAGMENT_IDS)
    gated = build_system_prompt()
    assert len(gated) < len(full)
    assert "人脉与披露" not in gated
    assert "红线" not in gated
    assert "## 目标" in gated


def test_module_off_yields_empty_system_prompt(tmp_path: Path) -> None:
    yaml_path = _write_context_yaml(
        tmp_path / "module_off.yaml",
        module_on=False,
        **{cid: True for cid in ALL_CONTEXT_IDS},
    )
    reload_modules_config(yaml_path)
    assert build_system_prompt() == ""
    assert force_save_prompt() == ""
    assert is_context_enabled(CTX_SYSTEM_ROLE_AND_GOAL) is False


def test_suggested_replies_fragment_toggle(tmp_path: Path) -> None:
    bits = {cid: True for cid in SYSTEM_FRAGMENT_IDS}
    bits[CTX_SYSTEM_SUGGESTED_REPLIES] = False
    yaml_path = _write_context_yaml(tmp_path / "no_suggest.yaml", **bits)
    reload_modules_config(yaml_path)
    prompt = build_system_prompt()
    assert "<suggested_replies>" not in prompt
    assert "可点击建议回复" not in prompt


def test_force_save_prompt_gate(tmp_path: Path) -> None:
    yaml_path = _write_context_yaml(
        tmp_path / "no_force.yaml",
        **{CTX_FORCE_SAVE_PROMPT: False},
    )
    reload_modules_config(yaml_path)
    assert force_save_prompt() == ""
    assert force_save_prompt(enabled=True) == FORCE_SAVE_PROMPT


def test_turn_known_profile_gate(tmp_path: Path) -> None:
    profile = profile_from_flat(
        user_id="u1",
        group_id="g1",
        doing="做 AI 教育",
        need="找渠道",
        offer="内容能力",
    )
    save_profile(tmp_path, profile)

    msg = known_profile_system_message(
        base_dir=tmp_path, user_id="u1", group_id="g1", enabled=True
    )
    assert msg is not None
    assert "做 AI 教育" in msg.content

    yaml_path = _write_context_yaml(
        tmp_path / "no_known.yaml",
        **{CTX_TURN_KNOWN_PROFILE: False},
    )
    reload_modules_config(yaml_path)
    assert (
        known_profile_system_message(
            base_dir=tmp_path, user_id="u1", group_id="g1"
        )
        is None
    )


def test_turn_referral_gate(tmp_path: Path) -> None:
    meta = {
        "referral_context": {
            "applicant_id": 100,
            "applicant_name": "张志远",
            "intro_once": True,
            "status": "dispatched",
        }
    }
    assert referral_context_system_message(meta, enabled=True) is not None

    yaml_path = _write_context_yaml(
        tmp_path / "no_ref.yaml",
        **{CTX_TURN_REFERRAL: False},
    )
    reload_modules_config(yaml_path)
    assert referral_context_system_message(meta) is None


def test_turn_prior_recommendation_gate(tmp_path: Path) -> None:
    hint = RevisitHint(
        has_prior_invite=True,
        candidate_names=["Alice"],
        topic_summary="找联合创始人",
    )
    assert prior_recommendation_system_content(hint, enabled=True) is not None

    yaml_path = _write_context_yaml(
        tmp_path / "no_prior.yaml",
        **{CTX_TURN_PRIOR_RECOMMENDATION: False},
    )
    reload_modules_config(yaml_path)
    assert prior_recommendation_system_content(hint) is None


def test_member_system_prompt_respects_yaml(tmp_path: Path) -> None:
    from apps.group_agent_api.agent_factory.agent import member_system_prompt

    bits = {cid: True for cid in SYSTEM_FRAGMENT_IDS}
    bits[CTX_SYSTEM_NETWORK_DONTS] = False
    yaml_path = _write_context_yaml(tmp_path / "member.yaml", **bits)
    reload_modules_config(yaml_path)
    prompt = member_system_prompt()
    assert "人脉与披露" not in prompt
    assert "## 目标" in prompt
