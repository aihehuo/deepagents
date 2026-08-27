"""Brief INV · mod.brain.invite_copy — default on ≡ today; off → empty invite / no enrich."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from apps.group_agent_api.agent_factory.invite import (
    CHECK_SCAFFOLD,
    MODULE_ID,
    enrich_candidate_with_single_copy,
    generate_invite_with_optional_llm,
    invite_copy_enabled,
    invite_scaffold_enabled,
    should_emit_invite_artifact,
)
from apps.group_agent_api.agent_factory.module_config import (
    load_modules_config,
    reload_modules_config,
    reset_modules_config_cache,
)
from apps.group_agent_api.agent_factory.profile_schema import (
    DisclosureLevel,
    profile_from_flat,
)


def _write_modules_yaml(
    path: Path,
    *,
    invite_copy: bool = True,
    invite_scaffold: bool = True,
    invite_llm_polish: bool = True,
) -> Path:
    text = f"""version: 1
preset: current
checks:
  chk.invite_scaffold: {'true' if invite_scaffold else 'false'}
  chk.invite_llm_polish: {'true' if invite_llm_polish else 'false'}
modules:
  mod.brain.reply_grounding: true
  {MODULE_ID}: {'true' if invite_copy else 'false'}
"""
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _reset_modules_config() -> Any:
    reset_modules_config_cache()
    yield
    reset_modules_config_cache()


def _profile():
    return profile_from_flat(
        user_id="u_inv",
        group_id="g_inv",
        doing="智能宠物喂食器",
        need="联网与 App 固件",
        offer="工厂与供应链",
    )


def _candidate() -> dict[str, Any]:
    return {
        "user_id": "c1",
        "display_name": "张三",
        "doing": {
            "value": "做 IoT 固件",
            "disclosure": DisclosureLevel.confirmed_public.value,
        },
        "is_reachable": True,
        "match_evidence": [{"summary": "方向接近固件"}],
    }


def test_default_yaml_invite_copy_on() -> None:
    cfg = load_modules_config()
    assert cfg.is_module_enabled(MODULE_ID) is True
    assert cfg.invite_copy_enabled() is True
    assert cfg.invite_scaffold_enabled() is True
    assert cfg.is_check_enabled(CHECK_SCAFFOLD) is True
    assert invite_copy_enabled() is True
    assert invite_scaffold_enabled() is True


def test_module_off_blocks_emit_and_empty_invite(tmp_path: Path) -> None:
    yaml_path = _write_modules_yaml(tmp_path / "off.yaml", invite_copy=False)
    reload_modules_config(yaml_path)

    assert invite_copy_enabled() is False
    assert invite_scaffold_enabled() is False
    assert (
        should_emit_invite_artifact(
            match_status="matched",
            match_reason=None,
            candidate_count=1,
        )
        is False
    )

    result = generate_invite_with_optional_llm(
        profile=_profile(),
        candidates=[_candidate()],
        match_status="matched",
        willing_to_at=True,
        user_id="u_inv",
        group_id="g_inv",
        use_llm=False,
    )
    assert result.text == ""
    assert result.ok is False
    assert "invite_scaffold_off" in result.violations


def test_scaffold_check_off_blocks_emit(tmp_path: Path) -> None:
    yaml_path = _write_modules_yaml(
        tmp_path / "scaffold_off.yaml",
        invite_copy=True,
        invite_scaffold=False,
    )
    reload_modules_config(yaml_path)

    assert invite_copy_enabled() is True
    assert invite_scaffold_enabled() is False
    assert (
        should_emit_invite_artifact(
            match_status="matched",
            match_reason=None,
            candidate_count=1,
        )
        is False
    )
    result = generate_invite_with_optional_llm(
        profile=_profile(),
        candidates=[_candidate()],
        match_status="matched",
        willing_to_at=True,
        use_llm=False,
    )
    assert result.text == ""
    assert result.ok is False


def test_module_on_default_produces_invite_text(tmp_path: Path) -> None:
    yaml_path = _write_modules_yaml(tmp_path / "on.yaml", invite_copy=True)
    reload_modules_config(yaml_path)

    assert (
        should_emit_invite_artifact(
            match_status="matched",
            match_reason=None,
            candidate_count=1,
        )
        is True
    )
    result = generate_invite_with_optional_llm(
        profile=_profile(),
        candidates=[_candidate()],
        match_status="matched",
        willing_to_at=True,
        use_llm=False,
    )
    assert result.ok is True
    assert result.text
    assert "@" in result.text or result.kind == "undirected"


def test_module_off_skips_per_candidate_enrich(tmp_path: Path) -> None:
    yaml_path = _write_modules_yaml(tmp_path / "enrich_off.yaml", invite_copy=False)
    reload_modules_config(yaml_path)

    raw = _candidate()
    enriched = enrich_candidate_with_single_copy(raw, _profile())
    assert "invite_text" not in enriched
    assert "forward_copy" not in enriched
    assert "quick_connect_copy" not in enriched
    assert "match_highlights" not in enriched


def test_module_on_enriches_per_candidate(tmp_path: Path) -> None:
    yaml_path = _write_modules_yaml(tmp_path / "enrich_on.yaml", invite_copy=True)
    reload_modules_config(yaml_path)

    enriched = enrich_candidate_with_single_copy(_candidate(), _profile())
    assert enriched.get("invite_text")
    assert enriched.get("forward_copy")
    assert enriched.get("quick_connect_copy")
    assert isinstance(enriched.get("match_highlights"), list)
