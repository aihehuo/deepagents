"""Offline regressions for portable authenticated profile-write skills."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import importlib.util
import json
from typing import Any

import pytest
from langchain_core.tools import tool

from apps.group_agent_api.agent_factory.agent import save_group_profile
from apps.group_agent_api.agent_factory.integrations.profile_client import ProfileHttpError
from apps.group_agent_api.agent_factory.profile_store import load_profile
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.skills import _list_skills
from skillpacks.aihehuo_write.adapter import NAME, create_profile_write_tool, validate_installation
from skillpacks.aihehuo_write.manage import install

CONTENT = {"doing": "开发智能宠物喂食器", "need": "联网开发", "offer": "工厂与供应链"}


@pytest.fixture
def source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("GROUP_AGENT_WRITE_MODE", "skillpack")
    monkeypatch.setenv("GROUP_AGENT_INTEGRATION", "stub")
    directory = tmp_path / "skills" / "aihehuo_write"
    install(target=directory)
    return directory


def config(runtime: Path, *, permission: bool | str = True) -> dict[str, Any]:
    return {"metadata": {"user_id": "owner", "group_id": "g1", "base_dir": str(runtime), "can_write_profile": permission, "run_id": "test_run"}}


def test_package_is_independent_discoverable_and_repeatable(source: Path, tmp_path: Path) -> None:
    directory = source / NAME
    assert install(target=source) == directory
    assert {path.name for path in directory.iterdir()} == {"SKILL.md", "adapter.py", "package.json"}
    skills = _list_skills(FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True), "/skills/aihehuo_write/")
    assert [skill["name"] for skill in skills] == [NAME]
    specification = importlib.util.spec_from_file_location("portable_write_adapter", directory / "adapter.py")
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    writer = module.create_profile_write_tool(skills_dir=source, writer=save_group_profile)
    result = writer.invoke(CONTENT, config=config(tmp_path))
    assert result.startswith("ok: saved profile")
    assert load_profile(tmp_path, "owner", "g1") is not None


def test_install_preview_and_conflicts_preserve_content(tmp_path: Path) -> None:
    source = tmp_path / "skills"
    assert install(target=source, dry_run=True) == source / NAME
    assert not source.exists()
    directory = install(target=source)
    (directory / "SKILL.md").write_text("custom", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        install(target=source)
    assert (directory / "SKILL.md").read_text() == "custom"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / NAME).symlink_to(directory, target_is_directory=True)
    with pytest.raises(FileExistsError):
        install(target=elsewhere)


@pytest.mark.parametrize("permission", [False, "false", "0", "", "unknown"])
def test_permission_disabled_never_writes(source: Path, tmp_path: Path, *, permission: bool | str) -> None:
    writer = create_profile_write_tool(skills_dir=source, writer=save_group_profile)
    assert writer.invoke(CONTENT, config=config(tmp_path, permission=permission)) == "error: write_capability_disabled"
    assert not (tmp_path / "users").exists()


def test_missing_authorization_and_identity_fail_closed(source: Path, tmp_path: Path) -> None:
    writer = create_profile_write_tool(skills_dir=source, writer=save_group_profile)
    untrusted = config(tmp_path)
    del untrusted["metadata"]["can_write_profile"]
    assert writer.invoke(CONTENT, config=untrusted) == "error: write_capability_disabled"
    missing = config(tmp_path)
    del missing["metadata"]["user_id"]
    assert writer.invoke(CONTENT, config=missing) == "error: missing trusted user_id or group_id"
    assert not (tmp_path / "users").exists()


def test_model_cannot_replace_owner_and_update_uses_same_profile(source: Path, tmp_path: Path) -> None:
    writer = create_profile_write_tool(skills_dir=source, writer=save_group_profile)
    assert {"user_id", "group_id", "run_id", "can_write_profile"}.isdisjoint(writer.tool_call_schema.model_fields)
    result = writer.invoke({**CONTENT, "user_id": "victim", "group_id": "g2", "can_write_profile": True}, config=config(tmp_path))
    assert result.startswith("ok:")
    assert writer.invoke({**CONTENT, "need": "产品销售渠道"}, config=config(tmp_path)).startswith("ok:")
    profile = load_profile(tmp_path, "owner", "g1")
    assert profile is not None
    assert profile.need.value == "产品销售渠道"
    assert load_profile(tmp_path, "victim", "g2") is None


def test_writer_schema_must_not_expose_identity(source: Path) -> None:
    @tool
    def save_group_profile(user_id: str, doing: str) -> str:
        """Unsafe host writer with model-selected identity."""
        del doing, user_id
        return "ok:"

    with pytest.raises(ValueError, match="inject identity"):
        create_profile_write_tool(skills_dir=source, writer=save_group_profile)


def test_removed_package_is_checked_again_before_writing(source: Path, tmp_path: Path) -> None:
    writer = create_profile_write_tool(skills_dir=source, writer=save_group_profile)
    (source / NAME / "package.json").unlink()
    assert writer.invoke(CONTENT, config=config(tmp_path)).startswith("error: write_skill_unavailable")
    assert not (tmp_path / "users").exists()
    with pytest.raises(ValueError, match="complete"):
        validate_installation(skills_dir=source)


def test_tampered_manifest_is_rejected(source: Path) -> None:
    manifest = source / NAME / "package.json"
    document = json.loads(manifest.read_text())
    document["tools"].append("delete_profile")
    manifest.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid installed"):
        create_profile_write_tool(skills_dir=source, writer=save_group_profile)


@pytest.mark.parametrize("status", ["created", "updated", "idempotent", "stale_ignored", "fence_rejected", "failure"])
def test_host_acknowledgements_are_preserved(source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str) -> None:
    monkeypatch.setenv("GROUP_AGENT_INTEGRATION", "http")
    calls: list[dict[str, Any]] = []

    def persist(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        if status == "failure":
            msg = "remote_rejected"
            raise ProfileHttpError(msg)
        return {"status": status, "profile_version": 1}

    monkeypatch.setattr("apps.group_agent_api.agent_factory.agent.persist_group_profile", persist)
    writer = create_profile_write_tool(skills_dir=source, writer=save_group_profile)
    result = writer.invoke(CONTENT, config=config(tmp_path))
    assert len(calls) == 1
    assert calls[0]["profile"].user_id == "owner"
    assert calls[0]["run_id"] == "test_run"
    if status in {"created", "updated", "idempotent"}:
        assert result.startswith("ok: saved profile to database")
        assert load_profile(tmp_path, "owner", "g1") is not None
    else:
        assert result.startswith("ok: profile_superseded" if status == "stale_ignored" else "error:")
        assert load_profile(tmp_path, "owner", "g1") is None


def test_duplicate_run_keeps_binding_and_accepts_idempotent_ack(source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROUP_AGENT_INTEGRATION", "http")
    bindings: list[tuple[str, str, str]] = []

    def persist(**kwargs: Any) -> dict[str, Any]:
        profile = kwargs["profile"]
        bindings.append((kwargs["run_id"], profile.user_id, profile.group_id))
        return {"status": "created" if len(bindings) == 1 else "idempotent", "profile_version": 1}

    monkeypatch.setattr("apps.group_agent_api.agent_factory.agent.persist_group_profile", persist)
    writer = create_profile_write_tool(skills_dir=source, writer=save_group_profile)
    assert writer.invoke(CONTENT, config=config(tmp_path)).startswith("ok:")
    assert writer.invoke(CONTENT, config=config(tmp_path)).startswith("ok:")
    assert bindings == [("test_run", "owner", "g1")] * 2
    profile = load_profile(tmp_path, "owner", "g1")
    assert profile is not None
    assert profile.profile_version == 1
