"""Offline regressions for independently composable member public reads."""

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from apps.group_agent_api.agent_factory import agent as factory
from apps.group_agent_api.agent_factory.read_skills import member_read_configuration
from apps.group_agent_api.agent_factory.search_tool import extract_search_this_turn
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.skills import _list_skills
from skillpacks.aihehuo import integration
from skillpacks.aihehuo.manage import install


@pytest.mark.parametrize(
    ("read", "write", "expected"),
    [
        (True, True, ["save_group_profile", "query_aihehuo_public_data"]),
        (True, False, ["query_aihehuo_public_data"]),
        (False, True, ["save_group_profile"]),
        (False, False, []),
    ],
)
def test_public_graph_composition(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    read: bool,
    write: bool,
    expected: list[str],
) -> None:
    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
    install(["get-ai-platform-help"], target=tmp_path / "skills" / "aihehuo")
    calls: list[dict[str, Any]] = []

    def capture(**kwargs: Any) -> object:
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(factory, "create_deep_agent", capture)
    checkpointer = object()
    factory.create_member_agent(
        runtime=tmp_path,
        model=object(),
        checkpointer=checkpointer,
        include_search=read,
        include_save=write,
    )
    configuration = calls[0]
    assert [tool.name for tool in configuration["tools"]] == expected
    assert configuration["checkpointer"] is checkpointer
    assert configuration["skills"] == (["/skills/aihehuo/"] if read else [])
    if read:
        assert "不能称为已验证的群成员匹配" in configuration["system_prompt"]
        discovered = _list_skills(configuration["backend"], configuration["skills"][0])
        assert [skill["name"] for skill in discovered] == ["get-ai-platform-help"]


def test_installed_selection_controls_public_resources(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
    source = tmp_path / "skills" / "aihehuo"
    install(["get-ai-cofounders", "get-ai-platform-help"], target=source)
    requested: list[str] = []

    def fetch(path: str, *, output_format: str) -> str:
        del output_format
        requested.append(path)
        return json.dumps({"data": [], "meta": {"has_more": False}})

    monkeypatch.setattr(integration, "fetch", fetch)
    tools, paths, _ = member_read_configuration(runtime=tmp_path, enabled=True)
    result = tools[0].invoke({"resource": "users", "q": "杭州技术合伙人"})
    assert json.loads(result)["data"] == []
    assert requested[0].startswith("/ai/users.json?q=")
    assert "not installed" in tools[0].invoke({"resource": "projects"})
    assert len(requested) == 1
    discovered = _list_skills(FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True), paths[0])
    assert {skill["name"] for skill in discovered} == {"get-ai-cofounders", "get-ai-platform-help"}


def test_public_configuration_fails_without_packages_and_rolls_back(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
    with pytest.raises(ValueError, match="Install at least one"):
        member_read_configuration(runtime=tmp_path, enabled=True)
    assert member_read_configuration(runtime=tmp_path, enabled=False) == ([], [], "")
    source = tmp_path / "skills" / "aihehuo"
    install(["get-ai-platform-help"], target=source)
    (source / "get-ai-platform-help" / "package.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid installed"):
        member_read_configuration(runtime=tmp_path, enabled=True)
    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "legacy")
    tools, paths, prompt = member_read_configuration(runtime=tmp_path, enabled=True)
    assert [tool.name for tool in tools] == ["search_candidates"]
    assert paths == []
    assert prompt == ""
    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "typo")
    with pytest.raises(ValueError, match="GROUP_AGENT_READ_MODE"):
        member_read_configuration(runtime=tmp_path, enabled=False)


def test_public_result_is_not_interpreted_as_group_match() -> None:
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "query_aihehuo_public_data",
                    "args": {"resource": "users"},
                    "id": "public1",
                }
            ],
        ),
        ToolMessage(
            content=json.dumps({"status": "matched", "candidates": [{"name": "公开资料"}]}),
            name="query_aihehuo_public_data",
            tool_call_id="public1",
        ),
    ]
    result = extract_search_this_turn(messages, 0)
    assert result.called is False
    assert result.candidates == []


def test_public_reads_are_not_granted_to_admin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
    install(["get-ai-platform-help"], target=tmp_path / "skills" / "aihehuo")
    calls: list[dict[str, Any]] = []

    def capture(**kwargs: Any) -> object:
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(factory, "create_deep_agent", capture)
    factory.create_agent(base_dir=tmp_path, model=object(), checkpointer=object())
    assert "query_aihehuo_public_data" in [tool.name for tool in calls[0]["tools"]]
    assert "query_aihehuo_public_data" not in [tool.name for tool in calls[1]["tools"]]
    assert calls[1].get("skills", []) == []


def test_query_failure_does_not_call_legacy_search(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
    install(["get-ai-platform-help"], target=tmp_path / "skills" / "aihehuo")

    def fail(path: str, *, output_format: str) -> str:
        del output_format, path
        msg = "HTTP 429"
        raise RuntimeError(msg)

    monkeypatch.setattr(integration, "fetch", fail)
    tools, _, _ = member_read_configuration(runtime=tmp_path, enabled=True)
    assert [tool.name for tool in tools] == ["query_aihehuo_public_data"]
    assert "HTTP 429" in tools[0].invoke({"resource": "faqs"})
