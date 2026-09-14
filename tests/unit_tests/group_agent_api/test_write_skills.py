"""Regression tests for independently installed member write capabilities."""

from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from starlette.requests import Request

from apps.group_agent_api.agent_factory import agent as factory
from apps.group_agent_api.agent_factory.profile_store import load_profile
from apps.group_agent_api.agent_factory.write_skills import member_write_configuration
from apps.group_agent_api.app.async_manager import _attempt_deterministic_profile_save
from apps.group_agent_api.app.session import resolve_trusted_session
from deepagents.middleware.skills import _list_skills
from skillpacks.aihehuo.manage import install as install_reads
from skillpacks.aihehuo_write.manage import install as install_write


@pytest.mark.parametrize(("read", "write"), [(True, True), (True, False), (False, True), (False, False)])
def test_read_and_write_packages_are_composed_independently(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, read: bool, write: bool) -> None:
    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
    monkeypatch.setenv("GROUP_AGENT_WRITE_MODE", "skillpack")
    install_reads(["get-ai-platform-help"], target=tmp_path / "skills" / "aihehuo")
    install_write(target=tmp_path / "skills" / "aihehuo_write")
    calls: list[dict[str, Any]] = []

    def capture(**kwargs: Any) -> object:
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(factory, "create_deep_agent", capture)
    factory.create_member_agent(
        runtime=tmp_path,
        model=object(),
        checkpointer=object(),
        include_search=read,
        include_save=write,
    )
    configuration = calls[0]
    if read and write:
        prompt = configuration["system_prompt"]
        assert "不声称已联系、保存" not in prompt
        assert "只有保存工具确认成功后才能确认画像已保存" in prompt
    expected_tools = (["save_group_profile"] if write else []) + (["query_aihehuo_public_data"] if read else [])
    assert [tool.name for tool in configuration["tools"]] == expected_tools
    expected_skills = ({"get-ai-platform-help"} if read else set()) | ({"save-ai-user-profile"} if write else set())
    discovered = {skill["name"] for path in configuration["skills"] for skill in _list_skills(configuration["backend"], path)}
    assert discovered == expected_skills


def test_missing_write_package_fails_closed_and_legacy_rolls_back(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GROUP_AGENT_WRITE_MODE", "skillpack")
    with pytest.raises(ValueError, match="complete"):
        member_write_configuration(runtime=tmp_path, enabled=True, writer=factory.save_group_profile)
    tools, paths, _ = member_write_configuration(runtime=tmp_path, enabled=False, writer=factory.save_group_profile)
    assert tools == paths == []
    monkeypatch.setenv("GROUP_AGENT_WRITE_MODE", "legacy")
    tools, paths, _ = member_write_configuration(runtime=tmp_path, enabled=True, writer=factory.save_group_profile)
    assert tools == [factory.save_group_profile]
    assert paths == []
    monkeypatch.setenv("GROUP_AGENT_WRITE_MODE", "unknown")
    with pytest.raises(ValueError, match="GROUP_AGENT_WRITE_MODE"):
        member_write_configuration(runtime=tmp_path, enabled=False, writer=factory.save_group_profile)


async def test_disabled_member_writes_remove_permission_tools_and_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GROUP_AGENT_INTEGRATION", "stub")
    monkeypatch.setenv("GROUP_AGENT_WRITE_MODE", "disabled")
    session = await resolve_trusted_session(
        Request({"type": "http", "method": "POST", "path": "/chat", "headers": []}),
        body_user_id="owner",
        body_group_id="g1",
        body_membership="in_group",
        body_unionid=None,
        body_group_token=None,
        body_user_token=None,
    )
    assert session.capabilities.can_search is True
    assert session.capabilities.can_write_profile is False
    tools, sources, _ = member_write_configuration(runtime=tmp_path, enabled=True, writer=factory.save_group_profile)
    assert tools == sources == []
    configuration = {"metadata": {"user_id": "owner", "group_id": "g1", "base_dir": str(tmp_path), "can_write_profile": "true"}}
    content = {"doing": "开发智能宠物喂食器", "need": "联网开发", "offer": "工厂与供应链"}
    assert factory.save_group_profile.invoke(content, config=configuration) == "error: write_capability_disabled"
    messages: list[Any] = []
    attempted = await _attempt_deterministic_profile_save(
        message="我在做智能宠物喂食器，缺联网与 App，有工厂与供应链",  # noqa: RUF001  # Chinese text intentionally uses Chinese punctuation.
        config=configuration,
        messages=messages,
    )
    assert attempted is False
    assert messages == []
    assert not (tmp_path / "users").exists()


async def test_deterministic_save_uses_installed_write_package(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GROUP_AGENT_INTEGRATION", "stub")
    monkeypatch.setenv("GROUP_AGENT_WRITE_MODE", "skillpack")
    install_write(target=tmp_path / "skills" / "aihehuo_write")
    configuration = {"metadata": {"user_id": "owner", "group_id": "g1", "base_dir": str(tmp_path), "can_write_profile": "true"}}
    messages: list[Any] = []
    assert await _attempt_deterministic_profile_save(
        message="doing:开发智能宠物喂食器; need:联网开发; offer:工厂与供应链",
        config=configuration,
        messages=messages,
    )
    assert str(messages[-1].content).startswith("ok:")
    assert load_profile(tmp_path, "owner", "g1") is not None


async def test_read_only_fallback_cannot_bypass_write_permission(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GROUP_AGENT_WRITE_MODE", "skillpack")
    messages: list[Any] = []
    attempted = await _attempt_deterministic_profile_save(
        message="doing:开发智能宠物喂食器; need:联网开发; offer:工厂与供应链",
        config={"metadata": {"base_dir": str(tmp_path), "can_write_profile": "false"}},
        messages=messages,
    )
    assert attempted is False
    assert messages == []
    assert not (tmp_path / "users").exists()


async def test_installed_write_tool_executes_in_real_member_graph(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GROUP_AGENT_INTEGRATION", "stub")
    monkeypatch.setenv("GROUP_AGENT_MODEL_MODE", "stub")
    monkeypatch.setenv("GROUP_AGENT_WRITE_MODE", "skillpack")
    install_write(target=tmp_path / "skills" / "aihehuo_write")
    graph = factory.create_member_agent(
        runtime=tmp_path,
        checkpointer=InMemorySaver(),
        include_search=False,
        include_save=True,
    )
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="我在做 AI startup，需要 Co-founder，可以提供 Python, product。请保存本人画像。")]},  # noqa: RUF001  # Chinese text intentionally uses Chinese punctuation.
        {
            "configurable": {"thread_id": "write-package-test"},
            "recursion_limit": 16,
            "metadata": {"user_id": "owner", "group_id": "g1", "base_dir": str(tmp_path), "can_write_profile": "true", "run_match": "false"},
        },
    )
    results = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert len(results) == 1
    assert results[0].name == "save_group_profile"
    assert str(results[0].content).startswith("ok:")
    assert load_profile(tmp_path, "owner", "g1") is not None
