from __future__ import annotations

# ruff: noqa: S101, ANN202, ANN001, RUF059, D103, PLR2004, TC003, PTH211, I001, D100, RUF100, TC002, FLY002, S108, RUF001, E501, PLR0915, SIM117

from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from apps.wu_tanchang_api.app.endpoints.chat import chat
from apps.wu_tanchang_api.app.models import ChatRequest
from apps.wu_tanchang_api.app.state import AppState
from apps.wu_tanchang_api.config import get_selected_provider, resolve_model_config
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import Field

from deepagents.graph import create_deep_agent
from deepagents.middleware.filesystem import FilesystemPermission
from deepagents.middleware.subagents import SubAgent


class _FakeCheckpointer:
    def __init__(self, checkpoint: dict[str, Any] | None = None) -> None:
        self.checkpoint = checkpoint

    async def aget(self, _config: dict[str, Any]) -> dict[str, Any] | None:
        return self.checkpoint


class _FakeAgent:
    def __init__(self, checkpointer: _FakeCheckpointer) -> None:
        self.checkpointer = checkpointer

    async def ainvoke(
        self, _input: dict[str, Any], _config: dict[str, Any]
    ) -> dict[str, Any]:
        return {"messages": []}


class _FakeModel:
    def __init__(self, profile: dict[str, Any] | None = None) -> None:
        self.profile = profile


def _state(
    checkpointer: _FakeCheckpointer,
    tmp_path: Path,
) -> AppState:
    agent = _FakeAgent(checkpointer)
    return AppState(
        agents={"default": agent},
        agent_configs={},
        default_agent="default",
        checkpoints_path=str(tmp_path / "checkpoints.pkl"),
        thread_locks={},
        backend_root=str(tmp_path),
    )


async def test_chat_calls_agent_normally_after_delivered(tmp_path: Path) -> None:
    from langchain_core.messages import ToolMessage

    checkpoint = {
        "channel_values": {
            "messages": [
                ToolMessage(
                    content="material_delivered",
                    name="mark_material_delivered",
                    tool_call_id="call_1",
                )
            ]
        }
    }
    checkpointer = _FakeCheckpointer(checkpoint)

    response = await chat(
        ChatRequest(user_id="u1", conversation_id="c1", message="继续聊"),
        _state(checkpointer, tmp_path),
    )

    # Should not short-circuit to _GUIDE_MESSAGE, but call the agent (which returns empty messages in FakeAgent)
    assert response.reply == ""


def test_agent_has_filesystem_middleware(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from apps.wu_tanchang_api.agent_factory import agent

    import deepagents._models

    captured: dict[str, Any] = {}

    def fake_create_model(**_kwargs: Any) -> _FakeModel:
        return _FakeModel(profile={})

    def fake_create_deep_agent(**_kwargs: Any) -> object:
        captured.update(_kwargs)
        return object()

    monkeypatch.setattr(deepagents._models, "resolve_model", lambda m: m)
    monkeypatch.setattr(agent, "create_model", fake_create_model)
    monkeypatch.setattr(agent, "create_deep_agent", fake_create_deep_agent)
    monkeypatch.setattr(agent, "default_runtime_dir", lambda: tmp_path / "runtime")

    (tmp_path / "workspace" / "skills" / "local").mkdir(parents=True, exist_ok=True)
    created_agent, checkpoints_path = agent.create_agent(
        backend_root=tmp_path, provider="deepseek"
    )

    assert created_agent is not None
    assert checkpoints_path == tmp_path / "runtime" / "checkpoints.pkl"

    kb_spec = next(
        spec for spec in captured["subagents"] if spec["name"] == "kb_analyst"
    )
    assert kb_spec["skills"] == ["/workspace/skills/local/"]


def test_config_resolves_env_references(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.json"
    env_path = tmp_path / ".env"
    config_path.write_text(
        """
{
  "model_provider": "env:WU_API_MODEL_PROVIDER",
  "default_model_provider": "openai_compatible",
  "providers": {
    "openai_compatible": {
      "api_type": "openai-compatible",
      "base_url": "env:WU_OPENAI_COMPATIBLE_BASE_URL",
      "api_key": "env:WU_OPENAI_COMPATIBLE_API_KEY",
      "main_agent_model": "env:WU_OPENAI_COMPATIBLE_MAIN_AGENT_MODEL",
      "max_tokens": "env:WU_OPENAI_COMPATIBLE_MAX_TOKENS",
      "timeout_s": "env:WU_OPENAI_COMPATIBLE_TIMEOUT_S",
      "temperature": "env:WU_OPENAI_COMPATIBLE_TEMPERATURE",
      "max_input_tokens": "env:WU_OPENAI_COMPATIBLE_MAX_INPUT_TOKENS"
    }
  }
}
""",
        encoding="utf-8",
    )
    env_path.write_text(
        "\n".join(
            [
                "WU_API_MODEL_PROVIDER=openai_compatible",
                "WU_OPENAI_COMPATIBLE_BASE_URL=https://example.test/v1",
                "WU_OPENAI_COMPATIBLE_API_KEY=test-key",
                "WU_OPENAI_COMPATIBLE_MAIN_AGENT_MODEL=test-main",
                "WU_OPENAI_COMPATIBLE_MAX_TOKENS=1234",
                "WU_OPENAI_COMPATIBLE_TIMEOUT_S=12.5",
                "WU_OPENAI_COMPATIBLE_TEMPERATURE=0.4",
                "WU_OPENAI_COMPATIBLE_MAX_INPUT_TOKENS=9999",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("WU_API_CONFIG", str(config_path))
    monkeypatch.setenv("WU_API_ENV_FILE", str(env_path))

    # Clean any pre-existing env vars that would override the .env file loading
    for key in [
        "WU_API_MODEL_PROVIDER",
        "WU_OPENAI_COMPATIBLE_BASE_URL",
        "WU_OPENAI_COMPATIBLE_API_KEY",
        "WU_OPENAI_COMPATIBLE_MAIN_AGENT_MODEL",
        "WU_OPENAI_COMPATIBLE_MAX_TOKENS",
        "WU_OPENAI_COMPATIBLE_TIMEOUT_S",
        "WU_OPENAI_COMPATIBLE_TEMPERATURE",
        "WU_OPENAI_COMPATIBLE_MAX_INPUT_TOKENS",
    ]:
        monkeypatch.delenv(key, raising=False)

    assert get_selected_provider() == "openai_compatible"

    model_config = resolve_model_config(
        provider="openai_compatible", model_name_suffix="MAIN_AGENT_MODEL"
    )
    assert model_config.api_type == "openai-compatible"
    assert model_config.base_url == "https://example.test/v1"
    assert model_config.api_key == "test-key"
    assert model_config.model == "test-main"
    assert model_config.max_tokens == 1234
    assert model_config.timeout_s == 12.5
    assert model_config.temperature == 0.4
    assert model_config.max_input_tokens == 9999


def test_create_model_dispatches_by_api_type(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from apps.wu_tanchang_api.agent_factory import model_builder

    calls: list[tuple[str, dict[str, Any]]] = []

    class FakeChatModel:
        def __init__(self, **kwargs: Any) -> None:
            self.profile: dict[str, Any] = {}
            self.kwargs = kwargs

    class FakeOpenAI(FakeChatModel):
        def __init__(self, **kwargs: Any) -> None:
            calls.append(("openai", kwargs))
            super().__init__(**kwargs)

    class FakeAnthropic(FakeChatModel):
        def __init__(self, **kwargs: Any) -> None:
            calls.append(("anthropic", kwargs))
            super().__init__(**kwargs)

    config_path = tmp_path / "config.json"
    config_path.write_text(
        """
{
  "default_model_provider": "openai_compatible",
  "providers": {
    "openai_compatible": {
      "api_type": "openai-compatible",
      "base_url": "https://openai-compatible.test/v1",
      "api_key": "openai-key",
      "main_agent_model": "openai-main",
      "max_input_tokens": 111
    },
    "anthropic_compatible": {
      "api_type": "anthropic-compatible",
      "base_url": "https://anthropic-compatible.test",
      "api_key": "anthropic-key",
      "main_agent_model": "anthropic-main",
      "max_input_tokens": 222
    }
  }
}
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("WU_API_CONFIG", str(config_path))
    monkeypatch.setenv("WU_API_ENV_FILE", str(tmp_path / "missing.env"))
    monkeypatch.setattr(model_builder, "ChatOpenAI", FakeOpenAI)
    monkeypatch.setattr(model_builder, "ChatAnthropic", FakeAnthropic)

    openai_model = model_builder.create_model(
        provider="openai_compatible", model_name_suffix="MAIN_AGENT_MODEL"
    )
    anthropic_model = model_builder.create_model(
        provider="anthropic_compatible", model_name_suffix="MAIN_AGENT_MODEL"
    )

    assert calls[0][0] == "openai"
    assert calls[0][1]["model"] == "openai-main"
    assert openai_model.profile["max_input_tokens"] == 111
    assert calls[1][0] == "anthropic"
    assert calls[1][1]["model"] == "anthropic-main"
    assert anthropic_model.profile["max_input_tokens"] == 222


def test_default_config_uses_model_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WU_API_CONFIG", raising=False)
    monkeypatch.delenv("WU_API_MODEL_PROVIDER", raising=False)
    monkeypatch.delenv("WU_DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("WU_DASHSCOPE_MAIN_AGENT_MODEL", raising=False)
    monkeypatch.setenv("WU_API_ENV_FILE", "/tmp/missing-wu-api.env")

    assert get_selected_provider() == "qwen"

    model_config = resolve_model_config(
        provider="qwen", model_name_suffix="MAIN_AGENT_MODEL"
    )
    assert model_config.model == "qwen-flash"
    assert model_config.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert model_config.api_key is None
    assert model_config.max_input_tokens == 1000000


def test_task_tool_blocks_kb_analyst_after_delivered() -> None:
    from unittest.mock import MagicMock

    from langchain.tools import ToolRuntime
    from langchain_core.messages import ToolMessage

    from deepagents.middleware.subagents import _build_task_tool

    # Create dummy specs
    mock_runnable = MagicMock()
    mock_runnable.with_config.return_value = mock_runnable
    mock_runnable.invoke.return_value = {"messages": [MagicMock()]}
    spec = {"name": "kb_analyst", "description": "KB", "runnable": mock_runnable}

    # Build the task tool
    tool = _build_task_tool([spec])

    # Case 1: Materials not delivered yet
    mock_runtime_ok = MagicMock(spec=ToolRuntime)
    mock_runtime_ok.state = {"messages": []}
    mock_runtime_ok.tool_call_id = "call-1"
    mock_runtime_ok.config = {}
    # Should attempt to execute and not return the block string
    res_ok = tool.func(
        description="hello", subagent_type="kb_analyst", runtime=mock_runtime_ok
    )
    assert (
        res_ok
        != "知识库已经检索过且会议准备材料已完成交付。禁止再次调用 kb_analyst 或检索知识库。请直接根据已交付的材料和对话历史回答用户的问题，并引导用户预约吴探长一对一深聊。"
    )

    # Case 2: Materials delivered
    mock_runtime_blocked = MagicMock(spec=ToolRuntime)
    delivered_msg = ToolMessage(
        content="done", name="mark_material_delivered", tool_call_id="tool-1"
    )
    mock_runtime_blocked.state = {"messages": [delivered_msg]}
    mock_runtime_blocked.tool_call_id = "call-2"
    mock_runtime_blocked.config = {}

    res_blocked = tool.func(
        description="hello", subagent_type="kb_analyst", runtime=mock_runtime_blocked
    )
    assert (
        res_blocked
        == "知识库已经检索过且会议准备材料已完成交付。禁止再次调用 kb_analyst 或检索知识库。请直接根据已交付的材料和对话历史回答用户的问题，并引导用户预约吴探长一对一深聊。"
    )

    # Case 3: Materials delivered (YC Workspace)
    mock_runtime_blocked_yc = MagicMock(spec=ToolRuntime)
    mock_runtime_blocked_yc.state = {"messages": [delivered_msg]}
    mock_runtime_blocked_yc.tool_call_id = "call-3"
    mock_runtime_blocked_yc.config = {
        "configurable": {"thread_id": "wt::yc01::u1::c1"},
        "metadata": {"agent_name": "yc01"},
    }

    res_blocked_yc = tool.func(
        description="hello", subagent_type="kb_analyst", runtime=mock_runtime_blocked_yc
    )
    assert (
        res_blocked_yc
        == "知识库已经检索过且会议准备材料已完成交付。禁止再次调用 kb_analyst 或检索知识库。请直接根据已交付的材料和对话历史回答用户的问题，并引导用户预约YC老师一对一深聊。"
    )


def test_accountant_middleware_resets_tool_count_per_turn() -> None:
    from types import SimpleNamespace

    from langchain_core.messages import ToolMessage
    from langgraph.types import Command

    from deepagents.middleware.accountant import AccountantMiddleware

    state = {"tool_call_count": 1, "accountant_turn_id": "turn-1"}
    runtime = SimpleNamespace(
        state=state,
        config={"configurable": {"deepagents_turn_id": "turn-2"}},
    )
    request = SimpleNamespace(
        runtime=runtime,
        tool_call={"name": "sample_tool", "id": "call-1"},
    )

    middleware = AccountantMiddleware(max_tool_calls=1)

    def handler(_request: object) -> ToolMessage:
        return ToolMessage(content="ok", tool_call_id="call-1")

    result = middleware.wrap_tool_call(request, handler)

    assert isinstance(result, Command)
    assert result.update["tool_call_count"] == 1
    assert result.update["accountant_turn_id"] == "turn-2"
    assert state["tool_call_count"] == 0
    assert state["accountant_turn_id"] == "turn-2"


def test_accountant_middleware_ignores_metadata_turn_id() -> None:
    from types import SimpleNamespace

    from langchain_core.messages import ToolMessage

    from deepagents.middleware.accountant import AccountantMiddleware

    state = {"tool_call_count": 1, "accountant_turn_id": "turn-1"}
    runtime = SimpleNamespace(
        state=state,
        config={"metadata": {"deepagents_turn_id": "turn-2"}},
    )
    request = SimpleNamespace(
        runtime=runtime,
        tool_call={"name": "sample_tool", "id": "call-1"},
    )

    middleware = AccountantMiddleware(max_tool_calls=1)

    def handler(_request: object) -> ToolMessage:
        return ToolMessage(content="ok", tool_call_id="call-1")

    result = middleware.wrap_tool_call(request, handler)

    assert isinstance(result, ToolMessage)
    assert "Tool call limit exceeded" in str(result.content)
    assert state["tool_call_count"] == 1
    assert state["accountant_turn_id"] == "turn-1"


def test_ensure_runtime_workspace_isolation(tmp_path: Path) -> None:
    import os

    from apps.wu_tanchang_api.agent_factory.utils import ensure_runtime_workspace

    # 1. Setup source workspace structure
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "kb").mkdir()
    (src_dir / "kb" / "doc.txt").write_text("kb source", encoding="utf-8")

    default_skill_dir = (
        tmp_path / "skills" / "default" / "kb_analyst" / "wu-tanchang-kb"
    )
    default_skill_dir.mkdir(parents=True)
    (default_skill_dir / "SKILL.md").write_text(
        "Default skill referencing kb/index.json and kb/METHOD.md", encoding="utf-8"
    )

    # Add a custom workspace folder
    custom_ws = tmp_path / "workspace_custom"
    custom_ws.mkdir()
    (custom_ws / "identity.md").write_text("identity source", encoding="utf-8")
    (custom_ws / "kb").mkdir()
    (custom_ws / "kb" / "doc.txt").write_text("tenant kb source", encoding="utf-8")
    local_skill_dir = custom_ws / "skills" / "local" / "custom-skill"
    local_skill_dir.mkdir(parents=True)
    (local_skill_dir / "SKILL.md").write_text(
        "Local skill referencing kb/local.md", encoding="utf-8"
    )

    # 2. Deploy to runtime
    runtime_dir = tmp_path / "runtime"
    ensure_runtime_workspace(workspace_src=src_dir, runtime_dir=runtime_dir)

    # 3. Assert target directory existence and content
    # Root kb/skills must NOT exist anymore
    assert not (runtime_dir / "kb").exists()
    assert not (runtime_dir / "skills").exists()

    assert (runtime_dir / "workspace_custom" / "identity.md").read_text(
        encoding="utf-8"
    ) == "identity source"

    # Assert default and local skills are copied and paths are dynamically formatted to /workspace_custom/kb/
    default_skill_content = (
        runtime_dir
        / "workspace_custom"
        / "skills"
        / "default"
        / "kb_analyst"
        / "wu-tanchang-kb"
        / "SKILL.md"
    ).read_text(encoding="utf-8")
    assert (
        "Default skill referencing /workspace_custom/kb/index.json and /workspace_custom/kb/METHOD.md"
        in default_skill_content
    )

    local_skill_content = (
        runtime_dir
        / "workspace_custom"
        / "skills"
        / "local"
        / "custom-skill"
        / "SKILL.md"
    ).read_text(encoding="utf-8")
    assert (
        "Local skill referencing /workspace_custom/kb/local.md" in local_skill_content
    )
    assert not (
        runtime_dir / "workspace_custom" / "skills" / "default" / "default"
    ).exists()
    assert not (
        runtime_dir / "workspace_custom" / "skills" / "local" / "local"
    ).exists()

    # Verify symlink status (if symlinks are supported by filesystem/OS)
    try:
        os.symlink(src_dir / "kb" / "doc.txt", tmp_path / "test_link")
        symlink_supported = True
    except (OSError, PermissionError):
        symlink_supported = False

    if symlink_supported:
        # Custom workspaces must NEVER be symlinks to preserve write isolation
        assert not (runtime_dir / "workspace_custom").is_symlink()
        # Custom workspace kb should be a symlink to custom_ws/kb
        assert (runtime_dir / "workspace_custom" / "kb").is_symlink()
        # Custom workspace skills must be a directory (since it is a copy)
        assert not (runtime_dir / "workspace_custom" / "skills").is_symlink()

    # 4. Modify files in runtime and check isolation
    # Note: custom workspace files can be modified and must be isolated.
    (runtime_dir / "workspace_custom" / "identity.md").write_text(
        "identity modified", encoding="utf-8"
    )

    # Assert runtime changed
    assert (runtime_dir / "workspace_custom" / "identity.md").read_text(
        encoding="utf-8"
    ) == "identity modified"

    # Assert source did NOT change!
    assert (custom_ws / "identity.md").read_text(encoding="utf-8") == "identity source"


def test_ensure_runtime_workspace_removes_old_skills_symlink(tmp_path: Path) -> None:
    import os
    from apps.wu_tanchang_api.agent_factory.utils import ensure_runtime_workspace

    # 1. Setup source workspace structure
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "kb").mkdir()

    # Add a custom workspace folder
    custom_ws = tmp_path / "workspace_custom"
    custom_ws.mkdir()
    (custom_ws / "identity.md").write_text("identity source", encoding="utf-8")

    # 2. Setup target runtime dir with a pre-existing symlink for skills
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    tenant_runtime_dir = runtime_dir / "workspace_custom"
    tenant_runtime_dir.mkdir()
    
    # Create a pre-existing symlink for dest/skills pointing to some other directory
    dummy_dir = tmp_path / "dummy_dir"
    dummy_dir.mkdir()
    os.symlink(dummy_dir, tenant_runtime_dir / "skills")
    
    assert (tenant_runtime_dir / "skills").is_symlink()

    # 3. Deploy to runtime
    ensure_runtime_workspace(workspace_src=src_dir, runtime_dir=runtime_dir)

    # 4. Assert that skills is no longer a symlink but a directory
    assert not (tenant_runtime_dir / "skills").is_symlink()
    assert (tenant_runtime_dir / "skills").is_dir()


def test_task_tool_blocks_kb_analyst_after_delivered_dynamic_owner() -> None:
    import json
    from unittest.mock import MagicMock

    from langchain.tools import ToolRuntime
    from langchain_core.messages import ToolMessage

    from deepagents.backends.protocol import BackendProtocol
    from deepagents.middleware.subagents import _build_task_tool

    # Create dummy specs
    mock_runnable = MagicMock()
    mock_runnable.with_config.return_value = mock_runnable
    mock_runnable.invoke.return_value = {"messages": [MagicMock()]}
    spec = {"name": "kb_analyst", "description": "KB", "runnable": mock_runnable}

    # 1. Test reading from workspace subdirectory based on MEMORY.md mapping
    # Mock backend
    mock_backend = MagicMock(spec=BackendProtocol)

    # ls("/") returns workspace directories
    mock_backend.ls.return_value = MagicMock(
        entries=[{"path": "/workspace"}, {"path": "/workspace_1"}]
    )

    # Mock read files
    def mock_read(path):
        from deepagents.backends.protocol import ReadResult

        if path == "/workspace_1/MEMORY.md":
            return ReadResult(
                file_data={"content": "- Agent id: yc01", "encoding": "utf-8"}
            )
        if path == "/workspace/MEMORY.md":
            return ReadResult(
                file_data={"content": "- Agent id: andy01", "encoding": "utf-8"}
            )
        if path == "/workspace_1/owner.json":
            return ReadResult(
                file_data={
                    "content": json.dumps({"owner_name": "动态YC老师"}),
                    "encoding": "utf-8",
                }
            )
        if path == "/workspace/owner.json":
            return ReadResult(
                file_data={
                    "content": json.dumps({"owner_name": "动态吴探长"}),
                    "encoding": "utf-8",
                }
            )
        if path == "/owner.json":
            return ReadResult(
                file_data={
                    "content": json.dumps({"owner_name": "根目录Owner"}),
                    "encoding": "utf-8",
                }
            )
        return ReadResult(error="file not found")

    mock_backend.read.side_effect = mock_read

    # Build the task tool with the mock backend
    tool = _build_task_tool([spec], backend=mock_backend)

    # Setup runtime with delivered message and YC metadata
    delivered_msg = ToolMessage(
        content="done", name="mark_material_delivered", tool_call_id="tool-1"
    )
    mock_runtime = MagicMock(spec=ToolRuntime)
    mock_runtime.state = {"messages": [delivered_msg]}
    mock_runtime.tool_call_id = "call-1"
    mock_runtime.config = {
        "configurable": {"thread_id": "wt::yc01::u1::c1"},
        "metadata": {"agent_name": "yc01"},
    }

    res = tool.func(
        description="hello", subagent_type="kb_analyst", runtime=mock_runtime
    )
    assert "动态YC老师" in res

    # 2. Test reading from default /owner.json
    mock_backend_default = MagicMock(spec=BackendProtocol)
    mock_backend_default.ls.side_effect = Exception("ls not implemented")
    mock_backend_default.read.side_effect = mock_read

    tool_default = _build_task_tool([spec], backend=mock_backend_default)
    mock_runtime_default = MagicMock(spec=ToolRuntime)
    mock_runtime_default.state = {"messages": [delivered_msg]}
    mock_runtime_default.tool_call_id = "call-2"
    mock_runtime_default.config = {
        "configurable": {"thread_id": "wt::andy01::u1::c1"},
        "metadata": {"agent_name": "andy01"},
    }

    res_default = tool_default.func(
        description="hello", subagent_type="kb_analyst", runtime=mock_runtime_default
    )
    assert "根目录Owner" in res_default


def test_create_agent_loads_local_skills_dynamically(tmp_path: Path) -> None:
    from unittest.mock import patch

    from apps.wu_tanchang_api.agent_factory.agent import create_agent
    from apps.wu_tanchang_api.config import WuAgentConfig

    # 1. Setup workspace folders with a local skill
    workspace_dir = tmp_path / "workspace_test"
    workspace_dir.mkdir()
    (workspace_dir / "skills").mkdir()
    (workspace_dir / "skills" / "local").mkdir()
    (workspace_dir / "skills" / "local" / "test-skill").mkdir()
    (workspace_dir / "skills" / "local" / "test-skill" / "SKILL.md").write_text(
        "---\nname: test-skill\ndescription: Test\n---\n", encoding="utf-8"
    )
    for name in ["IDENTITY.md", "SOUL.md", "AGENTS.md", "USER.md"]:
        (workspace_dir / name).write_text("dummy", encoding="utf-8")

    # Mock default_runtime_dir to point to tmp_path
    with (
        patch(
            "apps.wu_tanchang_api.agent_factory.agent.default_runtime_dir",
            return_value=tmp_path / "runtime",
        ),
        patch(
            "apps.wu_tanchang_api.agent_factory.agent.create_deep_agent"
        ) as mock_create,
    ):
        cfg = WuAgentConfig(
            name="test_agent",
            model="qwen-flash",
            provider="qwen",
            max_tokens=800,
            workspace="workspace_test",
        )
        create_agent(backend_root=tmp_path, agent_config=cfg)

        # Check that mock_create was called with subagents having the local skill path
        args, kwargs = mock_create.call_args
        subagents = kwargs.get("subagents", [])
        kb_sub = next(s for s in subagents if s["name"] == "kb_analyst")

        assert "/workspace_test/skills/local/" in kb_sub["skills"]


def test_strict_multi_tenant_isolation(tmp_path: Path) -> None:
    """Test strict multi-tenant database isolation, fail-closed connection routing, and permissions."""
    from apps.wu_tanchang_api.agent_factory.kb_search import (
        get_note_content,
        _clients_cache,
    )
    from apps.wu_tanchang_api.agent_factory.agent import (
        register_active_agent,
        unregister_active_agent,
        create_agent,
    )
    from apps.wu_tanchang_api.config import WuAgentConfig
    import sqlite3
    import os
    from unittest.mock import MagicMock, patch

    # 1. Setup two tenant directories in tmp_path/runtime
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    ws_a = runtime_dir / "workspace_a"
    ws_b = runtime_dir / "workspace_b"
    for ws in (ws_a, ws_b):
        ws.mkdir()
        (ws / "kb").mkdir()
        (ws / "kb" / "chroma").mkdir()
        (ws / "kb" / "chroma" / "dummy.bin").write_text("dummy", encoding="utf-8")

    # 2. Initialize tenant A SQLite database
    db_a_path = ws_a / "kb" / "kb.db"
    conn_a = sqlite3.connect(db_a_path)
    conn_a.execute(
        "CREATE TABLE notes (id TEXT PRIMARY KEY, title TEXT, brand TEXT, content TEXT, raw_path TEXT);"
    )
    conn_a.execute(
        "INSERT INTO notes VALUES ('a01', 'Brand A Title', 'Brand A', 'Content of Tenant A', 'a.md');"
    )
    conn_a.commit()
    conn_a.close()

    # 3. Initialize tenant B SQLite database
    db_b_path = ws_b / "kb" / "kb.db"
    conn_b = sqlite3.connect(db_b_path)
    conn_b.execute(
        "CREATE TABLE notes (id TEXT PRIMARY KEY, title TEXT, brand TEXT, content TEXT, raw_path TEXT);"
    )
    conn_b.execute(
        "INSERT INTO notes VALUES ('b01', 'Brand B Title', 'Brand B', 'Content of Tenant B', 'b.md');"
    )
    conn_b.commit()
    conn_b.close()

    # 4. Create mock active agents
    class MockBackend:
        def __init__(self, root_dir: str) -> None:
            self.root_dir = root_dir

    class MockAgent:
        def __init__(self, workspace_name: str, backend_root: str) -> None:
            self.workspace_name = workspace_name
            self.backend = MockBackend(backend_root)

    agent_a = MockAgent("workspace_a", str(runtime_dir))
    agent_b = MockAgent("workspace_b", str(runtime_dir))

    register_active_agent("thread_a", agent_a)
    register_active_agent("thread_b", agent_b)

    # Clear clients cache and environment keys to ensure clean routing
    _clients_cache.clear()
    with patch.dict("os.environ", {}):
        if "WU_KB_DB_PATH" in os.environ:
            del os.environ["WU_KB_DB_PATH"]

        try:
            # 5. Patch chromadb to avoid external dependencies
            with patch("chromadb.PersistentClient") as mock_chroma:
                mock_chroma.return_value.get_collection.return_value = MagicMock()

                # Test A: tenant A accesses its own note
                res_a_ok = get_note_content.invoke(
                    {"note_id": "a01"},
                    config={"configurable": {"thread_id": "thread_a"}},
                )
                assert "Content of Tenant A" in res_a_ok

                # Test B: tenant A tries to access tenant B's note (should not find it)
                res_a_fail = get_note_content.invoke(
                    {"note_id": "b01"},
                    config={"configurable": {"thread_id": "thread_a"}},
                )
                assert "错误: 未找到 ID 为 'b01' 的笔记" in res_a_fail

                # Test C: tenant B accesses its own note
                res_b_ok = get_note_content.invoke(
                    {"note_id": "b01"},
                    config={"configurable": {"thread_id": "thread_b"}},
                )
                assert "Content of Tenant B" in res_b_ok

                # Test D: tenant B tries to access tenant A's note (should not find it)
                res_b_fail = get_note_content.invoke(
                    {"note_id": "a01"},
                    config={"configurable": {"thread_id": "thread_b"}},
                )
                assert "错误: 未找到 ID 为 'a01' 的笔记" in res_b_fail

                # Test E: Fail Closed on missing thread_id / active agent session
                res_fail_closed = get_note_content.invoke(
                    {"note_id": "a01"},
                    config={"configurable": {"thread_id": "missing_thread"}},
                )
                assert "Multi-tenant routing error" in res_fail_closed

        finally:
            # Cleanup registry and cache
            unregister_active_agent("thread_a")
            unregister_active_agent("thread_b")
            _clients_cache.clear()

    # 6. Test subagent permissions are correctly configured to deny writes to kb/skills and deny all access to /kb/
    # Setup dummy persona files for create_agent
    (tmp_path / "IDENTITY.md").write_text("dummy", encoding="utf-8")
    (tmp_path / "SOUL.md").write_text("dummy", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("dummy", encoding="utf-8")
    (tmp_path / "USER.md").write_text("dummy", encoding="utf-8")

    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"default_model_provider": "qwen", "providers": {"qwen": {"api_type": "openai-compatible", "base_url": "https://dashscope.test/v1", "api_key": "fake-key", "main_agent_model": "qwen-flash"}}}',
        encoding="utf-8",
    )

    with patch(
        "apps.wu_tanchang_api.agent_factory.agent.create_deep_agent"
    ) as mock_create_agent:
        with patch.dict("os.environ", {"WU_API_CONFIG": str(config_path)}):
            cfg = WuAgentConfig(
                name="test_permissions",
                model="qwen-flash",
                provider="qwen",
                max_tokens=800,
                workspace="workspace_permissions",
            )
            create_agent(backend_root=tmp_path, agent_config=cfg)

            # Assert permissions are correctly added to kb_analyst
            args, kwargs = mock_create_agent.call_args
            subagents = kwargs.get("subagents", [])
            kb_sub = next(s for s in subagents if s["name"] == "kb_analyst")
            assert "permissions" in kb_sub

            perms = kb_sub["permissions"]
            # We expect two rules:
            # 1. Deny read/write on /kb/**
            # 2. Deny write on /workspace*/kb/**, /workspace*/skills/**, /skills/**
            rule_kb_deny = next(p for p in perms if "/kb/**" in p.paths)
            assert rule_kb_deny.mode == "deny"
            assert "read" in rule_kb_deny.operations
            assert "write" in rule_kb_deny.operations

            rule_write_deny = next(p for p in perms if "/workspace*/kb/**" in p.paths)
            assert rule_write_deny.mode == "deny"
            assert "write" in rule_write_deny.operations
            assert "read" not in rule_write_deny.operations

            # Validate the rules behavior using the SDK helper
            from deepagents.middleware.filesystem import _check_fs_permission

            assert _check_fs_permission(perms, "read", "/kb/index.json") == "deny"
            assert _check_fs_permission(perms, "write", "/kb/index.json") == "deny"
            assert (
                _check_fs_permission(perms, "write", "/workspace_aihehuo/kb/kb.db")
                == "deny"
            )
            assert (
                _check_fs_permission(
                    perms, "write", "/workspace_1/skills/local/skill.py"
                )
                == "deny"
            )
            assert (
                _check_fs_permission(
                    perms, "write", "/skills/default/kb_analyst/wu-tanchang-kb/SKILL.md"
                )
                == "deny"
            )
            assert (
                _check_fs_permission(perms, "read", "/workspace_aihehuo/kb/METHOD.md")
                == "allow"
            )
            assert (
                _check_fs_permission(
                    perms, "read", "/workspace_1/skills/local/yc-kb/SKILL.md"
                )
                == "allow"
            )


class _FixedFakeChatModel(GenericFakeChatModel):
    """Fixed version of GenericFakeChatModel that properly handles bind_tools.

    Without excluding `messages` from pydantic serialization, LangSmith tracing
    (which dumps the model via `model_dump(mode="json")`) consumes the iterator
    before `_generate` pulls from it.
    """

    messages: Iterator[AIMessage | str] = Field(exclude=True)

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Override bind_tools to return self."""
        return self


def test_compiled_subagent_tool_stack_permissions() -> None:
    """Validate filesystem permission enforcement through the real SDK tool stack.

    Uses ``create_deep_agent`` with a ``kb_analyst`` subagent that mirrors the
    production permission rules (deny read/write on ``/kb/**``, deny write on
    ``/workspace*/kb/**``, ``/workspace*/skills/**``, ``/skills/**``).

    The subagent model attempts three file operations:
      - ``read_file``  on ``/workspace_test/kb/METHOD.md``  → allowed
      - ``write_file`` on ``/workspace_test/kb/new.txt``    → denied
      - ``read_file``  on ``/kb/index.json``                → denied

    Each tool call is a separate "turn" for the fake model so the sequential
    tool-response loop works correctly with ``_FixedFakeChatModel``.
    """
    # -- Subagent model: 3 tool calls (each followed by seeing the response),
    #    then a final summary.
    subagent_model = _FixedFakeChatModel(
        messages=iter(
            [
                # Turn 1: read_file on allowed path
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "read_file",
                            "args": {"file_path": "/workspace_test/kb/METHOD.md"},
                            "id": "sub_call_1",
                            "type": "tool_call",
                        }
                    ],
                ),
                # Turn 2: write_file on denied path (workspace kb write)
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "write_file",
                            "args": {
                                "file_path": "/workspace_test/kb/new.txt",
                                "content": "should not be written",
                            },
                            "id": "sub_call_2",
                            "type": "tool_call",
                        }
                    ],
                ),
                # Turn 3: read_file on denied path (/kb/**)
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "read_file",
                            "args": {"file_path": "/kb/index.json"},
                            "id": "sub_call_3",
                            "type": "tool_call",
                        }
                    ],
                ),
                # Final response summarising outcomes
                AIMessage(
                    content=(
                        "read /workspace_test/kb/METHOD.md succeeded; "
                        "write /workspace_test/kb/new.txt permission denied; "
                        "read /kb/index.json permission denied"
                    ),
                ),
            ]
        )
    )

    # -- Parent model: delegate to kb_analyst, then finish.
    parent_model = _FixedFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "task",
                            "args": {
                                "description": "Check KB files",
                                "subagent_type": "kb_analyst",
                            },
                            "id": "call_task",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="Done."),
            ]
        )
    )

    kb_analyst: SubAgent = {
        "name": "kb_analyst",
        "description": "Knowledge base analyst subagent.",
        "system_prompt": "You analyse knowledge base files.",
        "model": subagent_model,
        "permissions": [
            # Deny all access to the root /kb/ directory
            FilesystemPermission(
                operations=["read", "write"], paths=["/kb/**"], mode="deny"
            ),
            # Deny writes to workspace kb and skills directories
            FilesystemPermission(
                operations=["write"],
                paths=["/workspace*/kb/**", "/workspace*/skills/**", "/skills/**"],
                mode="deny",
            ),
        ],
    }

    agent = create_deep_agent(
        model=parent_model,
        subagents=[kb_analyst],
    )
    result = agent.invoke(
        {"messages": [HumanMessage(content="Check the knowledge base files")]}
    )

    # Collect all messages produced during the run
    all_messages = result["messages"]
    tool_messages = [msg for msg in all_messages if msg.type == "tool"]

    # The parent should see a `task` ToolMessage whose content is the
    # subagent's final AIMessage — which mentions the denials.
    task_result = next(m for m in tool_messages if m.name == "task")
    assert "permission denied" in task_result.content

    # The SDK no longer exposes subagent-internal tool messages in the parent
    # message history. Detailed rule matching is covered above via
    # _check_fs_permission; this integration assertion verifies the parent sees
    # the permission-aware subagent result through the task tool.
    assert "write /workspace_test/kb/new.txt permission denied" in task_result.content
    assert "read /kb/index.json permission denied" in task_result.content


def test_run_aihehuo_skill_script_validation(tmp_path) -> None:
    """Test validation of run_aihehuo_skill_script tool."""
    from unittest.mock import patch
    from apps.wu_tanchang_api.agent_factory.agent import run_aihehuo_skill_script

    # Setup dummy script structure inside tmp_path so it passes the file exists check
    skill_dir = tmp_path / "workspace" / "skills" / "local_aihehuo" / "get-ai-blog"
    skill_dir.mkdir(parents=True, exist_ok=True)
    script_path = skill_dir / "run.py"
    script_path.write_text("print('ok')\n", encoding="utf-8")

    # Test invalid skill name validation (does not require script to exist as it fails early)
    res = run_aihehuo_skill_script.func(
        skill_name="get-ai-blog; rm -rf /",
        arguments=[],
        config={"configurable": {"thread_id": "test-thread"}},
    )
    assert "错误：不支持的技能名称" in res

    # Test invalid arguments validation (requires script to exist, so patch runtime dir)
    with patch("apps.wu_tanchang_api.agent_factory.utils.default_runtime_dir", return_value=tmp_path):
        res = run_aihehuo_skill_script.func(
            skill_name="get-ai-blog",
            arguments=["--page", "1; rm -rf /"],
            config={"configurable": {"thread_id": "test-thread"}},
        )
        assert "错误：参数中包含非法字符" in res


def test_run_aihehuo_skill_script_execution(tmp_path) -> None:
    """Test execution of run_aihehuo_skill_script tool with a dummy script."""
    from unittest.mock import patch
    from apps.wu_tanchang_api.agent_factory.agent import run_aihehuo_skill_script

    # Setup dummy script structure inside tmp_path:
    # tmp_path / "workspace" / "skills" / "local_aihehuo" / "get-ai-blog" / "run.py"
    skill_dir = tmp_path / "workspace" / "skills" / "local_aihehuo" / "get-ai-blog"
    skill_dir.mkdir(parents=True, exist_ok=True)
    script_path = skill_dir / "run.py"

    script_path.write_text(
        "import sys\n"
        "print('Args:', sys.argv[1:])\n"
        "sys.exit(0)\n",
        encoding="utf-8",
    )

    with patch("apps.wu_tanchang_api.agent_factory.utils.default_runtime_dir", return_value=tmp_path):
        res = run_aihehuo_skill_script.func(
            skill_name="get-ai-blog",
            arguments=["--page", "2", "test_val"],
            config={"configurable": {"thread_id": "test-thread"}},
        )
        assert "Args: ['--page', '2', 'test_val']" in res

    # Test script execution failure (non-zero exit code)
    script_path.write_text(
        "import sys\n"
        "print('Some error occurred', file=sys.stderr)\n"
        "sys.exit(1)\n",
        encoding="utf-8",
    )
    with patch("apps.wu_tanchang_api.agent_factory.utils.default_runtime_dir", return_value=tmp_path):
        res = run_aihehuo_skill_script.func(
            skill_name="get-ai-blog",
            arguments=[],
            config={"configurable": {"thread_id": "test-thread"}},
        )
        assert "错误：脚本执行失败" in res
        assert "Exit code 1" in res

