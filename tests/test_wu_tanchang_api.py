from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from apps.wu_tanchang_api.config import get_selected_provider, resolve_model_config
from apps.wu_tanchang_api.app.endpoints.chat import chat
from apps.wu_tanchang_api.app.models import ChatRequest
from apps.wu_tanchang_api.app.state import AppState


class _FakeCheckpointer:
    def __init__(self, checkpoint: dict[str, Any] | None = None) -> None:
        self.checkpoint = checkpoint

    async def aget(self, _config: dict[str, Any]) -> dict[str, Any] | None:
        return self.checkpoint


class _FakeAgent:
    def __init__(self, checkpointer: _FakeCheckpointer) -> None:
        self.checkpointer = checkpointer

    async def ainvoke(self, _input: dict[str, Any], _config: dict[str, Any]) -> dict[str, Any]:
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
                ToolMessage(content="material_delivered", name="mark_material_delivered", tool_call_id="call_1")
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


def test_agent_has_filesystem_middleware(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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

    created_agent, checkpoints_path = agent.create_agent(backend_root=tmp_path, provider="deepseek")

    assert created_agent is not None
    assert checkpoints_path == tmp_path / "runtime" / "checkpoints.pkl"

    kb_spec = next(spec for spec in captured["subagents"] if spec["name"] == "kb_analyst")
    assert kb_spec["skills"] == ["/skills/"]


def test_config_resolves_env_references(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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

    model_config = resolve_model_config(provider="openai_compatible", model_name_suffix="MAIN_AGENT_MODEL")
    assert model_config.api_type == "openai-compatible"
    assert model_config.base_url == "https://example.test/v1"
    assert model_config.api_key == "test-key"
    assert model_config.model == "test-main"
    assert model_config.max_tokens == 1234
    assert model_config.timeout_s == 12.5
    assert model_config.temperature == 0.4
    assert model_config.max_input_tokens == 9999


def test_create_model_dispatches_by_api_type(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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

    openai_model = model_builder.create_model(provider="openai_compatible", model_name_suffix="MAIN_AGENT_MODEL")
    anthropic_model = model_builder.create_model(provider="anthropic_compatible", model_name_suffix="MAIN_AGENT_MODEL")

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

    model_config = resolve_model_config(provider="qwen", model_name_suffix="MAIN_AGENT_MODEL")
    assert model_config.model == "qwen-flash"
    assert model_config.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert model_config.api_key is None
    assert model_config.max_input_tokens == 1000000


def test_task_tool_blocks_kb_analyst_after_delivered() -> None:
    from deepagents.middleware.subagents import _build_task_tool
    from langchain.tools import ToolRuntime
    from langchain_core.messages import ToolMessage
    from unittest.mock import MagicMock

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
    res_ok = tool.func(description="hello", subagent_type="kb_analyst", runtime=mock_runtime_ok)
    assert res_ok != "知识库已经检索过且会议准备材料已完成交付。禁止再次调用 kb_analyst 或检索知识库。请直接根据已交付的材料和对话历史回答用户的问题，并引导用户预约吴探长一对一深聊。"

    # Case 2: Materials delivered
    mock_runtime_blocked = MagicMock(spec=ToolRuntime)
    delivered_msg = ToolMessage(content="done", name="mark_material_delivered", tool_call_id="tool-1")
    mock_runtime_blocked.state = {"messages": [delivered_msg]}
    mock_runtime_blocked.tool_call_id = "call-2"
    mock_runtime_blocked.config = {}

    res_blocked = tool.func(description="hello", subagent_type="kb_analyst", runtime=mock_runtime_blocked)
    assert res_blocked == "知识库已经检索过且会议准备材料已完成交付。禁止再次调用 kb_analyst 或检索知识库。请直接根据已交付的材料和对话历史回答用户的问题，并引导用户预约吴探长一对一深聊。"

    # Case 3: Materials delivered (YC Workspace)
    mock_runtime_blocked_yc = MagicMock(spec=ToolRuntime)
    mock_runtime_blocked_yc.state = {"messages": [delivered_msg]}
    mock_runtime_blocked_yc.tool_call_id = "call-3"
    mock_runtime_blocked_yc.config = {
        "configurable": {"thread_id": "wt::yc01::u1::c1"},
        "metadata": {"agent_name": "yc01"}
    }

    res_blocked_yc = tool.func(description="hello", subagent_type="kb_analyst", runtime=mock_runtime_blocked_yc)
    assert res_blocked_yc == "知识库已经检索过且会议准备材料已完成交付。禁止再次调用 kb_analyst 或检索知识库。请直接根据已交付的材料和对话历史回答用户的问题，并引导用户预约YC老师一对一深聊。"


def test_accountant_middleware_resets_tool_count_per_turn() -> None:
    from types import SimpleNamespace

    from deepagents.middleware.accountant import AccountantMiddleware
    from langchain_core.messages import ToolMessage
    from langgraph.types import Command

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

    from deepagents.middleware.accountant import AccountantMiddleware
    from langchain_core.messages import ToolMessage

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
    from apps.wu_tanchang_api.agent_factory.utils import ensure_runtime_workspace
    import os

    # 1. Setup source workspace structure
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "kb").mkdir()
    (src_dir / "kb" / "doc.txt").write_text("kb source", encoding="utf-8")

    (src_dir / "skills").mkdir()
    (src_dir / "skills" / "skill.txt").write_text("skills source", encoding="utf-8")

    # Add a custom workspace folder
    custom_ws = tmp_path / "workspace_custom"
    custom_ws.mkdir()
    (custom_ws / "identity.md").write_text("identity source", encoding="utf-8")

    # 2. Deploy to runtime
    runtime_dir = tmp_path / "runtime"
    ensure_runtime_workspace(workspace_src=src_dir, runtime_dir=runtime_dir)

    # 3. Assert target directory existence
    assert (runtime_dir / "kb" / "doc.txt").read_text(encoding="utf-8") == "kb source"
    assert (runtime_dir / "skills" / "skill.txt").read_text(encoding="utf-8") == "skills source"
    assert (runtime_dir / "workspace_custom" / "identity.md").read_text(encoding="utf-8") == "identity source"

    # Verify symlink status (if symlinks are supported by filesystem/OS)
    try:
        os.symlink(src_dir / "kb" / "doc.txt", tmp_path / "test_link")
        symlink_supported = True
    except (OSError, PermissionError):
        symlink_supported = False

    if symlink_supported:
        # Both 'kb' and 'skills' should be symlinks since they are read-only
        assert (runtime_dir / "kb").is_symlink()
        assert (runtime_dir / "skills").is_symlink()
        # Custom workspaces must NEVER be symlinks to preserve write isolation
        assert not (runtime_dir / "workspace_custom").is_symlink()

    # 4. Modify files in runtime and check isolation
    # Note: custom workspace files can be modified and must be isolated.
    (runtime_dir / "workspace_custom" / "identity.md").write_text("identity modified", encoding="utf-8")

    # Assert runtime changed
    assert (runtime_dir / "workspace_custom" / "identity.md").read_text(encoding="utf-8") == "identity modified"

    # Assert source did NOT change!
    assert (custom_ws / "identity.md").read_text(encoding="utf-8") == "identity source"


def test_task_tool_blocks_kb_analyst_after_delivered_dynamic_owner() -> None:
    from deepagents.middleware.subagents import _build_task_tool
    from deepagents.backends.protocol import BackendProtocol
    from langchain.tools import ToolRuntime
    from langchain_core.messages import ToolMessage
    from unittest.mock import MagicMock
    import json

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
        entries=[
            {"path": "/workspace"},
            {"path": "/workspace_1"}
        ]
    )

    # Mock read files
    def mock_read(path):
        from deepagents.backends.protocol import ReadResult
        if path == "/workspace_1/MEMORY.md":
            return ReadResult(file_data={"content": "- Agent id: yc01", "encoding": "utf-8"})
        elif path == "/workspace/MEMORY.md":
            return ReadResult(file_data={"content": "- Agent id: andy01", "encoding": "utf-8"})
        elif path == "/workspace_1/owner.json":
            return ReadResult(file_data={"content": json.dumps({"owner_name": "动态YC老师"}), "encoding": "utf-8"})
        elif path == "/workspace/owner.json":
            return ReadResult(file_data={"content": json.dumps({"owner_name": "动态吴探长"}), "encoding": "utf-8"})
        elif path == "/owner.json":
            return ReadResult(file_data={"content": json.dumps({"owner_name": "根目录Owner"}), "encoding": "utf-8"})
        return ReadResult(error="file not found")

    mock_backend.read.side_effect = mock_read

    # Build the task tool with the mock backend
    tool = _build_task_tool([spec], backend=mock_backend)

    # Setup runtime with delivered message and YC metadata
    delivered_msg = ToolMessage(content="done", name="mark_material_delivered", tool_call_id="tool-1")
    mock_runtime = MagicMock(spec=ToolRuntime)
    mock_runtime.state = {"messages": [delivered_msg]}
    mock_runtime.tool_call_id = "call-1"
    mock_runtime.config = {
        "configurable": {"thread_id": "wt::yc01::u1::c1"},
        "metadata": {"agent_name": "yc01"}
    }

    res = tool.func(description="hello", subagent_type="kb_analyst", runtime=mock_runtime)
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
        "metadata": {"agent_name": "andy01"}
    }

    res_default = tool_default.func(description="hello", subagent_type="kb_analyst", runtime=mock_runtime_default)
    assert "根目录Owner" in res_default
