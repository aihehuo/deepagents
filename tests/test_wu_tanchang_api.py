from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from deepagents.middleware.filesystem import FilesystemMiddleware
from fastapi import HTTPException

from apps.wu_tanchang_api.config import get_selected_provider, resolve_model_config
from apps.wu_tanchang_api.agent_factory.intake_agent import MAX_INTAKE_ROUNDS
from apps.wu_tanchang_api.app.endpoints.brief import get_brief
from apps.wu_tanchang_api.app.endpoints.chat import chat
from apps.wu_tanchang_api.app.models import BriefRequest, ChatRequest
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


@dataclass
class _FakeModel:
    profile: dict[str, Any] | None = None


def _state(
    checkpointer: _FakeCheckpointer,
    tmp_path: Path,
    canvas_store: dict[str, dict[str, Any]] | None = None,
) -> AppState:
    agent = _FakeAgent(checkpointer)
    return AppState(
        agents={"default": agent},
        agent_configs={},
        default_agent="default",
        checkpoints_path=str(tmp_path / "checkpoints.pkl"),
        thread_locks={},
        backend_root=str(tmp_path),
        expertise_dir=str(tmp_path / "expertise"),
        canvas_store=canvas_store or {},
    )


async def test_brief_returns_current_canvas(tmp_path: Path) -> None:
    canvas = {
        "meta": {"intake_complete": True, "round_count": 6},
        "conversation_points": ["确认上海烘焙店的预算与商圈"],
        "main_challenges": ["预算和店型之间需要重新匹配"],
        "solution_directions": ["用库内烘焙案例校准选址和定价"],
        "relevant_cases": [{"brand": "Punch Monday", "id": "wu-punch-monday", "why_relevant": "平价烘焙模型"}],
        "open_questions": [],
    }
    thread_id = "wt::default::u1::c1"
    canvas_store = {
        thread_id: {
            "canvas": canvas,
            "brief_summary": "用户计划在上海做烘焙店，需重点讨论预算、选址和定价。",
            "intake_complete": True,
            "analysis_timestamp": "2026-06-09T00:00:00+00:00",
        }
    }
    checkpointer = _FakeCheckpointer(None)

    response = await get_brief(
        BriefRequest(user_id="u1", conversation_id="c1"),
        _state(checkpointer, tmp_path, canvas_store=canvas_store),
    )

    assert response.thread_id == thread_id
    assert response.canvas == canvas
    assert response.brief_summary == "用户计划在上海做烘焙店，需重点讨论预算、选址和定价。"
    assert response.current_round == 6
    assert response.intake_complete is True


async def test_chat_rejects_after_intake_complete(tmp_path: Path) -> None:
    thread_id = "wt::default::u1::c1"
    canvas_store = {
        thread_id: {
            "canvas": {"meta": {"intake_complete": True}},
            "intake_complete": True,
        }
    }
    checkpointer = _FakeCheckpointer(None)

    with pytest.raises(HTTPException) as exc:
        await chat(
            ChatRequest(user_id="u1", conversation_id="c1", message="继续聊"),
            _state(checkpointer, tmp_path, canvas_store=canvas_store),
        )

    assert exc.value.status_code == 400
    assert exc.value.detail["error"] == "intake_complete"
    assert exc.value.detail["thread_id"] == thread_id


def test_agent_has_filesystem_middleware(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from apps.wu_tanchang_api.agent_factory import intake_agent

    captured: dict[str, Any] = {}

    def fake_create_model(**_kwargs: Any) -> _FakeModel:
        return _FakeModel(profile={})

    def fake_create_deep_agent(**_kwargs: Any) -> object:
        captured.update(_kwargs)
        return object()

    monkeypatch.setattr(intake_agent, "create_model", fake_create_model)
    monkeypatch.setattr(intake_agent, "create_deep_agent", fake_create_deep_agent)
    monkeypatch.setattr(intake_agent, "default_runtime_dir", lambda: tmp_path / "runtime")

    agent, checkpoints_path = intake_agent.create_agent(backend_root=tmp_path, provider="deepseek")

    assert agent is not None
    assert checkpoints_path == tmp_path / "runtime" / "checkpoints.pkl"
    assert any(isinstance(item, FilesystemMiddleware) for item in captured["middleware"])


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
    assert model_config.model == "qwen-plus"
    assert model_config.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert model_config.api_key is None
    assert model_config.max_input_tokens == 1000000
