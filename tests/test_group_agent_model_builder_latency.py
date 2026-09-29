"""Caller-specific provider budgets for Group Agent model clients."""

from typing import Any

from apps.group_agent_api.agent_factory import model_builder


def test_create_model_applies_explicit_timeout_and_retry_budget(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class FakeChat:
        profile: dict[str, Any] = {}

    def fake_chat_openai(**kwargs: Any) -> FakeChat:
        captured.update(kwargs)
        return FakeChat()

    monkeypatch.setenv("GROUP_AGENT_PROVIDER", "deepseek")
    monkeypatch.setenv("GROUP_AGENT_MODEL_MODE", "real")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(model_builder, "ChatOpenAI", fake_chat_openai)

    model_builder.create_model(timeout_s=25, max_retries=0)

    assert captured["timeout"] == 25
    assert captured["max_retries"] == 0


def test_create_model_preserves_shared_retry_defaults(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class FakeChat:
        profile: dict[str, Any] = {}

    monkeypatch.setenv("GROUP_AGENT_PROVIDER", "deepseek")
    monkeypatch.setenv("GROUP_AGENT_MODEL_MODE", "real")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("GROUP_AGENT_TIMEOUT_S", "41")
    monkeypatch.setenv("GROUP_AGENT_MAX_RETRIES", "3")
    monkeypatch.setattr(
        model_builder,
        "ChatOpenAI",
        lambda **kwargs: captured.update(kwargs) or FakeChat(),
    )

    model_builder.create_model()

    assert captured["timeout"] == 41
    assert captured["max_retries"] == 3
