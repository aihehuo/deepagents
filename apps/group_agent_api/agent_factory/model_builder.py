"""Minimal model builder for group_agent_api (env-driven, openai-compatible)."""

from __future__ import annotations

import logging
import os
from typing import Any

from langchain_openai import ChatOpenAI

_logger = logging.getLogger("uvicorn.error")


_STUB_SAVE_ACK = "已根据对话内容整理并保存你的群画像（doing/need/offer）。"
_STUB_DEFAULT_PROFILE = {
    "doing": "Building an AI startup",
    "need": "Co-founder",
    "offer": "Python, product",
}


def _stub_profile_for(user_id: str, group_id: str) -> dict[str, str]:
    """Resolve deterministic doing/need/offer for the stub tool-call.

    When a fixture level is active, mirror the caller's fixture profile so the
    downstream match pipeline stays fixture-accurate; otherwise fall back to a
    fixed default. Never raises — the stub must stay deterministic.
    """
    test_lvl = os.environ.get("GROUP_AGENT_TEST_LEVEL")
    if test_lvl and user_id and group_id:
        try:
            from apps.group_agent_api.fixtures.loader import load_fixture

            ds = load_fixture(test_lvl)
            member = ds.members.get(f"{group_id}:{user_id}")
            if member and isinstance(member.profile, dict):
                p = member.profile
                return {
                    "doing": str(p.get("doing") or _STUB_DEFAULT_PROFILE["doing"]),
                    "need": str(p.get("need") or _STUB_DEFAULT_PROFILE["need"]),
                    "offer": str(p.get("offer") or _STUB_DEFAULT_PROFILE["offer"]),
                }
        except Exception as exc:  # noqa: BLE001
            _logger.warning("stub_profile_lookup_failed error_type=%s", type(exc).__name__)
    return dict(_STUB_DEFAULT_PROFILE)


def _build_stub_model() -> Any:
    """Deterministic, message-aware stub that emits a REAL save_group_profile tool call.

    Behaviour per turn:
    - First model step (no prior ToolMessage in the tail): return an AIMessage with a
      `save_group_profile` tool call carrying fixture-accurate doing/need/offer, so the
      deep-agent react loop actually executes the tool and persists the profile.
    - After the tool result (tail is a ToolMessage): return a short natural-language ack,
      ending the turn.

    This replaces the old FakeListChatModel, which only returned canned text and never
    triggered a tool call — hiding whether profile persistence really happened.
    """
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessage, ToolMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    class StubGroupAgentChatModel(BaseChatModel):
        # Metadata carried on the invoke config; captured so the tool-call args
        # can mirror the caller's fixture profile.
        _last_metadata: dict[str, Any] = {}

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            metadata = getattr(run_manager, "metadata", None) or {}
            user_id = str(metadata.get("user_id") or "")
            group_id = str(metadata.get("group_id") or "")

            tail = messages[-1] if messages else None
            if isinstance(tail, ToolMessage):
                msg = AIMessage(content=_STUB_SAVE_ACK)
            else:
                prof = _stub_profile_for(user_id, group_id)
                msg = AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "save_group_profile",
                            "args": {
                                "doing": prof["doing"],
                                "need": prof["need"],
                                "offer": prof["offer"],
                                "doing_disclosure": "confirmed_public",
                                "need_disclosure": "confirmed_public",
                                "offer_disclosure": "confirmed_public",
                            },
                            "id": "stub_save_group_profile",
                        }
                    ],
                )
            return ChatResult(generations=[ChatGeneration(message=msg)])

        @property
        def _llm_type(self) -> str:
            return "stub-group-agent-chat-model"

        def bind_tools(self, tools, **kwargs):
            return self

    fake = StubGroupAgentChatModel()
    fake.profile = {"max_input_tokens": 32000}
    return fake


def create_model(*, log_prefix: str = "[GroupAgentModel]") -> ChatOpenAI:
    """Create chat model from GROUP_AGENT_* / shared provider env vars."""
    provider = os.environ.get("GROUP_AGENT_PROVIDER", "deepseek").strip().lower()
    prefix = provider.upper()

    model = (
        os.environ.get("GROUP_AGENT_MODEL")
        or os.environ.get(f"{prefix}_MAIN_AGENT_MODEL")
        or os.environ.get(f"{prefix}_MODEL")
        or "deepseek-chat"
    )
    base_url = (
        os.environ.get("GROUP_AGENT_BASE_URL")
        or os.environ.get(f"{prefix}_BASE_URL")
        or ""
    )
    provider_fallback_key = None
    if provider in {"qwen", "dashscope"}:
        provider_fallback_key = os.environ.get("DASHSCOPE_API_KEY")
    elif provider == "deepseek":
        provider_fallback_key = os.environ.get("DEEPSEEK_API_KEY")

    api_key = (
        os.environ.get("GROUP_AGENT_API_KEY")
        or os.environ.get(f"{prefix}_API_KEY")
        or provider_fallback_key
        or "EMPTY"
    )
    max_tokens = int(os.environ.get("GROUP_AGENT_MAX_TOKENS", "2000"))
    timeout_s = float(os.environ.get("GROUP_AGENT_TIMEOUT_S", "60"))
    temperature_raw = os.environ.get("GROUP_AGENT_TEMPERATURE", "0.3")
    temperature = float(temperature_raw) if temperature_raw != "" else None

    kwargs: dict[str, object] = {
        "model": model,
        "max_tokens": max_tokens,
        "timeout": timeout_s,
        "api_key": api_key,
    }
    if base_url:
        kwargs["base_url"] = base_url
    if temperature is not None:
        kwargs["temperature"] = temperature

    _logger.info(
        "%s provider=%s model=%s base_url_set=%s",
        log_prefix,
        provider,
        model,
        bool(base_url),
    )
    integration = (os.environ.get("GROUP_AGENT_INTEGRATION") or "stub").strip().lower()
    model_mode = (os.environ.get("GROUP_AGENT_MODEL_MODE") or "").strip().lower()
    env = (os.environ.get("GROUP_AGENT_ENV") or os.environ.get("APP_ENV") or "development").strip().lower()

    if model_mode == "stub":
        if integration == "http" or env in {"production", "prod"}:
            raise RuntimeError("GROUP_AGENT_MODEL_MODE=stub is strictly forbidden in http/production integration")
        return _build_stub_model()

    if api_key == "EMPTY" or not api_key:
        raise RuntimeError(
            f"Missing LLM API key for provider '{provider}'. "
            "Real LLM API key is required when GROUP_AGENT_MODEL_MODE!=stub."
        )

    chat = ChatOpenAI(**kwargs)
    if not hasattr(chat, "profile") or chat.profile is None:
        chat.profile = {}
    if isinstance(chat.profile, dict):
        chat.profile["max_input_tokens"] = int(
            os.environ.get("GROUP_AGENT_MAX_INPUT_TOKENS", "32000")
        )
    return chat
