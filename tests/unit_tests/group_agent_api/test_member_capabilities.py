"""Regression tests for read-only member execution."""

from pathlib import Path
from typing import Any, ClassVar

import pytest
from langchain_core.messages import AIMessage

from apps.group_agent_api.agent_factory.capability import CapabilityTier, SessionCapabilities
from apps.group_agent_api.agent_factory.integrations.membership_client import MembershipResult
from apps.group_agent_api.agent_factory.integrations.principal import SessionPrincipal
from apps.group_agent_api.app.endpoints import chat as chat_ep
from apps.group_agent_api.app.models import ChatRequest
from apps.group_agent_api.app.session import TrustedSession
from apps.group_agent_api.app.state import AppState


class RecordingAgent:
    """Record invocations without performing writes or network requests."""

    def __init__(self) -> None:
        self.configs: list[dict[str, Any]] = []

    async def aget_state(self, config: dict[str, Any]) -> object:
        del config

        class Snapshot:
            values: ClassVar[dict[str, Any]] = {"messages": []}

        return Snapshot()

    async def ainvoke(self, payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        self.configs.append(config)
        return {"messages": [*payload["messages"], AIMessage(content="了解你的需求了。")]}


@pytest.mark.parametrize("can_search", [False, True])
async def test_read_only_chat_selects_graph_and_skips_save_retry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, can_search: bool) -> None:
    tier = CapabilityTier.in_group if can_search else CapabilityTier.unknown
    session = TrustedSession(
        principal=SessionPrincipal("u1", None, None, "signed_oauth_principal"),
        membership=MembershipResult(tier=tier, source="test"),
        group_id="g1",
        group_token=None,
        capabilities=SessionCapabilities(tier=tier, can_search=can_search, can_write_profile=False),
    )

    async def resolve_session(*args: Any, **kwargs: Any) -> TrustedSession:
        del args, kwargs
        return session

    monkeypatch.setattr(chat_ep, "resolve_trusted_session", resolve_session)
    full = RecordingAgent()
    read = RecordingAgent()
    empty = RecordingAgent()
    state = AppState(agent=full, base_dir=tmp_path, search_only_agent=read, no_tool_agent=empty)
    await chat_ep.chat(
        ChatRequest(
            user_id="u1",
            group_id="g1",
            message="我在做智能宠物喂食器，缺联网与 App，有工厂与供应链",  # noqa: RUF001  # Chinese text intentionally uses Chinese punctuation.
            metadata={"can_write_profile": "true"},
        ),
        state,
    )
    selected = read if can_search else empty
    assert len(selected.configs) == 1
    assert selected.configs[0]["metadata"]["can_write_profile"] == "false"
    assert full.configs == []
    assert not (tmp_path / "users").exists()
