"""REQ-004 unit tests: schema, forced persist, assert/retry, no-match surface."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
from apps.group_agent_api.agent_factory.agent import FORCE_SAVE_PROMPT, create_agent
from apps.group_agent_api.agent_factory.profile_schema import (
    DisclosureLevel,
    GroupProfile,
    ProfileField,
    profile_from_flat,
)
from apps.group_agent_api.agent_factory.profile_store import (
    ProfileStoreError,
    alert_persist_failure,
    assert_profile_persisted,
    disk_profile_path,
    load_profile,
    save_profile,
    virtual_profile_path,
)
from apps.group_agent_api.app.endpoints import chat as chat_ep
from apps.group_agent_api.app.endpoints import profile as profile_ep
from apps.group_agent_api.app.models import ChatRequest
from apps.group_agent_api.app.state import AppState
from apps.group_agent_api.app.utils import thread_id
from langchain_core.messages import AIMessage, HumanMessage


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_profile_schema_requires_three_dims() -> None:
    profile = profile_from_flat(
        user_id="mock_u1",
        group_id="mock_g1",
        doing="智能宠物喂食器",
        need="联网与 App",
        offer="工厂与供应链",
    )
    assert profile.is_complete()
    assert profile.doing.disclosure == DisclosureLevel.inferred_unconfirmed
    dumped = profile.to_storage_dict()
    assert dumped["doing"]["value"] == "智能宠物喂食器"
    assert dumped["schema_version"] == 1


def test_profile_field_rejects_empty() -> None:
    with pytest.raises(Exception):
        ProfileField(value="   ")


def test_path_isolation_per_group() -> None:
    assert virtual_profile_path("u1", "g1") == "/users/u1/groups/g1/profile.json"
    assert virtual_profile_path("u1", "g2") == "/users/u1/groups/g2/profile.json"
    with pytest.raises(ProfileStoreError):
        virtual_profile_path("../x", "g1")
    with pytest.raises(ProfileStoreError):
        virtual_profile_path("..", "g1")
    with pytest.raises(ProfileStoreError):
        virtual_profile_path(".", "g1")
    with pytest.raises(ProfileStoreError):
        virtual_profile_path("u.1", "g1")


@pytest.mark.asyncio
async def test_chat_ignores_metadata_identity_override(tmp_path: Path) -> None:
    """Client must not override user_id/group_id/base_dir via metadata."""
    agent = _FakeAgent(base_dir=tmp_path, save_on_attempt=1)
    evil = tmp_path / "evil_root"
    evil.mkdir()
    resp = await chat_ep.chat(
        ChatRequest(
            user_id="mock_u1",
            group_id="mock_g1",
            message="做喂食器，缺联网，有工厂",
            metadata={
                "user_id": "victim",
                "group_id": "other",
                "base_dir": str(evil),
            },
        ),
        _state(agent, tmp_path),
    )
    assert resp.profile_persisted is True
    assert resp.profile_path == "/users/mock_u1/groups/mock_g1/profile.json"
    assert load_profile(tmp_path, "mock_u1", "mock_g1") is not None
    assert load_profile(tmp_path, "victim", "other") is None
    assert not (evil / "users").exists()


@pytest.mark.asyncio
async def test_chat_detects_stale_profile_without_update(tmp_path: Path) -> None:
    save_profile(
        tmp_path,
        profile_from_flat(
            user_id="mock_u1",
            group_id="mock_g1",
            doing="旧方向",
            need="旧需求",
            offer="旧资源",
        ),
    )
    agent = _FakeAgent(base_dir=tmp_path, save_on_attempt=99)
    resp = await chat_ep.chat(
        ChatRequest(
            user_id="mock_u1",
            group_id="mock_g1",
            message="方向变了",
        ),
        _state(agent, tmp_path),
    )
    assert resp.profile_persisted is False
    assert resp.persist_alert == "stale_profile_not_updated"
    assert agent.invoke_count == 2



# ---------------------------------------------------------------------------
# Forced persist + assert
# ---------------------------------------------------------------------------


def test_save_and_assert_profile(tmp_path: Path) -> None:
    profile = profile_from_flat(
        user_id="mock_u1",
        group_id="mock_g1",
        doing="做硬件",
        need="找技术合伙人",
        offer="有工厂",
        need_disclosure="match_only",
    )
    path = save_profile(tmp_path, profile)
    assert path.exists()
    assert path == disk_profile_path(tmp_path, "mock_u1", "mock_g1")

    loaded = load_profile(tmp_path, "mock_u1", "mock_g1")
    assert loaded is not None
    assert loaded.need.disclosure == DisclosureLevel.match_only

    result = assert_profile_persisted(tmp_path, "mock_u1", "mock_g1")
    assert result.ok
    assert result.path == "/users/mock_u1/groups/mock_g1/profile.json"

    # group isolation: other group missing
    other = assert_profile_persisted(tmp_path, "mock_u1", "mock_g2")
    assert not other.ok
    assert other.reason == "missing_file"


def test_assert_rejects_empty_file(tmp_path: Path) -> None:
    path = disk_profile_path(tmp_path, "u1", "g1")
    path.parent.mkdir(parents=True)
    path.write_text("", encoding="utf-8")
    result = assert_profile_persisted(tmp_path, "u1", "g1")
    assert not result.ok
    assert result.reason == "empty_file"


def test_alert_persist_failure_logs_error(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.ERROR):
        alert_persist_failure(
            user_id="u1", group_id="g1", attempt=1, reason="missing_file"
        )
    assert any("ALERT" in r.message and "profile_persist_failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Chat post-assert + retry (fake agent, no LLM)
# ---------------------------------------------------------------------------


class _FakeCheckpointer:
    def flush(self) -> None:
        return None

    def delete_thread(self, _tid: str) -> None:
        return None


class _FakeAgent:
    """Configurable fake: first invoke skips save; second can save."""

    def __init__(
        self,
        *,
        base_dir: Path,
        save_on_attempt: int = 2,
        replies: list[str] | None = None,
    ) -> None:
        self.base_dir = base_dir
        self.save_on_attempt = save_on_attempt
        self.replies = replies or ["先聊聊你在做什么？", "已记下你的画像。"]
        self.invoke_count = 0
        self.checkpointer = _FakeCheckpointer()
        self.messages: list[Any] = []

    async def aget_state(self, _config: dict[str, Any]) -> Any:
        class _S:
            values = {"messages": list(self.messages)}

        return _S()

    async def ainvoke(
        self, payload: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        self.invoke_count += 1
        human = payload["messages"][0]
        self.messages.append(human)
        reply_idx = min(self.invoke_count - 1, len(self.replies) - 1)
        ai = AIMessage(content=self.replies[reply_idx])
        self.messages.append(ai)

        if self.invoke_count >= self.save_on_attempt:
            meta = config.get("metadata") or {}
            profile = profile_from_flat(
                user_id=str(meta["user_id"]),
                group_id=str(meta["group_id"]),
                doing="智能宠物喂食器",
                need="联网与 App",
                offer="工厂与供应链",
            )
            save_profile(self.base_dir, profile)

        return {"messages": list(self.messages)}


def _state(agent: Any, tmp_path: Path) -> AppState:
    return AppState(
        agent=agent,
        base_dir=tmp_path,
        checkpoints_path=str(tmp_path / "checkpoints.pkl"),
    )


@pytest.mark.asyncio
async def test_chat_retries_when_profile_missing(tmp_path: Path) -> None:
    agent = _FakeAgent(base_dir=tmp_path, save_on_attempt=2)
    resp = await chat_ep.chat(
        ChatRequest(
            user_id="mock_u1",
            group_id="mock_g1",
            message="我在做喂食器，缺联网，有工厂",
        ),
        _state(agent, tmp_path),
    )
    assert agent.invoke_count == 2
    assert isinstance(agent.messages[-2], HumanMessage)
    assert FORCE_SAVE_PROMPT in str(agent.messages[-2].content)
    assert resp.profile_persisted is True
    assert resp.assert_attempts == 2
    assert resp.profile_path == "/users/mock_u1/groups/mock_g1/profile.json"
    assert resp.persist_alert is None
    assert resp.thread_id == thread_id(
        user_id="mock_u1", group_id="mock_g1", conversation_id="default"
    )


@pytest.mark.asyncio
async def test_chat_alerts_when_retry_still_fails(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    agent = _FakeAgent(base_dir=tmp_path, save_on_attempt=99)  # never saves
    with caplog.at_level(logging.ERROR):
        resp = await chat_ep.chat(
            ChatRequest(
                user_id="mock_u1",
                group_id="mock_g1",
                message="随便",
            ),
            _state(agent, tmp_path),
        )
    assert resp.profile_persisted is False
    assert resp.assert_attempts == 2
    assert resp.persist_alert == "missing_file"
    assert any("ALERT" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_chat_succeeds_first_turn_when_tool_writes(tmp_path: Path) -> None:
    agent = _FakeAgent(base_dir=tmp_path, save_on_attempt=1)
    resp = await chat_ep.chat(
        ChatRequest(
            user_id="mock_u1",
            group_id="mock_g1",
            message="做喂食器，缺联网，有工厂",
        ),
        _state(agent, tmp_path),
    )
    assert agent.invoke_count == 1
    assert resp.profile_persisted is True
    assert resp.assert_attempts == 1


@pytest.mark.asyncio
async def test_get_profile_queryable(tmp_path: Path) -> None:
    save_profile(
        tmp_path,
        profile_from_flat(
            user_id="mock_u1",
            group_id="mock_g1",
            doing="A",
            need="B",
            offer="C",
        ),
    )
    resp = await profile_ep.get_profile(
        user_id="mock_u1", group_id="mock_g1", state=_state(object(), tmp_path)
    )
    assert resp.exists is True
    assert resp.profile is not None
    assert resp.profile["doing"]["value"] == "A"


def test_create_agent_wires_save_tool(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    class _FakeModel:
        profile: dict = {}

    def fake_create_deep_agent(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.agent.create_deep_agent",
        fake_create_deep_agent,
    )
    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.agent.create_model",
        lambda: _FakeModel(),
    )

    agent, ckpt = create_agent(base_dir=tmp_path, model=_FakeModel())
    assert agent is not None
    assert ckpt == tmp_path / "checkpoints.pkl"
    tool_names = [getattr(t, "name", None) for t in captured.get("tools") or []]
    assert "save_group_profile" in tool_names
    prompt = captured.get("system_prompt") or ""
    # Red lines in prompt
    assert "不得推荐" in prompt or "候选人" in prompt
    assert "@" in prompt or "邀请词" in prompt


def test_no_match_tools_in_agent_factory_source() -> None:
    """Static guard: slice 1 must not wire matching / member search."""
    src = (
        Path(__file__).resolve().parents[1]
        / "apps"
        / "group_agent_api"
        / "agent_factory"
        / "agent.py"
    ).read_text(encoding="utf-8")
    banned = [
        "aihehuo_search_members",
        "partner_query",
        "recommended_partners",
        "vector_search",
        "generate_proposal",
    ]
    for token in banned:
        assert token not in src, f"banned match surface leaked: {token}"
