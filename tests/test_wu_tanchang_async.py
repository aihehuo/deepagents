from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fastapi import HTTPException
from langchain_core.messages import ToolMessage

from apps.wu_tanchang_api.app.callbacks import CallbackUrlError, validate_callback_url
from apps.wu_tanchang_api.app.endpoints import async_chat, chat
from apps.wu_tanchang_api.app.models import CallWuTanchangAsyncRequest, ChatRequest
from apps.wu_tanchang_api.app.state import AppState


class FakeThread:
    def __init__(self) -> None:
        self.started = False

    def start(self) -> None:
        self.started = True

    def is_alive(self) -> bool:
        return self.started


class FakeCheckpointer:
    def __init__(self, messages: list[Any] | None = None) -> None:
        self.messages = messages or []

    async def aget(self, _config: dict[str, Any]) -> dict[str, Any]:
        return {"channel_values": {"messages": self.messages}}


class FakeAgent:
    def __init__(self, messages: list[Any] | None = None) -> None:
        self.checkpointer = FakeCheckpointer(messages)


def test_callback_url_requires_allowed_base_url() -> None:
    assert (
        validate_callback_url("http://host.docker.internal:3001/wu_tanchang_callbacks/1/default")
        == "http://host.docker.internal:3001/wu_tanchang_callbacks/1/default"
    )

    for url in [
        "http://host.docker.internal:9999/wu_tanchang_callbacks/1/default",
        "http://host.docker.internal:3001/admin",
        "http://169.254.169.254/latest/meta-data",
        "file:///tmp/callback",
    ]:
        with pytest.raises(CallbackUrlError):
            validate_callback_url(url)


def test_call_async_rejects_duplicate_active_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_thread = FakeThread()
    monkeypatch.setattr(async_chat, "build_callback_thread", lambda **_kwargs: fake_thread)

    state = AppState(agents={"default": FakeAgent()}, default_agent="default")
    req = CallWuTanchangAsyncRequest(
        user_id="u1",
        conversation_id="c1",
        message="hello",
        callback_url="http://host.docker.internal:3001/wu_tanchang_callbacks/u1/c1",
    )

    first = asyncio.run(async_chat.call_async(req, state))
    assert first.success is True
    assert fake_thread.started is True

    second = asyncio.run(async_chat.call_async(req, state))
    assert second.success is False
    assert second.message == "Agent run already in progress for this conversation"


def test_call_async_delivered_material_sends_guide_without_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    callbacks: list[dict[str, Any]] = []

    class ImmediateThread:
        def __init__(self, target: Any, **_kwargs: Any) -> None:
            self.target = target

        def start(self) -> None:
            self.target()

    monkeypatch.setattr(async_chat.threading, "Thread", ImmediateThread)
    monkeypatch.setattr(async_chat, "invoke_callback", lambda _url, payload: callbacks.append(payload) or False)

    delivered = ToolMessage(content="done", name="mark_material_delivered", tool_call_id="tool-1")
    state = AppState(agents={"default": FakeAgent([delivered])}, default_agent="default")
    req = CallWuTanchangAsyncRequest(
        user_id="u2",
        conversation_id="c2",
        message="hello again",
        callback_url="http://host.docker.internal:3001/wu_tanchang_callbacks/u2/c2",
    )

    response = asyncio.run(async_chat.call_async(req, state))

    assert response.success is True
    assert "guide message callback scheduled" in response.message
    assert [payload["type"] for payload in callbacks] == ["message", "status"]
    assert callbacks[0]["message_id"] == callbacks[1]["message_id"]
    assert callbacks[1]["status"] == "stream_completed"
    assert state.active_agent_runs == {}


def test_call_async_bad_max_active_env_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_thread = FakeThread()
    monkeypatch.setenv("WU_CALLBACK_MAX_ACTIVE_STREAMS", "not-an-int")
    monkeypatch.setattr(async_chat, "build_callback_thread", lambda **_kwargs: fake_thread)

    state = AppState(agents={"default": FakeAgent()}, default_agent="default")
    req = CallWuTanchangAsyncRequest(
        user_id="u3",
        conversation_id="c3",
        message="hello",
        callback_url="http://host.docker.internal:3001/wu_tanchang_callbacks/u3/c3",
    )

    response = asyncio.run(async_chat.call_async(req, state))

    assert response.success is True
    assert fake_thread.started is True


def test_chat_rejects_when_call_async_run_is_active() -> None:
    state = AppState(agents={"default": FakeAgent()}, default_agent="default")
    tid = "wt::default::u4::c4"
    assert state.try_start_agent_run(tid, "call_async") is True

    req = ChatRequest(user_id="u4", conversation_id="c4", message="hello")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(chat.chat(req, state))

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail["error"] == "stream_in_progress"
