from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient


@dataclass
class _FakeAIMessage:
    type: str
    content: str


class _FakeCheckpointer:
    def __init__(self) -> None:
        self.deleted_threads: list[str] = []

    def delete_thread(self, thread_id: str) -> None:
        self.deleted_threads.append(thread_id)


class _FakeAgent:
    def __init__(self, *, checkpointer: _FakeCheckpointer) -> None:
        self.checkpointer = checkpointer
        self.last_ainvoke_input: dict[str, Any] | None = None
        self.last_ainvoke_config: dict[str, Any] | None = None

    async def ainvoke(self, input: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        self.last_ainvoke_input = input
        self.last_ainvoke_config = config

        # Mirror the API's behavior: reply is taken from the last ai message content.
        user_msg = ""
        msgs = input.get("messages") or []
        if msgs:
            user_msg = str(getattr(msgs[-1], "content", ""))

        return {"messages": [_FakeAIMessage(type="ai", content=f"echo: {user_msg}")]}


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    # Import inside the fixture so monkeypatching affects the app module used by TestClient.
    import apps.business_cofounder_api.app as app_module

    # Ensure a clean module-level state per test.
    app_module._state = None

    checkpointer = _FakeCheckpointer()
    agent = _FakeAgent(checkpointer=checkpointer)
    checkpoints_path = tmp_path / "checkpoints.pkl"

    def _fake_create_business_cofounder_agent(*, agent_id: str) -> tuple[object, Path]:
        return agent, checkpoints_path

    monkeypatch.setattr(app_module, "create_business_cofounder_agent", _fake_create_business_cofounder_agent)

    with TestClient(app_module.app) as c:
        # Expose fakes for assertions (without relying on globals).
        c._fake_agent = agent  # type: ignore[attr-defined]
        c._fake_checkpointer = checkpointer  # type: ignore[attr-defined]
        yield c


def test_health_ok(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "ok"
    assert payload["checkpoints_path"].endswith("checkpoints.pkl")


def test_chat_returns_thread_id_and_reply(client: TestClient) -> None:
    resp = client.post(
        "/chat",
        json={
            "user_id": "u1",
            "conversation_id": "c1",
            "message": "Hello API",
            "metadata": {"source": "pytest"},
        },
    )
    assert resp.status_code == 200
    payload = resp.json()

    assert payload["user_id"] == "u1"
    assert payload["conversation_id"] == "c1"
    assert payload["thread_id"] == "bc::u1::c1"
    assert payload["reply"] == "echo: Hello API"

    # Validate the agent got the expected thread_id + metadata in the LangGraph config.
    agent = client._fake_agent  # type: ignore[attr-defined]
    assert agent.last_ainvoke_config is not None
    assert agent.last_ainvoke_config["configurable"]["thread_id"] == "bc::u1::c1"
    assert agent.last_ainvoke_config["metadata"]["user_id"] == "u1"
    assert agent.last_ainvoke_config["metadata"]["source"] == "pytest"


def test_chat_defaults_conversation_id_to_default(client: TestClient) -> None:
    resp = client.post(
        "/chat",
        json={"user_id": "u2", "message": "Hi"},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["conversation_id"] == "default"
    assert payload["thread_id"] == "bc::u2::default"


def test_reset_deletes_thread_in_checkpointer(client: TestClient) -> None:
    # Create a conversation first (not strictly required, but mirrors real usage).
    _ = client.post("/chat", json={"user_id": "u3", "conversation_id": "c9", "message": "start"})

    resp = client.post("/reset", json={"user_id": "u3", "conversation_id": "c9"})
    assert resp.status_code == 200
    payload = resp.json()

    assert payload["ok"] is True
    assert payload["thread_id"] == "bc::u3::c9"

    checkpointer = client._fake_checkpointer  # type: ignore[attr-defined]
    assert "bc::u3::c9" in checkpointer.deleted_threads


