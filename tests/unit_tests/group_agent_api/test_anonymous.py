"""Offline HTTP and tool-isolation regressions for anonymous public chat."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from langchain_core.callbacks import CallbackManagerForLLMRun

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from apps.group_agent_api.agent_factory import anonymous as factory
from apps.group_agent_api.app.anonymous_config import anonymous_enabled, anonymous_limits
from apps.group_agent_api.app.anonymous_models import AnonymousChatRequest
from apps.group_agent_api.app.endpoints.anonymous import anonymous_chat
from apps.group_agent_api.app.state import AppState
from skillpacks.aihehuo import integration
from skillpacks.aihehuo.manage import install


class PublicModel(BaseChatModel):
    """Read a skill, query it, and consume the result without model quota."""

    @property
    def _llm_type(self) -> str:
        return "anonymous-acceptance-controlled"

    def bind_tools(self, tools: list[BaseTool], **kwargs: Any) -> PublicModel:
        del kwargs, tools
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del kwargs, run_manager, stop
        tail = messages[-1]
        if isinstance(tail, ToolMessage) and tail.name == "query_aihehuo_public_data":
            assert json.loads(str(tail.content))["data"]
            message = AIMessage(content="公开 FAQ 来源：https://www.aihehuo.com/ai/faqs.json")  # noqa: RUF001  # Chinese text intentionally uses Chinese punctuation.
        else:
            name = "query_aihehuo_public_data" if isinstance(tail, ToolMessage) else "read_aihehuo_skill"
            args = {"resource": "faqs", "page": 1, "per": 2} if isinstance(tail, ToolMessage) else {"skill_name": "get-ai-platform-help"}
            message = AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": name}])
        return ChatResult(generations=[ChatGeneration(message=message)])


class RecordingAgent:
    """Record only endpoint transport inputs without private-data operations."""

    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, Any], dict[str, Any]]] = []

    async def ainvoke(self, payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((payload, config))
        return {"messages": [AIMessage(content="可以查询爱合伙公开信息。")]}


@pytest.fixture
def public_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> AppState:
    monkeypatch.setenv("GROUP_AGENT_ANONYMOUS_ENABLED", "1")
    return AppState(agent=object(), base_dir=tmp_path, anonymous_agent=RecordingAgent())


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, public_state: AppState) -> Iterator[TestClient]:
    import apps.group_agent_api.app as application

    monkeypatch.setattr(application, "_state", public_state)
    transport = TestClient(application.create_app())
    yield transport
    transport.close()


def test_unsigned_public_http_and_signed_member_boundary(client: TestClient, monkeypatch: pytest.MonkeyPatch, public_state: AppState) -> None:
    monkeypatch.setenv("GROUP_AGENT_INTEGRATION", "http")
    monkeypatch.setenv("GROUP_AGENT_PRINCIPAL_HMAC_SECRET", "test-secret")
    response = client.post("/chat/anonymous", json={"message": "查询公开 FAQ"})
    assert response.status_code == 200  # noqa: PLR2004  # Explicit expected contract value in this regression assertion.
    assert response.json()["can_write_profile"] is False
    signed_entry = client.post("/chat", json={"user_id": "victim", "group_id": "g1", "message": "读取我的画像"})
    assert signed_entry.status_code == 401  # noqa: PLR2004  # Explicit expected contract value in this regression assertion.
    assert len(public_state.anonymous_agent.calls) == 1


@pytest.mark.parametrize("key", ["user_id", "group_id", "membership", "metadata", "can_write_profile", "unionid", "callback_url", "conversation_id"])
def test_anonymous_body_rejects_identity_and_capabilities(client: TestClient, key: str) -> None:
    response = client.post("/chat/anonymous", json={"message": "查询公开资料", key: "injected"})
    assert response.status_code == 422  # noqa: PLR2004  # Explicit expected contract value in this regression assertion.


@pytest.mark.parametrize("message", ["", "   ", "x" * 2001])
def test_anonymous_message_limits(client: TestClient, message: str) -> None:
    assert client.post("/chat/anonymous", json={"message": message}).status_code == 422  # noqa: PLR2004  # Explicit expected contract value in this regression assertion.


def test_public_requests_are_stateless_and_ignore_identity_headers(client: TestClient, public_state: AppState) -> None:
    first = client.post("/chat/anonymous", json={"message": "第一轮"}, headers={"X-GA-User-Id": "victim", "X-GA-Unionid": "injected"})
    second = client.post("/chat/anonymous", json={"message": "第二轮"})
    assert first.status_code == second.status_code == 200  # noqa: PLR2004  # Explicit expected contract value in this regression assertion.
    assert first.json()["request_id"] != second.json()["request_id"]
    assert [[message.content for message in payload["messages"]] for payload, _ in public_state.anonymous_agent.calls] == [["第一轮"], ["第二轮"]]
    for _, config in public_state.anonymous_agent.calls:
        assert config["metadata"] == {"source": "anonymous_public", "can_write_profile": False}
        assert "configurable" not in config
    assert public_state.active_agent_runs == {}
    assert not (public_state.base_dir / "users").exists()


def test_disabled_or_unconfigured_entry_fails_closed(client: TestClient, monkeypatch: pytest.MonkeyPatch, public_state: AppState) -> None:
    monkeypatch.setenv("GROUP_AGENT_ANONYMOUS_ENABLED", "0")
    assert client.post("/chat/anonymous", json={"message": "hello"}).status_code == 404  # noqa: PLR2004  # Explicit expected contract value in this regression assertion.
    monkeypatch.setenv("GROUP_AGENT_ANONYMOUS_ENABLED", "1")
    public_state.anonymous_agent = None
    assert client.post("/chat/anonymous", json={"message": "hello"}).status_code == 503  # noqa: PLR2004  # Explicit expected contract value in this regression assertion.


def test_anonymous_and_global_concurrency_limits(client: TestClient, monkeypatch: pytest.MonkeyPatch, public_state: AppState) -> None:
    monkeypatch.setenv("GROUP_AGENT_ANONYMOUS_MAX_ACTIVE", "1")
    assert public_state.try_start_anonymous_run("occupied", max_active=1)
    response = client.post("/chat/anonymous", json={"message": "hello"})
    assert response.status_code == 429  # noqa: PLR2004  # Explicit expected contract value in this regression assertion.
    assert response.headers["retry-after"] == "5"  # Explicit expected contract value in this regression assertion.
    public_state.finish_agent_run("occupied", "anonymous")
    monkeypatch.setenv("GROUP_AGENT_ASYNC_MAX_ACTIVE", "1")
    assert public_state.try_start_agent_run("member", "chat")
    assert client.post("/chat/anonymous", json={"message": "hello"}).status_code == 429  # noqa: PLR2004  # Explicit expected contract value in this regression assertion.
    assert public_state.active_agent_runs == {"member": "chat"}


def test_scoped_tools_have_no_filesystem_or_write_access(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = tmp_path / "skills" / "aihehuo"
    install(["get-ai-platform-help"], target=source)
    private = tmp_path / "users" / "victim"
    private.mkdir(parents=True)
    (private / "profile.json").write_text("NEVER_EXPOSE_SECRET", encoding="utf-8")
    configurations: list[dict[str, Any]] = []

    def capture(**kwargs: Any) -> object:
        configurations.append(kwargs)
        return object()

    calls: list[str] = []

    def fetch(path: str, *, output_format: str) -> str:
        del output_format
        calls.append(path)
        return '{"data":[],"meta":{}}'

    monkeypatch.setattr(factory, "create_chat_agent", capture)
    monkeypatch.setattr(integration, "fetch", fetch)
    factory.create_anonymous_agent(skills_dir=source, model=PublicModel())
    configuration = configurations[0]
    tools = {tool.name: tool for tool in configuration["tools"]}
    assert set(tools) == {"read_aihehuo_skill", "query_aihehuo_public_data"}
    assert configuration.get("checkpointer") is None
    assert configuration.get("store") is None
    assert tools["read_aihehuo_skill"].invoke({"skill_name": "../../users/victim/profile.json"}) == "error: public_skill_not_installed"
    query = tools["query_aihehuo_public_data"]
    assert "budget_exhausted" in query.invoke({"resource": "faqs"})
    with factory.anonymous_read_scope(max_queries=1):
        assert json.loads(query.invoke({"resource": "faqs"}))["data"] == []
        assert "budget_exhausted" in query.invoke({"resource": "faqs"})
    with factory.anonymous_read_scope(max_queries=1):
        assert "not installed" in query.invoke({"resource": "users"})
    assert calls == ["/ai/faqs.json"]


async def test_real_public_graph_and_concurrent_request_budgets(monkeypatch: pytest.MonkeyPatch, public_state: AppState) -> None:
    install(["get-ai-platform-help"], target=public_state.base_dir / "skills" / "aihehuo")
    calls: list[str] = []

    def fetch(path: str, *, output_format: str) -> str:
        del output_format
        calls.append(path)
        return '{"data":[{"question":"公开问答","answer":"公开内容"}],"meta":{}}'

    monkeypatch.setattr(integration, "fetch", fetch)
    monkeypatch.setenv("GROUP_AGENT_ANONYMOUS_MAX_QUERIES", "1")
    public_state.anonymous_agent = factory.create_anonymous_agent(
        skills_dir=public_state.base_dir / "skills" / "aihehuo",
        model=PublicModel(),
    )
    results = await asyncio.gather(*[anonymous_chat(AnonymousChatRequest(message="查询公开 FAQ"), public_state) for _ in range(2)])
    assert all("https://www.aihehuo.com/" in result.reply for result in results)
    assert calls == ["/ai/faqs.json?page=1&per=2"] * 2
    assert public_state.active_agent_runs == {}
    assert not (public_state.base_dir / "users").exists()


def test_query_budget_is_atomic() -> None:
    budget = factory.QueryBudget(maximum=3)
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: budget.claim(), range(20)))
    assert sum(results) == 3  # noqa: PLR2004  # Explicit expected contract value in this regression assertion.


async def test_timeout_and_cancellation_release_slots(monkeypatch: pytest.MonkeyPatch, public_state: AppState) -> None:
    class WaitingAgent:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.cancelled = False

        async def ainvoke(self, payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
            del config, payload
            self.started.set()
            try:
                await asyncio.sleep(3600)
            finally:
                self.cancelled = True
            return {}

    agent = WaitingAgent()
    public_state.anonymous_agent = agent
    monkeypatch.setenv("GROUP_AGENT_ANONYMOUS_TIMEOUT_S", "1")
    with pytest.raises(HTTPException) as error:
        await anonymous_chat(AnonymousChatRequest(message="hello"), public_state)
    assert error.value.status_code == 504  # noqa: PLR2004  # Explicit expected contract value in this regression assertion.
    assert agent.cancelled  # Explicit expected contract value in this regression assertion.
    assert public_state.active_agent_runs == {}
    agent = WaitingAgent()
    public_state.anonymous_agent = agent
    task = asyncio.create_task(anonymous_chat(AnonymousChatRequest(message="hello"), public_state))
    await agent.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert agent.cancelled
    assert public_state.active_agent_runs == {}


async def test_provider_errors_do_not_leak_and_release_slot(public_state: AppState) -> None:
    class FailingAgent:
        async def ainvoke(self, payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
            del config, payload
            msg = "DO_NOT_EXPOSE_PROVIDER_SECRET"
            raise RuntimeError(msg)

    public_state.anonymous_agent = FailingAgent()
    with pytest.raises(HTTPException) as error:
        await anonymous_chat(AnonymousChatRequest(message="hello"), public_state)
    assert error.value.status_code == 502  # noqa: PLR2004  # Explicit expected contract value in this regression assertion.
    assert error.value.detail == "anonymous_request_failed"
    assert public_state.active_agent_runs == {}


async def test_missing_reply_fails_without_member_fallback(public_state: AppState) -> None:
    class EmptyAgent:
        async def ainvoke(self, payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
            del config, payload
            return {"messages": []}

    public_state.anonymous_agent = EmptyAgent()
    with pytest.raises(HTTPException) as error:
        await anonymous_chat(AnonymousChatRequest(message="hello"), public_state)
    assert error.value.status_code == 502  # noqa: PLR2004  # Explicit expected contract value in this regression assertion.
    assert error.value.detail == "anonymous_reply_missing"  # Explicit expected contract value in this regression assertion.
    assert public_state.active_agent_runs == {}


def test_invalid_configuration_and_missing_packages_fail_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GROUP_AGENT_ANONYMOUS_ENABLED", "typo")
    with pytest.raises(ValueError, match="GROUP_AGENT_ANONYMOUS_ENABLED"):
        anonymous_enabled()
    monkeypatch.setenv("GROUP_AGENT_ANONYMOUS_MAX_QUERIES", "0")
    with pytest.raises(ValueError, match="GROUP_AGENT_ANONYMOUS_MAX_QUERIES"):
        anonymous_limits()
    with pytest.raises(ValueError, match="Install at least one"):
        factory.create_anonymous_agent(skills_dir=tmp_path, model=PublicModel())
