"""Opt-in, low-volume public API acceptance without production callbacks."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.tools import BaseTool

import json
import os
from pathlib import Path
from typing import Any, NoReturn

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver

from skillpacks.aihehuo import integration
from skillpacks.aihehuo.client import HOST, build_path
from skillpacks.aihehuo.integration import create_aihehuo_tool
from skillpacks.aihehuo.manage import install

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("AIHEHUO_PUBLIC_LIVE_TEST") != "1",
        reason="Set AIHEHUO_PUBLIC_LIVE_TEST=1 to make public HTTPS requests",
    ),
]

PACKS = [
    "get-ai-cofounders",
    "get-ai-opportunities",
    "get-ai-project-partnerships",
    "get-ai-communities",
    "get-ai-startup-digest",
    "get-ai-platform-help",
]


@pytest.fixture(scope="module")
def runtime(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("aihehuo-live")
    install(PACKS, target=directory / "skills" / "aihehuo")
    return directory


@pytest.fixture(scope="module")
def public_tool(runtime: Path) -> BaseTool:
    return create_aihehuo_tool(skills_dir=runtime / "skills" / "aihehuo")


@pytest.fixture(scope="module")
def evidence() -> list[dict[str, Any]]:
    return []


@pytest.fixture(scope="module", autouse=True)
def save_evidence(evidence: list[dict[str, Any]]) -> Iterator[None]:
    yield
    target = os.environ.get("AIHEHUO_LIVE_REPORT")
    if target:
        Path(target).write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(public_tool: BaseTool, evidence: list[dict[str, Any]], resource: str, **params: Any) -> dict[str, Any]:
    """Record schema and pagination only, without storing public profiles."""
    result = public_tool.invoke({"resource": resource, "per": 2, **params})
    assert not result.startswith("AIHEHUO query failed"), result
    document = json.loads(result)
    assert isinstance(document, dict)
    data = document.get("data")
    if resource == "group-members":
        assert isinstance(data, dict)
        assert str(data["id"]) == params["event_id"]
        assert isinstance(data.get("members"), list)
        items = data["members"]
    else:
        assert isinstance(data, list)
        items = data
    assert isinstance(document.get("meta"), dict)
    assert len(items) <= 2  # noqa: PLR2004  # Explicit expected contract value in this regression assertion.
    assert isinstance(document["meta"].get("has_more"), bool)
    query = {"per": 2, **params}
    event_id = query.pop("event_id", None)
    evidence.append(
        {
            "resource": resource,
            "url": f"https://{HOST}{build_path(resource, params=query, event_id=event_id)}",
            "count": len(items),
            "meta": document["meta"],
            "fields": sorted(items[0]) if items else [],
            "data_shape": "group_with_members" if resource == "group-members" else "list",
            "requested_per": 2,
        }
    )
    return document


@pytest.fixture(scope="module")
def groups(public_tool: BaseTool, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    return read_json(public_tool, evidence, "groups")


@pytest.mark.parametrize("resource", ["users", "ideas", "projects", "blog", "faqs"])
def test_live_list_schema_and_sources(public_tool: BaseTool, evidence: list[dict[str, Any]], resource: str) -> None:
    document = read_json(public_tool, evidence, resource)
    for item in document["data"]:
        assert isinstance(item, dict)
        if resource == "faqs" and item.get("url") is None:
            assert isinstance(item.get("question"), str)
            assert isinstance(item.get("answer"), str)
            assert item.get("source")
            continue
        assert isinstance(item.get("url"), str)
        assert item["url"].startswith("https://www.aihehuo.com/")


def test_live_group_members(public_tool: BaseTool, evidence: list[dict[str, Any]], groups: dict[str, Any]) -> None:
    assert groups["data"], "No public group available for member endpoint acceptance"
    group = groups["data"][0]
    event_id = str(group["id"])
    assert event_id.isascii()
    assert event_id.isdecimal()
    read_json(public_tool, evidence, "group-members", event_id=event_id)


def test_live_pagination(public_tool: BaseTool, evidence: list[dict[str, Any]], groups: dict[str, Any]) -> None:
    assert groups["meta"]["has_more"], "No second group page available for acceptance"
    page = groups["meta"]["next_page"]
    assert isinstance(page, int)
    assert page > 1
    second = read_json(public_tool, evidence, "groups", page=page)
    assert second["meta"]["current_page"] == page
    assert {item["id"] for item in groups["data"]}.isdisjoint({item["id"] for item in second["data"]})


def test_live_explicit_first_page_size(public_tool: BaseTool, evidence: list[dict[str, Any]]) -> None:
    first = read_json(public_tool, evidence, "groups", page=1)
    assert first["meta"]["per_page"] == 2, "Explicit page=1&per=2 was not honored"  # noqa: PLR2004  # Explicit expected contract value in this regression assertion.


def test_live_weekly(public_tool: BaseTool, evidence: list[dict[str, Any]]) -> None:
    text = public_tool.invoke({"resource": "weekly"})
    assert not text.startswith("AIHEHUO query failed"), text
    assert text.strip()
    assert "https://www.aihehuo.com/" in text
    evidence.append({"resource": "weekly", "url": "https://www.aihehuo.com/ai/blog/weekly_markdown", "characters": len(text)})


@pytest.mark.parametrize("resource", ["users", "ideas"])
def test_live_semantic_query(public_tool: BaseTool, evidence: list[dict[str, Any]], resource: str) -> None:
    read_json(public_tool, evidence, resource, q="技术合伙人")


def test_live_empty_keyword(public_tool: BaseTool, evidence: list[dict[str, Any]]) -> None:
    document = read_json(public_tool, evidence, "faqs", keyword="aihehuo_acceptance_no_match_7d952ef0")
    assert document["data"] == []
    assert document["meta"]["has_more"] is False


class PublicQueryModel(BaseChatModel):
    """Exercise real graph execution without model quota or uncontrolled calls."""

    invocations: int = 0

    @property
    def _llm_type(self) -> str:
        return "public-acceptance-controlled"

    def bind_tools(self, tools: list[BaseTool], **kwargs: Any) -> PublicQueryModel:
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
        self.invocations += 1
        assert self.invocations <= 2, "Acceptance must finish in two model steps"  # noqa: PLR2004  # Explicit expected contract value in this regression assertion.
        if isinstance(messages[-1], ToolMessage):
            document = json.loads(str(messages[-1].content))
            assert isinstance(document["data"], list)
            message = AIMessage(content="已读取公开 FAQ。来源：https://www.aihehuo.com/ai/faqs.json")  # noqa: RUF001  # Chinese text intentionally uses Chinese punctuation.
        else:
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "query_aihehuo_public_data",
                        "args": {"resource": "faqs", "per": 2},
                        "id": "acceptance-public",
                    }
                ],
            )
        return ChatResult(generations=[ChatGeneration(message=message)])


async def test_live_member_graph_with_controlled_model(monkeypatch: pytest.MonkeyPatch, runtime: Path, evidence: list[dict[str, Any]]) -> None:
    from apps.group_agent_api.agent_factory.agent import create_member_agent
    from apps.group_agent_api.agent_factory.integrations import match_backend
    from apps.group_agent_api.agent_factory.search_tool import extract_search_this_turn

    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    calls: list[str] = []
    fetch = integration.fetch

    def tracked_fetch(path: str, *, output_format: str) -> str:
        assert not calls, "Acceptance must make only one public query"
        calls.append(path)
        return fetch(path, output_format=output_format)

    def reject_legacy(*args: Any, **kwargs: Any) -> NoReturn:
        del args, kwargs
        pytest.fail("Public mode must not invoke the production matcher")

    monkeypatch.setattr("skillpacks.aihehuo.integration.fetch", tracked_fetch)
    monkeypatch.setattr(match_backend, "run_match", reject_legacy)
    graph = create_member_agent(
        runtime=runtime,
        model=PublicQueryModel(),
        checkpointer=InMemorySaver(),
        include_search=True,
        include_save=False,
    )
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="请查询爱合伙公开 FAQ")]},
        {
            "configurable": {"thread_id": "public-acceptance"},
            "recursion_limit": 16,
            "metadata": {
                "user_id": "acceptance_u1",
                "group_id": "acceptance_g1",
                "membership": "in_group",
                "base_dir": str(runtime),
                "can_write_profile": "false",
            },
        },
    )
    messages = result["messages"]
    results = [message for message in messages if isinstance(message, ToolMessage)]
    assert len(results) == 1
    assert results[0].name == "query_aihehuo_public_data"
    assert json.loads(str(results[0].content))["data"]
    assert calls == ["/ai/faqs.json?per=2"]
    assert not extract_search_this_turn(messages, 0).called
    assert not (runtime / "users").exists()
    evidence.append({"scenario": "member_graph_controlled_model", "public_requests": calls, "legacy_match_calls": 0, "profile_written": False})


@pytest.mark.skipif(
    os.environ.get("AIHEHUO_REAL_MODEL_TEST") != "1",
    reason="Also set AIHEHUO_REAL_MODEL_TEST=1 with explicit provider/model/base configuration",
)
async def test_live_member_graph_with_real_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, evidence: list[dict[str, Any]]) -> None:
    from apps.group_agent_api.agent_factory.agent import create_member_agent
    from apps.group_agent_api.agent_factory.integrations import match_backend
    from apps.group_agent_api.agent_factory.model_builder import create_model
    from apps.group_agent_api.app.startup import _load_dotenv_if_present

    _load_dotenv_if_present()
    for name in ("GROUP_AGENT_PROVIDER", "GROUP_AGENT_MODEL", "GROUP_AGENT_BASE_URL"):
        assert os.environ.get(name), f"Explicit {name} is required for real model acceptance"
    monkeypatch.setenv("GROUP_AGENT_MODEL_MODE", "real")
    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setenv("GROUP_AGENT_MAX_TOKENS", "1000")
    monkeypatch.setenv("GROUP_AGENT_TIMEOUT_S", "30")
    install(["get-ai-platform-help"], target=tmp_path / "skills" / "aihehuo")
    requests: list[str] = []
    fetch = integration.fetch

    def bounded_fetch(path: str, *, output_format: str) -> str:
        assert len(requests) < 2, "Real model acceptance allows at most two public requests"  # noqa: PLR2004  # Explicit expected contract value in this regression assertion.
        requests.append(path)
        return fetch(path, output_format=output_format)

    def reject_legacy(*args: Any, **kwargs: Any) -> NoReturn:
        del args, kwargs
        pytest.fail("Real model public mode must not invoke the production matcher")

    monkeypatch.setattr(integration, "fetch", bounded_fetch)
    monkeypatch.setattr(match_backend, "run_match", reject_legacy)
    graph = create_member_agent(
        runtime=tmp_path,
        model=create_model(),
        checkpointer=InMemorySaver(),
        include_search=True,
        include_save=False,
    )
    result = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "请通过已安装的技能查询爱合伙公开 FAQ，只查询一次，使用 page=1、per=2。"  # noqa: RUF001  # Chinese punctuation in the prompt.
                        "简要回答什么是爱合伙，并给出实际来源链接。不要写入画像。"  # noqa: RUF001  # Chinese punctuation in the prompt.
                    )
                )
            ]
        },
        {
            "configurable": {"thread_id": "public-real-model-acceptance"},
            "recursion_limit": 20,
            "metadata": {
                "user_id": "acceptance_u1",
                "group_id": "acceptance_g1",
                "membership": "in_group",
                "base_dir": str(tmp_path),
                "can_write_profile": "false",
            },
        },
    )
    tool_results = [message for message in result["messages"] if isinstance(message, ToolMessage) and message.name == "query_aihehuo_public_data"]
    assert tool_results, "Real model did not choose the public query tool"
    errors = [str(message.content) for message in tool_results if str(message.content).startswith("AIHEHUO query failed")]
    documents = [json.loads(str(message.content)) for message in tool_results if not str(message.content).startswith("AIHEHUO query failed")]
    assert any(document.get("data") for document in documents)
    assert "https://www.aihehuo.com/" in str(result["messages"][-1].content)
    assert not (tmp_path / "users").exists()
    evidence.append(
        {
            "scenario": "member_graph_real_model",
            "public_requests": requests,
            "legacy_match_calls": 0,
            "profile_written": False,
            "rejected_tool_queries": errors,
        }
    )


@pytest.mark.skipif(
    os.environ.get("AIHEHUO_REAL_MODEL_TEST") != "1",
    reason="Real model anonymous HTTP acceptance is explicitly opt-in",
)
def test_live_anonymous_http_with_real_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, evidence: list[dict[str, Any]]) -> None:
    from fastapi.testclient import TestClient

    import apps.group_agent_api.app as application
    from apps.group_agent_api.agent_factory import profile_store
    from apps.group_agent_api.agent_factory.anonymous import create_anonymous_agent
    from apps.group_agent_api.agent_factory.integrations import match_backend
    from apps.group_agent_api.agent_factory.model_builder import create_model
    from apps.group_agent_api.app.state import AppState

    for name in ("GROUP_AGENT_PROVIDER", "GROUP_AGENT_MODEL", "GROUP_AGENT_BASE_URL"):
        assert os.environ.get(name), f"Explicit {name} is required"
    monkeypatch.setenv("GROUP_AGENT_MODEL_MODE", "real")
    monkeypatch.setenv("GROUP_AGENT_MAX_TOKENS", "1000")
    monkeypatch.setenv("GROUP_AGENT_TIMEOUT_S", "30")
    monkeypatch.setenv("GROUP_AGENT_ANONYMOUS_ENABLED", "1")
    monkeypatch.setenv("GROUP_AGENT_ANONYMOUS_MAX_QUERIES", "2")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    source = tmp_path / "skills" / "aihehuo"
    install(["get-ai-platform-help"], target=source)
    requests: list[str] = []
    fetch = integration.fetch

    def tracked_fetch(path: str, *, output_format: str) -> str:
        requests.append(path)
        return fetch(path, output_format=output_format)

    def reject_private(*args: Any, **kwargs: Any) -> NoReturn:
        del args, kwargs
        pytest.fail("Anonymous HTTP must not access profiles or production matching")

    monkeypatch.setattr(integration, "fetch", tracked_fetch)
    monkeypatch.setattr(profile_store, "load_profile", reject_private)
    monkeypatch.setattr(profile_store, "save_profile", reject_private)
    monkeypatch.setattr(match_backend, "run_match", reject_private)
    state = AppState(
        agent=object(),
        base_dir=tmp_path,
        anonymous_agent=create_anonymous_agent(skills_dir=source, model=create_model()),
    )
    monkeypatch.setattr(application, "_state", state)
    client = TestClient(application.create_app())
    try:
        response = client.post(
            "/chat/anonymous",
            json={
                "message": "请通过已安装的技能查询一次公开 FAQ，使用 page=1、per=2。简要介绍什么是爱合伙，并给出真实来源链接。",  # noqa: RUF001  # Chinese text intentionally uses Chinese punctuation.
            },
        )
    finally:
        client.close()
    assert response.status_code == 200, response.text  # noqa: PLR2004  # Explicit expected contract value in this regression assertion.
    body = response.json()
    assert body["can_write_profile"] is False
    assert "https://www.aihehuo.com/" in body["reply"]
    assert 1 <= len(requests) <= 2  # noqa: PLR2004  # Explicit expected contract value in this regression assertion.
    assert not (tmp_path / "users").exists()
    assert state.active_agent_runs == {}
    evidence.append(
        {
            "scenario": "anonymous_http_real_model",
            "public_requests": requests,
            "http_status": 200,
            "signed_identity_provided": False,
            "profile_accessed": False,
            "profile_written": False,
            "legacy_match_calls": 0,
        }
    )
