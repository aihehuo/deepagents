"""Real tool-free graph and shared-core action isolation without providers."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from apps.group_agent_api.agent_factory.agent import save_group_profile
from apps.group_agent_api.agent_factory.cafe_agent import create_cafe_agent
from apps.group_agent_api.agent_factory.host_context import active_host_context, current_host_context
from apps.group_agent_api.agent_factory.host_policy import HostActionRejected
from apps.group_agent_api.agent_factory.integrations.match_backend import run_match
from apps.group_agent_api.agent_factory.integrations.profile_client import persist_group_profile
from apps.group_agent_api.agent_factory.invite import generate_invite_with_optional_llm
from apps.group_agent_api.agent_factory.profile_schema import profile_from_flat
from apps.group_agent_api.agent_factory.profile_store import disk_profile_path, load_profile, save_profile
from apps.group_agent_api.agent_factory.search_tool import search_candidates
from apps.group_agent_api.app import async_manager
from apps.group_agent_api.app.async_manager import execute_async_run_core
from apps.group_agent_api.app.models import AsyncCallRequest
from apps.group_agent_api.app.state import AppState
from tests.test_group_agent_host_da001 import _host, _request
from tests.test_group_agent_venue_da001 import _session


class ChatModel(BaseChatModel):
    """Provider-free model recording the actual final graph input and tools."""

    captured: list[list[Any]] = []
    bindings: list[list[Any]] = []
    output: AIMessage = AIMessage(content="只是聊聊也很好。")
    fail_once: bool = False
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "host-runtime-test"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ChatModel:
        """Record tools offered by a real graph.

        Args:
            tools: Model tool declarations.
            kwargs: Provider-specific binding options.
        """
        self.bindings.append(list(tools))
        return self

    def _generate(self, messages: list[Any], stop: Any = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        self.captured.append(messages)
        self.calls += 1
        if self.fail_once and self.calls == 1:
            raise RuntimeError("synthetic-provider-failure")
        return ChatResult(generations=[ChatGeneration(message=self.output.model_copy(deep=True))])


def _state(tmp_path: Path, model: ChatModel, *, saver: Any | None = None) -> AppState:
    legacy = create_agent(model=ChatModel(), tools=[], checkpointer=saver or InMemorySaver())
    return AppState(agent=legacy, base_dir=tmp_path, greeting_model=model)


def _profile(tmp_path: Path) -> Any:
    profile = profile_from_flat(user_id="101", group_id="global", doing="旧机器人项目", need="技术合伙人", offer="研发")
    save_profile(tmp_path, profile)
    return profile


@pytest.mark.parametrize("message", ["随便坐坐", "帮我找一个技术合伙人", "选2", "这里只是假设案例，别保存", "今天聊摄影，不聊旧机器人项目"])
async def test_core_has_no_tool_search_save_or_invite_side_effects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, message: str) -> None:
    profile = _profile(tmp_path)
    path = disk_profile_path(tmp_path, "101", "global")
    before = path.read_bytes()
    model = ChatModel(captured=[], bindings=[])
    state = _state(tmp_path, model)
    # Poison all orchestrator side-effect/restart paths. Any use must fail.
    forbidden: dict[str, MagicMock] = {}
    for name in ("assert_profile_persisted", "_should_force_profile_save", "_attempt_deterministic_profile_save", "generate_invite_with_optional_llm", "prior_recommendation_system_content", "parse_revisit_from_metadata", "_profile_save_succeeded_this_turn"):
        forbidden[name] = MagicMock(side_effect=AssertionError(name + " must not run"))
        monkeypatch.setattr(async_manager, name, forbidden[name])
    monkeypatch.setenv("GROUP_AGENT_TEST_LEVEL", "L1")
    from apps.group_agent_api.agent_factory import search_tool
    search_harvest = MagicMock(side_effect=AssertionError("no search harvest"))
    monkeypatch.setattr(search_tool, "extract_search_this_turn", search_harvest)
    raw = _request(index=2)
    raw["message"] = message
    from tests.test_group_agent_venue_da001 import _refingerprint
    req = AsyncCallRequest.model_validate(_refingerprint(raw))
    emit = AsyncMock(return_value=True)
    gate = MagicMock()
    await execute_async_run_core(req=req, session=_session(req), state=state, tid="same-authorized-thread", emit_callback=emit, side_effect_gate=gate)
    assert path.read_bytes() == before
    assert load_profile(tmp_path, "101", "global") == profile
    assert all(mock.call_count == 0 for mock in forbidden.values())
    assert search_harvest.call_count == 0
    assert not any(model.bindings)
    assert "tools" not in state.cafe_agent.get_graph().nodes
    final = next(call.args[1] for call in emit.await_args_list if call.args[0] == "final")
    assert final["profile_persisted"] is False
    assert final["profile_status"] == "host_read_only"
    assert final["candidates"] == [] and final["search_log"] is None
    assert final["invite_text"] is None and final["mentioned_user_ids"] == []
    assert final["profile"] is None
    assert all(call.args[0] not in {"match", "invite", "profile_force_save", "profile_deterministic_save"} for call in gate.call_args_list)
    assert current_host_context() is None


@pytest.mark.parametrize("action", ["save_tool", "save_store", "save_http", "search_tool", "match_backend", "invite", "profile_cas"])
async def test_direct_business_boundaries_reject_before_io(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: str) -> None:
    import requests

    http = MagicMock(side_effect=AssertionError("business HTTP must not run"))
    monkeypatch.setattr(requests.sessions.Session, "request", http)
    profile = _profile(tmp_path)
    before = {str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    config = {"metadata": {"user_id": "101", "group_id": "global", "base_dir": str(tmp_path), "run_match": True}}
    with active_host_context(_host(tmp_path)):
        with pytest.raises(HostActionRejected):
            if action == "save_tool":
                save_group_profile.invoke({"doing": "新项目", "need": "合伙人", "offer": "资金"}, config)
            elif action == "save_store":
                save_profile(tmp_path, profile)
            elif action == "save_http":
                persist_group_profile(profile=profile, run_id="synthetic", base_url="http://127.0.0.1:9", secret="PUBLIC-TEST-KEY")
            elif action == "search_tool":
                search_candidates.invoke({"query": "技术合伙人"}, config)
            elif action == "match_backend":
                run_match(query="技术", group_id="global")
            elif action == "profile_cas":
                from apps.group_agent_api.execution.active_fence import commit_profile_write_allowed

                commit_profile_write_allowed(user_id="101", group_id="global")
            else:
                generate_invite_with_optional_llm(profile=profile, candidates=[], match_status="empty", willing_to_at=True)
    assert before == {str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    http.assert_not_called()


@pytest.mark.parametrize("name", ["save_group_profile", "search_candidates", "write_file", "task", "write_todos"])
async def test_provider_tool_calls_rejected_without_any_execution(tmp_path: Path, name: str) -> None:
    model = ChatModel(output=AIMessage(content="", tool_calls=[{"id": "malicious", "name": name, "args": {}}]))
    state = _state(tmp_path, model)
    req = AsyncCallRequest.model_validate(_request(index=2))
    emit = AsyncMock(return_value=True)
    with pytest.raises(HostActionRejected, match="tool_call_rejected"):
        await execute_async_run_core(req=req, session=_session(req), state=state, tid="malicious-thread", emit_callback=emit)
    assert not any(call.args[0] == "final" for call in emit.await_args_list)
    assert not any(path.is_file() for path in tmp_path.rglob("*"))
    assert current_host_context() is None


async def test_old_checkpoint_pending_tool_is_not_resumed_or_history_deleted(tmp_path: Path) -> None:
    executions: list[str] = []

    @tool
    def search_candidates(query: str) -> str:
        """Poison tool to detect accidental resume of the original graph."""
        executions.append(query)
        raise AssertionError("old pending tool must not execute")

    saver = InMemorySaver()
    pending_model = ChatModel(output=AIMessage(content="之前的想法", tool_calls=[{"id": "old-call", "name": "search_candidates", "args": {"query": "robot"}}]))
    legacy = create_agent(model=pending_model, tools=[search_candidates], checkpointer=saver, interrupt_before=["tools"])
    config = {"configurable": {"thread_id": "preserved-thread"}}
    await legacy.ainvoke({"messages": [HumanMessage(content="之前聊机器人")]}, config)
    prior = await legacy.aget_state(config)
    assert prior.next == ("tools",)
    model = ChatModel(captured=[], bindings=[])
    state = AppState(agent=legacy, base_dir=tmp_path, greeting_model=model)
    req = AsyncCallRequest.model_validate(_request(index=2))
    await execute_async_run_core(req=req, session=_session(req), state=state, tid="preserved-thread", emit_callback=AsyncMock(return_value=True))
    assert executions == []
    saved = await state.cafe_agent.aget_state(config)
    assert any(item.content == "之前聊机器人" for item in saved.values["messages"])
    assert any(getattr(item, "tool_calls", None) for item in saved.values["messages"])
    assert any(item.content == "只是聊聊也很好。" for item in saved.values["messages"])
    assert not any(isinstance(item, ToolMessage) or getattr(item, "tool_calls", None) for item in model.captured[-1])
    assert not saved.next


async def test_day_night_and_failed_retry_preserve_scope_and_readonly_policy(tmp_path: Path) -> None:
    model = ChatModel(captured=[], bindings=[], fail_once=True)
    state = _state(tmp_path, model)
    req = AsyncCallRequest.model_validate(_request(index=2))
    frozen = req.model_dump()
    with pytest.raises(RuntimeError, match="synthetic-provider-failure"):
        await execute_async_run_core(req=req, session=_session(req), state=state, tid="retry-thread", emit_callback=AsyncMock(return_value=True))
    await execute_async_run_core(req=req, session=_session(req), state=state, tid="retry-thread", emit_callback=AsyncMock(return_value=True))
    assert req.model_dump() == frozen
    day = _host(tmp_path)
    night = _host(tmp_path, index=1)
    graph = state.cafe_agent
    async def invoke(host: Any, tid: str) -> None:
        with active_host_context(host):
            await graph.ainvoke({"messages": [HumanMessage(content="今天只是坐坐")]}, {"configurable": {"thread_id": tid}})
    await asyncio.gather(invoke(day, "day-thread"), invoke(night, "night-thread"))
    assert any("日间 Barista" in str(items[0].content) for items in model.captured)
    assert any("夜间 Bartender" in str(items[0].content) for items in model.captured)
    assert not any(model.bindings)
    assert current_host_context() is None


async def test_legacy_policy_can_still_write_and_host_graph_requires_context(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    with active_host_context(_host(tmp_path, version=1)):
        assert save_profile(tmp_path, profile).exists()
    with active_host_context(None):
        graph = create_cafe_agent(model=ChatModel(), checkpointer=InMemorySaver())
        with pytest.raises(HostActionRejected, match="context_required"):
            await graph.ainvoke({"messages": [HumanMessage(content="hello")]}, {"configurable": {"thread_id": "untrusted"}})


@pytest.mark.parametrize("missing", ["model", "checkpointer"])
def test_missing_host_executor_fails_closed(missing: str) -> None:
    with pytest.raises(HostActionRejected, match="executor_unavailable"):
        create_cafe_agent(model=None if missing == "model" else ChatModel(), checkpointer=None if missing == "checkpointer" else InMemorySaver())


async def test_host_model_requires_worker_fence_when_durable(tmp_path: Path) -> None:
    from apps.group_agent_api.execution.active_fence import FenceRejectedError, clear_active_fence, set_active_fence

    model = ChatModel(captured=[])
    graph = create_cafe_agent(model=model, checkpointer=InMemorySaver())
    tokens = set_active_fence(None, require=True)
    try:
        with active_host_context(_host(tmp_path)):
            with pytest.raises(FenceRejectedError, match="fence_required"):
                await graph.ainvoke({"messages": [HumanMessage(content="hello")]}, {"configurable": {"thread_id": "missing-fence"}})
        assert model.calls == 0
    finally:
        clear_active_fence(tokens)
