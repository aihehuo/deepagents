"""Cafe Host v4 paid search execution and pending-offer tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from apps.group_agent_api.agent_factory.intent_routing.canonical import canonical_hash
from apps.group_agent_api.app.async_manager import execute_async_run_core
from apps.group_agent_api.app.cafe_host_search import (
    bound_search_credential,
    canonical_query_plan,
    message_sha256,
    routing_snapshot_hash,
)
from apps.group_agent_api.app.chat_access import (
    ChatAccessContext,
    ChatAccessRejected,
    active_chat_access,
)
from apps.group_agent_api.app.execution_modes import calculate_v4_fingerprint
from apps.group_agent_api.app.models import AsyncCallRequest
from apps.group_agent_api.app.utils import thread_id
from skillpacks.aihehuo.integration import _enforce_chat_access
from skillpacks.aihehuo.manage import install
from tests.test_cafe_host_access_da001 import _active_with_token, _v4_body
from tests.test_group_agent_host_da002 import ChatModel, _profile, _state
from tests.test_group_agent_venue_da001 import _session
from tests.unit_tests.group_agent_api.planner_fixtures import (
    PlannerModel,
    StructuredPlanner,
    opportunity_query,
    people_query,
    proposal,
)

TOKEN = "synthetic-user-bearer"


def _enable_search(monkeypatch: pytest.MonkeyPatch, user_id: str) -> None:
    monkeypatch.setenv("CAFE_HOST_V4_SEARCH_EXECUTION_ENABLED", "1")
    monkeypatch.setenv("CAFE_HOST_V4_SEARCH_USER_ALLOWLIST", user_id)
    monkeypatch.setenv("CAFE_HOST_V4_SEARCH_LOG_SCHEMA_VERSION", "1")
    monkeypatch.setenv("CAFE_HOST_V4_SEARCH_RESULT_SCHEMA_VERSION", "1")


def _ready(intent: str, message: str) -> tuple[AsyncCallRequest, Any]:
    req, session = _active_with_token(TOKEN)
    body = req.model_dump()
    body["message"] = message
    body["chat_access"]["confirmed_search_intent"] = intent
    body["request_fingerprint"] = calculate_v4_fingerprint(body)
    return AsyncCallRequest.model_validate(body), session


def _install(tmp_path: Path) -> None:
    install(
        (
            "get-ai-cofounders",
            "get-ai-opportunities",
            "get-ai-project-partnerships",
        ),
        target=tmp_path / "skills" / "aihehuo",
    )


def _fake_fetch(calls: list[str]):
    def fetch(path: str, **_kwargs: object) -> str:
        calls.append(path)
        if "/users" in path:
            payload = {
                "data": [{"user_id": "101", "name": "周餐饮", "bio": "上海餐饮品牌"}]
            }
        elif "/ideas" in path:
            payload = {
                "data": [{"id": 11, "title": "餐饮新品牌", "summary": "社区餐饮"}]
            }
        elif "/projects" in path:
            payload = {"data": [{"id": 21, "title": "餐饮项目", "summary": "寻找合作"}]}
        else:
            payload = {"data": []}
        return json.dumps(payload, ensure_ascii=False)

    return fetch


async def _run(
    req: AsyncCallRequest,
    session: Any,
    tmp_path: Path,
    planner: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, dict[str, Any]]]:
    import skillpacks.aihehuo.integration as integration

    calls: list[str] = []
    monkeypatch.setattr(integration, "fetch", _fake_fetch(calls))
    model = ChatModel(
        captured=[], bindings=[], output=AIMessage(content="周餐饮在做餐饮品牌。")
    )
    state = _state(tmp_path, model)
    state.quality_model = planner
    _profile(tmp_path)
    _install(tmp_path)
    emitted: list[tuple[str, dict[str, Any]]] = []

    async def emit(event: str, payload: dict[str, Any]) -> bool:
        emitted.append((event, payload))
        return True

    await execute_async_run_core(
        req=req,
        session=session,
        state=state,
        tid=thread_id(
            user_id=req.user_id,
            group_id=req.group_id,
            conversation_id=req.conversation_id,
            episode_id=req.episode_id,
            execution_mode=req.execution_mode,
        ),
        emit_callback=emit,
    )
    emitted.append(("__fetch__", {"paths": calls}))
    return emitted


def _final(emitted: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    return next(payload for event, payload in emitted if event == "final")


def _paths(emitted: list[tuple[str, dict[str, Any]]]) -> list[str]:
    return next(payload["paths"] for event, payload in emitted if event == "__fetch__")


@pytest.mark.asyncio
async def test_flag_off_keeps_confirmed_search_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    req, session = _ready("people_search", "可以，帮我找上海的餐饮合伙人")
    emitted = await _run(
        req,
        session,
        tmp_path,
        PlannerModel(StructuredPlanner([people_query("上海 餐饮 合伙人")])),
        monkeypatch,
    )
    final = _final(emitted)
    assert final["chat_access_control"]["code"] == "search_capability_unavailable"
    assert final["candidates"] == []
    assert "pending_search_offer" not in final
    assert _paths(emitted) == []


@pytest.mark.asyncio
async def test_ordinary_offer_does_not_call_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CAFE_HOST_PENDING_SEARCH_OFFER_ENABLED", "1")
    req = AsyncCallRequest.model_validate(_v4_body())
    session = _session(req)
    emitted = await _run(
        req,
        session,
        tmp_path,
        PlannerModel(StructuredPlanner([people_query("上海 餐饮 合伙人")])),
        monkeypatch,
    )
    final = _final(emitted)
    assert final["chat_access_control"]["code"] == "chat_session_required"
    assert "记下这次查找" in final["reply"]
    assert "正在" not in final["reply"]
    offer = final["pending_search_offer"]
    assert offer["intent"] == "people_search"
    assert offer["query_plan_hash"] == canonical_hash(offer["query_plan"])
    assert offer["source_message_sha256"] == message_sha256(req.message or "")
    assert final["candidates"] == []
    assert _paths(emitted) == []


@pytest.mark.asyncio
async def test_confirmed_people_opportunity_and_project_keep_primary_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    req, session = _ready("people_search", "可以，帮我找上海的餐饮合伙人")
    _enable_search(monkeypatch, req.user_id)
    people = await _run(
        req,
        session,
        tmp_path,
        PlannerModel(StructuredPlanner([people_query("上海 餐饮 合伙人")])),
        monkeypatch,
    )
    people_final = _final(people)
    assert "chat_access_control" not in people_final
    assert people_final["candidates"][0]["display_name"] == "周餐饮"
    assert people_final["search_projection"]["bar_entries"][0]["id"] == "101"
    assert len(people_final["search_projection"]["bar_entries"]) <= 3
    assert (
        people_final["search_receipt"]["projection_digest"]
        == people_final["search_projection"]["digest"]
    )
    assert any("/users" in path for path in _paths(people))

    opp_req, opp_session = _ready("opportunity_search", "可以，帮我找餐饮创业机会")
    opportunity = await _run(
        opp_req,
        opp_session,
        tmp_path,
        PlannerModel(StructuredPlanner([opportunity_query("餐饮 创业机会")])),
        monkeypatch,
    )
    opp_final = _final(opportunity)
    assert opp_final["opportunities"][0]["kind"] == "ideas"
    assert opp_final["candidates"] == []
    assert all("/projects" not in path for path in _paths(opportunity))

    project_req, project_session = _ready(
        "project_partnership_search", "可以，帮我找餐饮项目合作"
    )
    projects = await _run(
        project_req,
        project_session,
        tmp_path,
        PlannerModel(
            StructuredPlanner(
                [
                    proposal(
                        intent="project_partnership_search", projects="餐饮 项目合作"
                    )
                ]
            )
        ),
        monkeypatch,
    )
    project_final = _final(projects)
    assert project_final["projects"][0]["id"] == "21"
    assert "opportunities" not in project_final
    assert all("/ideas" not in path for path in _paths(projects))


@pytest.mark.asyncio
async def test_continuation_uses_frozen_plan_not_client_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    req, session = _ready("people_search", "可以，帮我找上海的餐饮合伙人")
    _enable_search(monkeypatch, req.user_id)
    plan = canonical_query_plan("people_search", {"users": "上海 餐饮 合伙人"})
    body = req.model_dump()
    body["trusted_search"] = {
        "version": 1,
        "status": "ready",
        "pending_search_ref": "psr_synthetic_0001",
        "query_plan": plan,
        "query_plan_hash": canonical_hash(plan),
        "routing_snapshot_hash": routing_snapshot_hash(plan),
        "original_message_sha256": message_sha256(body["message"]),
    }
    body["request_fingerprint"] = calculate_v4_fingerprint(body)
    ready = AsyncCallRequest.model_validate(body)
    emitted = await _run(
        ready,
        session,
        tmp_path,
        PlannerModel(
            StructuredPlanner([proposal(action="reply", intent="conversation")])
        ),
        monkeypatch,
    )
    final = _final(emitted)
    assert "chat_access_control" not in final
    assert any("/users" in path for path in _paths(emitted))

    rewritten = dict(body)
    rewritten["message"] = "随便聊聊别的话题"
    rewritten["request_fingerprint"] = calculate_v4_fingerprint(rewritten)
    blocked = await _run(
        AsyncCallRequest.model_validate(rewritten),
        session,
        tmp_path,
        PlannerModel(StructuredPlanner([people_query("上海 餐饮 合伙人")])),
        monkeypatch,
    )
    assert (
        _final(blocked)["chat_access_control"]["code"] == "pending_search_unavailable"
    )
    assert _paths(blocked) == []


def test_wrapper_rejects_ordinary_and_unbound_active_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ordinary = ChatAccessContext.from_request(
        AsyncCallRequest.model_validate(_v4_body())
    )
    assert ordinary is not None
    with active_chat_access(ordinary):
        with pytest.raises(ChatAccessRejected, match="chat_session_required"):
            _enforce_chat_access("users")
    req, _session = _active_with_token(TOKEN)
    _enable_search(monkeypatch, req.user_id)
    access = ChatAccessContext.from_request(req)
    assert access is not None
    with active_chat_access(access):
        with pytest.raises(ChatAccessRejected, match="search_capability_unavailable"):
            _enforce_chat_access("users")
        with bound_search_credential(TOKEN):
            _enforce_chat_access("users")
