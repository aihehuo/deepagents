"""Cafe Host model-selected reads stay behind a frozen server link."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from redis import Redis

from apps.group_agent_api.agent_factory.intent_routing.canonical import canonical_hash
from apps.group_agent_api.app.async_manager import execute_async_run_core
from apps.group_agent_api.app.cafe_host_model_tools import (
    LINK_MODEL_TOOLS,
    LINK_PLANNER,
    active_cafe_host_link,
    cafe_host_link_for_admission,
    current_cafe_host_link,
    decision_from_tool_args,
    should_reuse_shown_results,
)
from apps.group_agent_api.execution.active_fence import (
    ActiveAttemptFence,
    clear_active_fence,
    set_active_fence,
)
from apps.group_agent_api.execution.config import load_durable_queue_config
from apps.group_agent_api.execution.crypto import digest_id
from apps.group_agent_api.execution.models import ExecutionRecord, ExecutionStatus
from apps.group_agent_api.execution.redis_store import ExecutionStore
from apps.group_agent_worker.event_loop import run_on_worker_event_loop
from apps.group_agent_api.app.execution_modes import (
    calculate_v4_fingerprint,
    v4_queue_binding,
)
from apps.group_agent_api.app.models import AsyncCallRequest
from apps.group_agent_api.app.utils import thread_id
from apps.group_agent_api.execution.admission import build_trusted_envelope
from skillpacks.aihehuo.manage import install
from tests.test_cafe_host_access_da001 import _v4_body
from tests.test_cafe_host_access_da002 import _enable_search, _final, _paths, _session
from tests.test_group_agent_host_da002 import ChatModel, _profile, _state


TOKEN = "synthetic-user-bearer"


class BoomPlanner:
    """Fail if the retired planner is invoked on the new link."""

    def invoke(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("planner must not run")

    async def ainvoke(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("planner must not run")


class ScriptedModel(ChatModel):
    """Return a fixed sequence of model messages."""

    script: list[Any] = []

    def _generate(self, messages: list[Any], stop: Any = None, run_manager: Any = None, **kwargs: Any):  # type: ignore[override]
        from langchain_core.outputs import ChatGeneration, ChatResult

        self.captured.append(messages)
        self.calls += 1
        output = self.script[min(self.calls - 1, len(self.script) - 1)] if self.script else self.output
        return ChatResult(generations=[ChatGeneration(message=output.model_copy(deep=True))])


def _install(tmp_path: Path) -> None:
    install(
        (
            "get-ai-cofounders",
            "get-ai-opportunities",
            "get-ai-project-partnerships",
            "get-ai-communities",
            "get-ai-startup-digest",
            "get-ai-platform-help",
        ),
        target=tmp_path / "skills" / "aihehuo",
    )


def _active(
    message: str,
    *,
    run_id: str | None = None,
    idempotency_key: str | None = None,
) -> tuple[AsyncCallRequest, Any]:
    body = _v4_body(active=True, confirmed=False)
    body["message"] = message
    if run_id is not None:
        body["run_id"] = run_id
    if idempotency_key is not None:
        body["idempotency_key"] = idempotency_key
    body["credential_binding"] = {
        "user_token_sha256": hashlib.sha256(TOKEN.encode()).hexdigest(),
        "group_token_sha256": "-",
    }
    body["request_fingerprint"] = calculate_v4_fingerprint(body)
    req = AsyncCallRequest.model_validate(body)
    base = _session(req)
    session = replace(
        base,
        principal=replace(base.principal, user_token=TOKEN, group_token=None),
        group_token=None,
    )
    return req, session


def _ordinary(
    message: str, *, run_id: str | None = None, idempotency_key: str | None = None
) -> tuple[AsyncCallRequest, Any]:
    body = _v4_body()
    body["message"] = message
    if run_id is not None:
        body["run_id"] = run_id
    if idempotency_key is not None:
        body["idempotency_key"] = idempotency_key
    body["request_fingerprint"] = calculate_v4_fingerprint(body)
    req = AsyncCallRequest.model_validate(body)
    return req, _session(req)


def _fake_fetch(calls: list[str]):
    def fetch(path: str, **_kwargs: object) -> str:
        calls.append(path)
        if "/users" in path:
            payload = {"data": [{"user_id": "101", "name": "周餐饮", "bio": "上海餐饮品牌"}]}
        elif "/groups" in path or "wechat_groups" in path:
            payload = {"data": [{"id": "e763", "name": "上海创业社群", "summary": "公开社群"}]}
        elif "/faqs" in path or "faq" in path:
            payload = {"data": [{"question": "如何实名认证", "answer": "在个人资料页提交证件。"}]}
        else:
            payload = {"data": []}
        return json.dumps(payload, ensure_ascii=False)

    return fetch


def _search_event_queries(emitted: list[tuple[str, dict[str, Any]]]) -> list[str]:
    queries: list[str] = []
    for event, payload in emitted:
        if event != "search_log" or not isinstance(payload, dict):
            continue
        body = payload.get("search_log")
        source = body if isinstance(body, dict) else payload
        queries.append(str(source.get("query") or ""))
    return queries


async def _run(
    req: AsyncCallRequest,
    session: Any,
    tmp_path: Path,
    model: ScriptedModel,
    monkeypatch: pytest.MonkeyPatch,
    *,
    state: Any | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    import skillpacks.aihehuo.integration as integration

    calls: list[str] = []
    monkeypatch.setattr(integration, "fetch", _fake_fetch(calls))
    if state is None:
        state = _state(tmp_path, model)
    else:
        state.greeting_model = model
    state.quality_model = BoomPlanner()
    state.planner_model = BoomPlanner()
    _profile(tmp_path)
    _install(tmp_path)
    emitted: list[tuple[str, dict[str, Any]]] = []

    async def emit(event: str, payload: dict[str, Any]) -> bool:
        emitted.append((event, payload))
        return True

    with active_cafe_host_link(LINK_MODEL_TOOLS):
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
    emitted.append(("__fetch__", {"paths": calls, "model_calls": model.calls}))
    return emitted


def test_admission_freezes_link_and_ignores_later_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    req = AsyncCallRequest.model_validate(_v4_body())
    monkeypatch.delenv("CAFE_HOST_MODEL_SELECTED_TOOLS", raising=False)
    assert cafe_host_link_for_admission(req) == LINK_PLANNER
    monkeypatch.setenv("CAFE_HOST_MODEL_SELECTED_TOOLS", "1")
    assert cafe_host_link_for_admission(req) == LINK_MODEL_TOOLS
    envelope = build_trusted_envelope(req=req, session=_session(req), thread_id="synthetic-thread")
    assert envelope["cafe_host_link"] == LINK_MODEL_TOOLS
    monkeypatch.delenv("CAFE_HOST_MODEL_SELECTED_TOOLS", raising=False)
    assert envelope["cafe_host_link"] == LINK_MODEL_TOOLS


def test_tool_schema_rejects_foreign_fields() -> None:
    decision = decision_from_tool_args("search_people", {"query": "上海 餐饮合伙人"})
    assert decision.primary_intent == "people_search"
    assert decision.action == "execute"
    assert decision.resource_queries.users == "上海 餐饮合伙人"
    assert decision.decision_source == "rules"
    with pytest.raises(ValueError, match="invalid_arguments"):
        decision_from_tool_args("search_people", {"query": "上海", "user_id": "1"})
    with pytest.raises(ValueError, match="invalid_arguments"):
        decision_from_tool_args("search_people", {"query": "   "})
    digest = decision_from_tool_args("read_startup_digest", {})
    assert digest.primary_intent == "startup_digest"
    assert not digest.resource_queries.non_empty()


@pytest.mark.asyncio
async def test_greeting_does_not_search_or_call_planner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = ScriptedModel(captured=[], bindings=[], script=[AIMessage(content="你好，随便坐。")])
    req, session = _ordinary("你好")
    emitted = await _run(req, session, tmp_path, model, monkeypatch)
    final = _final(emitted)
    assert "随便坐" in final["reply"]
    assert "chat_access_control" not in final
    assert final["candidates"] == []
    assert _paths(emitted) == []
    assert emitted[-1][1]["model_calls"] == 1
    progress = [payload.get("display_code") for event, payload in emitted if event == "progress"]
    assert "searching_public_people" not in progress


@pytest.mark.asyncio
async def test_explicit_community_request_without_tool_cannot_invent_groups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = ScriptedModel(
        captured=[], bindings=[],
        script=[AIMessage(content="爱合伙平台有真实活跃的沪上餐饮新势力社群，每月三场活动。")],
    )
    req, session = _ordinary("我想进入上海或者周边地区的创业群")
    emitted = await _run(req, session, tmp_path, model, monkeypatch)
    final = _final(emitted)
    assert "没有完成公开数据查询" in final["reply"]
    assert "沪上餐饮新势力" not in final["reply"]
    assert _paths(emitted) == []


@pytest.mark.asyncio
async def test_missed_explicit_community_lookup_gets_one_model_tool_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = ScriptedModel(
        captured=[], bindings=[],
        script=[
            AIMessage(content="平台有沪上餐饮新势力社群。"),
            AIMessage(content="", tool_calls=[
                {"name": "search_communities", "id": "call-groups", "args": {"query": "上海 周边 创业社群"}}
            ]),
            AIMessage(content="我查到了上海创业社群。"),
        ],
    )
    req, session = _active("我想进入上海或者周边地区的创业群")
    _enable_search(monkeypatch, req.user_id)
    emitted = await _run(req, session, tmp_path, model, monkeypatch)
    assert model.calls == 3
    assert any("wechat_groups" in path for path in _paths(emitted))
    assert "沪上餐饮新势力" not in _final(emitted)["reply"]


def test_reuse_requires_same_resource_and_a_result_reference() -> None:
    marker = SystemMessage(
        content='[CLIENT-RECOMMENDATION-CARDS]\n'
        '[{"card_ref":"candidate_1","display_name":"周餐饮"}]'
        '\n[/CLIENT-RECOMMENDATION-CARDS]'
    )
    messages = [marker, HumanMessage(content="找上海餐饮合伙人")]
    assert should_reuse_shown_results(messages, "第一个人怎么样", "search_people")
    assert not should_reuse_shown_results(messages, "我想进入上海的创业群", "search_communities")
    assert not should_reuse_shown_results(messages, "你来调用工具搜索", "search_communities")
    assert not should_reuse_shown_results(messages, "请找北京的合伙人", "search_people")
    assert not should_reuse_shown_results(messages, "帮我找和刚才第一个类似的人", "search_people")


@pytest.mark.asyncio
async def test_active_people_search_freezes_before_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = ScriptedModel(
        captured=[],
        bindings=[],
        script=[
            AIMessage(
                content="",
                tool_calls=[{"name": "search_people", "id": "call-people", "args": {"query": "上海 餐饮合伙人"}}],
            ),
            AIMessage(content="我查到了周餐饮，可以点开卡片看公开资料。"),
        ],
    )
    req, session = _active("帮我找上海的餐饮合伙人")
    _enable_search(monkeypatch, req.user_id)
    emitted = await _run(req, session, tmp_path, model, monkeypatch)
    final = _final(emitted)
    assert "周餐饮" in final["reply"]
    assert "可以点开卡片" in final["reply"]
    assert final["search_log"] is None
    queries = _search_event_queries(emitted)
    assert queries and "上海 餐饮合伙人" in queries[0]
    assert final["candidates"]
    assert "chat_access_control" not in final
    assert any("/users" in path for path in _paths(emitted))
    assert "searching_public_people" in [
        payload.get("display_code") for event, payload in emitted if event == "progress"
    ]


@pytest.mark.asyncio
async def test_final_search_log_uses_frozen_query_when_schema_is_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = ScriptedModel(
        captured=[],
        bindings=[],
        script=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_people",
                        "id": "call-people",
                        "args": {"query": "上海 餐饮合伙人"},
                    }
                ],
            ),
            AIMessage(content="我查到了周餐饮，可以点开卡片看公开资料。"),
        ],
    )
    req, session = _active("帮我找上海的餐饮合伙人")
    _enable_search(monkeypatch, req.user_id)
    monkeypatch.delenv("CAFE_HOST_V4_SEARCH_LOG_SCHEMA_VERSION", raising=False)
    final = _final(await _run(req, session, tmp_path, model, monkeypatch))
    assert final["search_log"]["query"] == "上海 餐饮合伙人"
    assert final["search_log"]["query"] != req.message


@pytest.mark.asyncio
async def test_ordinary_paid_search_creates_offer_without_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CAFE_HOST_PENDING_SEARCH_OFFER_ENABLED", "1")
    model = ScriptedModel(
        captured=[],
        bindings=[],
        script=[
            AIMessage(
                content="",
                tool_calls=[{"name": "search_people", "id": "call-paid", "args": {"query": "上海 餐饮合伙人"}}],
            ),
            AIMessage(content="我已经帮你搜索到了三位合伙人。"),
        ],
    )
    req, session = _ordinary("帮我找上海的餐饮合伙人")
    emitted = await _run(req, session, tmp_path, model, monkeypatch)
    final = _final(emitted)
    assert final["chat_access_control"]["code"] == "chat_session_required"
    assert final["pending_search_offer"]["intent"] == "people_search"
    offer = final["pending_search_offer"]
    assert offer["query_plan"]["queries"]["users"] == "上海 餐饮合伙人"
    assert offer["query_plan_hash"] == canonical_hash(offer["query_plan"])
    assert final["candidates"] == []
    assert _paths(emitted) == []
    assert "搜索到了" not in final["reply"]


@pytest.mark.asyncio
async def test_ordinary_community_and_digest_need_drink_but_faq_stays_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CAFE_HOST_PENDING_SEARCH_OFFER_ENABLED", "1")
    community = ScriptedModel(
        captured=[],
        bindings=[],
        script=[
            AIMessage(
                content="",
                tool_calls=[{"name": "search_communities", "id": "call-group", "args": {"query": "上海 创业社群"}}],
            ),
            AIMessage(content="我查到了上海创业社群。"),
        ],
    )
    req, session = _ordinary("帮我找上海的创业社群")
    emitted = await _run(req, session, tmp_path, community, monkeypatch)
    final = _final(emitted)
    assert final["chat_access_control"]["code"] == "chat_session_required"
    assert final["pending_search_offer"]["intent"] == "community_search"
    assert _paths(emitted) == []

    digest = ScriptedModel(
        captured=[],
        bindings=[],
        script=[
            AIMessage(
                content="",
                tool_calls=[{"name": "read_startup_digest", "id": "call-digest", "args": {}}],
            ),
            AIMessage(content="我已经读到了近期博客。"),
        ],
    )
    req, session = _ordinary("最近有哪些创业博客")
    emitted = await _run(req, session, tmp_path, digest, monkeypatch)
    final = _final(emitted)
    assert final["chat_access_control"]["code"] == "chat_session_required"
    assert final["pending_search_offer"]["intent"] == "startup_digest"
    assert _paths(emitted) == []

    faq = ScriptedModel(
        captured=[],
        bindings=[],
        script=[
            AIMessage(
                content="",
                tool_calls=[{"name": "search_platform_faq", "id": "call-faq", "args": {"query": "实名认证"}}],
            ),
            AIMessage(content="我按公开说明回答。"),
        ],
    )
    req, session = _ordinary("爱合伙平台如何完成实名认证")
    emitted = await _run(req, session, tmp_path, faq, monkeypatch)
    final = _final(emitted)
    assert "chat_access_control" not in final
    assert "pending_search_offer" not in final
    assert any("faq" in path for path in _paths(emitted))


@pytest.mark.asyncio
async def test_active_community_and_digest_can_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for tool_name, message, args, expected_path in (
        ("search_communities", "帮我找上海创业社群", {"query": "上海 创业社群"}, "group"),
        ("read_startup_digest", "最近有哪些创业博客", {}, "blog"),
    ):
        model = ScriptedModel(
            captured=[],
            bindings=[],
            script=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": tool_name, "id": "call-read", "args": args}],
                ),
                AIMessage(content="我按刚才查到的公开结果回答。"),
            ],
        )
        req, session = _active(message)
        _enable_search(monkeypatch, req.user_id)
        emitted = await _run(req, session, tmp_path, model, monkeypatch)
        final = _final(emitted)
        assert "chat_access_control" not in final
        assert any(expected_path in path for path in _paths(emitted))
        resolved = [payload for event, payload in emitted if event == "progress" and payload.get("phase") == "intent_resolved"]
        assert len(resolved) == 1
        assert resolved[0]["intent"] in {"community_search", "startup_digest"}
        assert resolved[0]["routing_snapshot_hash"].startswith("sha256:")
        assert resolved[0]["plan_id"] is None
        assert resolved[0]["plan_step_id"] is None
        assert resolved[0]["request_index"] is None
        tool_progress = [payload for event, payload in emitted if event == "progress" and payload.get("phase") == "tool_started"]
        assert tool_progress and tool_progress[0]["plan_id"] is not None


@pytest.mark.asyncio
async def test_parallel_and_write_tools_do_not_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = ScriptedModel(
        captured=[],
        bindings=[],
        script=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "search_people", "id": "call-a", "args": {"query": "上海"}},
                    {"name": "search_projects", "id": "call-b", "args": {"query": "餐饮"}},
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[{"name": "search_people", "id": "call-again", "args": {"query": "北京 投资人"}}],
            ),
        ],
    )
    req, session = _active("同时找人和项目")
    _enable_search(monkeypatch, req.user_id)
    emitted = await _run(req, session, tmp_path, model, monkeypatch)
    assert _paths(emitted) == []
    assert "北京" not in _final(emitted)["reply"]

    write = ScriptedModel(
        captured=[],
        bindings=[],
        script=[
            AIMessage(
                content="",
                tool_calls=[{"name": "save_group_profile", "id": "call-write", "args": {"doing": "x"}}],
            ),
            AIMessage(content="这个我做不了。"),
        ],
    )
    req, session = _active("帮我保存画像")
    emitted = await _run(req, session, tmp_path, write, monkeypatch)
    assert _paths(emitted) == []


@pytest.mark.asyncio
async def test_invalid_tool_arguments_do_not_fail_the_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = ScriptedModel(
        captured=[],
        bindings=[],
        script=[
            AIMessage(
                content="",
                invalid_tool_calls=[
                    {
                        "name": "search_people",
                        "id": "bad-json",
                        "args": "{",
                        "error": "bad json",
                    }
                ],
            ),
            AIMessage(content="这个条件我没法查，你换个说法。"),
        ],
    )
    req, session = _active("帮我找人")
    _enable_search(monkeypatch, req.user_id)
    emitted = await _run(req, session, tmp_path, model, monkeypatch)
    final = _final(emitted)
    assert _paths(emitted) == []
    assert "换个说法" in final["reply"]
    assert "chat_access_control" not in final


@pytest.mark.asyncio
async def test_follow_up_reuses_shown_results_until_user_asks_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = ScriptedModel(
        captured=[],
        bindings=[],
        script=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_people",
                        "id": "call-1",
                        "args": {"query": "上海 餐饮合伙人"},
                    }
                ],
            ),
            AIMessage(content="我查到了周餐饮，可以点开卡片看公开资料。"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_people",
                        "id": "call-2",
                        "args": {"query": "北京 投资人"},
                    }
                ],
            ),
            AIMessage(content="还是刚才的周餐饮。"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_people",
                        "id": "call-3",
                        "args": {"query": "北京 投资人"},
                    }
                ],
            ),
            AIMessage(content="按新条件又查了一次。"),
        ],
    )
    state = _state(tmp_path, model)
    first_req, session = _active("帮我找上海的餐饮合伙人")
    _enable_search(monkeypatch, first_req.user_id)
    first = await _run(first_req, session, tmp_path, model, monkeypatch, state=state)
    assert any("/users" in path for path in _paths(first))

    follow_req, _session_unused = _active(
        "第一个人怎么样",
        run_id="00000000-0000-4000-8000-000000000301",
        idempotency_key="synthetic-access-cafe-follow",
    )
    follow = await _run(follow_req, session, tmp_path, model, monkeypatch, state=state)
    assert _paths(follow) == []
    assert "北京" not in _final(follow)["reply"]
    assert "周餐饮" in _final(follow)["reply"]

    again_req, _session_unused = _active(
        "再搜一下北京的合伙人",
        run_id="00000000-0000-4000-8000-000000000302",
        idempotency_key="synthetic-access-cafe-again",
    )
    again = await _run(again_req, session, tmp_path, model, monkeypatch, state=state)
    assert any("/users" in path for path in _paths(again))


@pytest.mark.asyncio
async def test_people_history_cannot_suppress_new_community_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = ScriptedModel(
        captured=[], bindings=[],
        script=[
            AIMessage(content="", tool_calls=[
                {"name": "search_people", "id": "call-people", "args": {"query": "上海 餐饮合伙人"}}
            ]),
            AIMessage(content="我查到了周餐饮，可以查看公开卡片。"),
            AIMessage(content="", tool_calls=[
                {"name": "search_communities", "id": "call-groups", "args": {"query": "上海 周边 创业社群"}}
            ]),
            AIMessage(content="我查到了上海创业社群。"),
        ],
    )
    state = _state(tmp_path, model)
    first_req, session = _active("找上海餐饮合伙人")
    _enable_search(monkeypatch, first_req.user_id)
    first = await _run(first_req, session, tmp_path, model, monkeypatch, state=state)
    assert any("/users" in path for path in _paths(first))
    next_req, _ = _active(
        "我想进入上海或者周边地区的创业群",
        run_id="00000000-0000-4000-8000-000000000303",
        idempotency_key="synthetic-access-cafe-groups",
    )
    second = await _run(next_req, session, tmp_path, model, monkeypatch, state=state)
    assert any("wechat_groups" in path for path in _paths(second))
    assert "chat_access_control" not in _final(second)


@pytest.mark.asyncio
async def test_ordinary_community_search_after_people_results_reaches_drink_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CAFE_HOST_PENDING_SEARCH_OFFER_ENABLED", "1")
    model = ScriptedModel(
        captured=[], bindings=[],
        script=[
            AIMessage(content="", tool_calls=[
                {"name": "search_people", "id": "call-people", "args": {"query": "上海 餐饮合伙人"}}
            ]),
            AIMessage(content="我查到了周餐饮，可以查看公开卡片。"),
            AIMessage(content="", tool_calls=[
                {"name": "search_communities", "id": "call-groups", "args": {"query": "上海 周边 创业社群"}}
            ]),
            AIMessage(content="这次没有执行搜索。"),
        ],
    )
    state = _state(tmp_path, model)
    first_req, first_session = _active("找上海餐饮合伙人")
    _enable_search(monkeypatch, first_req.user_id)
    await _run(first_req, first_session, tmp_path, model, monkeypatch, state=state)
    next_req, next_session = _ordinary(
        "你来调用工具搜索上海周边创业群",
        run_id="00000000-0000-4000-8000-000000000305",
        idempotency_key="synthetic-access-cafe-groups-ordinary",
    )
    emitted = await _run(next_req, next_session, tmp_path, model, monkeypatch, state=state)
    final = _final(emitted)
    assert final["chat_access_control"]["code"] == "chat_session_required"
    assert final["pending_search_offer"]["intent"] == "community_search"
    assert _paths(emitted) == []


@pytest.mark.asyncio
async def test_historical_person_followup_cannot_invent_profile_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = ScriptedModel(
        captured=[], bindings=[],
        script=[
            AIMessage(content="", tool_calls=[
                {"name": "search_people", "id": "call-person", "args": {"query": "上海 餐饮合伙人"}}
            ]),
            AIMessage(content="我查到了周餐饮，可以查看公开卡片。"),
            AIMessage(content="周餐饮有五年连锁餐饮经历，还持有智能排班专利。"),
        ],
    )
    state = _state(tmp_path, model)
    first_req, session = _active("找上海餐饮合伙人")
    _enable_search(monkeypatch, first_req.user_id)
    await _run(first_req, session, tmp_path, model, monkeypatch, state=state)
    follow_req, _ = _active(
        "我想看看第一个结果",
        run_id="00000000-0000-4000-8000-000000000304",
        idempotency_key="synthetic-access-cafe-person-detail",
    )
    follow = await _run(follow_req, session, tmp_path, model, monkeypatch, state=state)
    reply = _final(follow)["reply"]
    assert "周餐饮" in reply
    assert "专利" not in reply
    assert "五年" not in reply
    assert _paths(follow) == []


@pytest.mark.asyncio
async def test_model_reply_cannot_name_people_outside_the_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = ScriptedModel(
        captured=[],
        bindings=[],
        script=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_people",
                        "id": "call-people",
                        "args": {"query": "上海 餐饮合伙人"},
                    }
                ],
            ),
            AIMessage(content="另外李明也合适，周餐饮也可以。"),
        ],
    )
    req, session = _active("帮我找上海的餐饮合伙人")
    _enable_search(monkeypatch, req.user_id)
    final = _final(await _run(req, session, tmp_path, model, monkeypatch))
    assert "李明" not in final["reply"]
    assert "周餐饮" in final["reply"]


REDIS_URL = os.environ.get("GROUP_AGENT_TEST_REDIS_URL", "redis://127.0.0.1:6379/15")


@pytest.fixture
def durable(monkeypatch: pytest.MonkeyPatch):
    prefix = f"ga:test:cafe003:{uuid.uuid4().hex[:10]}"
    key = base64.b64encode(bytes(range(32))).decode()
    monkeypatch.setenv("GROUP_AGENT_DURABLE_QUEUE_ENABLED", "1")
    monkeypatch.setenv("GROUP_AGENT_REDIS_URL", REDIS_URL)
    monkeypatch.setenv("GROUP_AGENT_REDIS_PREFIX", prefix)
    monkeypatch.setenv("GROUP_AGENT_QUEUE_PAYLOAD_KEYS", f"cafe-k1:{key}")
    monkeypatch.setenv("GROUP_AGENT_QUEUE_PAYLOAD_CURRENT_VERSION", "cafe-k1")
    monkeypatch.setenv("GROUP_AGENT_LEASE_TTL_S", "30")
    monkeypatch.setenv("GROUP_AGENT_HEARTBEAT_INTERVAL_S", "5")
    monkeypatch.setenv("GROUP_AGENT_VISIBILITY_TIMEOUT_S", "240")
    monkeypatch.setenv("GROUP_AGENT_TASK_SOFT_LIMIT_S", "150")
    monkeypatch.setenv("GROUP_AGENT_TASK_HARD_LIMIT_S", "180")
    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
    config = load_durable_queue_config(require_enabled=True)
    assert config is not None
    client = Redis.from_url(REDIS_URL, decode_responses=True)
    try:
        client.ping()
    except Exception as exc:
        pytest.skip(f"isolated Redis unavailable: {exc}")
    store = ExecutionStore(client, config)
    yield store
    for redis_key in client.scan_iter(match=f"{prefix}:*", count=200):
        client.delete(redis_key)


def _seed(store: ExecutionStore, req: AsyncCallRequest):
    store.create_or_get(
        record=ExecutionRecord(
            run_id=req.run_id,
            idempotency_key=req.idempotency_key,
            request_schema_version=4,
            request_fingerprint=req.request_fingerprint,
            queue_schema_version=4,
            status=ExecutionStatus.ACCEPTED,
            created_at=store.redis_time(),
            conversation_id=req.conversation_id,
            user_id_digest=digest_id(req.user_id),
            group_id_digest=digest_id(req.group_id),
            mode_binding=v4_queue_binding(req),
        )
    )
    store.mark_queued(req.run_id, expected_status="accepted")
    outcome = store.claim_lease(
        run_id=req.run_id,
        conversation_id=req.conversation_id,
        owner="cafe-worker",
    )
    assert outcome.claim is not None
    return outcome.claim


def _activate(store: ExecutionStore, claim: Any, conversation_id: str):
    return set_active_fence(
        ActiveAttemptFence(
            store=store,
            claim=claim,
            conversation_id=conversation_id,
            cancel_event=threading.Event(),
        ),
        require=True,
    )


def _take_over(store: ExecutionStore, claim: Any, req: AsyncCallRequest):
    store.schedule_retry(
        claim,
        conversation_id=req.conversation_id,
        next_attempt_at=store.redis_time() - 1,
        error_code="worker_lost",
    )
    assert store.promote_retry_wait(req.run_id) == "ok"
    store.mark_queued(req.run_id, expected_status="enqueue_failed")
    outcome = store.claim_lease(
        run_id=req.run_id,
        conversation_id=req.conversation_id,
        owner="cafe-worker-2",
    )
    assert outcome.claim is not None
    return outcome.claim


def test_worker_loop_copies_frozen_link(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CAFE_HOST_MODEL_SELECTED_TOOLS", raising=False)
    seen: dict[str, str] = {}

    async def probe() -> None:
        seen["link"] = current_cafe_host_link()

    with active_cafe_host_link(LINK_MODEL_TOOLS):
        run_on_worker_event_loop(probe())
    assert seen["link"] == LINK_MODEL_TOOLS
    assert current_cafe_host_link() == LINK_PLANNER
    assert cafe_host_link_for_admission(
        AsyncCallRequest.model_validate(_v4_body())
    ) == LINK_PLANNER


@pytest.mark.asyncio
async def test_takeover_reuses_frozen_people_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, durable: ExecutionStore
) -> None:
    req, session = _active("帮我找上海的餐饮合伙人")
    _enable_search(monkeypatch, req.user_id)
    first_claim = _seed(durable, req)
    tokens = _activate(durable, first_claim, req.conversation_id)
    model = ScriptedModel(
        captured=[],
        bindings=[],
        script=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_people",
                        "id": "call-people",
                        "args": {"query": "上海 餐饮合伙人"},
                    }
                ],
            ),
            AIMessage(content="我查到了周餐饮，可以点开卡片看公开资料。"),
        ],
    )
    try:
        first = await _run(req, session, tmp_path, model, monkeypatch)
    finally:
        clear_active_fence(tokens)
    assert any("/users" in path for path in _paths(first))
    first_queries = _search_event_queries(first)
    assert first_queries and "上海 餐饮合伙人" in first_queries[0]
    first_query = first_queries[0]

    monkeypatch.delenv("CAFE_HOST_MODEL_SELECTED_TOOLS", raising=False)
    second_claim = _take_over(durable, first_claim, req)
    tokens = _activate(durable, second_claim, req.conversation_id)
    assert durable.get(req.run_id).routing_snapshot_hash
    replay = ScriptedModel(
        captured=[],
        bindings=[],
        script=[
            AIMessage(
                content="我改查北京。",
                tool_calls=[
                    {
                        "name": "search_people",
                        "id": "call-other",
                        "args": {"query": "北京 投资人"},
                    }
                ],
            )
        ],
    )
    try:
        second = await _run(req, session, tmp_path, replay, monkeypatch)
    finally:
        clear_active_fence(tokens)
    assert _paths(second) == []
    final = _final(second)
    assert replay.calls >= 1
    assert final["candidates"][0]["display_name"] == "周餐饮"
    assert "北京" not in final["reply"]
    second_queries = _search_event_queries(second)
    assert all("北京" not in query for query in second_queries)
    if second_queries:
        assert second_queries[0] == first_query


@pytest.mark.asyncio
async def test_takeover_replays_the_same_pending_offer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, durable: ExecutionStore
) -> None:
    monkeypatch.setenv("CAFE_HOST_PENDING_SEARCH_OFFER_ENABLED", "1")
    req, session = _ordinary("帮我找上海的餐饮合伙人")
    first_claim = _seed(durable, req)
    tokens = _activate(durable, first_claim, req.conversation_id)
    model = ScriptedModel(
        captured=[],
        bindings=[],
        script=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_people",
                        "id": "call-paid",
                        "args": {"query": "上海 餐饮合伙人"},
                    }
                ],
            ),
            AIMessage(content="我已经帮你搜索到了三位合伙人。"),
        ],
    )
    try:
        first = await _run(req, session, tmp_path, model, monkeypatch)
    finally:
        clear_active_fence(tokens)
    offer = _final(first)["pending_search_offer"]
    assert offer["query_plan_hash"] == canonical_hash(offer["query_plan"])

    second_claim = _take_over(durable, first_claim, req)
    tokens = _activate(durable, second_claim, req.conversation_id)
    replay = ScriptedModel(
        captured=[],
        bindings=[],
        script=[AIMessage(content="换成找北京的项目。")],
    )
    try:
        second = await _run(req, session, tmp_path, replay, monkeypatch)
    finally:
        clear_active_fence(tokens)
    again = _final(second)["pending_search_offer"]
    assert again["query_plan_hash"] == offer["query_plan_hash"]
    assert again["query_plan"] == offer["query_plan"]
    assert _paths(second) == []
    assert replay.calls == 1
