from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import ValidationError

from apps.group_agent_api.agent_factory.intent_routing.classifier import (
    aclassify_intent,
)
from apps.group_agent_api.agent_factory.intent_routing.context import (
    IntentContextScope,
    completed_intent_context_message,
    extract_completed_context,
)
from apps.group_agent_api.agent_factory.intent_routing.resolver import (
    aresolve_intent,
    intent_classifier_timeout_s,
    intent_routing_mode,
    record_intent_outcome,
    record_intent_prediction,
    resolve_intent,
)
from apps.group_agent_api.agent_factory.intent_routing.schema import IntentDecisionV1
from apps.group_agent_api.agent_factory.capability import (
    CapabilityTier,
    SessionCapabilities,
)
from apps.group_agent_api.agent_factory.integrations.membership_client import (
    MembershipResult,
)
from apps.group_agent_api.agent_factory.integrations.principal import SessionPrincipal
from apps.group_agent_api.app.models import AsyncCallRequest, ChatRequest
from apps.group_agent_api.app.session import TrustedSession
from apps.group_agent_api.app.state import AppState


def _scope() -> IntentContextScope:
    return IntentContextScope(
        thread_id="ga::u::global::group_agent::c::episode-1",
        user_id="u",
        execution_mode="group_agent",
        episode_id="episode-1",
    )


@pytest.mark.parametrize(
    ("message", "intent", "action"),
    [
        ("有没有上海做餐饮的技术合伙人", "people_search", "execute"),
        ("找餐饮项目正在招募的合作席位", "project_partnership_search", "execute"),
        ("看看最近的 AI 创业机会", "opportunity_search", "execute"),
        ("浏览最新创业机会", "opportunity_search", "execute"),
        ("加入杭州创业社群", "community_search", "execute"),
        ("看看本周创业周报和文章", "startup_digest", "execute"),
        ("如何发布餐饮项目招合伙人", "platform_help", "execute"),
        ("找正在做餐饮项目的人", "people_search", "execute"),
        ("先别查，只是聊聊怎么找合伙人", "conversation", "conversation"),
        ("找合伙人是不是很难", "conversation", "conversation"),
        ("把我的所在地改成上海", "profile_update", "delegate_existing_flow"),
        ("你好，今天想聊聊创业", "conversation", "conversation"),
    ],
)
def test_frozen_rule_replays(message: str, intent: str, action: str) -> None:
    result = resolve_intent(message, execution_mode="group_agent", can_search=True)
    assert result.decision.primary_intent == intent
    assert result.decision.action == action


def test_read_policy_rejects_cafe_and_missing_permission() -> None:
    cafe = resolve_intent("帮我找合伙人", execution_mode="cafe_host", can_search=True)
    denied = resolve_intent(
        "帮我找合伙人", execution_mode="group_agent", can_search=False
    )
    assert cafe.decision.primary_intent == "people_search"
    assert cafe.decision.action == "reject"
    assert denied.decision.action == "reject"
    assert cafe.decision.query_plan_ref is None
    assert denied.decision.required_skill is None


def test_decision_schema_is_closed_and_action_bound() -> None:
    valid = resolve_intent(
        "帮我找上海的合伙人", execution_mode="group_agent", can_search=True
    ).decision
    assert valid.model_dump(mode="json")["query_plan_ref"] == {
        "plan_version": 1,
        "plan_id": "people-search-v1",
    }
    with pytest.raises(ValidationError):
        IntentDecisionV1.model_validate({**valid.model_dump(), "unknown": True})
    with pytest.raises(ValidationError):
        IntentDecisionV1.model_validate(
            {
                **valid.model_dump(),
                "required_skill": None,
                "query_plan_ref": None,
            }
        )


def _completed_people_search(*, success: bool = True) -> list[Any]:
    call = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "read-1",
                "name": "query_aihehuo_public_data",
                "args": {"resource": "users", "location": "北京", "role": "技术合伙人"},
            }
        ],
    )
    result = ToolMessage(
        content='{"status":"succeeded","items":[]}' if success else "error: timed_out",
        tool_call_id="read-1",
        name="query_aihehuo_public_data",
    )
    return [call, result, AIMessage(content="查询完成。")]


def test_context_inherits_only_completed_tool_arguments() -> None:
    messages = _completed_people_search()
    context = extract_completed_context(messages, scope=_scope())
    assert context is not None
    assert context.intent == "people_search"
    assert context.constraints.location == ("北京",)

    result = resolve_intent(
        "上海的呢",
        execution_mode="group_agent",
        can_search=True,
        prior_messages=messages,
        scope=_scope(),
    )
    assert result.context_inherited is True
    assert result.decision.primary_intent == "people_search"
    assert result.decision.constraints.location == ("上海",)
    assert result.decision.decision_source == "context"

    excluded = extract_completed_context(
        _completed_people_search(success=False), scope=_scope()
    )
    assert excluded is None
    incomplete = extract_completed_context(
        _completed_people_search()[:-1], scope=_scope()
    )
    assert incomplete is None
    failed_scope = IntentContextScope(
        thread_id=_scope().thread_id,
        user_id="u",
        execution_mode="group_agent",
        episode_id="episode-1",
        run_status="failed",
    )
    assert extract_completed_context(messages, scope=failed_scope) is None


def test_server_context_requires_completion_hash_and_exact_scope() -> None:
    decision = resolve_intent(
        "找上海餐饮创业者，优先有供应链经验",
        execution_mode="group_agent",
        can_search=True,
    ).decision
    marker = completed_intent_context_message(
        decision=decision,
        scope=_scope(),
        routing_snapshot_hash="sha256:" + "1" * 64,
        plan_id="people-search-v1",
        receipt_ids=["sha256:" + "2" * 64],
        user_message="找上海餐饮创业者，优先有供应链经验",
    )
    completed = [
        marker,
        HumanMessage(content="找上海餐饮创业者，优先有供应链经验"),
        AIMessage(content="查询完成。"),
    ]

    context = extract_completed_context(completed, scope=_scope())
    assert context is not None
    assert context.intent == "people_search"
    assert context.constraints.capability == ("供应链经验",)
    assert extract_completed_context([marker], scope=_scope()) is None
    assert extract_completed_context(
        completed,
        scope=IntentContextScope(
            thread_id="other-thread",
            user_id="u",
            execution_mode="group_agent",
            episode_id="episode-1",
        ),
    ) is None

    marker.additional_kwargs["group_agent_completed_intent_context_v1"][
        "plan_id"
    ] = "platform-help-v1"
    assert extract_completed_context(completed, scope=_scope()) is None


def test_failed_server_marker_cannot_be_completed_by_later_chat() -> None:
    decision = resolve_intent(
        "找上海技术合伙人",
        execution_mode="group_agent",
        can_search=True,
    ).decision
    marker = completed_intent_context_message(
        decision=decision,
        scope=_scope(),
        routing_snapshot_hash="sha256:" + "1" * 64,
        plan_id="people-search-v1",
        receipt_ids=["sha256:" + "2" * 64],
        user_message="找上海技术合伙人",
    )
    failed_then_chat = [
        marker,
        HumanMessage(content="找上海技术合伙人"),
        HumanMessage(content="今天天气怎么样"),
        AIMessage(content="今天天气不错。"),
    ]
    mismatched_same_turn = [
        marker,
        HumanMessage(content="今天天气怎么样"),
        AIMessage(content="今天天气不错。"),
    ]

    assert extract_completed_context(failed_then_chat, scope=_scope()) is None
    assert extract_completed_context(mismatched_same_turn, scope=_scope()) is None


def test_context_negation_removes_one_constraint_without_cancelling_search() -> None:
    result = resolve_intent(
        "不要北京",
        execution_mode="group_agent",
        can_search=True,
        prior_messages=_completed_people_search(),
        scope=_scope(),
    )
    assert result.decision.primary_intent == "people_search"
    assert result.decision.constraints.location == ()
    assert result.decision.negations == ("location:北京",)


def test_full_negation_never_inherits_completed_search() -> None:
    result = resolve_intent(
        "先不找了",
        execution_mode="group_agent",
        can_search=True,
        prior_messages=_completed_people_search(),
        scope=_scope(),
    )
    assert result.context_inherited is False
    assert result.decision.primary_intent == "conversation"
    assert result.decision.action == "conversation"


def test_ambiguous_object_requires_context_or_clarification() -> None:
    without_context = resolve_intent(
        "上海有什么合适的吗？",
        execution_mode="group_agent",
        can_search=True,
    )
    with_context = resolve_intent(
        "上海有什么合适的吗？",
        execution_mode="group_agent",
        can_search=True,
        prior_messages=_completed_people_search(),
        scope=_scope(),
    )
    assert without_context.decision.primary_intent == "clarification"
    assert without_context.decision.action == "clarify"
    assert with_context.context_inherited is True
    assert with_context.decision.primary_intent == "people_search"
    assert with_context.decision.constraints.location == ("上海",)


def test_new_complete_intent_overrides_inherited_route() -> None:
    result = resolve_intent(
        "换成帮我找杭州创业社群",
        execution_mode="group_agent",
        can_search=True,
        prior_messages=_completed_people_search(),
        scope=_scope(),
    )
    assert result.context_inherited is False
    assert result.decision.primary_intent == "community_search"
    assert result.decision.constraints.location == ("杭州",)


class _Structured:
    def __init__(self, outputs: list[Any], *, delay: float = 0.0) -> None:
        self.outputs = outputs
        self.delay = delay
        self.calls = 0

    def invoke(self, _: Any) -> Any:
        output = self.outputs[min(self.calls, len(self.outputs) - 1)]
        self.calls += 1
        return output

    async def ainvoke(self, _: Any) -> Any:
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.outputs[min(self.calls - 1, len(self.outputs) - 1)]


class _ClassifierModel:
    def __init__(self, structured: _Structured) -> None:
        self.structured = structured

    def with_structured_output(self, _: Any) -> _Structured:
        return self.structured


def test_classifier_schema_failure_retries_once_then_clarifies() -> None:
    structured = _Structured([{"bad": True}])
    result = resolve_intent(
        "帮我找一下",
        execution_mode="group_agent",
        can_search=True,
        classifier_model=_ClassifierModel(structured),
    )
    assert structured.calls == 2
    assert result.classifier_status == "invalid"
    assert result.decision.primary_intent == "clarification"
    assert result.decision.action == "clarify"


def test_valid_classifier_is_bounded_and_cannot_supply_policy_fields() -> None:
    structured = _Structured(
        [
            {
                "version": 1,
                "candidates": [{"intent": "community_search", "confidence": "medium"}],
                "evidence_spans": ["创业社群", "不在原文"],
                "missing_object": None,
                "classifier_version": "intent-classifier-v1",
            }
        ]
    )
    result = resolve_intent(
        "创业社群",
        execution_mode="group_agent",
        can_search=True,
        classifier_model=_ClassifierModel(structured),
    )
    assert result.classifier_status == "ok"
    assert result.decision.primary_intent == "community_search"
    assert result.decision.action == "execute"
    assert result.decision.evidence_spans == ("创业社群",)
    assert result.decision.required_skill == "get-ai-communities"


@pytest.mark.parametrize(
    "classification",
    [
        {
            "version": 1,
            "candidates": [{"intent": "people_search", "confidence": "high"}],
            "evidence_spans": [],
            "missing_object": "要找的人或资源",
            "classifier_version": "intent-classifier-v1",
        },
        {
            "version": 1,
            "candidates": [
                {"intent": "people_search", "confidence": "high"},
                {"intent": "community_search", "confidence": "high"},
            ],
            "evidence_spans": [],
            "missing_object": None,
            "classifier_version": "intent-classifier-v1",
        },
    ],
)
def test_classifier_missing_object_or_top_tie_clarifies(
    classification: dict[str, Any],
) -> None:
    result = resolve_intent(
        "帮我找一下",
        execution_mode="group_agent",
        can_search=True,
        classifier_model=_ClassifierModel(_Structured([classification])),
    )
    assert result.classifier_status == "ok"
    assert result.decision.primary_intent == "clarification"
    assert result.decision.action == "clarify"


async def test_classifier_timeout_fails_to_clarification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROUP_AGENT_INTENT_CLASSIFIER_TIMEOUT_S", "0.05")
    structured = _Structured(
        [
            {
                "version": 1,
                "candidates": [{"intent": "people_search", "confidence": "high"}],
                "evidence_spans": [],
                "missing_object": None,
                "classifier_version": "intent-classifier-v1",
            }
        ],
        delay=0.2,
    )
    result = await aresolve_intent(
        "帮我找一下",
        execution_mode="group_agent",
        can_search=True,
        classifier_model=_ClassifierModel(structured),
    )
    assert result.classifier_status == "timed_out"
    assert result.decision.primary_intent == "clarification"


async def test_sync_async_rule_paths_are_identical() -> None:
    sync = resolve_intent(
        "找正在做餐饮项目的人", execution_mode="group_agent", can_search=True
    )
    async_result = await aresolve_intent(
        "找正在做餐饮项目的人", execution_mode="group_agent", can_search=True
    )
    assert sync.decision == async_result.decision
    assert sync.signal_ids == async_result.signal_ids


def test_long_input_stays_bounded_and_linear() -> None:
    message = "普通描述" * 100_000 + " 有没有上海的技术合伙人"
    started = time.perf_counter()
    result = resolve_intent(message, execution_mode="group_agent", can_search=True)
    assert result.decision.primary_intent == "people_search"
    assert time.perf_counter() - started < 1.0


def test_observe_logs_have_confusion_fields_without_message_or_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, Any]] = []

    def record(*_: Any, **kwargs: Any) -> None:
        captured.append(kwargs["detail"])

    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.intent_routing.resolver.record_decision_point",
        record,
    )
    result = resolve_intent(
        "有没有上海的餐饮合伙人", execution_mode="group_agent", can_search=True
    )
    record_intent_prediction(result, run_id="run-1", thread_id="thread-1")
    record_intent_outcome(
        result, observed_route="search", run_id="run-1", thread_id="thread-1"
    )
    assert captured[1]["matched"] is True
    wire = repr(captured)
    assert "有没有上海" not in wire
    assert "evidence_spans" not in wire


def test_observe_config_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROUP_AGENT_INTENT_ROUTING_MODE", "invalid")
    with pytest.raises(RuntimeError, match="off, observe, or enforce"):
        intent_routing_mode()
    monkeypatch.setenv("GROUP_AGENT_INTENT_ROUTING_MODE", "enforce")
    assert intent_routing_mode() == "enforce"
    monkeypatch.setenv("GROUP_AGENT_INTENT_CLASSIFIER_TIMEOUT_S", "bad")
    with pytest.raises(RuntimeError, match="must be numeric"):
        intent_classifier_timeout_s()


async def test_classifier_helper_reports_timeout_without_raw_exception() -> None:
    result = await aclassify_intent(
        "模糊请求",
        model=_ClassifierModel(_Structured([{}], delay=0.2)),
        timeout_s=0.05,
    )
    assert result.status == "timed_out"
    assert result.classification is None


class _EndpointAgent:
    def __init__(self, prediction_recorded: list[bool]) -> None:
        self.prediction_recorded = prediction_recorded

    async def aget_state(self, _: Any) -> Any:
        return SimpleNamespace(values={"messages": []})

    async def ainvoke(self, _: Any, __: Any) -> dict[str, Any]:
        assert self.prediction_recorded == [True]
        return {"messages": [AIMessage(content="普通回复。")]}


async def test_chat_observe_runs_before_first_agent_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from apps.group_agent_api.app.endpoints import chat as chat_endpoint

    recorded: list[bool] = []
    agent = _EndpointAgent(recorded)
    state = AppState(agent=object(), base_dir=tmp_path, no_tool_agent=agent)
    session = TrustedSession(
        principal=SessionPrincipal("user-1", None, None, "body_stub"),
        membership=MembershipResult(tier=CapabilityTier.not_in_group, source="stub"),
        group_id="group-1",
        group_token=None,
        capabilities=SessionCapabilities(
            tier=CapabilityTier.not_in_group,
            can_search=False,
            can_write_profile=False,
        ),
    )

    async def resolve(*_: Any, **__: Any) -> TrustedSession:
        return session

    def record(*_: Any, **__: Any) -> None:
        recorded.append(True)

    monkeypatch.setenv("GROUP_AGENT_INTENT_ROUTING_MODE", "observe")
    monkeypatch.setattr(chat_endpoint, "resolve_trusted_session", resolve)
    monkeypatch.setattr(chat_endpoint, "record_intent_prediction", record)
    response = await chat_endpoint.chat(
        ChatRequest(
            user_id="user-1",
            group_id="group-1",
            message="普通聊天",
            run_match=False,
            run_invite=False,
        ),
        state,
    )
    assert response.reply
    assert recorded == [True]


async def test_worker_observe_runs_before_first_agent_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from apps.group_agent_api.app import async_manager

    recorded: list[bool] = []
    agent = _EndpointAgent(recorded)
    state = AppState(agent=object(), base_dir=tmp_path, no_tool_agent=agent)
    session = TrustedSession(
        principal=SessionPrincipal("user-1", None, None, "body_stub"),
        membership=MembershipResult(tier=CapabilityTier.not_in_group, source="stub"),
        group_id="group-1",
        group_token=None,
        capabilities=SessionCapabilities(
            tier=CapabilityTier.not_in_group,
            can_search=False,
            can_write_profile=False,
        ),
    )
    events: list[str] = []

    async def emit(event: str, _: dict[str, Any]) -> bool:
        events.append(event)
        return True

    def record(*_: Any, **__: Any) -> None:
        recorded.append(True)

    monkeypatch.setenv("GROUP_AGENT_INTENT_ROUTING_MODE", "observe")
    monkeypatch.setattr(async_manager, "record_intent_prediction", record)
    await async_manager._execute_core_agent(
        req=AsyncCallRequest(
            run_id="run-1",
            idempotency_key="intent-observe-1",
            user_id="user-1",
            group_id="group-1",
            conversation_id="conversation-1",
            message="普通聊天",
            callback_url="http://example.invalid/group_agent_callbacks/run-1",
            run_match=False,
            run_invite=False,
        ),
        session=session,
        state=state,
        tid="worker-intent-observe",
        emit_callback=emit,
    )
    assert events[-1] == "final"
    assert recorded == [True]
