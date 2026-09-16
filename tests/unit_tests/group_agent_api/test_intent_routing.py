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
    intent_constraints_from_profile,
)
from apps.group_agent_api.agent_factory.intent_routing.resolver import (
    IntentResolution,
    aresolve_intent,
    intent_classifier_timeout_s,
    intent_routing_mode,
    record_intent_outcome,
    record_intent_prediction,
    resolve_intent,
)
from apps.group_agent_api.agent_factory.intent_routing.schema import (
    IntentConstraintsV1,
    IntentDecisionV1,
)
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
    ],
)
def test_frozen_rule_replays(message: str, intent: str, action: str) -> None:
    result = resolve_intent(message, execution_mode="group_agent", can_search=True)
    assert result.decision.primary_intent == intent
    assert result.decision.action == action


def test_ordinary_chat_without_classifier_fail_closes_to_clarify() -> None:
    result = resolve_intent(
        "你好，今天想聊聊创业", execution_mode="group_agent", can_search=True
    )
    assert result.decision.primary_intent == "clarification"
    assert result.decision.action == "clarify"
    assert result.signal_ids == ("no_business_action",)
    assert result.classifier_status == "unavailable"
    assert result.routing_class == "clarification_fallback"


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
    assert (
        extract_completed_context(
            completed,
            scope=IntentContextScope(
                thread_id="other-thread",
                user_id="u",
                execution_mode="group_agent",
                episode_id="episode-1",
            ),
        )
        is None
    )

    marker.additional_kwargs["group_agent_completed_intent_context_v1"]["plan_id"] = (
        "platform-help-v1"
    )
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
        self.last_prompts: Any = None

    def invoke(self, prompts: Any) -> Any:
        self.last_prompts = prompts
        output = self.outputs[min(self.calls, len(self.outputs) - 1)]
        self.calls += 1
        return output

    async def ainvoke(self, prompts: Any) -> Any:
        self.last_prompts = prompts
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.outputs[min(self.calls - 1, len(self.outputs) - 1)]


class _CapturingStructured(_Structured):
    """Alias kept for Phase 1 prompt-context assertions."""


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
    assert result.routing_class == "clarification_fallback"
    assert _query_plan_would_run(result) is False


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
    assert result.routing_class == "semantic_planner"
    assert _query_plan_would_run(result) is True
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
    assert result.routing_class == "semantic_planner"
    assert _query_plan_would_run(result) is False


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
    assert result.routing_class == "clarification_fallback"
    assert _query_plan_would_run(result) is False


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
    assert captured[0]["routing_class"] == "l0_positive"
    assert captured[0]["query_plan_would_run"] is True
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


def test_intent_resolution_public_constructor_omits_routing_class() -> None:
    decision = resolve_intent(
        "找上海餐饮运营合伙人", execution_mode="group_agent", can_search=True
    ).decision
    rebuilt = IntentResolution(
        decision=decision,
        latency_ms=0,
        signal_ids=("action_object:people",),
        classifier_status="not_needed",
        classifier_attempts=0,
        context_inherited=False,
    )
    assert rebuilt.routing_class == "l0_positive"


def _query_plan_would_run(result: Any) -> bool:
    return (
        result.decision.action == "execute"
        and result.decision.query_plan_ref is not None
    )


def _assert_routing_diagnostics(
    result: Any,
    *,
    source: str,
    classifier_status: str,
    inherited: bool,
    action: str,
    query_plan: bool,
    routing_class: str,
    signal_ids: tuple[str, ...] | None = None,
    classifier_attempts: int | None = None,
) -> None:
    assert result.decision.decision_source == source
    assert result.classifier_status == classifier_status
    assert result.context_inherited is inherited
    assert result.decision.action == action
    assert _query_plan_would_run(result) is query_plan
    assert result.routing_class == routing_class
    if signal_ids is not None:
        assert result.signal_ids == signal_ids
    if classifier_attempts is not None:
        assert result.classifier_attempts == classifier_attempts


def test_phase0_explicit_chinese_people_search_is_l0_positive() -> None:
    structured = _Structured(
        [
            {
                "version": 1,
                "candidates": [{"intent": "conversation", "confidence": "high"}],
                "evidence_spans": [],
                "missing_object": None,
                "classifier_version": "intent-classifier-v1",
            }
        ]
    )
    model = _ClassifierModel(structured)
    result = resolve_intent(
        "找上海餐饮运营合伙人",
        execution_mode="group_agent",
        can_search=True,
        classifier_model=model,
    )
    assert result.decision.primary_intent == "people_search"
    assert structured.calls == 0
    _assert_routing_diagnostics(
        result,
        source="rules",
        classifier_status="not_needed",
        inherited=False,
        action="execute",
        query_plan=True,
        routing_class="l0_positive",
        signal_ids=("action_object:people",),
        classifier_attempts=0,
    )
    assert result.decision.query_plan_ref.plan_id == "people-search-v1"


def _classification(
    intent: str,
    *,
    confidence: str = "high",
    missing_object: str | None = None,
    evidence: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "version": 1,
        "candidates": [{"intent": intent, "confidence": confidence}],
        "evidence_spans": evidence or [],
        "missing_object": missing_object,
        "classifier_version": "intent-classifier-v1",
    }


def _completed_shanghai_ops_people_search() -> list[Any]:
    call = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "read-sh",
                "name": "query_aihehuo_public_data",
                "args": {
                    "resource": "users",
                    "location": "上海",
                    "industry": "餐饮",
                    "role": "运营合伙人",
                    "capability": "运营经验",
                },
            }
        ],
    )
    result = ToolMessage(
        content='{"status":"succeeded","items":[]}',
        tool_call_id="read-sh",
        name="query_aihehuo_public_data",
    )
    return [call, result, AIMessage(content="查询完成。")]


def _shanghai_ops_profile() -> IntentConstraintsV1:
    return IntentConstraintsV1(
        location=("Shanghai",),
        industry=("restaurant",),
        role=("operations",),
    )


def test_phase1_english_people_search_reaches_query_plan() -> None:
    structured = _Structured([_classification("people_search", evidence=["Shanghai"])])
    profile = _shanghai_ops_profile()
    result = resolve_intent(
        "Can you find restaurant operators in Shanghai?",
        execution_mode="group_agent",
        can_search=True,
        classifier_model=_ClassifierModel(structured),
        profile_constraints=profile,
    )
    assert result.decision.primary_intent == "people_search"
    assert result.decision.constraints.location == ("Shanghai",)
    assert result.decision.constraints.industry == ("restaurant",)
    assert result.decision.constraints.role == ("operations",)
    _assert_routing_diagnostics(
        result,
        source="classifier",
        classifier_status="ok",
        inherited=False,
        action="execute",
        query_plan=True,
        routing_class="semantic_planner",
        signal_ids=("no_business_action",),
        classifier_attempts=1,
    )
    assert result.decision.query_plan_ref.plan_id == "people-search-v1"
    assert structured.calls == 1


def test_phase1_english_search_with_trusted_context_executes_people_search() -> None:
    structured = _Structured([_classification("people_search", evidence=["search"])])
    result = resolve_intent(
        "Can you help me do a search",
        execution_mode="group_agent",
        can_search=True,
        prior_messages=_completed_shanghai_ops_people_search(),
        scope=_scope(),
        classifier_model=_ClassifierModel(structured),
    )
    assert result.decision.primary_intent == "people_search"
    assert result.decision.constraints.location == ("上海",)
    assert result.decision.constraints.industry == ("餐饮",)
    _assert_routing_diagnostics(
        result,
        source="classifier",
        classifier_status="ok",
        inherited=False,
        action="execute",
        query_plan=True,
        routing_class="semantic_planner",
        signal_ids=("no_business_action",),
        classifier_attempts=1,
    )


def test_phase1_english_search_without_usable_context_clarifies() -> None:
    structured = _Structured([_classification("people_search", evidence=["search"])])
    result = resolve_intent(
        "Can you help me do a search",
        execution_mode="group_agent",
        can_search=True,
        classifier_model=_ClassifierModel(structured),
    )
    assert result.decision.primary_intent == "clarification"
    assert result.decision.constraints.location == ()
    _assert_routing_diagnostics(
        result,
        source="classifier",
        classifier_status="ok",
        inherited=False,
        action="clarify",
        query_plan=False,
        routing_class="semantic_planner",
        signal_ids=("no_business_action",),
        classifier_attempts=1,
    )


def test_phase1_profile_constraints_merge_into_authoritative_decision() -> None:
    structured = _Structured([_classification("people_search")])
    result = resolve_intent(
        "Can you help me do a search",
        execution_mode="group_agent",
        can_search=True,
        classifier_model=_ClassifierModel(structured),
        profile_constraints=_shanghai_ops_profile(),
    )
    assert result.decision.primary_intent == "people_search"
    assert result.decision.constraints == _shanghai_ops_profile()
    assert _query_plan_would_run(result) is True


def test_phase1_ordinary_conversation_via_classifier() -> None:
    structured = _Structured([_classification("conversation")])
    result = resolve_intent(
        "你好",
        execution_mode="group_agent",
        can_search=True,
        classifier_model=_ClassifierModel(structured),
    )
    assert result.decision.primary_intent == "conversation"
    _assert_routing_diagnostics(
        result,
        source="classifier",
        classifier_status="ok",
        inherited=False,
        action="conversation",
        query_plan=False,
        routing_class="semantic_planner",
        signal_ids=("no_business_action",),
        classifier_attempts=1,
    )


def test_phase1_explicit_search_negation_is_hard_veto() -> None:
    structured = _Structured([_classification("people_search")])
    result = resolve_intent(
        "先别查，只是聊聊",
        execution_mode="group_agent",
        can_search=True,
        prior_messages=_completed_people_search(),
        scope=_scope(),
        classifier_model=_ClassifierModel(structured),
    )
    assert result.decision.primary_intent == "conversation"
    assert structured.calls == 0
    _assert_routing_diagnostics(
        result,
        source="rules",
        classifier_status="not_needed",
        inherited=False,
        action="conversation",
        query_plan=False,
        routing_class="l0_conversation",
        signal_ids=("explicit_negation",),
    )
    assert "先别查" in result.decision.negations or result.decision.negations


def test_phase1_english_explicit_negation_is_hard_veto() -> None:
    structured = _Structured([_classification("people_search")])
    result = resolve_intent(
        "Do not search, just chat",
        execution_mode="group_agent",
        can_search=True,
        classifier_model=_ClassifierModel(structured),
        profile_constraints=_shanghai_ops_profile(),
    )
    assert result.decision.primary_intent == "conversation"
    assert result.decision.action == "conversation"
    assert structured.calls == 0
    assert result.signal_ids == ("explicit_negation",)
    assert _query_plan_would_run(result) is False


def test_phase1_elliptical_followup_after_completed_query_inherits_context() -> None:
    structured = _Structured([_classification("opportunity_search")])
    result = resolve_intent(
        "北京的呢",
        execution_mode="group_agent",
        can_search=True,
        prior_messages=_completed_people_search(),
        scope=_scope(),
        classifier_model=_ClassifierModel(structured),
    )
    assert result.decision.primary_intent == "people_search"
    assert result.decision.constraints.location == ("北京",)
    assert structured.calls == 0
    _assert_routing_diagnostics(
        result,
        source="context",
        classifier_status="not_needed",
        inherited=True,
        action="execute",
        query_plan=True,
        routing_class="context_inheritance",
        classifier_attempts=0,
    )


def test_phase1_replace_location_followup_inherits_without_beijing() -> None:
    result = resolve_intent(
        "不要北京，改成上海的",
        execution_mode="group_agent",
        can_search=True,
        prior_messages=_completed_people_search(),
        scope=_scope(),
    )
    assert result.decision.primary_intent == "people_search"
    assert result.decision.constraints.location == ("上海",)
    assert "location:北京" in result.decision.negations
    _assert_routing_diagnostics(
        result,
        source="context",
        classifier_status="not_needed",
        inherited=True,
        action="execute",
        query_plan=True,
        routing_class="context_inheritance",
    )


def test_phase1_classifier_unavailable_clarifies_without_public_request() -> None:
    result = resolve_intent(
        "帮我找一下",
        execution_mode="group_agent",
        can_search=True,
        classifier_model=None,
    )
    _assert_routing_diagnostics(
        result,
        source="classifier",
        classifier_status="unavailable",
        inherited=False,
        action="clarify",
        query_plan=False,
        routing_class="clarification_fallback",
        signal_ids=("search_missing_object",),
        classifier_attempts=0,
    )


def test_phase1_no_business_action_without_classifier_clarifies() -> None:
    result = resolve_intent(
        "Can you help me do a search",
        execution_mode="group_agent",
        can_search=True,
    )
    _assert_routing_diagnostics(
        result,
        source="classifier",
        classifier_status="unavailable",
        inherited=False,
        action="clarify",
        query_plan=False,
        routing_class="clarification_fallback",
        signal_ids=("no_business_action",),
        classifier_attempts=0,
    )


def test_phase1_classifier_context_clips_oversized_profile_tokens() -> None:
    from apps.group_agent_api.agent_factory.intent_routing.context import (
        build_classifier_context,
    )
    from apps.group_agent_api.agent_factory.intent_routing.prompts import (
        classifier_user_prompt,
    )

    oversized = IntentConstraintsV1(location=("上海" + ("x" * 1_000_000),))
    context = build_classifier_context(
        trusted=None,
        prior_messages=None,
        profile=oversized,
        current_negations=(),
    )
    assert len(context.profile_location) == 1
    assert len(context.profile_location[0]) == 64
    prompt = classifier_user_prompt("hello", context)
    assert len(prompt) < 5_000


def test_phase1_intent_constraints_from_profile_match_constraints() -> None:
    profile = SimpleNamespace(
        match_constraints=[
            {
                "field": "city",
                "operator": "in",
                "values": ["Shanghai", "x" * 200],
                "strength": "hard",
            },
            {
                "field": "industry",
                "operator": "in",
                "values": ["restaurant"],
                "strength": "hard",
            },
            {
                "field": "role",
                "operator": "in",
                "values": ["operations"],
                "strength": "soft",
            },
            {
                "field": "gender",
                "operator": "eq",
                "values": ["ignored"],
                "strength": "hard",
            },
        ]
    )
    constraints = intent_constraints_from_profile(profile)
    assert constraints is not None
    assert constraints.location == ("Shanghai", "x" * 64)
    assert constraints.industry == ("restaurant",)
    assert constraints.role == ("operations",)


def test_phase0_policy_rejection_is_observable_without_query_plan() -> None:
    cafe = resolve_intent(
        "找上海餐饮运营合伙人", execution_mode="cafe_host", can_search=True
    )
    denied = resolve_intent(
        "找上海餐饮运营合伙人", execution_mode="group_agent", can_search=False
    )
    for result in (cafe, denied):
        assert result.decision.primary_intent == "people_search"
        _assert_routing_diagnostics(
            result,
            source="rules",
            classifier_status="not_needed",
            inherited=False,
            action="reject",
            query_plan=False,
            routing_class="policy_rejection",
        )


def test_phase1_motivating_english_sequence_reaches_people_search() -> None:
    import json

    from apps.group_agent_api.agent_factory.intent_routing.execution import (
        prepare_intent_execution,
    )
    from apps.group_agent_api.agent_factory.intent_routing.plan_executor import (
        build_step_parameters,
    )

    fixture_path = (
        Path(__file__).parent / "fixtures" / "thin_harness_phase0_english_search.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    current = fixture["current_turn"]
    expected = fixture["phase1_with_classifier_and_profile"]
    profile_raw = fixture["phase1_structured_profile"]
    profile = IntentConstraintsV1(
        location=tuple(profile_raw["location"]),
        industry=tuple(profile_raw["industry"]),
        role=tuple(profile_raw["role"]),
    )
    assert current == "Can you help me do a search"
    assert all("token" not in turn["text"].lower() for turn in fixture["turns"])
    assert all("password" not in turn["text"].lower() for turn in fixture["turns"])

    prior = [
        HumanMessage(content=turn["text"])
        if turn["role"] == "user"
        else AIMessage(content=turn["text"])
        for turn in fixture["turns"][:-1]
    ]
    structured = _CapturingStructured(
        [_classification("people_search", evidence=["search"])]
    )
    result = resolve_intent(
        current,
        execution_mode="group_agent",
        can_search=True,
        prior_messages=prior,
        scope=_scope(),
        classifier_model=_ClassifierModel(structured),
        profile_constraints=profile,
    )
    assert result.decision.primary_intent == expected["primary_intent"]
    assert result.decision.query_plan_ref.plan_id == expected["query_plan_id"]
    assert result.decision.constraints.location == tuple(
        expected["constraints"]["location"]
    )
    assert result.decision.constraints.industry == tuple(
        expected["constraints"]["industry"]
    )
    assert result.decision.constraints.role == tuple(expected["constraints"]["role"])
    _assert_routing_diagnostics(
        result,
        source=expected["decision_source"],
        classifier_status=expected["classifier_status"],
        inherited=expected["context_inherited"],
        action=expected["action"],
        query_plan=expected["query_plan_would_run"],
        routing_class=expected["routing_class"],
        signal_ids=tuple(expected["signal_ids"]),
        classifier_attempts=expected["classifier_attempts"],
    )
    prompt_text = str(structured.last_prompts)
    assert "classifier_context=" in prompt_text
    assert "Shanghai" in prompt_text
    assert "restaurant" in prompt_text
    assert "operations" in prompt_text

    execution = prepare_intent_execution(
        result,
        original_message=current,
        run_id=None,
        request_fingerprint=None,
    )
    query = build_step_parameters(execution, execution.plan.steps[0])["q"]
    assert "Shanghai" in query
    assert "restaurant" in query
    assert "operations" in query
    assert query != current


def test_phase1_motivating_sequence_without_structured_object_clarifies() -> None:
    import json

    fixture_path = (
        Path(__file__).parent / "fixtures" / "thin_harness_phase0_english_search.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    expected = fixture["phase1_without_structured_object"]
    prior = [
        HumanMessage(content=turn["text"])
        if turn["role"] == "user"
        else AIMessage(content=turn["text"])
        for turn in fixture["turns"][:-1]
    ]
    result = resolve_intent(
        fixture["current_turn"],
        execution_mode="group_agent",
        can_search=True,
        prior_messages=prior,
        scope=_scope(),
        classifier_model=_ClassifierModel(
            _Structured([_classification("people_search")])
        ),
    )
    assert result.decision.primary_intent == expected["primary_intent"]
    _assert_routing_diagnostics(
        result,
        source=expected["decision_source"],
        classifier_status=expected["classifier_status"],
        inherited=expected["context_inherited"],
        action=expected["action"],
        query_plan=expected["query_plan_would_run"],
        routing_class=expected["routing_class"],
        signal_ids=tuple(expected["signal_ids"]),
        classifier_attempts=expected["classifier_attempts"],
    )


async def test_phase1_sync_async_matrix_paths_match() -> None:
    structured_sync = _Structured([_classification("people_search")])
    structured_async = _Structured([_classification("people_search")])
    message = "Can you help me do a search"
    prior = _completed_shanghai_ops_people_search()
    sync = resolve_intent(
        message,
        execution_mode="group_agent",
        can_search=True,
        prior_messages=prior,
        scope=_scope(),
        classifier_model=_ClassifierModel(structured_sync),
    )
    async_result = await aresolve_intent(
        message,
        execution_mode="group_agent",
        can_search=True,
        prior_messages=prior,
        scope=_scope(),
        classifier_model=_ClassifierModel(structured_async),
    )
    assert sync.decision == async_result.decision
    assert sync.signal_ids == async_result.signal_ids
    assert sync.classifier_status == async_result.classifier_status
    assert sync.routing_class == async_result.routing_class


def test_phase1_observe_logs_distinguish_routing_classes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, Any]] = []

    def record(*_: Any, **kwargs: Any) -> None:
        captured.append(kwargs["detail"])

    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.intent_routing.resolver.record_decision_point",
        record,
    )
    cases = [
        ("找上海餐饮运营合伙人", "l0_positive", True, None, None),
        (
            "Can you help me do a search",
            "semantic_planner",
            True,
            _ClassifierModel(_Structured([_classification("people_search")])),
            _shanghai_ops_profile(),
        ),
        ("帮我找一下", "clarification_fallback", False, None, None),
    ]
    for message, routing_class, query_plan, model, profile in cases:
        captured.clear()
        result = resolve_intent(
            message,
            execution_mode="group_agent",
            can_search=True,
            classifier_model=model,
            profile_constraints=profile,
        )
        record_intent_prediction(result, run_id="run-phase1", thread_id="thread-phase1")
        detail = captured[0]
        assert detail["routing_class"] == routing_class
        assert detail["query_plan_would_run"] is query_plan
        assert "decision_source" in detail
        assert "classifier_status" in detail
        assert "classifier_attempts" in detail
        assert "context_inherited" in detail
        assert "action" in detail
        assert message not in repr(detail)
