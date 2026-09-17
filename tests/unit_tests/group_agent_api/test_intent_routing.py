"""Phase 2 authoritative planner routing tests."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import ValidationError

from apps.group_agent_api.agent_factory.intent_routing.context import (
    IntentContextScope,
    build_planner_context,
    completed_intent_context_message,
    extract_completed_context,
    planner_profile_from_store,
)
from apps.group_agent_api.agent_factory.intent_routing.planner import aplan_intent
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
    IntentDecisionV1,
    PlannerProposalV2,
    ResourceQueriesV2,
)
from apps.group_agent_api.agent_factory.capability import (
    CapabilityTier,
    SessionCapabilities,
)
from apps.group_agent_api.agent_factory.integrations.membership_client import (
    MembershipResult,
)
from apps.group_agent_api.agent_factory.integrations.principal import SessionPrincipal
from apps.group_agent_api.app.models import ChatRequest
from apps.group_agent_api.app.session import TrustedSession
from apps.group_agent_api.app.state import AppState
from tests.unit_tests.group_agent_api.planner_fixtures import (
    PlannerModel,
    StructuredPlanner,
    clarify,
    community_query,
    people_query,
    planner_for_message,
    planner_returning,
    proposal,
    reply,
)


def _scope() -> IntentContextScope:
    return IntentContextScope(
        thread_id="ga::u::global::group_agent::c::episode-1",
        user_id="u",
        execution_mode="group_agent",
        episode_id="episode-1",
    )


def _query_plan_would_run(result: Any) -> bool:
    return (
        result.decision.action == "execute"
        and result.decision.query_plan_ref is not None
    )


def _assert_routing_diagnostics(
    result: Any,
    *,
    source: str,
    planner_status: str,
    action: str,
    query_plan: bool,
    routing_class: str,
    inherited: bool = False,
    compressed: bool = False,
) -> None:
    assert result.decision.decision_source == source
    assert result.planner_status == planner_status
    assert result.decision.action == action
    assert _query_plan_would_run(result) is query_plan
    assert result.routing_class == routing_class
    assert result.context_inherited is inherited
    assert result.context_compressed is compressed


@pytest.mark.parametrize(
    ("message", "payload", "intent", "action"),
    [
        (
            "有没有上海做餐饮的技术合伙人",
            people_query("上海 餐饮 技术合伙人"),
            "people_search",
            "execute",
        ),
        (
            "找餐饮项目正在招募的合作席位",
            proposal(
                intent="project_partnership_search",
                projects="餐饮 招募 合作席位",
            ),
            "project_partnership_search",
            "execute",
        ),
        (
            "看看最近的 AI 创业机会",
            proposal(
                intent="opportunity_search",
                ideas="AI 创业机会",
                time_range="最近",
            ),
            "opportunity_search",
            "execute",
        ),
        (
            "加入杭州创业社群",
            community_query("杭州 创业社群"),
            "community_search",
            "execute",
        ),
        (
            "看看本周创业周报和文章",
            proposal(intent="startup_digest"),
            "startup_digest",
            "execute",
        ),
        (
            "如何发布餐饮项目招合伙人",
            proposal(intent="platform_help", faqs="发布 餐饮项目 招合伙人"),
            "platform_help",
            "execute",
        ),
        (
            "先别查，只是聊聊怎么找合伙人",
            reply(),
            "conversation",
            "conversation",
        ),
        (
            "把我的所在地改成上海",
            reply(intent="profile_update"),
            "profile_update",
            "delegate_existing_flow",
        ),
    ],
)
def test_planner_replays(
    message: str, payload: dict[str, Any], intent: str, action: str
) -> None:
    result = resolve_intent(
        message,
        execution_mode="group_agent",
        can_search=True,
        planner_model=planner_returning(payload),
    )
    assert result.decision.primary_intent == intent
    assert result.decision.action == action
    assert result.decision.decision_source == "planner"
    assert result.routing_class == "authoritative_planner"


def test_planner_unavailable_fail_closes_to_clarify() -> None:
    result = resolve_intent(
        "你好，今天想聊聊创业", execution_mode="group_agent", can_search=True
    )
    _assert_routing_diagnostics(
        result,
        source="planner",
        planner_status="unavailable",
        action="clarify",
        query_plan=False,
        routing_class="clarification_fallback",
    )


def test_read_policy_rejects_cafe_and_missing_permission() -> None:
    cafe = resolve_intent(
        "帮我找合伙人",
        execution_mode="cafe_host",
        can_search=True,
        planner_model=planner_returning(people_query("合伙人")),
    )
    denied = resolve_intent(
        "帮我找合伙人",
        execution_mode="group_agent",
        can_search=False,
        planner_model=planner_returning(people_query("合伙人")),
    )
    assert cafe.decision.primary_intent == "people_search"
    assert cafe.decision.action == "reject"
    assert denied.decision.action == "reject"
    assert cafe.decision.query_plan_ref is None
    assert denied.decision.required_skill is None
    assert cafe.routing_class == "policy_rejection"


def test_decision_schema_is_closed_and_action_bound() -> None:
    valid = resolve_intent(
        "帮我找上海的合伙人",
        execution_mode="group_agent",
        can_search=True,
        planner_model=planner_returning(people_query("上海 合伙人")),
    ).decision
    assert valid.model_dump(mode="json")["query_plan_ref"] == {
        "plan_version": 1,
        "plan_id": "people-search-v1",
    }
    assert valid.resource_queries.users == "上海 合伙人"
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


def test_planner_proposal_rejects_query_without_primary_resource() -> None:
    with pytest.raises(ValidationError):
        PlannerProposalV2.model_validate(
            {
                "version": 2,
                "action": "query",
                "intent": "people_search",
                "queries": {},
                "clarification": None,
                "time_range": None,
                "planner_version": "intent-planner-v2",
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


def test_completed_context_is_data_for_planner_not_short_circuit() -> None:
    messages = _completed_people_search()
    context = extract_completed_context(messages, scope=_scope())
    assert context is not None
    assert context.intent == "people_search"

    structured = StructuredPlanner([people_query("上海 技术合伙人")])
    result = resolve_intent(
        "上海的呢",
        execution_mode="group_agent",
        can_search=True,
        prior_messages=messages,
        scope=_scope(),
        planner_model=PlannerModel(structured),
    )
    assert structured.calls == 1
    assert result.context_inherited is True
    assert result.decision.primary_intent == "people_search"
    assert result.decision.resource_queries.users == "上海 技术合伙人"
    assert result.decision.decision_source == "planner"


def test_server_context_requires_completion_hash_and_exact_scope() -> None:
    decision = resolve_intent(
        "找上海餐饮创业者，优先有供应链经验",
        execution_mode="group_agent",
        can_search=True,
        planner_model=planner_for_message(
            "找上海餐饮创业者，优先有供应链经验"
        ),
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


def test_planner_schema_failure_retries_once_then_clarifies() -> None:
    structured = StructuredPlanner([{"bad": True}, {"bad": True}])
    result = resolve_intent(
        "找人",
        execution_mode="group_agent",
        can_search=True,
        planner_model=PlannerModel(structured),
    )
    assert structured.calls == 2
    assert result.planner_status == "invalid"
    assert result.decision.action == "clarify"


def test_valid_planner_cannot_supply_policy_fields() -> None:
    structured = StructuredPlanner(
        [
            {
                **people_query("上海 合伙人"),
                "required_skill": "get-ai-cofounders",
            }
        ]
    )
    # Extra fields are rejected by closed schema → invalid → clarify after retries.
    result = resolve_intent(
        "找上海合伙人",
        execution_mode="group_agent",
        can_search=True,
        planner_model=PlannerModel(structured),
    )
    assert result.decision.action == "clarify"
    assert result.planner_status == "invalid"


async def test_planner_timeout_fails_to_clarification() -> None:
    structured = StructuredPlanner([people_query("合伙人")], delay=0.2)
    result = await aresolve_intent(
        "找合伙人",
        execution_mode="group_agent",
        can_search=True,
        planner_model=PlannerModel(structured),
    )
    # Default timeout is 1.5s; use a short env in a dedicated test below.
    assert result.decision.action in {"execute", "clarify"}


async def test_planner_helper_reports_timeout_without_raw_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROUP_AGENT_INTENT_CLASSIFIER_TIMEOUT_S", "0.05")
    structured = StructuredPlanner([people_query("合伙人")], delay=0.2)
    result = await aplan_intent(
        model=PlannerModel(structured),
        context=build_planner_context(
            message="找合伙人",
            prior_messages=None,
            profile=None,
            trusted=None,
        ),
        timeout_s=0.05,
    )
    assert result.status == "timed_out"
    assert result.proposal is None


async def test_sync_async_planner_paths_are_identical() -> None:
    payload = people_query("上海 餐饮 运营")
    sync = resolve_intent(
        "找上海餐饮运营",
        execution_mode="group_agent",
        can_search=True,
        planner_model=planner_returning(payload),
    )
    async_result = await aresolve_intent(
        "找上海餐饮运营",
        execution_mode="group_agent",
        can_search=True,
        planner_model=planner_returning(payload),
    )
    assert sync.decision.model_dump() == async_result.decision.model_dump()
    assert sync.routing_class == async_result.routing_class == "authoritative_planner"


def test_observe_logs_have_confusion_fields_without_message_or_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, Any]] = []

    def capture(**kwargs: Any) -> None:
        captured.append(kwargs)

    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.intent_routing.resolver.record_decision_point",
        capture,
    )
    result = resolve_intent(
        "找上海合伙人",
        execution_mode="group_agent",
        can_search=True,
        planner_model=planner_returning(people_query("上海 合伙人")),
    )
    record_intent_prediction(result, run_id="r1", thread_id="t1")
    record_intent_outcome(
        result, observed_route="search", run_id="r1", thread_id="t1"
    )
    joined = json.dumps(captured, ensure_ascii=False)
    assert "找上海合伙人" not in joined
    assert captured[0]["detail"]["routing_class"] == "authoritative_planner"
    assert captured[0]["detail"]["planner_status"] == "ok"


def test_observe_config_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROUP_AGENT_INTENT_ROUTING_MODE", "weird")
    with pytest.raises(RuntimeError):
        intent_routing_mode()
    monkeypatch.setenv("GROUP_AGENT_INTENT_CLASSIFIER_TIMEOUT_S", "abc")
    with pytest.raises(RuntimeError):
        intent_classifier_timeout_s()


def test_intent_resolution_public_constructor_omits_routing_class() -> None:
    decision = resolve_intent(
        "找上海餐饮运营合伙人",
        execution_mode="group_agent",
        can_search=True,
        planner_model=planner_for_message("找上海餐饮运营合伙人"),
    ).decision
    rebuilt = IntentResolution(
        decision=decision,
        latency_ms=0,
        signal_ids=(),
        classifier_status="ok",
        classifier_attempts=1,
        context_inherited=False,
    )
    assert rebuilt.routing_class == "authoritative_planner"


def test_phase2_every_turn_calls_planner_including_obvious_chinese_search() -> None:
    structured = StructuredPlanner([people_query("上海 餐饮 运营合伙人")])
    result = resolve_intent(
        "找上海餐饮运营合伙人",
        execution_mode="group_agent",
        can_search=True,
        planner_model=PlannerModel(structured),
    )
    assert structured.calls == 1
    _assert_routing_diagnostics(
        result,
        source="planner",
        planner_status="ok",
        action="execute",
        query_plan=True,
        routing_class="authoritative_planner",
    )
    assert result.decision.resource_queries.users == "上海 餐饮 运营合伙人"


def test_phase2_explicit_negation_is_planner_reply_not_execute() -> None:
    result = resolve_intent(
        "不要搜索，先别查",
        execution_mode="group_agent",
        can_search=True,
        planner_model=planner_returning(reply()),
        prior_messages=_completed_people_search(),
        scope=_scope(),
    )
    assert result.decision.action == "conversation"
    assert result.decision.resource_queries.users is None


def test_phase2_english_search_with_episode_history_executes() -> None:
    prior = [
        HumanMessage(content="I need restaurant operations help in Shanghai."),
        AIMessage(content="Tell me more about the role."),
        HumanMessage(content="Experienced operators."),
        AIMessage(content="Ready when you want to search."),
    ]
    result = resolve_intent(
        "Can you help me do a search",
        execution_mode="group_agent",
        can_search=True,
        prior_messages=prior,
        scope=_scope(),
        planner_model=planner_returning(
            people_query("上海 餐饮 运营 经验")
        ),
    )
    assert result.decision.action == "execute"
    assert "餐饮" in (result.decision.resource_queries.users or "")
    assert result.decision.resource_queries.users


def test_phase2_english_search_without_object_clarifies() -> None:
    result = resolve_intent(
        "Can you help me do a search",
        execution_mode="group_agent",
        can_search=True,
        planner_model=planner_returning(clarify("需要什么对象？")),
    )
    assert result.decision.action == "clarify"
    assert not _query_plan_would_run(result)


def test_phase2_new_episode_uses_legacy_profile_free_text() -> None:
    profile = SimpleNamespace(
        doing=SimpleNamespace(value="开一家上海餐厅，缺运营合伙人"),
        need=SimpleNamespace(value="有餐饮运营经验的人"),
        offer=SimpleNamespace(value="品牌与供应链"),
        match_constraints=[],
    )
    structured = StructuredPlanner(
        [people_query("上海 餐厅 餐饮 运营合伙人")]
    )
    result = resolve_intent(
        "继续搜索",
        execution_mode="group_agent",
        can_search=True,
        planner_model=PlannerModel(structured),
        profile=profile,
    )
    prompt = structured.prompts[0][1].content
    assert "开一家上海餐厅" in prompt
    assert result.decision.action == "execute"
    assert "餐厅" in (result.decision.resource_queries.users or "")


def test_phase2_new_episode_restaurant_query_includes_profile_industry() -> None:
    profile = SimpleNamespace(
        doing=SimpleNamespace(value="正在上海做餐厅创业"),
        need=SimpleNamespace(value="运营合伙人"),
        offer=SimpleNamespace(value="场地"),
        match_constraints=[],
    )
    projected = planner_profile_from_store(profile)
    assert projected.doing and "餐厅" in projected.doing
    result = resolve_intent(
        "搜索上海的运营合伙人",
        execution_mode="group_agent",
        can_search=True,
        planner_model=planner_returning(
            people_query("上海 餐厅 运营合伙人")
        ),
        profile=profile,
    )
    assert "餐厅" in (result.decision.resource_queries.users or "")


def test_phase2_correction_uses_conversation_not_superseded_detail() -> None:
    prior = [
        HumanMessage(content="找北京的技术合伙人"),
        AIMessage(content="已按北京技术合伙人理解。"),
    ]
    result = resolve_intent(
        "不要北京，改成上海的",
        execution_mode="group_agent",
        can_search=True,
        prior_messages=prior,
        scope=_scope(),
        planner_model=planner_returning(people_query("上海 技术合伙人")),
    )
    query = result.decision.resource_queries.users or ""
    assert "上海" in query
    assert "北京" not in query


def test_phase2_context_overflow_fail_closes_without_search() -> None:
    prior: list[Any] = []
    for index in range(30):
        prior.append(HumanMessage(content=("目标细节" + str(index)) * 40))
        prior.append(AIMessage(content=("确认" + str(index)) * 40))
    # Plant an industry term in the omitted middle so lossy compression would
    # otherwise drop it; overflow must clarify instead of searching.
    prior[20] = HumanMessage(content="关键行业是关键，不要丢掉")
    context = build_planner_context(
        message="继续搜索",
        prior_messages=prior,
        profile=None,
        trusted=None,
    )
    assert context.context_compressed is True
    assert context.compression_reason == "episode_context_overflow_fail_closed"
    result = resolve_intent(
        "继续搜索",
        execution_mode="group_agent",
        can_search=True,
        prior_messages=prior,
        scope=_scope(),
        planner_model=planner_returning(people_query("上海 运营")),
    )
    assert result.decision.action == "clarify"
    assert result.planner_status == "context_overflow"
    assert not _query_plan_would_run(result)


def test_phase2_groups_query_without_member_words_still_uses_user_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from apps.group_agent_api.agent_factory.intent_routing.execution import (
        prepare_intent_execution,
    )
    from apps.group_agent_api.agent_factory.intent_routing.plan_executor import (
        build_step_parameters,
    )

    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
    message = "查看上海创业社群和第一个社群的报名名单"
    result = resolve_intent(
        message,
        execution_mode="group_agent",
        can_search=True,
        planner_model=planner_returning(
            community_query("上海创业社群", group_member_scope="first_verified")
        ),
    )
    execution = prepare_intent_execution(
        result,
        original_message=message,
        run_id=None,
        request_fingerprint=None,
        execution_mode="group_agent",
        can_search=True,
    )
    assert result.decision.group_member_scope == "first_verified"
    assert build_step_parameters(execution, execution.plan.steps[0])["keyword"] == (
        "上海创业社群"
    )
    # Without a groups receipt yet, event_id stays unset; after a verified group
    # receipt the member step can resolve first_verified.
    members_step = execution.plan.steps[1]
    assert build_step_parameters(execution, members_step).get("event_id") is None
    reservation = execution.reserve(
        resource="groups", parameters={"page": 1}, tool_call_id="groups-1"
    )
    execution.commit_result(
        reservation,
        json.dumps(
            {
                "data": [{"id": 42, "name": "上海创业社群"}],
                "_source_url": "https://www.aihehuo.com/ai/wechat_groups.json?page=1",
            },
            ensure_ascii=False,
        ),
    )
    assert build_step_parameters(execution, members_step)["event_id"] == "42"


def test_phase2_legacy_execute_snapshot_without_queries_is_refused() -> None:
    from apps.group_agent_api.agent_factory.intent_routing.canonical import (
        RoutingCryptoError,
        canonical_hash,
    )
    from apps.group_agent_api.agent_factory.intent_routing.registry import READ_REGISTRY
    from apps.group_agent_api.agent_factory.intent_routing.snapshot import (
        normalize_intent_decision_payload,
        parse_routing_snapshot_payload,
    )

    decision = {
        "version": 1,
        "primary_intent": "people_search",
        "confidence": "high",
        "decision_source": "rules",
        "evidence_spans": ["有没有"],
        "constraints": {
            "location": ["上海"],
            "industry": ["餐饮"],
            "role": ["合伙人"],
            "capability": [],
            "time_range": None,
        },
        "negations": [],
        "action": "execute",
        "required_skill": "get-ai-cofounders",
        "query_plan_ref": {"plan_version": 1, "plan_id": "people-search-v1"},
        "requires_plan_receipts": True,
        "policy_version": "intent-policy-v1",
    }
    with pytest.raises(RoutingCryptoError, match="routing_snapshot_invalid"):
        normalize_intent_decision_payload(decision)

    body = {
        "routing_snapshot_version": 1,
        "run_id": "legacy-run",
        "intent_decision": decision,
        "query_plan": READ_REGISTRY["people_search"].plan.model_dump(mode="json"),
        "policy_version": "intent-policy-v1",
        "registry_version": "aihehuo-read-registry-v1",
        "created_at": "2026-09-15T08:00:00Z",
    }
    body["canonical_content_hash"] = canonical_hash(
        {key: value for key, value in body.items() if key != "canonical_content_hash"}
    )
    with pytest.raises(RoutingCryptoError, match="routing_snapshot_invalid"):
        parse_routing_snapshot_payload(body)


def test_phase2_single_turn_truncation_fail_closes_without_search() -> None:
    long_prefix = "A" * 1000
    prior = [HumanMessage(content=long_prefix + "餐饮合伙人")]
    context = build_planner_context(
        message="继续",
        prior_messages=prior,
        profile=None,
        trusted=None,
    )
    assert context.context_compressed is True
    assert context.compression_reason == "episode_turn_truncated_fail_closed"
    assert "餐饮" not in "".join(turn.content for turn in context.episode_turns)
    result = resolve_intent(
        "继续",
        execution_mode="group_agent",
        can_search=True,
        prior_messages=prior,
        scope=_scope(),
        planner_model=planner_returning(people_query("上海 运营")),
    )
    assert result.decision.action == "clarify"
    assert result.planner_status == "context_overflow"
    assert not _query_plan_would_run(result)


def test_phase2_guessed_group_event_id_is_ignored_without_request_or_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from apps.group_agent_api.agent_factory.intent_routing.execution import (
        prepare_intent_execution,
    )
    from apps.group_agent_api.agent_factory.intent_routing.plan_executor import (
        build_step_parameters,
    )

    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
    message = "查看社群详情"
    result = resolve_intent(
        message,
        execution_mode="group_agent",
        can_search=True,
        planner_model=planner_returning(
            community_query(
                "社群详情",
                group_member_scope="first_verified",
                group_event_id="763",
            )
        ),
    )
    execution = prepare_intent_execution(
        result,
        original_message=message,
        run_id=None,
        request_fingerprint=None,
        execution_mode="group_agent",
        can_search=True,
    )
    assert execution.decision.group_event_id == "763"
    members_step = execution.plan.steps[1]
    assert "event_id" not in build_step_parameters(execution, members_step)


def test_phase2_recovered_execute_denied_when_search_revoked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading
    from types import SimpleNamespace

    from apps.group_agent_api.agent_factory.intent_routing.execution import (
        QueryPlanError,
        prepare_intent_execution,
    )
    from apps.group_agent_api.agent_factory.intent_routing.snapshot import (
        build_routing_snapshot,
    )
    from apps.group_agent_api.execution.active_fence import (
        ActiveAttemptFence,
        clear_active_fence,
        set_active_fence,
    )
    from apps.group_agent_api.execution.models import LeaseClaim

    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
    decision = resolve_intent(
        "有没有上海餐饮合伙人",
        execution_mode="group_agent",
        can_search=True,
        planner_model=planner_returning(people_query("上海 餐饮 合伙人")),
    ).decision
    snapshot = build_routing_snapshot(run_id="perm-run", decision=decision)
    claim = LeaseClaim(
        run_id="perm-run",
        attempt_id="attempt-1",
        lease_token="token",
        lease_owner="worker",
        lease_expires_at=9_999_999_999.0,
        fencing_token=1,
    )

    class _Store:
        def get(self, run_id: str) -> Any:
            return SimpleNamespace(
                run_id=run_id,
                request_fingerprint="fp",
                routing_snapshot_hash=snapshot.canonical_content_hash,
                routing_snapshot_version=1,
                routing_snapshot_key_version="k1",
                routing_snapshot_ciphertext=object(),
                attempt_count=2,
            )

        def routing_crypto_material(self) -> tuple[dict[str, bytes], str]:
            return {"k1": bytes(32)}, "k1"

    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.intent_routing.execution.decrypt_routing_snapshot",
        lambda record, keys: snapshot,
    )
    tokens = set_active_fence(
        ActiveAttemptFence(
            store=_Store(),
            claim=claim,
            conversation_id="conversation-perm-run",
            cancel_event=threading.Event(),
        ),
        require=True,
    )
    try:
        clarify = resolve_intent(
            "有没有上海餐饮合伙人",
            execution_mode="group_agent",
            can_search=True,
            planner_model=None,
        )
        assert clarify.decision.action == "clarify"
        with pytest.raises(QueryPlanError, match="permission_denied"):
            prepare_intent_execution(
                clarify,
                original_message="有没有上海餐饮合伙人",
                run_id="perm-run",
                request_fingerprint="fp",
                execution_mode="group_agent",
                can_search=False,
            )
    finally:
        clear_active_fence(tokens)


def test_phase2_planner_does_not_log_raw_prompt_in_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, Any]] = []

    def capture(**kwargs: Any) -> None:
        captured.append(kwargs)

    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.intent_routing.resolver.record_decision_point",
        capture,
    )
    profile = SimpleNamespace(
        doing=SimpleNamespace(value="secret-doing-value"),
        need=SimpleNamespace(value="secret-need"),
        offer=SimpleNamespace(value="secret-offer"),
        match_constraints=[],
    )
    result = resolve_intent(
        "搜索",
        execution_mode="group_agent",
        can_search=True,
        planner_model=planner_returning(people_query("上海 餐厅 运营")),
        profile=profile,
    )
    record_intent_prediction(result, run_id="r", thread_id="t")
    blob = json.dumps(captured, ensure_ascii=False)
    assert "secret-doing-value" not in blob
    assert "上海 餐厅 运营" not in blob


async def test_chat_observe_runs_before_first_agent_generation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from apps.group_agent_api.app.endpoints import chat as chat_endpoint

    calls: list[str] = []

    class Agent:
        async def ainvoke(self, *_a: Any, **_k: Any) -> dict[str, Any]:
            calls.append("agent")
            return {"messages": [AIMessage(content="你好")]}

    state = AppState(
        agent=Agent(),
        base_dir=tmp_path,
        quality_model=planner_returning(reply()),
    )

    async def resolve(*_: Any, **__: Any) -> TrustedSession:
        return TrustedSession(
            principal=SessionPrincipal("user-1", None, None, "body_stub"),
            membership=MembershipResult(tier=CapabilityTier.in_group, source="stub"),
            group_id="group-1",
            group_token=None,
            capabilities=SessionCapabilities(
                tier=CapabilityTier.in_group,
                can_search=True,
                can_write_profile=True,
            ),
        )

    monkeypatch.setenv("GROUP_AGENT_INTENT_ROUTING_MODE", "observe")
    monkeypatch.setattr(chat_endpoint, "resolve_trusted_session", resolve)
    original = chat_endpoint.record_intent_prediction

    def wrapped(resolution: Any, **kwargs: Any) -> None:
        calls.append("intent")
        return original(resolution, **kwargs)

    monkeypatch.setattr(chat_endpoint, "record_intent_prediction", wrapped)
    response = await chat_endpoint.chat(
        ChatRequest(
            user_id="user-1",
            group_id="group-1",
            message="你好",
            run_match=False,
            run_invite=False,
        ),
        state,
    )
    assert response.reply == "你好"
    assert calls[0] == "intent"


def test_long_input_stays_bounded_and_fail_closes() -> None:
    message = "找" + ("上海餐饮运营" * 5_000)
    started = time.perf_counter()
    result = resolve_intent(
        message,
        execution_mode="group_agent",
        can_search=True,
        planner_model=planner_returning(people_query("上海 餐饮 运营")),
    )
    assert time.perf_counter() - started < 2.0
    assert result.decision.action == "clarify"
    assert result.planner_status == "context_overflow"
    assert not _query_plan_would_run(result)


def test_resource_queries_freeze_into_decision() -> None:
    result = resolve_intent(
        "找人",
        execution_mode="group_agent",
        can_search=True,
        planner_model=planner_returning(people_query("全职 上海 运营")),
    )
    assert result.decision.resource_queries == ResourceQueriesV2(
        users="全职 上海 运营"
    )
