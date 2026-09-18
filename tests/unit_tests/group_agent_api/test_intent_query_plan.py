"""INTENT-DA-002 QueryPlan, snapshot, and receipt tests."""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi import HTTPException
from langchain_core.messages import AIMessage, ToolMessage

from apps.group_agent_api.agent_factory.intent_routing.canonical import (
    RoutingCryptoError,
    canonical_hash,
    canonical_json_bytes,
    decrypt_canonical,
)
from apps.group_agent_api.agent_factory.intent_routing.context import IntentContextScope
from apps.group_agent_api.agent_factory.intent_routing.execution import (
    QueryPlanError,
    active_intent_execution,
    current_intent_execution,
    prepare_intent_execution,
    search_log_schema_enabled,
    search_result_schema_enabled,
)
from apps.group_agent_api.agent_factory.intent_routing.registry import (
    READ_REGISTRY,
    validate_registry,
)
from apps.group_agent_api.agent_factory.intent_routing.resolver import resolve_intent
from apps.group_agent_api.agent_factory.intent_routing.schema import IntentDecisionV1
from apps.group_agent_api.agent_factory.intent_routing.snapshot import (
    build_routing_snapshot,
    encrypt_routing_snapshot,
    routing_aad,
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
from tests.unit_tests.group_agent_api.planner_fixtures import (
    AdaptivePlannerModel,
    planner_for_message,
    planner_returning,
    people_query,
)


def _decision() -> IntentDecisionV1:
    return IntentDecisionV1.model_validate(
        {
            "version": 1,
            "primary_intent": "people_search",
            "confidence": "high",
            "decision_source": "planner",
            "evidence_spans": [],
            "constraints": {
                "location": [],
                "industry": [],
                "role": [],
                "capability": [],
                "time_range": None,
            },
            "negations": [],
            "action": "execute",
            "required_skill": "get-ai-cofounders",
            "query_plan_ref": {"plan_version": 1, "plan_id": "people-search-v1"},
            "requires_plan_receipts": True,
            "resource_queries": {
                "users": "上海 餐饮 合伙人",
                "ideas": None,
                "projects": None,
                "groups": None,
                "faqs": None,
            },
            "group_member_scope": "none",
            "group_event_id": None,
            "planner_version": "intent-planner-v2",
            "policy_version": "intent-policy-v1",
        }
    )


def test_frozen_routing_hash_aad_and_aead_vector() -> None:
    snapshot = build_routing_snapshot(
        run_id="run-intent-0001",
        decision=_decision(),
        created_at="2026-09-15T08:00:00Z",
    )
    # Phase 2 freezes planner resource_queries into the snapshot hash.
    assert snapshot.canonical_content_hash.startswith("sha256:")
    assert len(snapshot.canonical_content_hash) == 71
    assert snapshot.intent_decision.resource_queries.users == "上海 餐饮 合伙人"
    aad = routing_aad(
        run_id=snapshot.run_id,
        request_fingerprint=(
            "f09ae99c63e48a7ce2c076d5f32e5c57a5f01630746d6cf689515e421c65bd94"
        ),
        key_version="test-k1",
    )
    assert canonical_hash(aad) == (
        "sha256:c77fe1555cb24265fcbbc78888bac4c385cc2042ea13a7644b67ccde11ca6dea"
    )
    key = bytes(range(32))
    envelope = encrypt_routing_snapshot(
        snapshot,
        key=key,
        key_version="test-k1",
        request_fingerprint=aad["request_fingerprint"],
        nonce=bytes.fromhex("101112131415161718191a1b"),
    )
    restored = decrypt_canonical(
        envelope,
        keys={"test-k1": key},
        aad=aad,
        error_code="routing_snapshot_invalid",
    )
    assert restored == snapshot.model_dump(mode="json")
    with pytest.raises(RoutingCryptoError, match="routing_snapshot_invalid"):
        decrypt_canonical(
            envelope,
            keys={"test-k1": key},
            aad={**aad, "run_id": "other-run"},
            error_code="routing_snapshot_invalid",
        )


def test_canonical_rejects_float_unsafe_integer_and_surrogate() -> None:
    for value in (1.5, 9_007_199_254_740_992, "\ud800"):
        with pytest.raises(RoutingCryptoError, match="schema_invalid"):
            canonical_json_bytes({"value": value})


def test_registry_is_six_unique_closed_plans() -> None:
    validate_registry(
        installed_skills={entry.skill for entry in READ_REGISTRY.values()}
    )
    assert len(READ_REGISTRY) == 6
    assert len({entry.plan.plan_id for entry in READ_REGISTRY.values()}) == 6
    with pytest.raises(RuntimeError, match="query_plan_invalid"):
        validate_registry(installed_skills={"get-ai-cofounders"})


def test_public_query_timeout_budget_allows_fifteen_seconds_per_request() -> None:
    for entry in READ_REGISTRY.values():
        budget = entry.plan.budget
        assert budget.max_elapsed_ms == budget.max_requests * 15_000
        assert budget.max_elapsed_ms <= 60_000

    execution = _execution()
    reservation = execution.reserve(
        resource="users", parameters={"page": 1}, tool_call_id="timeout-budget"
    )
    assert reservation.hard_timeout_ms == 15_000
    assert execution.plan is not None
    assert execution.plan.budget.max_elapsed_ms == 30_000


def _execution(message: str = "有没有上海餐饮合伙人"):
    from tests.unit_tests.group_agent_api.planner_fixtures import planner_for_message

    resolution = resolve_intent(
        message,
        execution_mode="group_agent",
        can_search=True,
        planner_model=planner_for_message(message),
    )
    return prepare_intent_execution(
        resolution,
        original_message=message,
        run_id=None,
        request_fingerprint=None,
        execution_mode="group_agent",
        can_search=True,
    )


def _result(items: list[dict[str, object]], *, page: int = 1) -> str:
    import json

    return json.dumps(
        {
            "data": items,
            "_source_url": f"https://www.aihehuo.com/ai/users.json?page={page}",
        },
        ensure_ascii=False,
    )


def _install_route_skill(base_dir: Path) -> None:
    package = base_dir / "skills" / "aihehuo" / "get-ai-cofounders"
    package.mkdir(parents=True)
    (package / "SKILL.md").write_text(
        "# 创业合伙人搜索\n仅调用 users 公开资源。",
        encoding="utf-8",
    )
    (package / "package.json").write_text(
        json.dumps({"name": "get-ai-cofounders", "resources": ["users"]}),
        encoding="utf-8",
    )


def test_completed_execution_uses_app_ids_and_rejects_public_number_as_route_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from apps.group_agent_api.agent_factory.intent_routing.projection import (
        _public_candidate,
        project_intent_search,
    )

    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
    execution = _execution()
    reservation = execution.reserve(
        resource="users", parameters={"page": 1}, tool_call_id="projection"
    )
    execution.commit_result(
        reservation,
        _result(
            [
                {
                    "id": 999,
                    "number": 101,
                    "user_id": "999",
                    "name": "张三",
                    "source_group_id": "some-other-group",
                    "city": "上海",
                    "industry": "餐饮",
                    "title": "创业合伙人",
                    "bio": "餐饮供应链经验",
                    "phone": "13800000000",
                }
            ]
        ),
    )

    projected = project_intent_search(execution)
    assert projected.completed
    assert projected.result_count == 1
    assert projected.public_reads[0].records[0]["name"] == "张三"
    assert [candidate["user_id"] for candidate in projected.candidates] == ["999"]
    assert projected.candidates[0]["kind"] == "public"
    assert projected.candidates[0]["identity_kind"] == "app_user_id"
    assert "source_group_id" not in projected.candidates[0]
    assert "13800000000" not in str(projected.candidates)
    assert projected.public_result_names == ["张三"]

    assert _public_candidate({"id": "101", "number": 101, "name": "旧公网结果"}) is None


def test_ephemeral_required_receipt_empty_and_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
    execution = _execution()
    first = execution.reserve(
        resource="users",
        parameters={"page": 1, "per": 20},
        tool_call_id="tool-1",
    )
    receipt = execution.commit_result(first, _result([{"id": "u-1"}]))
    assert receipt.status == "succeeded"
    assert receipt.request_index == 1
    assert execution.required_steps_satisfied() is True

    second = execution.reserve(
        resource="users",
        parameters={"page": 2, "per": 20},
        tool_call_id="tool-2",
    )
    empty = execution.commit_result(second, _result([], page=2))
    assert empty.status == "empty"
    assert empty.result_count == 0
    with pytest.raises(QueryPlanError, match="query_budget_exhausted"):
        execution.reserve(
            resource="users",
            parameters={"page": 3},
            tool_call_id="tool-3",
        )


def test_failure_audit_does_not_satisfy_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
    execution = _execution()
    reservation = execution.reserve(
        resource="users", parameters={}, tool_call_id="tool-failed"
    )
    audit = execution.commit_failure(reservation, timed_out=True)
    assert audit.error_code == "tool_timed_out"
    assert execution.receipts == []
    assert execution.required_steps_satisfied() is False


def test_plan_budget_counts_cross_page_items_deduplicated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
    execution = _execution()
    first = execution.reserve(
        resource="users", parameters={"page": 1}, tool_call_id="page-1"
    )
    execution.commit_result(first, _result([{"id": "u-1"}], page=1))
    second = execution.reserve(
        resource="users", parameters={"page": 2}, tool_call_id="page-2"
    )
    receipt = execution.commit_result(
        second,
        _result([{"id": "u-1"}, {"id": "u-2"}], page=2),
    )
    assert receipt.result_count == 2
    assert execution.unique_result_count() == 2
    assert execution._ephemeral_budget is not None  # noqa: SLF001
    assert execution._ephemeral_budget.items_committed == 2  # noqa: SLF001


def test_duplicate_query_and_oversize_result_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
    duplicate = _execution()
    duplicate.reserve(resource="users", parameters={}, tool_call_id="first")
    with pytest.raises(QueryPlanError, match="query_budget_exhausted"):
        duplicate.reserve(resource="users", parameters={}, tool_call_id="duplicate")

    oversize = _execution()
    reservation = oversize.reserve(
        resource="users", parameters={}, tool_call_id="oversize"
    )
    with pytest.raises(QueryPlanError, match="query_budget_exhausted"):
        oversize.commit_result(
            reservation,
            _result([{"id": f"u-{index}"} for index in range(41)]),
        )
    assert oversize.receipts == []


def test_selected_skill_and_adopted_evidence_are_injected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from skillpacks.aihehuo import evidence as public_evidence

    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
    execution = _execution()
    reservation = execution.reserve(
        resource="users", parameters={}, tool_call_id="adopted"
    )
    receipt = execution.commit_result(reservation, _result([{"id": "u-1"}]))
    execution.adopted_receipt_ids.add(receipt.receipt_id)
    _install_route_skill(tmp_path)

    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        public_evidence,
        "record_result",
        lambda resource, content: recorded.append((resource, content)),
    )
    execution.restore_adopted_evidence()
    prompt = execution.route_prompt(skills_dir=tmp_path / "skills" / "aihehuo")
    assert "[SERVER-SELECTED-SKILL]" in prompt
    assert "仅调用 users 公开资源" in prompt
    assert "[ADOPTED-DURABLE-EVIDENCE]" in prompt
    assert receipt.receipt_id in prompt
    assert recorded and recorded[0][0] == "users"

    with pytest.raises(QueryPlanError, match="query_plan_invalid"):
        execution.route_prompt(skills_dir=tmp_path / "missing")


async def test_server_executor_runs_required_step_without_model_tool_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from apps.group_agent_api.agent_factory.intent_routing.plan_executor import (
        execute_query_plan,
    )
    from skillpacks.aihehuo import integration

    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
    _install_route_skill(tmp_path)
    monkeypatch.setattr(
        integration,
        "fetch",
        lambda path, **_kwargs: json.dumps(
            {
                "data": [{"id": "u-1", "name": "公开用户"}],
                "meta": {"has_more": False},
            },
            ensure_ascii=False,
        ),
    )
    execution = _execution()

    await execute_query_plan(
        execution, skills_dir=tmp_path / "skills" / "aihehuo"
    )

    assert execution.required_steps_satisfied()
    assert execution.receipts[0].tool_call_id == "server-users-primary-1"
    assert execution.receipts[0].resource == "users"
    prompt = execution.route_prompt(skills_dir=tmp_path / "skills" / "aihehuo")
    assert "[SERVER-VERIFIED-EVIDENCE]" in prompt
    assert "公开用户" in prompt


async def test_server_executor_paginates_after_active_steps_receive_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from apps.group_agent_api.agent_factory.intent_routing.plan_executor import (
        execute_query_plan,
    )
    from skillpacks.aihehuo import integration

    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
    for name, resources in (
        ("get-ai-opportunities", ["ideas", "projects"]),
        ("get-ai-project-partnerships", ["projects", "ideas"]),
    ):
        package = tmp_path / "skills" / "aihehuo" / name
        package.mkdir(parents=True)
        (package / "SKILL.md").write_text("test", encoding="utf-8")
        (package / "package.json").write_text(
            json.dumps({"name": name, "resources": resources}), encoding="utf-8"
        )
    paths: list[str] = []

    def fetch(path: str, **_kwargs: object) -> str:
        paths.append(path)
        return '{"data":[{"id":"x"}],"meta":{"has_more":true,"next_page":2}}'

    monkeypatch.setattr(integration, "fetch", fetch)
    execution = _execution("看看上海有哪些餐饮创业项目")
    await execute_query_plan(
        execution, skills_dir=tmp_path / "skills" / "aihehuo"
    )

    assert execution.required_steps_satisfied()
    assert [receipt.resource for receipt in execution.receipts[:2]] == [
        "ideas",
        "projects",
    ]
    assert len(paths) == execution.plan.budget.max_requests


def test_server_parameters_preserve_semantics_and_map_time_range() -> None:
    from apps.group_agent_api.agent_factory.intent_routing.plan_executor import (
        build_step_parameters,
    )

    message = "帮我找上海做餐饮的创业者，优先有供应链经验，最近的"
    execution = _execution(message)
    assert execution.decision.primary_intent == "people_search"
    parameters = build_step_parameters(execution, execution.plan.steps[0])

    assert parameters["q"] == "上海 餐饮 供应链经验 创业者"
    anchor = datetime.datetime.fromisoformat(
        execution.snapshot.created_at.replace("Z", "+00:00")
    )
    created_since = datetime.datetime.fromisoformat(
        parameters["created_since"].replace("Z", "+00:00")
    )
    assert anchor - created_since == datetime.timedelta(days=30)

    project_execution = _execution("看看上海有哪些餐饮创业项目")
    project_step = project_execution.plan.steps[1]
    assert build_step_parameters(project_execution, project_step)["keyword"] == (
        "上海 餐饮"
    )


def test_people_query_drops_request_scaffolding_but_keeps_search_details() -> None:
    from apps.group_agent_api.agent_factory.intent_routing.plan_executor import (
        build_step_parameters,
    )

    basic = _execution()
    assert build_step_parameters(basic, basic.plan.steps[0])["q"] == (
        "上海 餐饮 合伙人"
    )

    simple = _execution("请帮我搜索上海餐饮行业的运营合伙人")
    assert build_step_parameters(simple, simple.plan.steps[0])["q"] == (
        "上海 餐饮 运营合伙人"
    )

    execution = _execution("请帮我搜索上海餐饮行业的全职运营合伙人")
    assert build_step_parameters(execution, execution.plan.steps[0])["q"] == (
        "上海 餐饮 运营合伙人 全职"
    )


def test_followup_query_merges_resolved_context_constraints() -> None:
    from apps.group_agent_api.agent_factory.intent_routing.plan_executor import (
        build_step_parameters,
    )
    from tests.unit_tests.group_agent_api.planner_fixtures import (
        people_query,
        planner_returning,
    )

    prior_messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "prior-search",
                    "name": "query_aihehuo_public_data",
                    "args": {
                        "resource": "users",
                        "location": "北京",
                        "role": "技术合伙人",
                    },
                }
            ],
        ),
        ToolMessage(
            content='{"status":"succeeded","items":[]}',
            tool_call_id="prior-search",
            name="query_aihehuo_public_data",
        ),
        AIMessage(content="查询完成。"),
    ]
    resolution = resolve_intent(
        "上海的呢",
        execution_mode="group_agent",
        can_search=True,
        prior_messages=prior_messages,
        scope=IntentContextScope(
            thread_id="thread-1",
            user_id="user-1",
            execution_mode="group_agent",
            episode_id="episode-1",
        ),
        planner_model=planner_returning(people_query("上海 技术合伙人")),
    )
    execution = prepare_intent_execution(
        resolution,
        original_message="上海的呢",
        run_id=None,
        request_fingerprint=None,
        execution_mode="group_agent",
        can_search=True,
    )
    query = build_step_parameters(execution, execution.plan.steps[0])["q"]

    assert resolution.decision.decision_source == "planner"
    assert query == "上海 技术合伙人"

    negated_resolution = resolve_intent(
        "不要北京，改成上海的",
        execution_mode="group_agent",
        can_search=True,
        prior_messages=prior_messages,
        scope=IntentContextScope(
            thread_id="thread-1",
            user_id="user-1",
            execution_mode="group_agent",
            episode_id="episode-1",
        ),
        planner_model=planner_returning(people_query("上海 技术合伙人")),
    )
    negated_execution = prepare_intent_execution(
        negated_resolution,
        original_message="不要北京，改成上海的",
        run_id=None,
        request_fingerprint=None,
        execution_mode="group_agent",
        can_search=True,
    )
    negated_query = build_step_parameters(
        negated_execution, negated_execution.plan.steps[0]
    )["q"]

    assert negated_query == "上海 技术合伙人"


@pytest.mark.parametrize(
    ("message", "created_at", "expected"),
    [
        ("看看本月的创业机会", "2026-10-31T16:30:00Z", "2026-10-31T16:00:00.000Z"),
        ("看看本周的创业机会", "2026-09-06T16:30:00Z", "2026-09-06T16:00:00.000Z"),
    ],
)
def test_calendar_ranges_use_shanghai_boundaries(
    message: str, created_at: str, expected: str
) -> None:
    from apps.group_agent_api.agent_factory.intent_routing.plan_executor import (
        build_step_parameters,
    )

    execution = _execution(message)
    execution.snapshot = build_routing_snapshot(
        run_id=execution.snapshot.run_id,
        decision=execution.decision,
        created_at=created_at,
    )

    parameters = build_step_parameters(execution, execution.plan.steps[0])
    assert parameters["created_since"] == expected


async def test_community_plan_selects_verified_first_group_for_members(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from apps.group_agent_api.agent_factory.intent_routing.plan_executor import (
        execute_query_plan,
    )
    from skillpacks.aihehuo import integration

    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
    package = tmp_path / "skills" / "aihehuo" / "get-ai-communities"
    package.mkdir(parents=True)
    (package / "SKILL.md").write_text("test", encoding="utf-8")
    (package / "package.json").write_text(
        json.dumps(
            {
                "name": "get-ai-communities",
                "resources": ["groups", "group-members"],
            }
        ),
        encoding="utf-8",
    )
    paths: list[str] = []

    def fetch(path: str, **_kwargs: object) -> str:
        paths.append(path)
        if "/ai/wechat_groups.json" in path:
            return '{"data":[{"id":42,"name":"上海创业社群"}]}'
        if "/ai/wechat_groups/42.json" in path:
            return '{"data":{"id":42,"members":[{"id":7}]}}'
        raise AssertionError(path)

    monkeypatch.setattr(integration, "fetch", fetch)
    execution = _execution("查看上海创业社群和第一个社群的报名名单")
    assert execution.decision.primary_intent == "community_search"

    await execute_query_plan(
        execution, skills_dir=tmp_path / "skills" / "aihehuo"
    )

    assert [receipt.resource for receipt in execution.receipts] == [
        "groups",
        "group-members",
    ]
    assert any("/ai/wechat_groups/42.json" in path for path in paths)


def test_community_explicit_event_id_is_bounded_and_requires_member_scope() -> None:
    from apps.group_agent_api.agent_factory.intent_routing.plan_executor import (
        build_step_parameters,
    )

    selected = _execution("查看社群 ID 763 的报名名单")
    assert selected.decision.primary_intent == "community_search"
    member_step = selected.plan.steps[1]
    assert build_step_parameters(selected, member_step)["event_id"] == "763"

    details_only = _execution("查看社群 ID 763 的详情")
    assert "event_id" not in build_step_parameters(
        details_only, details_only.plan.steps[1]
    )


def test_prompt_evidence_has_one_global_byte_and_item_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
    _install_route_skill(tmp_path)
    execution = _execution()
    reservation = execution.reserve(
        resource="users", parameters={}, tool_call_id="large-evidence"
    )
    items = [
        {f"field_{field}": "外部可控文本" * 400 for field in range(40)}
        for _item in range(40)
    ]
    execution.commit_result(
        reservation,
        json.dumps(
            {
                "data": items,
                "_source_url": "https://www.aihehuo.com/ai/users.json?page=1",
            },
            ensure_ascii=False,
        ),
    )

    prompt = execution.route_prompt(skills_dir=tmp_path / "skills" / "aihehuo")
    evidence_json = prompt.split("[SERVER-VERIFIED-EVIDENCE]\n", 1)[1].split(
        "\n[/SERVER-VERIFIED-EVIDENCE]", 1
    )[0]
    assert len(evidence_json.encode("utf-8")) <= 96 * 1024
    assert len(json.loads(evidence_json)[0]["evidence"]) < 40
    assert "[SERVER-EVIDENCE-LIMIT]" in prompt


def test_profile_update_delegates_without_query_plan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tests.unit_tests.group_agent_api.planner_fixtures import planner_for_message

    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
    resolution = resolve_intent(
        "把我的所在地改成上海",
        execution_mode="group_agent",
        can_search=True,
        planner_model=planner_for_message("把我的所在地改成上海"),
    )
    execution = prepare_intent_execution(
        resolution,
        original_message="把我的所在地改成上海",
        run_id=None,
        request_fingerprint=None,
        execution_mode="group_agent",
        can_search=True,
    )
    assert execution.decision.action == "delegate_existing_flow"
    assert execution.plan is None
    assert execution.route_prompt(skills_dir=tmp_path) == ""


def test_triggered_conditional_step_is_required_for_plan_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
    message = "看看上海有哪些餐饮创业项目"
    execution = _execution(message)
    first = execution.reserve(
        resource="ideas", parameters={}, tool_call_id="ideas-call"
    )
    execution.commit_result(
        first,
        '{"data":[{"id":"idea-1"}],'
        '"_source_url":"https://www.aihehuo.com/ai/ideas.json?page=1"}',
    )
    assert execution.required_steps_satisfied() is False
    second = execution.reserve(
        resource="projects", parameters={}, tool_call_id="projects-call"
    )
    execution.commit_result(
        second,
        '{"data":[],"_source_url":"https://www.aihehuo.com/ai/projects.json?page=1"}',
    )
    assert execution.required_steps_satisfied() is True


def test_wrong_resource_and_parameters_fail_before_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
    execution = _execution()
    with pytest.raises(QueryPlanError, match="query_plan_invalid"):
        execution.reserve(resource="ideas", parameters={}, tool_call_id="wrong")
    with pytest.raises(QueryPlanError, match="query_plan_invalid"):
        execution.reserve(
            resource="users",
            parameters={"event_id": "42"},
            tool_call_id="wrong-parameter",
        )


def test_legacy_read_mode_cannot_satisfy_execute_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.unit_tests.group_agent_api.planner_fixtures import planner_for_message

    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "legacy")
    resolution = resolve_intent(
        "帮我找合伙人",
        execution_mode="group_agent",
        can_search=True,
        planner_model=planner_for_message("帮我找合伙人"),
    )
    with pytest.raises(QueryPlanError, match="query_plan_invalid"):
        prepare_intent_execution(
            resolution,
            original_message="帮我找合伙人",
            run_id=None,
            request_fingerprint=None,
            execution_mode="group_agent",
            can_search=True,
        )


def test_receipt_id_excludes_itself(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
    execution = _execution()
    reservation = execution.reserve(
        resource="users", parameters={}, tool_call_id="tool-id"
    )
    receipt = execution.commit_result(reservation, _result([{"id": "u-1"}]))
    body = receipt.model_dump(mode="json")
    body.pop("receipt_id")
    assert receipt.receipt_id == canonical_hash(body)
    assert receipt.receipt_id.startswith("sha256:")
    assert len(receipt.receipt_id) == len("sha256:") + hashlib.sha256().digest_size * 2


def test_public_tool_wrapper_blocks_wrong_resource_and_commits_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from skillpacks.aihehuo import integration

    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
    skills = tmp_path / "skills"
    package = skills / "get-ai-cofounders"
    package.mkdir(parents=True)
    (package / "SKILL.md").write_text("test", encoding="utf-8")
    (package / "package.json").write_text(
        json.dumps({"name": "get-ai-cofounders", "resources": ["users"]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        integration,
        "_catalog",
        lambda: {"get-ai-cofounders": {"resources": ["users"]}},
    )
    fetches: list[str] = []

    def fetch(path: str, **_: object) -> str:
        fetches.append(path)
        return '{"data":[{"id":"u-1"}]}'

    monkeypatch.setattr(integration, "fetch", fetch)
    tool = integration.create_aihehuo_tool(skills_dir=skills)
    execution = _execution()
    with active_intent_execution(execution):
        result = tool.invoke(
            {
                "name": "query_aihehuo_public_data",
                "args": {"resource": "users", "page": 1},
                "id": "tool-call-1",
                "type": "tool_call",
            }
        )
    assert "u-1" in result.content
    assert fetches == ["/ai/users.json?page=1"]
    assert execution.receipts[0].tool_call_id == "tool-call-1"
    assert execution.required_steps_satisfied()


@pytest.mark.parametrize("emitter_failure", ["rejected", "raised"])
async def test_public_tool_async_waits_for_tool_started_ack_before_fetch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    emitter_failure: str,
) -> None:
    from skillpacks.aihehuo import integration

    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
    skills = tmp_path / "skills"
    package = skills / "get-ai-cofounders"
    package.mkdir(parents=True)
    (package / "SKILL.md").write_text("test", encoding="utf-8")
    (package / "package.json").write_text(
        json.dumps({"name": "get-ai-cofounders", "resources": ["users"]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        integration,
        "_catalog",
        lambda: {"get-ai-cofounders": {"resources": ["users"]}},
    )
    fetches: list[str] = []
    monkeypatch.setattr(
        integration,
        "fetch",
        lambda path, **_kwargs: fetches.append(path) or '{"data":[]}',
    )

    async def emitter(_event: str, _payload: dict[str, Any]) -> bool:
        if emitter_failure == "raised":
            raise RuntimeError("transport failed")
        return False

    execution = _execution()
    execution.bind_progress_emitter(emitter, started_monotonic=0.0, emit_search_logs=True)
    tool = integration.create_aihehuo_tool(skills_dir=skills)
    with active_intent_execution(execution):
        result = await tool.ainvoke({"resource": "users", "page": 1})

    assert result == "AIHEHUO query rejected: callback_ack_invalid"
    assert fetches == []


async def test_public_tool_async_uses_fifteen_second_fetch_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from skillpacks.aihehuo import integration

    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
    _install_route_skill(tmp_path)
    skills = tmp_path / "skills" / "aihehuo"
    monkeypatch.setattr(
        integration,
        "_catalog",
        lambda: {"get-ai-cofounders": {"resources": ["users"]}},
    )
    timeouts: list[float] = []

    def fetch(_path: str, *, timeout: float, **_kwargs: object) -> str:
        timeouts.append(timeout)
        return '{"data":[{"id":"u-1"}]}'

    events: list[tuple[str, dict[str, Any]]] = []

    async def emitter(_event: str, _payload: dict[str, Any]) -> bool:
        events.append((_event, _payload))
        return True

    monkeypatch.setattr(integration, "fetch", fetch)
    execution = _execution()
    execution.bind_progress_emitter(emitter, started_monotonic=0.0, emit_search_logs=True)
    tool = integration.create_aihehuo_tool(skills_dir=skills)
    with active_intent_execution(execution):
        result = await tool.ainvoke({"resource": "users", "page": 1})

    assert "u-1" in result
    assert timeouts == [15.0]
    assert [event for event, _ in events[:2]] == ["progress", "search_log"]
    assert events[1][1]["search_log"]["query"] == "（无关键词）"
    assert events[1][1]["search_log"]["resource"] == "users"
    assert execution.required_steps_satisfied()


async def test_public_tool_async_search_log_ack_precedes_fetch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from skillpacks.aihehuo import integration

    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
    _install_route_skill(tmp_path)
    monkeypatch.setattr(
        integration,
        "_catalog",
        lambda: {"get-ai-cofounders": {"resources": ["users"]}},
    )
    fetches: list[str] = []
    monkeypatch.setattr(
        integration,
        "fetch",
        lambda path, **_kwargs: fetches.append(path) or '{"data":[]}',
    )
    events: list[tuple[str, dict[str, Any]]] = []

    async def emitter(event: str, payload: dict[str, Any]) -> bool:
        events.append((event, payload))
        return event != "search_log"

    execution = _execution()
    execution.bind_progress_emitter(emitter, started_monotonic=0.0, emit_search_logs=True)
    tool = integration.create_aihehuo_tool(skills_dir=tmp_path / "skills" / "aihehuo")
    with active_intent_execution(execution):
        result = await tool.ainvoke(
            {"resource": "users", "q": "上海 餐饮 运营", "page": 2}
        )

    assert result == "AIHEHUO query rejected: callback_ack_invalid"
    assert fetches == []
    assert [event for event, _ in events] == ["progress", "search_log"]
    assert events[1][1]["search_log"]["query"] == "q=上海 餐饮 运营"
    assert events[1][1]["search_log"]["page"] == 2


async def test_public_tool_emits_bounded_reviewed_search_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from skillpacks.aihehuo import integration

    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
    _install_route_skill(tmp_path)
    monkeypatch.setattr(
        integration,
        "_catalog",
        lambda: {"get-ai-cofounders": {"resources": ["users"]}},
    )
    monkeypatch.setattr(
        integration,
        "fetch",
        lambda _path, **_kwargs: json.dumps(
            {
                "data": [
                    {"id": "1", "name": "张三", "city": "上海", "industry": "餐饮", "bio": "餐饮合伙人", "phone": "13800138000"},
                    {"id": "2", "name": "李四", "city": "北京", "industry": "餐饮", "bio": "餐饮合伙人"},
                ]
            },
            ensure_ascii=False,
        ),
    )
    events: list[tuple[str, dict[str, Any]]] = []

    async def emitter(event: str, payload: dict[str, Any]) -> bool:
        events.append((event, payload))
        return True

    execution = _execution("有没有上海餐饮合伙人")
    execution.bind_progress_emitter(
        emitter, started_monotonic=0.0, emit_search_logs=True, emit_search_results=True
    )
    tool = integration.create_aihehuo_tool(skills_dir=tmp_path / "skills" / "aihehuo")
    with active_intent_execution(execution):
        await tool.ainvoke({"resource": "users", "q": "上海 餐饮 合伙人", "page": 1})

    assert [event for event, _ in events] == [
        "progress", "search_log", "progress", "search_result"
    ]
    result = events[-1][1]["search_result"]
    assert result["search_id"] == events[1][1]["search_log"]["search_id"]
    assert result["result_count"] == 2
    assert [item["review_status"] for item in result["items"]] == ["not_applicable", "not_applicable"]
    assert result["items"][1]["review_reasons"] == []
    assert "phone" not in result["items"][0]
    assert "13800138000" not in str(result)


async def test_search_diagnostics_are_v3_only_and_redact_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GROUP_AGENT_SEARCH_LOG_SCHEMA_VERSION", raising=False)
    assert search_log_schema_enabled() is False
    monkeypatch.setenv("GROUP_AGENT_SEARCH_LOG_SCHEMA_VERSION", "1")
    assert search_log_schema_enabled() is True
    monkeypatch.setenv("GROUP_AGENT_SEARCH_LOG_SCHEMA_VERSION", "invalid")
    assert search_log_schema_enabled() is False
    monkeypatch.delenv("GROUP_AGENT_SEARCH_RESULT_SCHEMA_VERSION", raising=False)
    assert search_result_schema_enabled() is False
    monkeypatch.setenv("GROUP_AGENT_SEARCH_RESULT_SCHEMA_VERSION", "1")
    assert search_result_schema_enabled() is True
    execution = _execution()
    events: list[str] = []

    async def emitter(event: str, _payload: dict[str, Any]) -> bool:
        events.append(event)
        return True

    execution.bind_progress_emitter(emitter, started_monotonic=0.0)
    reservation = await execution.reserve_async(
        resource="users", parameters={"q": "上海"}, tool_call_id="legacy-tool"
    )
    await execution.record_search_async(reservation)
    assert events == ["progress"]
    assert "secret" not in execution._search_query(
        {"q": "Bearer secret access"}
    )


async def test_sync_reserve_on_progress_loop_fails_closed() -> None:
    async def emitter(_event: str, _payload: dict[str, Any]) -> bool:
        return True

    execution = _execution()
    execution.bind_progress_emitter(emitter, started_monotonic=0.0)

    with pytest.raises(QueryPlanError, match="callback_ack_invalid"):
        execution.reserve(resource="users", parameters={}, tool_call_id="sync-call")
    assert execution._ephemeral_budget is not None
    assert execution._ephemeral_budget.requests_reserved == 0


class _RetryAgent:
    def __init__(self, *, complete_on_retry: bool, reply: str = "公开查询处理完成。") -> None:
        self.complete_on_retry = complete_on_retry
        self.reply = reply
        self.calls = 0
        self.inputs: list[Any] = []

    async def aget_state(self, _: Any) -> Any:
        return SimpleNamespace(values={"messages": []})

    async def ainvoke(self, invocation: Any, __: Any) -> dict[str, Any]:
        self.calls += 1
        self.inputs.append(invocation)
        execution = current_intent_execution()
        assert execution is not None
        if self.calls == 2 and self.complete_on_retry:
            reservation = await execution.reserve_async(
                resource="users", parameters={}, tool_call_id="retry-tool"
            )
            await execution.commit_result_async(reservation, _result([{"id": "u-1"}]))
        return {"messages": [AIMessage(content=self.reply)]}


class _PersistentAgent:
    def __init__(self, history: list[Any]) -> None:
        self.history = history

    async def aget_state(self, _: Any) -> Any:
        return SimpleNamespace(values={"messages": list(self.history)})

    async def ainvoke(self, invocation: Any, __: Any) -> dict[str, Any]:
        assert current_intent_execution() is not None
        self.history.extend(invocation["messages"])
        self.history.append(AIMessage(content="已根据公开结果完成查询。"))
        return {"messages": list(self.history)}


def _session() -> TrustedSession:
    return TrustedSession(
        principal=SessionPrincipal("user-1", None, None, "body_stub"),
        membership=MembershipResult(tier=CapabilityTier.in_group, source="stub"),
        group_id="group-1",
        group_token=None,
        capabilities=SessionCapabilities(
            tier=CapabilityTier.in_group,
            can_search=True,
            can_write_profile=False,
        ),
    )


def test_planner_failure_callback_does_not_claim_search() -> None:
    from apps.group_agent_worker.tasks import _terminal_error_payload
    from apps.group_agent_api.execution.models import RetryClass
    from apps.group_agent_api.execution.retry import classify_error

    payload = _terminal_error_payload("intent_classifier_invalid", timed_out=False)
    assert payload["error_code"] == "intent_classifier_invalid"
    assert "未执行搜索" in payload["message"]
    assert classify_error("intent_classifier_invalid") == RetryClass.PERMANENT


async def test_chat_planner_failure_stops_before_composing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from apps.group_agent_api.app.endpoints import chat as chat_endpoint

    agent = _RetryAgent(complete_on_retry=False)
    state = AppState(
        agent=agent,
        base_dir=tmp_path,
        search_only_agent=agent,
        write_only_agent=agent,
        no_tool_agent=agent,
        quality_model=planner_returning(people_query(""), people_query("")),
    )

    async def resolve(*_: Any, **__: Any) -> TrustedSession:
        return _session()

    monkeypatch.setenv("GROUP_AGENT_INTENT_ROUTING_MODE", "enforce")
    monkeypatch.setattr(chat_endpoint, "resolve_trusted_session", resolve)
    with pytest.raises(HTTPException) as raised:
        await chat_endpoint.chat(
            ChatRequest(
                user_id="user-1",
                group_id="group-1",
                message="能先帮我找人吗",
                run_match=False,
                run_invite=False,
            ),
            state,
        )
    assert raised.value.status_code == 503
    assert raised.value.detail["error"] == "intent_classifier_invalid"
    assert "未执行搜索" in raised.value.detail["message"]
    assert agent.calls == 0


async def test_async_planner_failure_stops_before_composing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from apps.group_agent_api.app import async_manager

    agent = _RetryAgent(complete_on_retry=False)
    state = AppState(
        agent=agent,
        base_dir=tmp_path,
        search_only_agent=agent,
        write_only_agent=agent,
        no_tool_agent=agent,
        quality_model=planner_returning(people_query(""), people_query("")),
    )
    events: list[tuple[str, dict[str, Any]]] = []

    async def emit(event: str, payload: dict[str, Any]) -> bool:
        events.append((event, payload))
        return True

    monkeypatch.setenv("GROUP_AGENT_INTENT_ROUTING_MODE", "enforce")
    monkeypatch.setenv(
        "GROUP_AGENT_CALLBACK_ALLOWED_BASE_URLS",
        "http://example.invalid/callbacks",
    )
    with pytest.raises(QueryPlanError) as raised:
        await async_manager._execute_core_agent(
            req=AsyncCallRequest(
                run_id="intent-planner-invalid",
                idempotency_key="intent-planner-invalid-key",
                user_id="user-1",
                group_id="group-1",
                conversation_id="conversation-1",
                message="能先帮我找人吗",
                callback_url="http://example.invalid/callbacks/intent-planner-invalid",
                run_match=False,
                run_invite=False,
            ),
            session=_session(),
            state=state,
            tid="intent-planner-invalid-thread",
            emit_callback=emit,
        )
    assert raised.value.code == "intent_classifier_invalid"
    assert agent.calls == 0
    assert all(event != "final" for event, _payload in events)


async def test_chat_enforce_executes_required_tool_before_composing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from apps.group_agent_api.app.endpoints import chat as chat_endpoint

    search_agent = _RetryAgent(complete_on_retry=True)
    compose_agent = _RetryAgent(complete_on_retry=True)
    state = AppState(
        agent=search_agent,
        base_dir=tmp_path,
        search_only_agent=search_agent,
        write_only_agent=compose_agent,
        no_tool_agent=compose_agent,
        quality_model=planner_for_message("帮我找技术合伙人"),
    )
    _install_route_skill(tmp_path)

    async def resolve(*_: Any, **__: Any) -> TrustedSession:
        return _session()

    monkeypatch.setenv("GROUP_AGENT_INTENT_ROUTING_MODE", "enforce")
    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
    from skillpacks.aihehuo import integration

    monkeypatch.setattr(
        integration,
        "fetch",
        lambda _path, **_kwargs: '{"data":[{"id":101,"user_id":101,"number":9659712,"name":"张三","bio":"技术合伙人"}],"meta":{"has_more":false}}',
    )
    monkeypatch.setattr(chat_endpoint, "resolve_trusted_session", resolve)
    response = await chat_endpoint.chat(
        ChatRequest(
            user_id="user-1",
            group_id="group-1",
            message="帮我找技术合伙人",
            run_match=False,
            run_invite=False,
        ),
        state,
    )
    assert response.reply == "公开查询处理完成。"
    assert response.match_reason == "intent_query_plan"
    assert response.match_status == "matched"
    assert [candidate["user_id"] for candidate in response.candidates] == ["101"]
    assert search_agent.calls == 0
    assert compose_agent.calls == 1
    prompt = "\n".join(
        str(message.content)
        for message in compose_agent.inputs[0]["messages"]
        if hasattr(message, "content")
    )
    assert "[SERVER-VERIFIED-EVIDENCE]" in prompt
    assert '\"id\":101' in prompt
    assert '\"name\":\"张三\"' in prompt


async def test_chat_empty_receipt_cannot_promise_future_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from apps.group_agent_api.app.endpoints import chat as chat_endpoint
    from skillpacks.aihehuo import integration

    agent = _RetryAgent(
        complete_on_retry=False,
        reply="我马上查公开人才库，稍等给你名单。",
    )
    state = AppState(
        agent=agent,
        base_dir=tmp_path,
        search_only_agent=agent,
        write_only_agent=agent,
        no_tool_agent=agent,
        quality_model=planner_for_message("帮我找技术合伙人"),
    )
    _install_route_skill(tmp_path)

    async def resolve(*_: Any, **__: Any) -> TrustedSession:
        return _session()

    monkeypatch.setenv("GROUP_AGENT_INTENT_ROUTING_MODE", "enforce")
    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
    paths: list[str] = []

    def fetch(path: str, **_kwargs: Any) -> str:
        paths.append(path)
        return '{"data":[],"meta":{"has_more":false}}'

    monkeypatch.setattr(integration, "fetch", fetch)
    monkeypatch.setattr(chat_endpoint, "resolve_trusted_session", resolve)
    response = await chat_endpoint.chat(
        ChatRequest(
            user_id="user-1", group_id="group-1", message="帮我找技术合伙人",
            run_match=False, run_invite=False,
        ),
        state,
    )
    assert agent.calls == 1
    assert len(paths) == 1
    assert "原始召回为0条" in str(agent.inputs[0]["messages"])
    assert response.reply.startswith("本次公开检索已完成，但没有返回")
    assert "稍等" not in response.reply
    assert response.match_status == "empty"


async def test_chat_enforce_stops_before_model_when_public_query_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from apps.group_agent_api.app.endpoints import chat as chat_endpoint

    agent = _RetryAgent(complete_on_retry=False)
    state = AppState(
        agent=agent,
        base_dir=tmp_path,
        search_only_agent=agent,
        write_only_agent=agent,
        no_tool_agent=agent,
        quality_model=planner_for_message("帮我找技术合伙人"),
    )
    _install_route_skill(tmp_path)

    async def resolve(*_: Any, **__: Any) -> TrustedSession:
        return _session()

    monkeypatch.setenv("GROUP_AGENT_INTENT_ROUTING_MODE", "enforce")
    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
    from skillpacks.aihehuo import integration

    monkeypatch.setattr(
        integration,
        "fetch",
        lambda _path, **_kwargs: (_ for _ in ()).throw(OSError("network failed")),
    )
    monkeypatch.setattr(chat_endpoint, "resolve_trusted_session", resolve)
    with pytest.raises(HTTPException) as raised:
        await chat_endpoint.chat(
            ChatRequest(
                user_id="user-1",
                group_id="group-1",
                message="帮我找技术合伙人",
                run_match=False,
                run_invite=False,
            ),
            state,
        )
    assert raised.value.status_code == 502
    assert raised.value.detail["error_message"] == "tool_failed"
    assert agent.calls == 0


async def test_async_enforce_executes_required_tool_before_composing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from apps.group_agent_api.app import async_manager

    search_agent = _RetryAgent(complete_on_retry=True)
    compose_agent = _RetryAgent(complete_on_retry=True)
    state = AppState(
        agent=search_agent,
        base_dir=tmp_path,
        search_only_agent=search_agent,
        write_only_agent=compose_agent,
        no_tool_agent=compose_agent,
        quality_model=planner_for_message("帮我找技术合伙人"),
    )
    _install_route_skill(tmp_path)
    events: list[tuple[str, dict[str, Any]]] = []

    async def emit(event: str, payload: dict[str, Any]) -> bool:
        events.append((event, payload))
        return True

    monkeypatch.setenv("GROUP_AGENT_INTENT_ROUTING_MODE", "enforce")
    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
    monkeypatch.setenv("GROUP_AGENT_GROUNDED_FINAL_ENABLED", "1")
    from skillpacks.aihehuo import integration

    monkeypatch.setattr(
        integration,
        "fetch",
        lambda _path, **_kwargs: '{"data":[{"id":101,"user_id":101,"number":9659712,"name":"张三","bio":"技术合伙人"}],"meta":{"has_more":false}}',
    )
    monkeypatch.setenv(
        "GROUP_AGENT_CALLBACK_ALLOWED_BASE_URLS",
        "http://example.invalid/group_agent_callbacks",
    )
    await async_manager._execute_core_agent(
        req=AsyncCallRequest(
            run_id="intent-enforce-async",
            idempotency_key="intent-enforce-async-key",
            user_id="user-1",
            group_id="group-1",
            conversation_id="conversation-1",
            message="帮我找技术合伙人",
            callback_url="http://example.invalid/group_agent_callbacks/intent-enforce-async",
            rollout_context={"protocol_mode": "grounded_v2"},
            run_match=False,
            run_invite=False,
        ),
        session=_session(),
        state=state,
        tid="intent-enforce-async-thread",
        emit_callback=emit,
    )
    assert search_agent.calls == 0
    assert compose_agent.calls == 1
    event_names = [event for event, _payload in events]
    assert event_names.index("progress") < event_names.index("final")
    progress_phases = [
        payload["phase"] for event, payload in events if event == "progress"
    ]
    assert progress_phases.index("tool_started") < progress_phases.index(
        "tool_completed"
    )
    assert events[-1][0] == "final"
    assert events[-1][1]["reply"] == "公开查询处理完成。"
    assert events[-1][1]["reply_mode"] == "recommendation"
    assert events[-1][1]["match_reason"] == "intent_query_plan"
    assert [candidate["user_id"] for candidate in events[-1][1]["candidates"]] == ["101"]
    assert events[-1][1]["grounded_final"]["candidate_count"] == 1


async def test_async_empty_receipt_cannot_promise_future_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from apps.group_agent_api.app import async_manager
    from skillpacks.aihehuo import integration

    agent = _RetryAgent(
        complete_on_retry=False,
        reply="我马上查公开人才库，稍等给你名单。",
    )
    state = AppState(
        agent=agent,
        base_dir=tmp_path,
        search_only_agent=agent,
        write_only_agent=agent,
        no_tool_agent=agent,
        quality_model=planner_for_message("帮我找技术合伙人"),
    )
    _install_route_skill(tmp_path)
    events: list[tuple[str, dict[str, Any]]] = []

    async def emit(event: str, payload: dict[str, Any]) -> bool:
        events.append((event, payload))
        return True

    monkeypatch.setenv("GROUP_AGENT_INTENT_ROUTING_MODE", "enforce")
    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
    monkeypatch.setenv("GROUP_AGENT_GROUNDED_FINAL_ENABLED", "1")
    monkeypatch.setenv(
        "GROUP_AGENT_CALLBACK_ALLOWED_BASE_URLS",
        "http://example.invalid/group_agent_callbacks",
    )
    paths: list[str] = []

    def fetch(path: str, **_kwargs: Any) -> str:
        paths.append(path)
        return '{"data":[],"meta":{"has_more":false}}'

    monkeypatch.setattr(integration, "fetch", fetch)
    await async_manager._execute_core_agent(
        req=AsyncCallRequest(
            run_id="intent-empty-async", idempotency_key="intent-empty-async-key",
            user_id="user-1", group_id="group-1", conversation_id="conversation-1",
            message="帮我找技术合伙人",
            callback_url="http://example.invalid/group_agent_callbacks/intent-empty-async",
            rollout_context={"protocol_mode": "grounded_v2"},
            run_match=False, run_invite=False,
        ),
        session=_session(), state=state, tid="intent-empty-async-thread",
        emit_callback=emit,
    )
    assert agent.calls == 1
    assert len(paths) == 1
    final = events[-1][1]
    assert final["reply"].startswith("本次公开检索已完成，但没有返回")
    assert "稍等" not in final["reply"]
    assert final["reply_mode"] == "no_match"
    assert final["grounded_final"]["reply"] == final["reply"]


async def test_async_consecutive_turns_restore_server_query_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from apps.group_agent_api.app import async_manager
    from skillpacks.aihehuo import integration

    history: list[Any] = []
    search_agent = _PersistentAgent(history)
    compose_agent = _PersistentAgent(history)
    state = AppState(
        agent=search_agent,
        base_dir=tmp_path,
        search_only_agent=search_agent,
        write_only_agent=compose_agent,
        no_tool_agent=compose_agent,
        quality_model=AdaptivePlannerModel(),
    )
    _install_route_skill(tmp_path)
    paths: list[str] = []

    def fetch(path: str, **_kwargs: object) -> str:
        paths.append(path)
        return '{"data":[],"meta":{"has_more":false}}'

    async def emit(_event: str, _payload: dict[str, Any]) -> bool:
        return True

    monkeypatch.setenv("GROUP_AGENT_INTENT_ROUTING_MODE", "enforce")
    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
    monkeypatch.setenv(
        "GROUP_AGENT_CALLBACK_ALLOWED_BASE_URLS",
        "http://example.invalid/callbacks",
    )
    monkeypatch.setattr(integration, "fetch", fetch)

    async def execute(run_id: str, message: str) -> None:
        await async_manager._execute_core_agent(
            req=AsyncCallRequest(
                run_id=run_id,
                idempotency_key=f"{run_id}-key",
                user_id="user-1",
                group_id="group-1",
                conversation_id="conversation-1",
                message=message,
                callback_url=f"http://example.invalid/callbacks/{run_id}",
                run_match=False,
                run_invite=False,
            ),
            session=_session(),
            state=state,
            tid="shared-intent-thread",
            emit_callback=emit,
        )

    await execute("context-run-1", "找上海餐饮创业者，优先有供应链经验")
    await execute("context-run-2", "北京的呢")
    await execute("context-run-3", "不要北京，改成上海的")

    assert len(paths) == 3
    queries = [parse_qs(urlparse(path).query)["q"][0] for path in paths]
    assert all("/ai/users.json" in path for path in paths)
    assert "北京" in queries[1]
    assert "餐饮" in queries[1]
    assert "供应链经验" in queries[1]
    assert "上海" in queries[2]
    assert "餐饮" in queries[2]
    assert "供应链经验" in queries[2]
    assert "北京" not in queries[2]


async def test_server_executor_validates_skill_before_public_fetch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from apps.group_agent_api.agent_factory.intent_routing.plan_executor import (
        execute_query_plan,
    )
    from skillpacks.aihehuo import integration

    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
    _install_route_skill(tmp_path)
    skill_file = (
        tmp_path / "skills" / "aihehuo" / "get-ai-cofounders" / "SKILL.md"
    )
    skill_file.write_text("", encoding="utf-8")
    fetches: list[str] = []
    monkeypatch.setattr(
        integration,
        "fetch",
        lambda path, **_kwargs: fetches.append(path) or '{"data":[]}',
    )

    with pytest.raises(QueryPlanError, match="query_plan_invalid"):
        await execute_query_plan(
            _execution(), skills_dir=tmp_path / "skills" / "aihehuo"
        )
    assert fetches == []
