"""INTENT-DA-002 QueryPlan, snapshot, and receipt tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException
from langchain_core.messages import AIMessage

from apps.group_agent_api.agent_factory.intent_routing.canonical import (
    RoutingCryptoError,
    canonical_hash,
    canonical_json_bytes,
    decrypt_canonical,
)
from apps.group_agent_api.agent_factory.intent_routing.execution import (
    QueryPlanError,
    active_intent_execution,
    current_intent_execution,
    prepare_intent_execution,
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


def _decision() -> IntentDecisionV1:
    return IntentDecisionV1.model_validate(
        {
            "version": 1,
            "primary_intent": "people_search",
            "confidence": "high",
            "decision_source": "rules",
            "evidence_spans": ["有没有", "上海", "餐饮合伙人"],
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
    )


def test_frozen_routing_hash_aad_and_aead_vector() -> None:
    snapshot = build_routing_snapshot(
        run_id="run-intent-0001",
        decision=_decision(),
        created_at="2026-09-15T08:00:00Z",
    )
    assert snapshot.canonical_content_hash == (
        "sha256:e32d221269b5442a19c79ad568f592364ee8bb8f319b02fe1678b5399c7d1dc3"
    )
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


def _execution(message: str = "有没有上海餐饮合伙人"):
    resolution = resolve_intent(
        message,
        execution_mode="group_agent",
        can_search=True,
    )
    return prepare_intent_execution(
        resolution,
        original_message=message,
        run_id=None,
        request_fingerprint=None,
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


def test_profile_update_delegates_without_query_plan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
    resolution = resolve_intent(
        "把我的所在地改成上海",
        execution_mode="group_agent",
        can_search=True,
    )
    execution = prepare_intent_execution(
        resolution,
        original_message="把我的所在地改成上海",
        run_id=None,
        request_fingerprint=None,
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
    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "legacy")
    resolution = resolve_intent(
        "帮我找合伙人", execution_mode="group_agent", can_search=True
    )
    with pytest.raises(QueryPlanError, match="query_plan_invalid"):
        prepare_intent_execution(
            resolution,
            original_message="帮我找合伙人",
            run_id=None,
            request_fingerprint=None,
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
    execution.bind_progress_emitter(emitter, started_monotonic=0.0)
    tool = integration.create_aihehuo_tool(skills_dir=skills)
    with active_intent_execution(execution):
        result = await tool.ainvoke({"resource": "users", "page": 1})

    assert result == "AIHEHUO query rejected: callback_ack_invalid"
    assert fetches == []


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
    def __init__(self, *, complete_on_retry: bool) -> None:
        self.complete_on_retry = complete_on_retry
        self.calls = 0

    async def aget_state(self, _: Any) -> Any:
        return SimpleNamespace(values={"messages": []})

    async def ainvoke(self, _: Any, __: Any) -> dict[str, Any]:
        self.calls += 1
        execution = current_intent_execution()
        assert execution is not None
        if self.calls == 2 and self.complete_on_retry:
            reservation = await execution.reserve_async(
                resource="users", parameters={}, tool_call_id="retry-tool"
            )
            await execution.commit_result_async(reservation, _result([{"id": "u-1"}]))
        return {"messages": [AIMessage(content="公开查询处理完成。")]}


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


async def test_chat_enforce_retries_required_tool_exactly_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from apps.group_agent_api.app.endpoints import chat as chat_endpoint

    agent = _RetryAgent(complete_on_retry=True)
    state = AppState(
        agent=agent,
        base_dir=tmp_path,
        search_only_agent=agent,
        write_only_agent=agent,
        no_tool_agent=agent,
    )
    _install_route_skill(tmp_path)

    async def resolve(*_: Any, **__: Any) -> TrustedSession:
        return _session()

    monkeypatch.setenv("GROUP_AGENT_INTENT_ROUTING_MODE", "enforce")
    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
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
    assert "查询范围：本次查询覆盖 1 页" in response.reply
    assert agent.calls == 2


async def test_chat_enforce_fails_after_one_controlled_retry(
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
    )
    _install_route_skill(tmp_path)

    async def resolve(*_: Any, **__: Any) -> TrustedSession:
        return _session()

    monkeypatch.setenv("GROUP_AGENT_INTENT_ROUTING_MODE", "enforce")
    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
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
    assert raised.value.detail["error_message"] == "required_tool_not_called"
    assert agent.calls == 2


async def test_async_enforce_retries_before_final(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from apps.group_agent_api.app import async_manager

    agent = _RetryAgent(complete_on_retry=True)
    state = AppState(
        agent=agent,
        base_dir=tmp_path,
        search_only_agent=agent,
        write_only_agent=agent,
        no_tool_agent=agent,
    )
    _install_route_skill(tmp_path)
    events: list[tuple[str, dict[str, Any]]] = []

    async def emit(event: str, payload: dict[str, Any]) -> bool:
        events.append((event, payload))
        return True

    monkeypatch.setenv("GROUP_AGENT_INTENT_ROUTING_MODE", "enforce")
    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
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
            run_match=False,
            run_invite=False,
        ),
        session=_session(),
        state=state,
        tid="intent-enforce-async-thread",
        emit_callback=emit,
    )
    assert agent.calls == 2
    assert events[-1][0] == "final"
    assert "查询范围：本次查询覆盖 1 页" in events[-1][1]["reply"]
