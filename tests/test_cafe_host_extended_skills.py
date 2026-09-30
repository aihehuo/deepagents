"""Cafe Host access coverage for the second paid-skill rollout."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from apps.group_agent_api.agent_factory.intent_routing.resolver import resolve_intent
from apps.group_agent_api.app.cafe_host_search import (
    build_pending_search_offer,
    canonical_query_plan,
)
from apps.group_agent_api.app.chat_access import (
    ChatAccessContext,
    ChatAccessRejected,
    active_chat_access,
)
from tests.unit_tests.group_agent_api.planner_fixtures import (
    community_query,
    planner_returning,
    proposal,
)


def _access(mode: str) -> ChatAccessContext:
    paid = (
        "community_search",
        "opportunity_search",
        "people_search",
        "project_partnership_search",
        "startup_digest",
    )
    return ChatAccessContext(
        mode=mode,  # type: ignore[arg-type]
        policy_version="chat-access-v2",
        revision=1,
        drink_session_id=1 if mode == "active" else None,
        pending_search_ref=None,
        search_confirmation_id=None,
        confirmed_search_intent=None,
        allowed_search_intents=paid if mode == "active" else (),
        restricted_search_intents=() if mode == "active" else paid,
        user_token_sha256="-",
        group_token_sha256="-",
        user_id="1",
    )


def test_new_paid_plans_use_existing_registry_shapes() -> None:
    community = canonical_query_plan(
        "community_search", {"groups": "上海 人工智能"}
    )
    assert community["plan_id"] == "community-search-v1"
    assert community["primary_resource"] == "groups"
    assert community["queries"] == {"groups": "上海 人工智能"}

    digest = canonical_query_plan("startup_digest", {})
    assert digest["plan_id"] == "startup-digest-v1"
    assert digest["primary_resource"] == "weekly"
    assert digest["queries"] == {}


@pytest.mark.parametrize(
    ("intent", "resource"),
    (("community_search", "groups"), ("startup_digest", "weekly")),
)
def test_new_paid_skills_require_an_active_drink(intent: str, resource: str) -> None:
    access = _access("ordinary")
    assert access.control_for_intent(intent) == "chat_session_required"
    with pytest.raises(ChatAccessRejected, match="chat_session_required"):
        access.assert_resource_allowed(intent=intent, resource=resource)


def test_platform_help_is_free_in_ordinary_mode() -> None:
    access = _access("ordinary")
    assert access.control_for_intent("platform_help") is None
    access.assert_resource_allowed(intent="platform_help", resource="faqs")
    access.guard_final(
        reply="本次平台 FAQ 查询已经完成。",
        receipt_intents=("platform_help",),
    )


@pytest.mark.parametrize(
    ("message", "draft", "intent"),
    (
        ("查找上海创业社群", community_query("上海 创业社群"), "community_search"),
        ("看看最新创业简报", proposal(intent="startup_digest"), "startup_digest"),
    ),
)
def test_active_drink_authorizes_new_paid_planner_decisions(
    monkeypatch: pytest.MonkeyPatch,
    message: str,
    draft: dict[str, object],
    intent: str,
) -> None:
    monkeypatch.setenv("CAFE_HOST_V4_SEARCH_EXECUTION_ENABLED", "1")
    monkeypatch.setenv("CAFE_HOST_V4_SEARCH_USER_ALLOWLIST", "1")
    with active_chat_access(_access("active")):
        result = resolve_intent(
            message,
            execution_mode="cafe_host",
            can_search=False,
            planner_model=planner_returning(draft),
        )
    assert result.decision.primary_intent == intent
    assert result.decision.action == "execute"


def test_ordinary_community_offer_preserves_group_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAFE_HOST_PENDING_SEARCH_OFFER_ENABLED", "1")
    draft = SimpleNamespace(
        action="query",
        intent="community_search",
        queries=SimpleNamespace(
            users=None,
            ideas=None,
            projects=None,
            groups="上海 创业社群",
            faqs=None,
        ),
    )
    offer = build_pending_search_offer(
        proposal=draft,
        decision_action="offer",
        confidence="high",
        planner_status="ok",
        run_id="run-1",
        message="查找上海创业社群",
        conversation_id="conversation-1",
        episode_id="episode-1",
    )
    assert offer is not None
    assert offer["intent"] == "community_search"
    assert offer["query_plan"]["queries"] == {"groups": "上海 创业社群"}
