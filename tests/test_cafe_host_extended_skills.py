"""Cafe Host access coverage for the second paid-skill rollout."""

from __future__ import annotations

import pytest

from apps.group_agent_api.app.cafe_host_search import canonical_query_plan
from apps.group_agent_api.app.chat_access import (
    ChatAccessContext,
    ChatAccessRejected,
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
