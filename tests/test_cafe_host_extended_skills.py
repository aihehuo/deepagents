"""Cafe Host access coverage for the second paid-skill rollout."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from apps.group_agent_api.agent_factory.intent_routing.resolver import resolve_intent
from apps.group_agent_api.agent_factory.intent_routing.blog_articles import weekly_articles
from apps.group_agent_api.agent_factory.intent_routing.execution import IntentExecution
from apps.group_agent_api.agent_factory.intent_routing.projection import (
    IntentSearchProjection,
)
from apps.group_agent_api.app.cafe_host_search import (
    build_pending_search_offer,
    build_search_projection,
    canonical_query_plan,
    grounded_digest_reply,
    grounded_platform_help_reply,
    soft_location_hint,
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
    reply,
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


def _location_hint_execution(
    message: str,
    query: str,
    *,
    hard_locations: tuple[str, ...] = (),
) -> SimpleNamespace:
    return SimpleNamespace(
        original_message=message,
        decision=SimpleNamespace(
            primary_intent="people_search",
            constraints=SimpleNamespace(location=hard_locations),
            resource_queries=SimpleNamespace(users=query),
        ),
    )


def test_soft_people_location_hint_only_for_unfiltered_single_target_city() -> None:
    hint = soft_location_hint(
        _location_hint_execution("帮我找上海的餐饮合伙人", "上海 餐饮 合伙人")
    )
    assert "结果可能包含其他城市" in hint
    assert "只找上海的候选人" in hint
    assert not soft_location_hint(
        _location_hint_execution(
            "只找上海的餐饮合伙人", "上海 餐饮 合伙人", hard_locations=("上海",)
        )
    )
    assert not soft_location_hint(
        _location_hint_execution("找上海或深圳的合伙人", "上海 深圳 合伙人")
    )
    assert not soft_location_hint(
        _location_hint_execution("我在上海，想找餐饮合伙人", "餐饮 合伙人")
    )
    assert not soft_location_hint(
        _location_hint_execution("不要上海的候选人", "上海 候选人")
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


def test_cafe_real_name_verification_routes_to_free_faq_even_if_planner_says_chat() -> None:
    with active_chat_access(_access("ordinary")):
        result = resolve_intent(
            "爱合伙平台如何完成实名认证",
            execution_mode="cafe_host",
            can_search=True,
            planner_model=planner_returning(reply()),
        )
    assert result.decision.action == "execute"
    assert result.decision.primary_intent == "platform_help"
    assert result.decision.resource_queries.faqs == "实名认证"

    # A legacy/unfrozen Cafe run must not gain a free read just by naming it.
    legacy = resolve_intent(
        "爱合伙平台如何完成实名认证",
        execution_mode="cafe_host",
        can_search=False,
        planner_model=planner_returning(reply()),
    )
    assert legacy.decision.action == "reject"


def test_general_advice_and_explicit_no_search_stay_in_conversation() -> None:
    with active_chat_access(_access("ordinary")):
        for message in (
            "我想入局餐饮行业，应该怎么做品牌定位？",
            "不要查询，直接聊聊爱合伙实名认证为什么重要",
        ):
            result = resolve_intent(
                message,
                execution_mode="cafe_host",
                can_search=True,
                planner_model=planner_returning(reply()),
            )
            assert result.decision.action == "conversation"


def test_real_name_faq_reply_uses_only_verified_current_receipt() -> None:
    execution = SimpleNamespace(
        receipts=[SimpleNamespace(resource="faqs", status="succeeded", receipt_id="faq-1")],
        evidence_by_receipt={"faq-1": [
            {"question": "如何完成实名认证", "answer": "在 App 的认证页面提交资料并等待审核。"}
        ]},
    )
    answer = grounded_platform_help_reply(execution)
    assert "如何完成实名认证" in answer
    assert "在 App 的认证页面提交资料并等待审核" in answer
    assert "没有平台" not in answer

    execution.evidence_by_receipt = {}
    unavailable = grounded_platform_help_reply(execution)
    assert "没有返回足以核实具体操作的答案" in unavailable
    assert "没有平台" not in unavailable


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


@pytest.mark.parametrize("draft", (community_query("爱合伙 社群 新闻"), proposal(intent="conversation")))
def test_news_about_the_community_routes_to_startup_digest(
    monkeypatch: pytest.MonkeyPatch, draft: dict[str, object]
) -> None:
    monkeypatch.setenv("CAFE_HOST_V4_SEARCH_EXECUTION_ENABLED", "1")
    monkeypatch.setenv("CAFE_HOST_V4_SEARCH_USER_ALLOWLIST", "1")
    with active_chat_access(_access("active")):
        result = resolve_intent(
            "最近爱合伙社群有什么新闻",
            execution_mode="cafe_host",
            can_search=False,
            planner_model=planner_returning(draft),
        )
    assert result.decision.action == "execute"
    assert result.decision.primary_intent == "startup_digest"
    assert result.decision.resource_queries.model_dump(exclude_none=True) == {}


def test_platform_news_uses_verified_weekly_articles_not_model_denial() -> None:
    """A completed blog read must answer with actual recent posts."""
    receipt = SimpleNamespace(
        resource="weekly", status="succeeded", receipt_id="weekly-1"
    )
    execution = SimpleNamespace(
        receipts=[receipt],
        evidence_by_receipt={
            "weekly-1": [
                {
                    "source_url": "https://www.aihehuo.com/ai/blog/weekly_markdown",
                    "content": (
                        "# 爱合伙博客 · 近 7 日文章汇总\n\n"
                        "## 1. 每日AI创业快讯 — 2026年9月30日\n\n"
                        "- **链接**：https://www.aihehuo.com/blog/9437-news-2026-09-30\n"
                        "- **摘要**：今日新闻。\n\n"
                        "## 2. 社群新动态\n\n"
                        "- **链接**：https://www.aihehuo.com/blog/9438-community\n"
                    ),
                }
            ]
        },
    )
    reply = grounded_digest_reply(execution)
    assert "爱合伙博客近 7 天有更新" in reply
    assert "每日AI创业快讯 — 2026年9月30日" in reply
    assert "社群新动态" in reply
    assert "博客杯垫" in reply
    assert "https://www.aihehuo.com/blog/9437-news-2026-09-30" not in reply
    assert "没有平台" not in reply


def test_weekly_markdown_projects_real_articles_into_search_result() -> None:
    markdown = (
        "# 爱合伙博客 · 近 7 日文章汇总\n\n"
        "## 1. 每日AI创业快讯 — 2026年9月30日\n\n"
        "- **链接**：https://www.aihehuo.com/blog/9437-news-2026-09-30\n"
        "- **摘要**：今日 AI 新闻。\n\n"
        "## 2. 项目优选\n\n"
        "- **链接**：https://www.aihehuo.com/blog/9436-projects\n"
        "- **摘要**：可合作项目。\n\n"
        "## 3. 外部伪造链接\n\n"
        "- **链接**：https://example.com/blog/evil\n"
    )
    articles = weekly_articles(markdown)
    assert [article["id"] for article in articles] == ["9437-news-2026-09-30", "9436-projects"]
    assert articles[0]["summary"] == "今日 AI 新闻。"

    execution = SimpleNamespace(
        evidence_by_receipt={"weekly-1": [{"content": markdown}]},
        _search_log_payload=lambda _reservation: {"search_log": {
            "search_id": "run:weekly-primary:1", "query": "（无关键词）",
            "resource": "weekly", "plan_step_id": "weekly-primary",
            "request_index": 1, "page": None,
        }},
    )
    reservation = SimpleNamespace(step=SimpleNamespace(resource="weekly"))
    receipt = SimpleNamespace(receipt_id="weekly-1")
    result = IntentExecution._search_result_payload(execution, reservation, receipt)["search_result"]
    assert result["result_count"] == 2
    assert result["items"][0] == {
        "id": "9437-news-2026-09-30",
        "url": "https://www.aihehuo.com/blog/9437-news-2026-09-30",
        "name": "每日AI创业快讯 — 2026年9月30日",
        "city": "", "industry": "", "role": "", "summary": "今日 AI 新闻。",
        "review_status": "not_applicable", "review_reasons": [],
    }


def test_weekly_read_without_extractable_posts_does_not_claim_no_news() -> None:
    execution = SimpleNamespace(
        receipts=[SimpleNamespace(resource="weekly", status="succeeded", receipt_id="weekly-1")],
        evidence_by_receipt={"weekly-1": [{"content": "# 爱合伙博客 · 近 7 日文章汇总"}]},
    )
    reply = grounded_digest_reply(execution)
    assert "没能提取出可展示的文章" in reply
    assert "不能据此判断没有新闻" in reply


def test_community_projection_keeps_opaque_group_ids_and_public_urls() -> None:
    receipt = SimpleNamespace(
        resource="groups",
        status="succeeded",
        receipt_id="receipt-1",
        source_urls=("https://www.aihehuo.com/ai/groups?q=%E7%88%B1%E5%90%88%E4%BC%99",),
    )
    execution = SimpleNamespace(
        decision=SimpleNamespace(action="execute", primary_intent="community_search"),
        receipts=[receipt],
        evidence_by_receipt={
            "receipt-1": [
                {
                    "id": "e763",
                    "name": "上海地域群",
                    "summary": "上海创业者交流社群",
                    "url": "https://www.aihehuo.com/groups/e763",
                }
            ]
        },
    )
    projection = build_search_projection(
        execution,
        IntentSearchProjection(completed=True, result_count=1),
    )
    assert projection is not None
    assert projection["bar_entries"] == [
        {
            "kind": "groups",
            "id": "e763",
            "title": "上海地域群",
            "summary": "上海创业者交流社群",
            "source_url": "https://www.aihehuo.com/groups/e763",
        }
    ]
