"""REQ-026 test: verify invite generation & assertions allow partnership intent, terms, and candidate doing labels."""

from __future__ import annotations

from apps.group_agent_api.agent_factory.invite_copy import (
    _has_partnership_language,
    assert_directed_invite,
    generate_invite_copy,
)
from apps.group_agent_api.agent_factory.profile_schema import profile_from_flat


def _req026_profile():
    return profile_from_flat(
        user_id="user_1",
        group_id="group_763",
        doing="推进爱合伙群智能体产品",
        need="连接熟悉社群运营与 AI 智能体落地的伙伴",
        offer="可提供产品设计和技术协作",
    )


def test_candidate_doing_with_partnership_demand_label_succeeds() -> None:
    """Candidate doing containing '合伙需求:' must generate valid non-empty invite with ok=True."""
    profile = _req026_profile()
    candidates = [
        {
            "user_id": "cand_394760",
            "display_name": "正心正念",
            "doing": {
                "value": "合伙需求: 熟悉产业互联网，能够对接产业链上下游相关的资源",
                "disclosure": "confirmed_public",
            },
            "need": {"value": "资源对接", "disclosure": "confirmed_public"},
            "offer": {"value": "全栈开发", "disclosure": "confirmed_public"},
        }
    ]

    res = generate_invite_copy(
        profile=profile,
        candidates=candidates,
        match_status="matched",
        willing_to_at=True,
    )

    assert res.kind == "directed"
    assert res.ok, f"Expected ok=True, got violations: {res.violations}"
    assert res.text.strip()
    assert "partnership_language" not in res.violations
    assert "cand_394760" in res.mentioned_user_ids


def test_explicit_partnership_intent_and_equity_terms_allowed() -> None:
    """Explicit terms like '找技术合伙人', '股份', '创业搭班子' must not trigger partnership_language."""
    assert not _has_partnership_language("我想找一位技术合伙人一起推进创业项目")
    assert not _has_partnership_language("早期出让部分股份，一起创业搭班子")

    cands = [
        {
            "user_id": "u123",
            "display_name": "张三",
            "doing": {"value": "后端开发", "disclosure": "confirmed_public"},
        }
    ]
    elements = {
        "who_doing": "我在寻找技术合伙人",
        "resources": "有前期启动资金与股份",
        "topic": "技术架构与合伙机制",
        "why_invite": "@u123 你公开资料里提到「后端开发」，基于公开信息值得聊一次以确认是否对得上——不一定合适",
        "low_pressure": "聊聊就好，不耽误大家太多时间，有合伙意向也可顺便交流",
    }
    text = "\n".join(elements.values())
    violations = assert_directed_invite(text=text, elements=elements, candidates=cands)
    assert "partnership_language" not in violations
    assert not violations


def test_aihehuo_brand_profile_remains_ok() -> None:
    """Profile doing with '爱合伙' brand name continues to pass cleanly."""
    profile = _req026_profile()
    candidates = [
        {
            "user_id": "cand_99",
            "display_name": "李四",
            "doing": {"value": "在爱合伙做社区运营", "disclosure": "confirmed_public"},
        }
    ]

    res = generate_invite_copy(
        profile=profile,
        candidates=candidates,
        match_status="matched",
        willing_to_at=True,
    )

    assert res.ok
    assert res.text.strip()
    assert "partnership_language" not in res.violations


def test_other_assertions_still_enforced() -> None:
    """Non-partnership guards (e.g. unmentioned @ or missing uncertainty) are still enforced."""
    cands = [
        {
            "user_id": "u1",
            "display_name": "王五",
            "doing": {"value": "前端", "disclosure": "confirmed_public"},
        }
    ]
    elements = {
        "who_doing": "我在做项目",
        "resources": "有经验",
        "topic": "前端技术",
        "why_invite": "@foreign_user 结论：你非常合适当合伙人",  # Missing uncertainty & unapproved @
        "low_pressure": "聊聊",
    }
    text = "\n".join(elements.values())
    violations = assert_directed_invite(text=text, elements=elements, candidates=cands)
    assert "partnership_language" not in violations
    assert "missing_uncertainty" in violations
    assert any(v.startswith("at_not_in_candidates") for v in violations)
