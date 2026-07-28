"""REQ-024 test: verify brand name '爱合伙' does not false-positive trigger partnership_language ban."""

from __future__ import annotations

from apps.group_agent_api.agent_factory.invite_copy import (
    _has_partnership_language,
    assert_directed_invite,
    generate_invite_copy,
)
from apps.group_agent_api.agent_factory.profile_schema import profile_from_flat


def _brand_profile():
    return profile_from_flat(
        user_id="mock_u1",
        group_id="mock_g1",
        doing="在爱合伙推进爱合伙群智能体研发与落地",
        need="技术架构与 AI Agent 经验伙伴",
        offer="爱合伙平台资源与测试环境",
    )


def _candidates():
    return [
        {
            "user_id": "mock_u2",
            "name": "李四",
            "doing": {"value": "AI Agent 开发", "disclosure": "confirmed_public"},
            "need": {"value": "产品落地场景", "disclosure": "confirmed_public"},
            "offer": {"value": "Python/FastAPI 架构", "disclosure": "confirmed_public"},
        }
    ]


def test_brand_name_aihehuo_not_flagged_as_partnership_language() -> None:
    """REQ-026: _has_partnership_language always returns False per boss decision."""
    assert not _has_partnership_language("在爱合伙推进爱合伙群智能体项目")
    assert not _has_partnership_language("爱合伙平台支持与资源")
    assert not _has_partnership_language("爱合伙平台不谈合伙，先聊交流")
    assert not _has_partnership_language("很适合你当合伙人")
    assert not _has_partnership_language("在爱合伙寻找合伙人")
    assert not _has_partnership_language("给股份，一起创业搭班子")
    assert not _has_partnership_language("爱合伙项目招合伙人")


def test_generate_invite_copy_with_brand_profile_succeeds() -> None:
    """generate_invite_copy must succeed (ok=True, text non-empty) for profiles containing '爱合伙'."""
    profile = _brand_profile()
    cands = _candidates()

    result = generate_invite_copy(
        profile=profile,
        candidates=cands,
        match_status="matched",
        willing_to_at=True,
    )

    assert result.kind == "directed"
    assert result.ok, f"expected ok=True, got violations: {result.violations}"
    assert result.text.strip()
    assert "partnership_language" not in result.violations
    assert "爱合伙" in result.text or "爱合伙" in result.elements.get("who_doing", "")


def test_assert_directed_invite_allows_partnership_pitch() -> None:
    """REQ-026: assert_directed_invite must allow partnership pitches without returning partnership_language."""
    cands = _candidates()
    elements = {
        "who_doing": "在爱合伙推进项目",
        "resources": "AI Agent",
        "topic": "Agent 架构",
        "why_invite": "@mock_u2，想请教几位一起对齐一下——不一定对得上，供参考，聊聊看就好",
        "low_pressure": "当合伙人一起创业",
    }
    text = "\n".join(elements.values())
    violations = assert_directed_invite(
        text=text,
        elements=elements,
        candidates=cands,
    )
    assert "partnership_language" not in violations
    assert not violations
