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
    """_has_partnership_language must allow '爱合伙' brand name while forbidding pushy partnership terms."""
    # Positive case: Brand name used in doing/project name
    assert not _has_partnership_language("在爱合伙推进爱合伙群智能体项目")
    assert not _has_partnership_language("爱合伙平台支持与资源")

    # Allowed negation with brand name
    assert not _has_partnership_language("爱合伙平台不谈合伙，先聊交流")

    # Negative case: Pushy partnership pitch
    assert _has_partnership_language("很适合你当合伙人")
    assert _has_partnership_language("在爱合伙寻找合伙人")
    assert _has_partnership_language("给股份，一起创业搭班子")
    assert _has_partnership_language("爱合伙项目招合伙人")


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


def test_assert_directed_invite_rejects_real_partnership_pitch_even_with_brand() -> None:
    """assert_directed_invite must still reject genuine partnership pitches even if '爱合伙' is present."""
    cands = _candidates()
    elements = {
        "who_doing": "在爱合伙推进项目",
        "resources": "AI Agent",
        "topic": "Agent 架构",
        "why_invite": "根据公开信息，值得聊",
        "low_pressure": "当合伙人一起创业",  # Contains forbidden phrase
    }
    text = "在爱合伙推进项目，包含 AI Agent 架构，根据公开信息，值得聊。当合伙人一起创业"
    violations = assert_directed_invite(
        text=text,
        elements=elements,
        candidates=cands,
    )
    assert "partnership_language" in violations
