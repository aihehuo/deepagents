"""REQ-040 automated tests: 4D state parsing and single-candidate Prompt copy generation engine."""

from __future__ import annotations

import re
from typing import Any

import pytest

from apps.group_agent_api.agent_factory.disclosure import filter_member_for_visibility
from apps.group_agent_api.agent_factory.integrations.match_client import _normalize_candidate
from apps.group_agent_api.agent_factory.invite_copy import (
    assert_directed_invite,
    generate_invite_copy,
)
from apps.group_agent_api.agent_factory.match_stub import MatchStub, PoolMember
from apps.group_agent_api.agent_factory.per_candidate_copy import (
    enrich_candidate_with_single_copy,
    enrich_candidates_with_single_copy,
    generate_single_candidate_copy,
)
from apps.group_agent_api.agent_factory.profile_schema import profile_from_flat


@pytest.fixture
def sample_profile():
    return profile_from_flat(
        user_id="u105",
        group_id="group_l1_alpha",
        doing="做 AI 智能体应用开发与落地",
        need="寻找精通 Python 与 LLM 架构的技术合伙人",
        offer="具备极强产品规划与 BD 经验",
        doing_disclosure="confirmed_public",
        need_disclosure="confirmed_public",
        offer_disclosure="confirmed_public",
    )


def test_match_client_parses_4d_state_and_is_masked():
    """Requirement 1: MatchClient parses same_group, wechat_reachable, app_registered, has_talked_with_agent, is_masked."""
    raw_item = {
        "user_id": "u101",
        "name": "张三",
        "doing": {"value": "智能硬件开发", "disclosure": "confirmed_public"},
        "same_group": True,
        "wechat_reachable": True,
        "app_registered": True,
        "has_talked_with_agent": True,
        "is_masked": False,
        "match_score": 0.88,
    }
    normalized = _normalize_candidate(raw_item, fallback_group="group_alpha")

    assert normalized["user_id"] == "u101"
    assert normalized["same_group"] is True
    assert normalized["wechat_reachable"] is True
    assert normalized["app_registered"] is True
    assert normalized["has_talked_with_agent"] is True
    assert normalized["is_masked"] is False

    raw_masked = {
        "user_id": "u202",
        "name": "李*",
        "doing": {"value": "芯片设计", "disclosure": "confirmed_public"},
        "same_group": False,
        "wechat_reachable": False,
        "app_registered": True,
        "has_talked_with_agent": False,
        "is_masked": True,
        "match_score": 0.72,
    }
    normalized_masked = _normalize_candidate(raw_masked, fallback_group="group_alpha")

    assert normalized_masked["user_id"] == "u202"
    assert normalized_masked["same_group"] is False
    assert normalized_masked["wechat_reachable"] is False
    assert normalized_masked["app_registered"] is True
    assert normalized_masked["has_talked_with_agent"] is False
    assert normalized_masked["is_masked"] is True


def test_filter_member_for_visibility_preserves_4d_fields():
    """Ensure filter_member_for_visibility passes through 4D state fields and is_masked."""
    member = {
        "user_id": "u102",
        "group_id": "g1",
        "display_name": "王五",
        "same_group": False,
        "wechat_reachable": True,
        "app_registered": False,
        "has_talked_with_agent": True,
        "is_masked": True,
        "doing": {"value": "SaaS 开发", "disclosure": "confirmed_public"},
    }
    filtered = filter_member_for_visibility(member)

    assert filtered["same_group"] is False
    assert filtered["wechat_reachable"] is True
    assert filtered["app_registered"] is False
    assert filtered["has_talked_with_agent"] is True
    assert filtered["is_masked"] is True


def test_single_candidate_copy_generator(sample_profile):
    """Requirement 2: Single candidate copy generator produces 4 dedicated fields with 100% coverage."""
    candidate = {
        "user_id": "u101",
        "display_name": "林知夏",
        "doing": {"value": "Python AI Agent 架构研发", "disclosure": "confirmed_public"},
        "same_group": True,
        "wechat_reachable": True,
        "app_registered": True,
        "has_talked_with_agent": True,
        "is_masked": False,
    }

    copy = generate_single_candidate_copy(sample_profile, candidate)

    # Assert all 4 required keys are present and non-empty
    assert "invite_text" in copy and copy["invite_text"]
    assert "match_highlights" in copy and len(copy["match_highlights"]) == 3
    assert "forward_copy" in copy and copy["forward_copy"]
    assert "quick_connect_copy" in copy and copy["quick_connect_copy"]

    # 1. invite_text五要素完备
    inv_text = copy["invite_text"]
    assert "我在做的项目" in inv_text
    assert "我能提供的资源或能力" in inv_text
    assert "想聊聊" in inv_text
    assert "@林知夏" in inv_text
    assert "不一定对得上" in inv_text or "聊聊看就好" in inv_text

    # 2. match_highlights为长度为3的数组
    highlights = copy["match_highlights"]
    assert isinstance(highlights, list)
    assert len(highlights) == 3
    assert "对方" in highlights[0]
    assert "需求" in highlights[1]
    assert "契合点" in highlights[2] or "优势" in highlights[2]

    # 3. forward_copy以第三人称格式撰写
    fwd = copy["forward_copy"]
    assert "Hi 林知夏" in fwd
    assert "爱合伙平台上有一位创业者正在做" in fwd or "正在做" in fwd
    assert "想请教" in fwd
    assert "是否方便为您做微信对接" in fwd or "对接" in fwd

    # 4. quick_connect_copy简洁礼貌
    qc = copy["quick_connect_copy"]
    assert "你好，在爱合伙看到你的" in qc
    assert "背景" in qc
    assert "希望能交流合作" in qc or "合作" in qc


def test_enrich_candidates_with_single_copy(sample_profile):
    """Verify enrich_candidates_with_single_copy populates all candidate cards in a list."""
    candidates = [
        {
            "user_id": "u101",
            "display_name": "周然",
            "doing": {"value": "嵌入式量产固件", "disclosure": "confirmed_public"},
            "same_group": True,
            "wechat_reachable": True,
            "app_registered": True,
            "has_talked_with_agent": True,
            "is_masked": False,
        },
        {
            "user_id": "u104",
            "display_name": "李工",
            "doing": {"value": "物联网模组选型", "disclosure": "confirmed_public"},
            "same_group": False,
            "wechat_reachable": True,
            "app_registered": True,
            "has_talked_with_agent": False,
            "is_masked": True,
        },
    ]

    enriched = enrich_candidates_with_single_copy(candidates, sample_profile)

    assert len(enriched) == 2
    for c in enriched:
        assert "invite_text" in c and c["invite_text"]
        assert "match_highlights" in c and len(c["match_highlights"]) == 3
        assert "forward_copy" in c and c["forward_copy"]
        assert "quick_connect_copy" in c and c["quick_connect_copy"]


def test_generate_invite_copy_integration(sample_profile):
    """Test full integration in generate_invite_copy where candidate list is returned enriched."""
    candidates = [
        {
            "user_id": "u101",
            "display_name": "周然",
            "doing": {"value": "智能固件量产", "disclosure": "confirmed_public"},
            "same_group": True,
            "wechat_reachable": True,
            "app_registered": True,
            "has_talked_with_agent": True,
            "is_masked": False,
        }
    ]

    res = generate_invite_copy(
        profile=sample_profile,
        candidates=candidates,
        match_status="matched",
        willing_to_at=True,
    )

    assert res.ok is True
    assert res.kind == "directed"
    assert "u101" in res.mentioned_user_ids
