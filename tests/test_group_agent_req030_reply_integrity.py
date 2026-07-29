"""Tests for REQ-030: reply integrity and avoiding template overwrites."""

from __future__ import annotations

from apps.group_agent_api.agent_factory.content_quality import (
    finalize_user_visible_reply,
)
from apps.group_agent_api.agent_factory.profile_schema import profile_from_flat
from apps.group_agent_api.agent_factory.revisit import RevisitHint


def test_finalize_user_visible_reply_preserves_custom_original_reply() -> None:
    profile = profile_from_flat(
        user_id="1",
        group_id="763",
        doing="做生鲜烧麦外带和外送",
        need="缺懂本地化运营的人",
        offer="提供种子客群",
    )
    custom_llm_reply = (
        "我搜索时使用的关键词包含「生鲜烧麦」、「外带外送」和「闪购运营」。"
    )

    result = finalize_user_visible_reply(
        original_reply=custom_llm_reply,
        profile=profile,
        profile_persisted=True,
        match_status="empty",
        candidate_count=0,
        delivery_kind="undirected",
        invite_ok=True,
        network_unlocked=True,
        revisit_hint=None,
    )

    assert custom_llm_reply in result
    assert "我理解并已更新画像" in result


def test_finalize_user_visible_reply_uses_template_when_original_reply_is_empty() -> None:
    profile = profile_from_flat(
        user_id="1",
        group_id="763",
        doing="做生鲜烧麦外带和外送",
        need="缺懂本地化运营的人",
        offer="提供种子客群",
    )

    result = finalize_user_visible_reply(
        original_reply="",
        profile=profile,
        profile_persisted=True,
        match_status="empty",
        candidate_count=0,
        delivery_kind="undirected",
        invite_ok=True,
        network_unlocked=True,
        revisit_hint=None,
    )

    assert "我理解并已更新画像" in result
    assert "这次暂未找到足够明确的本群人选" in result
