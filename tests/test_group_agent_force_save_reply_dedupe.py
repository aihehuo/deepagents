"""FORCE_SAVE reply merge must not duplicate user-visible bubbles."""

from __future__ import annotations

from apps.group_agent_api.app.endpoints.chat import (
    _merge_force_save_reply,
    _replies_substantially_same,
    _should_force_profile_save,
)


def test_merge_keeps_distinct_retry_text() -> None:
    merged = _merge_force_save_reply("首轮回复。", "强制保存轮回复。")
    assert "首轮回复。" in merged
    assert "强制保存轮回复。" in merged


def test_merge_dedupes_near_identical_retry() -> None:
    text = (
        "明白了，新方向：餐饮小店，生鲜外卖为主，互联网+内容打法。"
        "还缺 need / offer。先补一个，你最想优先解决的卡点是？"
    )
    merged = _merge_force_save_reply(text, text)
    assert merged == text
    assert merged.count("明白了，新方向") == 1


def test_merge_dedupes_when_retry_contains_first() -> None:
    first = "明白了，新方向：餐饮小店。"
    retry = first + "\n还缺 need。先补卡点？"
    assert _replies_substantially_same(first, retry)
    assert _merge_force_save_reply(first, retry) in (first, retry)


def test_should_not_force_save_on_stale_episode_without_save_attempt() -> None:
    assert (
        _should_force_profile_save(
            profile_ok=False,
            profile_status="failed",
            persist_alert="profile_stale_for_episode",
            messages=[],
            msg_count_before=0,
        )
        is False
    )


def test_should_force_save_when_no_profile_on_disk() -> None:
    assert (
        _should_force_profile_save(
            profile_ok=False,
            profile_status="failed",
            persist_alert="missing_profile",
            messages=[],
            msg_count_before=0,
        )
        is True
    )
