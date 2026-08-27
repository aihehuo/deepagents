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


def test_should_force_save_on_stale_when_user_asks_for_partners() -> None:
    assert (
        _should_force_profile_save(
            profile_ok=False,
            profile_status="failed",
            persist_alert="profile_stale_for_episode",
            messages=[],
            msg_count_before=0,
            user_message="你这边有合适的搭子吗？",
        )
        is True
    )


def test_should_force_save_on_stale_when_message_is_profile_bearing() -> None:
    dump = (
        "嗨呀，最近在做内容变现的选题，想找懂社群运营或者有落地经验的小伙伴聊聊～"
        "你这边有合适的搭子吗？"
    )
    assert (
        _should_force_profile_save(
            profile_ok=False,
            profile_status="failed",
            persist_alert="profile_stale_for_episode",
            messages=[],
            msg_count_before=0,
            user_message=dump,
        )
        is True
    )


def test_natural_chinese_dump_extracts_doing_need_offer() -> None:
    from apps.group_agent_api.app.async_manager import extract_explicit_profile_dimensions

    dump = (
        "你好，我正在做一个AI辅导工具，面向K12学生。"
        "目前我有技术开发能力，包括NLP和LLM，还有一个初步的原型。"
        "我需要找一个懂教育、教研或者有教培经验的合伙人。你这边有合适的吗？"
    )
    dims = extract_explicit_profile_dimensions(dump)
    assert dims is not None
    assert "AI辅导" in dims["doing"] or "K12" in dims["doing"]
    assert "教育" in dims["need"] or "教研" in dims["need"]
    assert "NLP" in dims["offer"] or "原型" in dims["offer"]
    assert extract_explicit_profile_dimensions("你好") is None
