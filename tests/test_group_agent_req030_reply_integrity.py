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


def test_finalize_user_visible_reply_deduplicates_when_next_step_is_original_reply() -> None:
    profile = profile_from_flat(
        user_id="1",
        group_id="763",
        doing="AI training project",
        need="ML engineer",
        offer="compute",
    )
    long_english_reply = (
        "Got it — you're working on an AI training project.\n"
        "To help match the right people, I need to clarify three things:\n"
        "1. Doing: What's the specific goal?\n"
        "2. Need: What's missing?\n"
        "3. Offer: What can you bring?"
    )

    result = finalize_user_visible_reply(
        original_reply=long_english_reply,
        profile=profile,
        profile_persisted=True,
        match_status="skipped",
        candidate_count=0,
        delivery_kind=None,
        invite_ok=None,
        network_unlocked=True,
        revisit_hint=None,
        match_reason="profile_too_thin",
    )

    # Must contain the reply exactly ONCE, not duplicated
    assert result.count("Got it — you're working on an AI training project.") == 1


def test_english_short_prefix_deduplication() -> None:
    from apps.group_agent_api.app.endpoints.chat import (
        _merge_force_save_reply,
        _replies_substantially_same,
    )

    a = "Yes.\n\nI help match people with the right collaborators — based on what they're doing, what they need, and what they can offer."
    b = "Yes.\n\nI help match people with the right collaborators — based on what they're doing, what they need, and what they can offer.\n\nGot it — 'AI training project' is a start."

    assert _replies_substantially_same(a, b) is True
    merged = _merge_force_save_reply(a, b)
    assert merged == b
    assert merged.count("I help match people") == 1


def test_extract_reply_ignores_pre_tool_call_intermediate_message() -> None:
    from langchain_core.messages import AIMessage, ToolMessage
    from apps.group_agent_api.app.endpoints.chat import _extract_reply

    msg1 = AIMessage(
        content="Yes.\n\nI don't have prior knowledge about you...",
        tool_calls=[{"name": "save_group_profile", "args": {}, "id": "call_1"}],
    )
    msg2 = ToolMessage(content='{"ok": true}', tool_call_id="call_1")
    msg3 = AIMessage(
        content="Got it — 'work on an AI project' is a start. To sharpen...",
        tool_calls=[],
    )

    messages = [msg1, msg2, msg3]
    extracted = _extract_reply(messages, 0)
    assert extracted == "Got it — 'work on an AI project' is a start. To sharpen..."


def test_extract_reply_never_joins_history_when_msg_count_before_is_zero() -> None:
    """Frontend export ga_116f189a…: each bubble was prior bubbles joined with \\n\\n."""
    from langchain_core.messages import AIMessage, HumanMessage
    from apps.group_agent_api.app.endpoints.chat import _extract_reply

    history_and_turn = [
        HumanMessage(content="Hi, can you speak English"),
        AIMessage(content="Yes."),
        HumanMessage(content="What do you know about"),
        AIMessage(
            content=(
                "Yes.\n\nI'm ready — just tell me what you're working on, "
                "what you need, or what you can offer. Let's get your profile filled in."
            )
        ),
        HumanMessage(content="I'm working on an AI project."),
        AIMessage(
            content=(
                "Yes.\n\nI'm ready — just tell me what you're working on, "
                "what you need, or what you can offer. Let's get your profile filled in.\n\n"
                "Got it — “AI project” is a start. To match you well, I need to dig "
                "just one level deeper."
            )
        ),
    ]

    # Bad/missing checkpoint count → whole thread passed as the extract window.
    extracted = _extract_reply(history_and_turn, 0)
    assert extracted.startswith("Got it —")
    assert extracted.count("Yes.") == 0
    assert "I'm ready" not in extracted


def test_extract_reply_peels_model_echo_of_prior_bubble() -> None:
    from langchain_core.messages import AIMessage, HumanMessage
    from apps.group_agent_api.app.endpoints.chat import _extract_reply

    prior = (
        "Yes.\n\nI'm ready — just tell me what you're working on, "
        "what you need, or what you can offer. Let's get your profile filled in."
    )
    newest = (
        "Got it — “AI project” is a start. To match you well, I need to dig "
        "just one level deeper."
    )
    messages = [
        HumanMessage(content="What do you know about"),
        AIMessage(content=prior),
        HumanMessage(content="I'm working on an AI project."),
        AIMessage(content=f"{prior}\n\n{newest}"),
    ]
    extracted = _extract_reply(messages, 2)
    assert extracted == newest


def test_extract_reply_uses_last_ai_without_joining_when_no_tools() -> None:
    from langchain_core.messages import AIMessage
    from apps.group_agent_api.app.endpoints.chat import _extract_reply

    messages = [
        AIMessage(content="Yes."),
        AIMessage(
            content=(
                "I'm ready — just tell me what you're working on, "
                "what you need, or what you can offer."
            )
        ),
    ]
    assert _extract_reply(messages, 0) == (
        "I'm ready — just tell me what you're working on, "
        "what you need, or what you can offer."
    )
