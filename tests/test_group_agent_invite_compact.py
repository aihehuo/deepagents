"""Invite copy must stay WeChat-paste short — no full profile dumps."""

from __future__ import annotations

from apps.group_agent_api.agent_factory.invite_copy import (
    MAX_INVITE_CHARS,
    compact_doing_for_invite,
    generate_invite_copy,
)
from apps.group_agent_api.agent_factory.profile_schema import profile_from_flat

_DUMP = (
    "所在地: 广东 深圳\n"
    "所在行业: 企业服务\n"
    "细分行业: 行业解决方案\n"
    "个人目标: AIOT产品经理，华为多年经验，前深圳市物联网协会副秘书长，"
    "长期关注智能硬件与产业数字化。\n"
    "具体介绍: 做AI漫画生产与分发，搭建AI文创产业运营平台，"
    "同时覆盖B端方案与内容分发。"
)


def test_compact_doing_prefers_具体介绍_and_truncates() -> None:
    hook = compact_doing_for_invite(_DUMP, max_chars=40)
    assert "所在地" not in hook
    assert "个人目标" not in hook
    assert "具体介绍" not in hook
    assert "AI漫画" in hook or "文创" in hook
    assert len(hook) <= 40


def test_directed_invite_from_profile_dump_stays_short() -> None:
    profile = profile_from_flat(
        user_id="u_me",
        group_id="g1",
        doing="推广爱合伙群智能体产品，找AI agent拓客合伙人",
        need="有AI agent客户拓展经验的合伙人",
        offer="产品设计与技术协作",
    )
    candidates = [
        {
            "user_id": "19887",
            "display_name": "候选甲",
            "source_group_id": "g1",
            "group_id": "g1",
            "doing": {"value": _DUMP, "disclosure": "confirmed_public"},
        },
        {
            "user_id": "61970",
            "display_name": "候选乙",
            "source_group_id": "g1",
            "group_id": "g1",
            "doing": {
                "value": (
                    "所在地: 上海\n"
                    "个人目标: 连续创业者，做过跨境电商与品牌出海。\n"
                    "具体介绍: 正在做跨境供应链数字化与品牌增长咨询。"
                ),
                "disclosure": "confirmed_public",
            },
        },
    ]
    result = generate_invite_copy(
        profile=profile,
        candidates=candidates,
        match_status="matched",
        willing_to_at=True,
    )
    assert result.ok
    assert result.kind == "directed"
    assert result.text
    assert len(result.text) <= MAX_INVITE_CHARS
    assert "所在地" not in result.text
    assert "个人目标" not in result.text
    assert "具体介绍" not in result.text
    assert "@19887" in result.text
    assert "@61970" in result.text


def test_three_candidate_invite_stays_within_budget() -> None:
    profile = profile_from_flat(
        user_id="u_me",
        group_id="g1",
        doing="x" * 80,
        need="y" * 80,
        offer="z" * 80,
    )
    candidates = [
        {
            "user_id": f"uid_{i}",
            "display_name": f"候选{i}",
            "source_group_id": "g1",
            "group_id": "g1",
            "doing": {
                "value": "具体介绍: " + ("长描述内容" * 20),
                "disclosure": "confirmed_public",
            },
        }
        for i in range(3)
    ]
    result = generate_invite_copy(
        profile=profile,
        candidates=candidates,
        match_status="matched",
        willing_to_at=True,
    )
    assert result.ok
    assert len(result.text) <= MAX_INVITE_CHARS
    assert result.text.count("@") == 3
