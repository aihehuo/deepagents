"""Invite copy: natural WeChat paste — @ people, no ads / no formal public cites."""

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

_AD = "梦日自助播报站项目找合伙人，微信18580542346"
_AD_TAG = "【大健康赛道】寻找投资人，已有健康管理app产品"


def test_compact_strips_ads_phones_and_field_dumps() -> None:
    hook = compact_doing_for_invite(_DUMP, max_chars=24)
    assert "所在地" not in hook
    assert "具体介绍" not in hook
    assert len(hook) <= 24

    ad = compact_doing_for_invite(_AD, max_chars=24)
    assert "185" not in ad
    assert "微信" not in ad
    assert "找合伙人" not in ad

    tagged = compact_doing_for_invite(_AD_TAG, max_chars=24)
    assert "投资人" not in tagged
    assert "大健康" in tagged


def test_directed_invite_natural_at_names_no_ad_quote() -> None:
    profile = profile_from_flat(
        user_id="u_me",
        group_id="g1",
        doing="推广爱合伙群智能体产品",
        need="有AI agent客户拓展经验的人",
        offer="产品设计与技术协作",
    )
    candidates = [
        {
            "user_id": "342662",
            "display_name": "严淑贤",
            "source_group_id": "g1",
            "group_id": "g1",
            "doing": {"value": _AD, "disclosure": "confirmed_public"},
        },
        {
            "user_id": "12798",
            "display_name": "Tomi",
            "source_group_id": "g1",
            "group_id": "g1",
            "doing": {"value": _AD_TAG, "disclosure": "confirmed_public"},
        },
        {
            "user_id": "362609",
            "display_name": "柳志强",
            "source_group_id": "g1",
            "group_id": "g1",
            "doing": {
                "value": "【DeepSeek大模型医疗AI智慧医院解决方案】招商",
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
    text = result.text
    assert len(text) <= MAX_INVITE_CHARS
    assert "@严淑贤" in text
    assert "@Tomi" in text
    assert "@柳志强" in text
    assert "@342662" not in text
    assert "公开资料" not in text
    assert "18580542346" not in text
    assert "微信" not in text
    assert "找合伙人" not in text
    assert "寻找投资人" not in text
    assert "AI agent" in text.lower() or "拓展" in text
    assert "不一定" in text or "供参考" in text


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
