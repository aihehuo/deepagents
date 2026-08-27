"""Contact scrub + clarifying-reply deferral for early match."""

from __future__ import annotations

from apps.group_agent_api.agent_factory.contact_scrub import (
    scrub_candidate_contacts,
    scrub_contact_text,
    scrub_display_name,
)
from apps.group_agent_api.agent_factory.profile_quality import (
    looks_like_clarifying_reply,
    looks_like_profile_bearing_message,
    should_defer_match_for_clarifying,
    wants_force_match,
)


def test_scrub_contact_text_removes_wechat_and_phone() -> None:
    raw = "做K12教研，微信：abc_12345，手机 13812345678，加我微信详谈"
    cleaned = scrub_contact_text(raw)
    assert "微信" not in cleaned
    assert "abc_12345" not in cleaned
    assert "13812345678" not in cleaned
    assert "加我微信" not in cleaned
    assert "K12" in cleaned or "教研" in cleaned


def test_scrub_display_name_replaces_handle_only() -> None:
    assert scrub_display_name("wx_hello_world", fallback_user_id="99") == "用户99"
    assert scrub_display_name("尹先生", fallback_user_id="99") == "尹先生"


def test_scrub_candidate_contacts_clears_doing_handles() -> None:
    cand = scrub_candidate_contacts(
        {
            "user_id": "42",
            "display_name": "张三 微信:foo_bar99",
            "doing": {
                "value": "英语教研，联系方式：13900001111",
                "disclosure": "confirmed_public",
            },
        }
    )
    assert "微信" not in cand["display_name"]
    assert "foo_bar99" not in cand["display_name"]
    assert "13900001111" not in cand["doing"]["value"]


def test_looks_like_clarifying_reply_defers_match() -> None:
    clarifying = (
        "已落库。\n\n接下来，帮你挖得更具体一点：\n"
        "你提到「线下教培机构经验」——能说说最近合作或任职过的机构类型和角色吗？"
        "是学科老师 / 教研主管？做过哪类课程？"
    )
    assert looks_like_clarifying_reply(clarifying) is True
    assert looks_like_clarifying_reply("本群已有 3 位公开信息与需求有交集的人选。") is False
    assert looks_like_clarifying_reply("是否现在启动匹配？") is True


def test_looks_like_clarifying_reply_defers_priority_ab_fork() -> None:
    """Real prod miss: one ？ + 「优先看 A 还是 B」while saying 可直接匹配."""
    reply = (
        "已落库：\n"
        "✅ doing：小红书职场成长内容博主（10万粉丝）\n"
        "✅ need：缺知识付费产品设计能力\n"
        "✅ offer：10万精准职场类粉丝\n\n"
        "接下来，我们可直接匹配——你希望优先看「有成熟知识付费产品从0到1经验」的人，"
        "还是更倾向「擅长社群冷启动+复购运营」的搭档？\n"
        "（这会影响匹配权重，我帮你对齐最相关管线）"
    )
    assert looks_like_clarifying_reply(reply) is True
    assert looks_like_clarifying_reply(
        "下一步：我已按这些条件在本群找到 3 位值得进一步聊的人选，并生成了定向邀请。"
    ) is False


def test_should_not_defer_match_when_profile_ok_or_user_asks_for_people() -> None:
    clarifying = "能再说说你希望优先看教研还是教培吗？"
    assert looks_like_clarifying_reply(clarifying) is True
    assert (
        should_defer_match_for_clarifying(
            reply=clarifying,
            user_message="我需要一个懂教育行业的，最好有教研经验。",
            profile_ok=False,
        )
        is True
    )
    assert (
        should_defer_match_for_clarifying(
            reply=clarifying,
            user_message="我需要一个懂教育行业的，最好有教研经验。",
            profile_ok=True,
        )
        is False
    )
    assert (
        should_defer_match_for_clarifying(
            reply=clarifying,
            user_message="你好，我正在做一个AI辅导工具。你这边有合适的吗？",
            profile_ok=False,
        )
        is False
    )
    assert wants_force_match("你这边有合适的吗？") is True


def test_profile_bearing_message_detects_complete_dump() -> None:
    dump = (
        "你好，我正在做一个AI辅导工具，面向K12学生。"
        "目前我有技术开发能力，包括NLP和LLM，还有一个初步的原型。"
        "我需要找一个懂教育、教研或者有教培经验的合伙人。你这边有合适的吗？"
    )
    assert looks_like_profile_bearing_message(dump) is True
    assert looks_like_profile_bearing_message("你好") is False
    assert looks_like_profile_bearing_message("我已经不充了") is False
