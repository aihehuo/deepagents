"""Contact scrub + clarifying-reply deferral for early match."""

from __future__ import annotations

from apps.group_agent_api.agent_factory.contact_scrub import (
    scrub_candidate_contacts,
    scrub_contact_text,
    scrub_display_name,
)
from apps.group_agent_api.agent_factory.profile_quality import looks_like_clarifying_reply


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
