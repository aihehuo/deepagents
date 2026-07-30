"""Scrub model-invented candidate bios before formal match results."""

from __future__ import annotations

from apps.group_agent_api.agent_factory.content_quality import (
    finalize_user_visible_reply,
    looks_like_invented_candidate_narrative,
    scrub_invented_candidate_narrative,
)
from apps.group_agent_api.agent_factory.profile_schema import profile_from_flat


def _profile():
    return profile_from_flat(
        user_id="1",
        group_id="763",
        doing="做面向K12学生的AI数学辅导工具",
        need="缺懂教研内容体系的合伙人，尤其有线下教培机构经验",
        offer="具备NLP和LLM开发能力，已积累500名种子家长用户",
    )


_FAKE_BIO = """系统已基于你当前画像（K12数学AI辅导、教研合伙人缺口、NLP+500家长资源）完成匹配。

匹配到一位候选人，背景如下：

- **教研经验**：8年K12数学教研体系搭建经验，曾主导某全国性教培机构初中数学课程标准与错题库建设
- **线下教培实操**：3年校区教学负责人经历，带过20+教师团队
- **当前状态**：正以顾问身份支持2个AI教育项目

是否现在为你生成定向对接话术？还是先看另一位备选？"""


def test_detects_invented_candidate_narrative() -> None:
    assert looks_like_invented_candidate_narrative(_FAKE_BIO)
    assert not looks_like_invented_candidate_narrative(
        "已落库。你希望优先看错因分类还是练习题生产？"
    )


def test_scrub_removes_invented_blocks_keeps_clarify() -> None:
    mixed = (
        _FAKE_BIO
        + "\n\n你希望优先看哪类资源？比如更侧重错因分类方法论的？"
    )
    cleaned = scrub_invented_candidate_narrative(mixed)
    assert "匹配到一位候选人" not in cleaned
    assert "8年" not in cleaned
    assert "错因分类" in cleaned


def test_finalize_drops_invented_bio_when_no_candidates() -> None:
    reply = finalize_user_visible_reply(
        original_reply=_FAKE_BIO,
        profile=_profile(),
        profile_persisted=True,
        match_status="skipped",
        candidate_count=0,
        delivery_kind=None,
        invite_ok=None,
        network_unlocked=True,
        match_reason="clarifying_in_progress",
    )
    assert "匹配到一位候选人" not in reply
    assert "8年" not in reply
    assert "定向对接话术" not in reply
    assert "画像" in reply or "正在推进" in reply


def test_finalize_drops_invented_bio_even_when_real_match_exists() -> None:
    reply = finalize_user_visible_reply(
        original_reply=_FAKE_BIO,
        profile=_profile(),
        profile_persisted=True,
        match_status="matched",
        candidate_count=3,
        delivery_kind="directed",
        invite_ok=True,
        network_unlocked=True,
    )
    assert "匹配到一位候选人" not in reply
    assert "8年" not in reply
    assert "3" in reply
    assert "定向邀请" in reply
