"""REQ-029 profile match-ready: length gate + LLM semantic + thin-skip degrade."""

from __future__ import annotations

from pathlib import Path

import pytest
from apps.group_agent_api.agent_factory.content_quality import finalize_user_visible_reply
from apps.group_agent_api.agent_factory.profile_quality import (
    MAX_THIN_SKIPS_BEFORE_DEGRADED,
    assess_length_and_role,
    decide_match_gate,
    wants_force_match,
)
from apps.group_agent_api.agent_factory.profile_schema import profile_from_flat


class _ReadyModel:
    def invoke(self, _msgs):
        class _M:
            content = (
                '{"ready":true,"score":80,"doing_ok":true,"need_ok":true,'
                '"offer_ok":true,"reasons":[],"gaps":[]}'
            )

        return _M()


class _ThinModel:
    def invoke(self, _msgs):
        class _M:
            content = (
                '{"ready":false,"score":20,"doing_ok":false,"need_ok":true,'
                '"offer_ok":true,"reasons":["doing_too_vague"],'
                '"gaps":["你在做的具体产品场景是什么？"]}'
            )

        return _M()


class _BadJsonModel:
    def invoke(self, _msgs):
        class _M:
            content = "not-json"

        return _M()


def _rich_profile(tmp: Path, uid: str = "u1", gid: str = "g1"):
    return profile_from_flat(
        user_id=uid,
        group_id=gid,
        doing="在做智能宠物喂食器硬件与 App",
        need="需要联网固件与量产供应链对接",
        offer="有工厂资源和硬件设计经验",
    )


def test_length_gate_rejects_short_fields() -> None:
    p = profile_from_flat(
        user_id="u1",
        group_id="g1",
        doing="创业",
        need="找人",
        offer="资源",
    )
    q = assess_length_and_role(p)
    assert not q.ready
    assert q.layer_failed == "length"
    assert any(r.endswith("_too_short") for r in q.reasons)


def test_length_gate_accepts_compact_chinese_doing() -> None:
    """Whitespace must not make「做 AI 教育」fail the doing length gate."""
    p = profile_from_flat(
        user_id="u1",
        group_id="g1",
        doing="做 AI 教育",
        need="缺能落地课程产品的教育产品经理",
        offer="有一个私域客户群",
    )
    q = assess_length_and_role(p)
    assert q.ready
    assert q.layer_failed is None


def test_wants_force_match() -> None:
    assert wants_force_match("先匹配一下吧")
    assert wants_force_match("不用再问了，先搜一下")
    assert not wants_force_match("我想再补充一下需求")


def test_decide_ready_allows_match(tmp_path: Path) -> None:
    profile = _rich_profile(tmp_path)
    d = decide_match_gate(
        profile=profile,
        model=_ReadyModel(),
        base_dir=tmp_path,
        message="继续",
        metadata={"episode_id": "ep1"},
    )
    assert d.allow_match
    assert d.match_reason is None
    assert d.quality.ready


def test_decide_thin_skips_then_degrades(tmp_path: Path) -> None:
    profile = _rich_profile(tmp_path)
    meta = {"episode_id": "ep_thin"}
    for i in range(MAX_THIN_SKIPS_BEFORE_DEGRADED):
        d = decide_match_gate(
            profile=profile,
            model=_ThinModel(),
            base_dir=tmp_path,
            message="补充一点",
            metadata=meta,
        )
        assert not d.allow_match
        assert d.match_reason == "profile_too_thin"
        assert d.thin_skip_count == i + 1

    d = decide_match_gate(
        profile=profile,
        model=_ThinModel(),
        base_dir=tmp_path,
        message="再试试",
        metadata=meta,
    )
    assert d.allow_match
    assert d.degraded
    assert d.match_reason == "profile_thin_degraded"


def test_force_match_degrades_immediately(tmp_path: Path) -> None:
    profile = _rich_profile(tmp_path)
    d = decide_match_gate(
        profile=profile,
        model=_ThinModel(),
        base_dir=tmp_path,
        message="先匹配",
        metadata={"episode_id": "ep_force"},
    )
    assert d.allow_match
    assert d.degraded
    assert d.match_reason == "profile_thin_degraded"


def test_llm_unavailable_does_not_count_thin_skip(tmp_path: Path) -> None:
    profile = _rich_profile(tmp_path)
    meta = {"episode_id": "ep_unavail"}
    d1 = decide_match_gate(
        profile=profile,
        model=_BadJsonModel(),
        base_dir=tmp_path,
        message="hi",
        metadata=meta,
    )
    assert not d1.allow_match
    assert d1.match_reason == "profile_quality_unavailable"
    assert d1.thin_skip_count == 0

    d2 = decide_match_gate(
        profile=profile,
        model=_BadJsonModel(),
        base_dir=tmp_path,
        message="hi",
        metadata=meta,
    )
    assert d2.thin_skip_count == 0


def test_finalize_asks_on_too_thin() -> None:
    profile = _rich_profile(Path("/tmp"))
    text = finalize_user_visible_reply(
        original_reply="",
        profile=profile,
        profile_persisted=True,
        match_status="skipped",
        candidate_count=0,
        delivery_kind=None,
        invite_ok=None,
        network_unlocked=True,
        match_reason="profile_too_thin",
        quality_gaps=["你最卡的是渠道还是技术？"],
    )
    assert "再确认" in text
    assert "最卡" in text
    assert "定向邀请" not in text


def test_finalize_keeps_agent_followup_when_too_thin() -> None:
    profile = _rich_profile(Path("/tmp"))
    text = finalize_user_visible_reply(
        original_reply="已记下：具体产品是 AI 单词记忆。还想确认你的目标用户是谁？",
        profile=profile,
        profile_persisted=True,
        match_status="skipped",
        candidate_count=0,
        delivery_kind=None,
        invite_ok=None,
        network_unlocked=True,
        match_reason="profile_too_thin",
        quality_gaps=["你在做的具体产品或场景，再多说一两句？"],
    )
    assert "AI 单词记忆" in text
    assert "再多说一两句" not in text or "若还没说到" in text
    assert "定向邀请" not in text


def test_finalize_thin_degraded_note() -> None:
    profile = _rich_profile(Path("/tmp"))
    text = finalize_user_visible_reply(
        original_reply="原始",
        profile=profile,
        profile_persisted=True,
        match_status="matched",
        candidate_count=2,
        delivery_kind="directed",
        invite_ok=True,
        network_unlocked=True,
        match_reason="profile_thin_degraded",
        quality_gaps=[],
    )
    assert "比较粗" in text
    assert "仅供参考" in text
