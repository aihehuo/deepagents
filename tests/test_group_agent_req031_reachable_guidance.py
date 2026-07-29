"""REQ-031 tests: Candidate reachability awareness and cross-group guidance logic."""

from __future__ import annotations

from typing import Any

from apps.group_agent_api.agent_factory.capability import CapabilityTier
from apps.group_agent_api.agent_factory.content_quality import (
    finalize_user_visible_reply,
)
from apps.group_agent_api.agent_factory.disclosure import (
    filter_member_for_visibility,
)
from apps.group_agent_api.agent_factory.guard import enforce_capability_guard
from apps.group_agent_api.agent_factory.integrations.match_client import (
    _normalize_candidate,
)
from apps.group_agent_api.agent_factory.invite_copy import (
    _build_directed_elements,
)
from apps.group_agent_api.agent_factory.match_stub import (
    MatchStub,
    PoolMember,
)
from apps.group_agent_api.agent_factory.profile_schema import (
    GroupProfile,
    ProfileField,
)


def test_filter_member_for_visibility_preserves_reachability_and_group_info() -> None:
    raw = {
        "user_id": "u201",
        "group_id": "g2",
        "display_name": "张三",
        "is_reachable": False,
        "group_info": {"id": "g2", "name": "AI创投群"},
        "doing": {"value": "大模型架构", "disclosure": "confirmed_public"},
    }
    visible = filter_member_for_visibility(raw)
    assert visible["user_id"] == "u201"
    assert visible["is_reachable"] is False
    assert visible["group_info"] == {"id": "g2", "name": "AI创投群"}


def test_normalize_candidate_preserves_reachability_and_group_info() -> None:
    raw = {
        "user_id": "u202",
        "group_id": "g3",
        "source_group_id": "g3",
        "display_name": "李四",
        "is_reachable": False,
        "group_info": {"id": "g3", "name": "黑客松群"},
        "doing": {"value": "前端开发", "disclosure": "confirmed_public"},
    }
    normalized = _normalize_candidate(raw, fallback_group="g1")
    assert normalized["user_id"] == "u202"
    assert normalized["is_reachable"] is False
    assert normalized["group_info"] == {"id": "g3", "name": "黑客松群"}


def test_capability_guard_allows_unreachable_cross_group_candidate() -> None:
    result = enforce_capability_guard(
        tier=CapabilityTier.in_group,
        reply="ok",
        candidates=[
            {
                "user_id": "u201",
                "group_id": "g2",
                "source_group_id": "g2",
                "display_name": "张三",
                "is_reachable": False,
                "group_info": {"id": "g2", "name": "AI创投群"},
                "doing": {"value": "大模型架构", "disclosure": "confirmed_public"},
            }
        ],
        caller_group_id="g1",
        user_id="u1",
    )
    assert not result.blocked
    assert len(result.candidates) == 1
    assert result.candidates[0]["user_id"] == "u201"
    assert result.candidates[0]["is_reachable"] is False
    assert not any("cross_group" in v for v in result.violations)


def test_build_directed_elements_skips_unreachable_candidate_at_mentions() -> None:
    profile = GroupProfile(
        user_id="u1",
        group_id="g1",
        doing=ProfileField(value="推进群智能体"),
        need=ProfileField(value="找架构师伙伴"),
        offer=ProfileField(value="全栈架构"),
    )
    candidates = [
        {
            "user_id": "u201",
            "display_name": "张三",
            "is_reachable": False,
            "group_info": {"id": "g2", "name": "AI创投群"},
            "doing": {"value": "大模型架构", "disclosure": "confirmed_public"},
        },
        {
            "user_id": "u101",
            "display_name": "王五",
            "is_reachable": True,
            "group_info": {"id": "g1", "name": "本群"},
            "doing": {"value": "后端架构", "disclosure": "confirmed_public"},
        },
    ]
    elements = _build_directed_elements(
        profile=profile, candidates=candidates, topic="想聊聊架构设计"
    )
    why = elements.get("why_invite", "")
    assert "@王五" in why
    assert "@张三" not in why


def test_finalize_user_visible_reply_generates_unreachable_guidance_prompt() -> None:
    profile = GroupProfile(
        user_id="u1",
        group_id="g1",
        doing=ProfileField(value="推进群智能体"),
        need=ProfileField(value="找架构师伙伴"),
        offer=ProfileField(value="全栈架构"),
    )
    candidates = [
        {
            "user_id": "u201",
            "display_name": "张三",
            "is_reachable": False,
            "group_info": {"id": "g2", "name": "AI创投群"},
            "doing": {"value": "大模型架构", "disclosure": "confirmed_public"},
        }
    ]
    reply = finalize_user_visible_reply(
        original_reply="",
        profile=profile,
        profile_persisted=True,
        match_status="matched",
        candidate_count=1,
        delivery_kind="directed",
        invite_ok=True,
        network_unlocked=True,
        candidates=candidates,
    )
    expected_prompt = "候选人【张三】在【AI创投群】，你目前不在该群，可以申请加入【AI创投群】"
    assert expected_prompt in reply


def test_match_stub_with_unreachable_candidate() -> None:
    stub = MatchStub(
        pool=[
            PoolMember(
                user_id="u201",
                group_id="g2",
                bound=True,
                display_name="张三",
                is_reachable=False,
                group_info={"id": "g2", "name": "AI创投群"},
                doing={"value": "LLM 引擎", "disclosure": "confirmed_public"},
                need={"value": "找合作", "disclosure": "confirmed_public"},
                offer={"value": "C++ 引擎", "disclosure": "confirmed_public"},
                keywords=["llm", "引擎"],
            )
        ]
    )
    result = stub.search(query="LLM 引擎", group_id="g1")
    assert result.status == "matched"
    assert len(result.candidates) == 1
    cand = result.candidates[0]
    assert cand["user_id"] == "u201"
    assert cand["is_reachable"] is False
    assert cand["group_info"] == {"id": "g2", "name": "AI创投群"}


def test_align_preserves_unreachable_cross_group_candidate() -> None:
    from apps.group_agent_api.agent_factory.integrations.group_bind import (
        align_match_to_trusted_group,
    )
    from apps.group_agent_api.agent_factory.match_stub import MatchResult

    raw = MatchResult(
        status="matched",
        group_id="763",
        query="教研合伙人",
        reason="matched_1",
        candidates=[
            {
                "user_id": "u201",
                "group_id": "999",
                "source_group_id": "999",
                "display_name": "张三",
                "is_reachable": False,
                "group_info": {
                    "group_id": "999",
                    "group_name": "AI创投群",
                    "apply_url": "https://m.aihehuo.com/group_share/e999",
                },
                "doing": {"value": "大模型架构", "disclosure": "confirmed_public"},
            }
        ],
    )
    aligned = align_match_to_trusted_group(raw, trusted_group_id="763")
    assert aligned.status == "matched"
    assert len(aligned.candidates) == 1
    cand = aligned.candidates[0]
    assert cand["is_reachable"] is False
    assert cand["source_group_id"] == "999"
    assert cand["group_info"]["group_name"] == "AI创投群"


def test_align_preserves_reachable_via_other_shared_group() -> None:
    from apps.group_agent_api.agent_factory.integrations.group_bind import (
        align_match_to_trusted_group,
    )
    from apps.group_agent_api.agent_factory.match_stub import MatchResult

    raw = MatchResult(
        status="matched",
        group_id="763",
        query="合伙人",
        reason="matched_1",
        candidates=[
            {
                "user_id": "u301",
                "group_id": "888",
                "source_group_id": "888",
                "display_name": "李四",
                "is_reachable": True,
                "group_info": {
                    "group_id": "888",
                    "group_name": "硬件群",
                    "apply_url": "https://m.aihehuo.com/group_share/e888",
                },
                "doing": {"value": "嵌入式", "disclosure": "confirmed_public"},
            }
        ],
    )
    aligned = align_match_to_trusted_group(raw, trusted_group_id="763")
    assert aligned.status == "matched"
    assert aligned.candidates[0]["source_group_id"] == "888"
    assert aligned.candidates[0]["is_reachable"] is True


def test_align_still_drops_foreign_without_reachability_flag() -> None:
    from apps.group_agent_api.agent_factory.integrations.group_bind import (
        align_match_to_trusted_group,
    )
    from apps.group_agent_api.agent_factory.match_stub import MatchResult

    raw = MatchResult(
        status="matched",
        group_id="763",
        query="q",
        reason="x",
        candidates=[
            {
                "user_id": "1",
                "group_id": "9999",
                "source_group_id": "9999",
                "display_name": "X",
                "doing": {"value": "固件", "disclosure": "confirmed_public"},
            }
        ],
    )
    aligned = align_match_to_trusted_group(raw, trusted_group_id="763")
    assert aligned.status == "empty"
    assert aligned.candidates == []


def test_align_then_guard_e2e_keeps_unreachable_candidate() -> None:
    """Production path: align → guard must not strip REQ-053 cross-group hits."""
    from apps.group_agent_api.agent_factory.integrations.group_bind import (
        align_match_to_trusted_group,
    )
    from apps.group_agent_api.agent_factory.match_stub import MatchResult

    raw = MatchResult(
        status="matched",
        group_id="g1",
        query="架构",
        reason="matched_2",
        candidates=[
            {
                "user_id": "u201",
                "group_id": "g2",
                "source_group_id": "g2",
                "display_name": "张三",
                "is_reachable": False,
                "group_info": {"group_id": "g2", "group_name": "AI创投群"},
                "doing": {"value": "大模型架构", "disclosure": "confirmed_public"},
            },
            {
                "user_id": "u101",
                "group_id": "g1",
                "source_group_id": "g1",
                "display_name": "王五",
                "is_reachable": True,
                "doing": {"value": "后端架构", "disclosure": "confirmed_public"},
            },
        ],
    )
    aligned = align_match_to_trusted_group(raw, trusted_group_id="g1")
    guarded = enforce_capability_guard(
        tier=CapabilityTier.in_group,
        reply="ok",
        candidates=aligned.candidates,
        caller_group_id="g1",
        user_id="u1",
    )
    assert not guarded.blocked
    ids = {c["user_id"] for c in guarded.candidates}
    assert ids == {"u201", "u101"}
    unreachable = next(c for c in guarded.candidates if c["user_id"] == "u201")
    assert unreachable["source_group_id"] == "g2"
    assert unreachable["is_reachable"] is False


def test_capability_guard_allows_reachable_via_other_shared_group() -> None:
    result = enforce_capability_guard(
        tier=CapabilityTier.in_group,
        reply="ok",
        candidates=[
            {
                "user_id": "u301",
                "group_id": "g_other",
                "source_group_id": "g_other",
                "display_name": "李四",
                "is_reachable": True,
                "group_info": {"group_id": "g_other", "group_name": "硬件群"},
                "doing": {"value": "嵌入式", "disclosure": "confirmed_public"},
            }
        ],
        caller_group_id="g1",
        user_id="u1",
    )
    assert not result.blocked
    assert len(result.candidates) == 1
    assert not any("cross_group" in v for v in result.violations)
