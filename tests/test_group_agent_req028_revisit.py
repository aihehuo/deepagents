"""REQ-028 revisit opener + prior_candidate_ids exclude (TSD-03 Path A)."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from apps.group_agent_api.agent_factory.capability import CapabilityTier
from apps.group_agent_api.agent_factory.content_quality import (
    finalize_and_guard_user_visible_reply,
    finalize_user_visible_reply,
)
from apps.group_agent_api.agent_factory.match_stub import MatchResult, MatchStub
from apps.group_agent_api.agent_factory.profile_schema import profile_from_flat
from apps.group_agent_api.agent_factory.revisit import (
    RevisitHint,
    build_revisit_opener,
    excluded_ids_for_match,
    normalize_prior_candidate_ids,
    parse_revisit_from_metadata,
    parse_revisit_hint,
)
from apps.group_agent_api.app.models import AsyncCallRequest


def _base_async(**overrides: Any) -> dict[str, Any]:
    body = {
        "run_id": "run_req028_1",
        "idempotency_key": "idem_req028_1",
        "user_id": "u105",
        "unionid": "union_105",
        "group_id": "group_l1_alpha",
        "conversation_id": "ga_group_l1_alpha_u105",
        "message": "我回来了",
        "callback_url": "http://micro-web.example.invalid:3000/group_agent_callbacks/run_req028_1",
        "metadata": {},
    }
    body.update(overrides)
    return body


def _profile():
    return profile_from_flat(
        user_id="u105",
        group_id="group_l1_alpha",
        doing="在做联网宠物喂食器固件创业",
        need="需要硬件联调与量产供应链伙伴",
        offer="有嵌入式开发与工厂对接经验",
    )


def test_async_metadata_accepts_micro_revisit_shape() -> None:
    req = AsyncCallRequest(
        **_base_async(
            metadata={
                "prior_candidate_ids": ["101", "102", 103],
                "revisit_hint": {
                    "has_prior_invite": True,
                    "candidate_names": ["周然", "李工"],
                    "topic_summary": "联网与固件",
                },
                "custom": "ok",
            }
        )
    )
    assert req.metadata["prior_candidate_ids"] == ["101", "102", 103]
    assert req.metadata["revisit_hint"]["has_prior_invite"] is True


def test_async_metadata_still_rejects_arbitrary_lists_and_candidate_injection() -> None:
    with pytest.raises(ValidationError):
        AsyncCallRequest(**_base_async(metadata={"invalid_list": [1, 2, 3]}))
    with pytest.raises(ValidationError):
        AsyncCallRequest(
            **_base_async(metadata={"candidates": [{"user_id": "x"}]})
        )
    with pytest.raises(ValidationError):
        AsyncCallRequest(
            **_base_async(
                metadata={
                    "revisit_hint": {
                        "has_prior_invite": True,
                        "evil": True,
                    }
                }
            )
        )
    with pytest.raises(ValidationError):
        AsyncCallRequest(
            **_base_async(metadata={"prior_candidate_ids": [True]})
        )


def test_parse_helpers_normalize_and_fail_closed() -> None:
    priors, hint = parse_revisit_from_metadata(
        {
            "prior_candidate_ids": ["101", "101", " ", 102, None],
            "revisit_hint": {
                "has_prior_invite": True,
                "candidate_names": ["周然", "", "x" * 80],
                "topic_summary": "联网与固件",
            },
        }
    )
    assert priors == ["101", "102"]
    assert hint.has_prior_invite is True
    assert hint.candidate_names == ("周然",)
    assert hint.topic_summary == "联网与固件"
    assert parse_revisit_hint("bad") == RevisitHint()
    assert normalize_prior_candidate_ids("bad") == []


def test_excluded_ids_for_match_merges_self_and_priors() -> None:
    assert excluded_ids_for_match(
        "u105",
        {"prior_candidate_ids": ["u101", "u105", "u102"]},
    ) == ["u105", "u101", "u102"]


def test_revisit_opener_mentions_prior_and_branches() -> None:
    opener = build_revisit_opener(
        RevisitHint(
            has_prior_invite=True,
            candidate_names=("周然", "李工"),
            topic_summary="联网与固件",
        )
    )
    assert opener is not None
    assert opener.startswith("上次我给你推荐过周然、李工")
    assert "联网与固件" in opener
    assert "有回音" in opener
    assert "换人" in opener and "换题" in opener and "开新一轮" in opener
    assert build_revisit_opener(RevisitHint(has_prior_invite=False)) is None


def test_known_match_system_content_lists_candidates() -> None:
    from apps.group_agent_api.agent_factory.revisit import known_match_system_content

    assert known_match_system_content(None) is None
    assert known_match_system_content(RevisitHint(has_prior_invite=False)) is None
    text = known_match_system_content(
        RevisitHint(
            has_prior_invite=True,
            candidate_names=("王梦波", "慢跑者", "王先生"),
            topic_summary="Marketing lead",
        )
    )
    assert text is not None
    assert "王梦波" in text and "慢跑者" in text and "王先生" in text
    assert "Marketing lead" in text
    assert "禁止说" in text or "no recommendations" in text
    assert "定向邀请词" in text


def test_finalize_with_revisit_hint_prefixes_opener() -> None:
    hint = RevisitHint(
        has_prior_invite=True,
        candidate_names=("周然",),
        topic_summary="固件",
    )
    reply = finalize_user_visible_reply(
        original_reply="你好",
        profile=_profile(),
        profile_persisted=True,
        match_status="skipped",
        candidate_count=0,
        delivery_kind=None,
        invite_ok=None,
        network_unlocked=True,
        revisit_hint=hint,
    )
    assert reply.startswith("上次我给你推荐过周然")
    assert "有回音" in reply
    assert "\n\n我理解并已更新画像" in reply
    assert "定向邀请" not in reply


def test_finalize_without_revisit_hint_unchanged() -> None:
    reply = finalize_user_visible_reply(
        original_reply="你好",
        profile=_profile(),
        profile_persisted=True,
        match_status="skipped",
        candidate_count=0,
        delivery_kind=None,
        invite_ok=None,
        network_unlocked=True,
        revisit_hint=RevisitHint(has_prior_invite=False),
    )
    assert not reply.startswith("上次")
    assert "我理解并已更新画像" in reply


def test_finalize_revisit_without_profile_still_surfaces_opener() -> None:
    reply = finalize_user_visible_reply(
        original_reply="先聊聊近况。",
        profile=None,
        profile_persisted=False,
        match_status="skipped",
        candidate_count=0,
        delivery_kind=None,
        invite_ok=None,
        network_unlocked=True,
        revisit_hint=RevisitHint(
            has_prior_invite=True,
            candidate_names=("周然",),
        ),
    )
    assert reply.startswith("上次我给你推荐过周然")
    assert "先聊聊近况" in reply


@pytest.mark.parametrize(
    "tier", [CapabilityTier.not_in_group, CapabilityTier.unknown]
)
def test_non_network_capability_suppresses_revisit_opener(
    tier: CapabilityTier,
) -> None:
    guarded = finalize_and_guard_user_visible_reply(
        tier=tier,
        caller_group_id="group_l1_alpha",
        user_id="u105",
        original_reply="补充一下目标。",
        profile=_profile(),
        profile_persisted=True,
        match_status="skipped",
        candidates=[],
        delivery_kind=None,
        invite_ok=None,
        revisit_hint=RevisitHint(
            has_prior_invite=True,
            candidate_names=("周然",),
        ),
    )
    assert "推荐" not in guarded.reply
    assert "上次" not in guarded.reply
    assert guarded.candidates == []


def test_match_stub_excludes_prior_candidate_ids() -> None:
    stub = MatchStub()
    pool = stub.reachable_pool("mock_g1")
    assert pool, "default stub pool must have reachable candidates"
    prior = pool[0].user_id
    result = stub.search(
        query=" ".join(pool[0].keywords) or pool[0].doing.get("value", "AI"),
        group_id="mock_g1",
        excluded_ids=excluded_ids_for_match(
            "caller_self", {"prior_candidate_ids": [prior]}
        ),
    )
    ids = [c["user_id"] for c in result.candidates]
    assert prior not in ids
    assert "caller_self" not in ids


def test_search_candidates_tool_passes_prior_exclusions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def _fake_run_match(**kwargs: Any) -> MatchResult:
        captured.update(kwargs)
        return MatchResult(
            status="empty",
            candidates=[],
            query=str(kwargs.get("query") or ""),
            group_id=str(kwargs.get("group_id") or ""),
            reason="none",
        )

    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.search_tool.match_backend.run_match",
        _fake_run_match,
    )
    from apps.group_agent_api.agent_factory.search_tool import search_candidates

    search_candidates.invoke(
        {"query": "python", "rank_query": "python"},
        config={
            "metadata": {
                "user_id": "u105",
                "group_id": "group_l1_alpha",
                "run_match": "true",
                "prior_candidate_ids": ["u101", "u102"],
            }
        },
    )
    assert captured["excluded_ids"] == ["u105", "u101", "u102"]
