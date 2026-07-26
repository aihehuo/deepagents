"""REQ-006 slice 2b tests: topic, directed/undirected invite, five-element assert."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from apps.group_agent_api.agent_factory.invite_copy import (
    assert_directed_invite,
    assert_undirected_invite,
    decide_delivery,
    derive_common_topic,
    generate_invite_copy,
)
from apps.group_agent_api.agent_factory.match_stub import MatchStub, set_match_stub
from apps.group_agent_api.agent_factory.profile_schema import profile_from_flat
from apps.group_agent_api.agent_factory.profile_store import save_profile
from apps.group_agent_api.app.endpoints import invite as invite_ep
from apps.group_agent_api.app.models import InviteRequest
from apps.group_agent_api.app.state import AppState


def _profile():
    return profile_from_flat(
        user_id="mock_u1",
        group_id="mock_g1",
        doing="智能宠物喂食器",
        need="联网与 App 固件",
        offer="工厂与供应链",
    )


def _public_candidates():
    stub = MatchStub()
    result = stub.search(
        query="智能宠物喂食器 联网 App 固件 工厂",
        group_id="mock_g1",
        excluded_ids=["mock_u1"],
    )
    return result.status, result.candidates


# ---------------------------------------------------------------------------
# Branching
# ---------------------------------------------------------------------------


def test_decide_delivery_mutual_exclusion() -> None:
    cands = [{"user_id": "a"}]
    assert decide_delivery(
        match_status="matched", candidates=cands, willing_to_at=True
    ) == "directed"
    assert decide_delivery(
        match_status="matched", candidates=cands, willing_to_at=False
    ) == "undirected"
    assert decide_delivery(
        match_status="weak", candidates=cands, willing_to_at=True
    ) == "undirected"
    assert decide_delivery(
        match_status="empty", candidates=[], willing_to_at=True
    ) == "undirected"


def test_derive_topic_not_vague() -> None:
    status, cands = _public_candidates()
    assert status in {"matched", "weak"}
    topic = derive_common_topic(_profile(), cands)
    assert "想认识一下" not in topic.topic
    assert "交流交流" not in topic.topic
    assert len(topic.topic) > 8


# ---------------------------------------------------------------------------
# FR-05 directed
# ---------------------------------------------------------------------------


def test_directed_invite_five_elements_and_at_subset() -> None:
    status, cands = _public_candidates()
    assert status == "matched" or len(cands) >= 1
    # Force matched path
    result = generate_invite_copy(
        profile=_profile(),
        candidates=cands,
        match_status="matched",
        willing_to_at=True,
    )
    assert result.kind == "directed"
    assert result.ok
    assert result.elements is not None
    for key in ("who_doing", "resources", "topic", "why_invite", "low_pressure"):
        assert result.elements[key].strip()
    assert "当合伙人" not in result.text
    assert "股份" not in result.text
    assert "不谈合伙" in result.text  # allowed negation
    assert "不一定" in result.text or "值得聊" in result.text
    for uid in result.mentioned_user_ids:
        assert uid in {c["user_id"] for c in cands}
    assert len(result.mentioned_user_ids) <= 3
    # @ names in text must be candidate display names
    violations = assert_directed_invite(
        text=result.text, elements=result.elements, candidates=cands
    )
    assert violations == []
    assert "@" in result.text


def test_directed_missing_element_blocked_then_retry(
    caplog: pytest.LogCaptureFixture,
) -> None:
    _, cands = _public_candidates()
    with caplog.at_level(logging.ERROR):
        result = generate_invite_copy(
            profile=_profile(),
            candidates=cands,
            match_status="matched",
            willing_to_at=True,
            _broken_first_draft=True,
        )
    assert result.ok
    assert result.assert_attempts == 2
    assert any("invite_assert_failed" in r.message for r in caplog.records)


def test_assert_rejects_partnership_and_foreign_at() -> None:
    elements = {
        "who_doing": "我在做X",
        "resources": "有Y",
        "topic": "想请教固件选型",
        "why_invite": "@外人 他很适合你当合伙人",
        "low_pressure": "聊聊",
    }
    text = "\n".join(elements.values())
    cands = [
        {
            "user_id": "mock_zhou",
            "display_name": "周然",
            "source_group_id": "mock_g1",
            "group_id": "mock_g1",
            "doing": {"value": "固件", "disclosure": "confirmed_public"},
        }
    ]
    v = assert_directed_invite(text=text, elements=elements, candidates=cands)
    assert "partnership_language" in v
    assert any(x.startswith("at_not_in_candidates") for x in v)


# ---------------------------------------------------------------------------
# FR-05B undirected · SC-05/06/07
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "willing", "expect_note"),
    [
        ("empty", True, "暂时没找到"),
        ("weak", True, "关联度一般"),
        ("matched", False, "不点名"),
    ],
)
def test_undirected_branches(status: str, willing: bool, expect_note: str) -> None:
    _, cands = _public_candidates()
    use_cands = cands if status != "empty" else []
    result = generate_invite_copy(
        profile=_profile(),
        candidates=use_cands,
        match_status=status,
        willing_to_at=willing,
    )
    assert result.kind == "undirected"
    assert result.ok
    assert result.mentioned_user_ids == []
    assert "@" not in result.text
    assert expect_note in (result.honest_note or result.text)
    assert assert_undirected_invite(text=result.text) == []


def test_undirected_at_leak_retried(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.ERROR):
        result = generate_invite_copy(
            profile=_profile(),
            candidates=[],
            match_status="empty",
            willing_to_at=True,
            _broken_first_draft=True,
        )
    assert result.ok
    assert "@" not in result.text
    assert result.assert_attempts == 2


# ---------------------------------------------------------------------------
# Disclosure: cannot use match_only in reasons
# ---------------------------------------------------------------------------


def test_topic_and_invite_ignore_non_public_candidate_fields() -> None:
    cands = [
        {
            "user_id": "x1",
            "display_name": "阿秘",
            "group_id": "mock_g1",
            "source_group_id": "mock_g1",
            "doing": {"value": "公开固件", "disclosure": "confirmed_public"},
            "need": {"value": "秘密融资缺口", "disclosure": "match_only"},
            "offer": {"value": "推断有钱", "disclosure": "inferred_unconfirmed"},
        }
    ]
    result = generate_invite_copy(
        profile=_profile(),
        candidates=cands,
        match_status="matched",
        willing_to_at=True,
    )
    assert "秘密融资" not in result.text
    assert "推断有钱" not in result.text
    assert "公开固件" in result.text


# ---------------------------------------------------------------------------
# HTTP endpoint
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_stub() -> None:
    set_match_stub(MatchStub())
    yield
    set_match_stub(MatchStub())


@pytest.mark.asyncio
async def test_invite_endpoint_directed(tmp_path: Path) -> None:
    save_profile(tmp_path, _profile())
    resp = await invite_ep.invite(
        InviteRequest(
            user_id="mock_u1",
            group_id="mock_g1",
            membership="in_group",
            willing_to_at=True,
            query="智能宠物喂食器 联网 固件 工厂",
        ),
        AppState(agent=object(), base_dir=tmp_path, checkpoints_path=""),
    )
    assert resp.delivery_kind == "directed"
    assert resp.invite_ok
    assert resp.invite_text
    assert len(resp.mentioned_user_ids) <= 3
    assert all(
        c["source_group_id"] == "mock_g1" for c in resp.candidates
    )


@pytest.mark.asyncio
async def test_invite_endpoint_sc05_undirected(tmp_path: Path) -> None:
    save_profile(tmp_path, _profile())
    resp = await invite_ep.invite(
        InviteRequest(
            user_id="mock_u1",
            group_id="mock_g1",
            membership="in_group",
            willing_to_at=True,
            match_status="empty",
            candidates=[],
        ),
        AppState(agent=object(), base_dir=tmp_path, checkpoints_path=""),
    )
    assert resp.delivery_kind == "undirected"
    assert "@" not in resp.invite_text
    assert "暂时没找到" in (resp.honest_note or "")
