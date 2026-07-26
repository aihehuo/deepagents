"""REQ-005 slice 2a tests: capability gate, match stub, disclosure, cross-group."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
from apps.group_agent_api.agent_factory.capability import (
    CapabilityTier,
    resolve_capability,
    unlocks_network,
)
from apps.group_agent_api.agent_factory.disclosure import (
    assert_visible_fields_public_only,
    filter_member_for_visibility,
)
from apps.group_agent_api.agent_factory.guard import enforce_capability_guard
from apps.group_agent_api.agent_factory.match_stub import (
    MatchStub,
    PoolMember,
    build_query_from_profile,
    get_match_stub,
    set_match_stub,
)
from apps.group_agent_api.agent_factory.profile_schema import profile_from_flat
from apps.group_agent_api.agent_factory.profile_store import save_profile
from apps.group_agent_api.app.endpoints import chat as chat_ep
from apps.group_agent_api.app.endpoints import match as match_ep
from apps.group_agent_api.app.models import ChatRequest, MatchRequest
from apps.group_agent_api.app.state import AppState
from langchain_core.messages import AIMessage, HumanMessage


# ---------------------------------------------------------------------------
# Capability · 单一集中定义
# ---------------------------------------------------------------------------


def test_resolve_capability_three_tiers() -> None:
    assert resolve_capability("in_group") is CapabilityTier.in_group
    assert resolve_capability("not_in_group") is CapabilityTier.not_in_group
    assert resolve_capability("unknown") is CapabilityTier.unknown
    assert resolve_capability(True) is CapabilityTier.in_group
    assert resolve_capability(False) is CapabilityTier.not_in_group
    assert resolve_capability(None) is CapabilityTier.unknown
    assert resolve_capability("garbage") is CapabilityTier.unknown
    assert unlocks_network(CapabilityTier.in_group)
    assert not unlocks_network(CapabilityTier.not_in_group)
    assert not unlocks_network(CapabilityTier.unknown)


# ---------------------------------------------------------------------------
# Guard · 三态零人脉
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tier", [CapabilityTier.not_in_group, CapabilityTier.unknown])
def test_guard_blocks_candidates_when_not_in_group(
    tier: CapabilityTier, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.ERROR):
        result = enforce_capability_guard(
            tier=tier,
            reply="群里有人值得认识 @周然",
            candidates=[
                {
                    "user_id": "x",
                    "group_id": "mock_g1",
                    "source_group_id": "mock_g1",
                    "display_name": "周然",
                    "doing": {
                        "value": "固件",
                        "disclosure": "confirmed_public",
                    },
                }
            ],
            caller_group_id="mock_g1",
            user_id="u1",
        )
    assert result.blocked
    assert result.candidates == []
    assert "candidates_present_without_in_group" in result.violations
    assert "[已拦截]" in result.reply
    assert any("capability_guard_blocked" in r.message for r in caplog.records)


def test_guard_allows_empty_when_not_in_group() -> None:
    result = enforce_capability_guard(
        tier=CapabilityTier.not_in_group,
        reply="先聊聊你在做什么",
        candidates=[],
        caller_group_id="mock_g1",
        user_id="u1",
    )
    assert result.ok
    assert not result.blocked
    assert result.candidates == []


def test_guard_blocks_cross_group_candidates() -> None:
    result = enforce_capability_guard(
        tier=CapabilityTier.in_group,
        reply="ok",
        candidates=[
            {
                "user_id": "spy",
                "group_id": "mock_g2",
                "source_group_id": "mock_g2",
                "display_name": "他群",
                "doing": {"value": "芯片", "disclosure": "confirmed_public"},
            }
        ],
        caller_group_id="mock_g1",
        user_id="u1",
    )
    assert result.blocked
    assert result.candidates == []
    assert any(v.startswith("cross_group:") for v in result.violations)


def test_guard_blocks_disclosure_leak() -> None:
    result = enforce_capability_guard(
        tier=CapabilityTier.in_group,
        reply="ok",
        candidates=[
            {
                "user_id": "m1",
                "group_id": "mock_g1",
                "source_group_id": "mock_g1",
                "display_name": "X",
                "need": {"value": "秘密缺口", "disclosure": "match_only"},
            }
        ],
        caller_group_id="mock_g1",
        user_id="u1",
    )
    assert result.blocked
    assert result.candidates == []
    assert any("disclosure_leak" in v for v in result.violations)


# ---------------------------------------------------------------------------
# Disclosure filter
# ---------------------------------------------------------------------------


def test_disclosure_strips_non_public_and_sensitive() -> None:
    raw = {
        "user_id": "m1",
        "group_id": "mock_g1",
        "display_name": "周然",
        "phone": "13800000000",
        "wechat": "wx_secret",
        "doing": {"value": "固件", "disclosure": "confirmed_public"},
        "need": {"value": "找搭档", "disclosure": "match_only"},
        "offer": {"value": "推断资金", "disclosure": "inferred_unconfirmed"},
        "profile_url": "/u/m1",
    }
    visible = filter_member_for_visibility(raw)
    assert "phone" not in visible
    assert "wechat" not in visible
    assert "doing" in visible
    assert "need" not in visible
    assert "offer" not in visible
    assert assert_visible_fields_public_only(visible) == []


# ---------------------------------------------------------------------------
# Match stub · 可触达池 / ≤3 / SC-05/06 / 越权
# ---------------------------------------------------------------------------


def test_match_stub_reachable_pool_filters_unbound_and_other_group() -> None:
    stub = MatchStub()
    pool = stub.reachable_pool("mock_g1")
    ids = {m.user_id for m in pool}
    assert "mock_zhou" in ids
    assert "mock_li" in ids
    assert "mock_unbound" not in ids
    assert "mock_other_group" not in ids


def test_match_stub_returns_candidates_capped_and_public_only() -> None:
    stub = MatchStub()
    result = stub.search(
        query="智能宠物喂食器 联网 App 固件 工厂",
        group_id="mock_g1",
        excluded_ids=["caller"],
    )
    assert result.status in {"matched", "weak"}
    assert 1 <= len(result.candidates) <= 3
    for c in result.candidates:
        assert c["source_group_id"] == "mock_g1"
        assert assert_visible_fields_public_only(c) == []
        assert "need" not in c or c["need"]["disclosure"] == "confirmed_public"
        # zhou's need is match_only → must not appear
        if c["user_id"] == "mock_zhou":
            assert "need" not in c


def test_match_stub_empty_sc05() -> None:
    stub = MatchStub(
        pool=[
            PoolMember(
                user_id="only_food",
                group_id="mock_g1",
                bound=True,
                display_name="吃货",
                doing={"value": "探店", "disclosure": "confirmed_public"},
                need={"value": "美食", "disclosure": "confirmed_public"},
                offer={"value": "点评", "disclosure": "confirmed_public"},
                keywords=["美食"],
            )
        ]
    )
    result = stub.search(query="航天发动机涡轮叶片", group_id="mock_g1")
    assert result.status == "empty"
    assert result.candidates == []
    assert result.reason == "sc05_no_suitable_match"


def test_match_stub_cannot_pull_other_group_via_query() -> None:
    stub = MatchStub()
    # Even with keywords matching other-group member, pool is group-scoped
    result = stub.search(query="芯片 ASIC 固件", group_id="mock_g1")
    ids = {c["user_id"] for c in result.candidates}
    assert "mock_other_group" not in ids


# ---------------------------------------------------------------------------
# Chat / Match endpoints
# ---------------------------------------------------------------------------


class _FakeCheckpointer:
    def flush(self) -> None:
        return None


class _FakeAgent:
    def __init__(self, *, base_dir: Path, save_on_attempt: int = 1) -> None:
        self.base_dir = base_dir
        self.save_on_attempt = save_on_attempt
        self.invoke_count = 0
        self.checkpointer = _FakeCheckpointer()
        self.messages: list[Any] = []

    async def aget_state(self, _config: dict[str, Any]) -> Any:
        class _S:
            values = {"messages": list(self.messages)}

        return _S()

    async def ainvoke(
        self, payload: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        self.invoke_count += 1
        self.messages.append(payload["messages"][0])
        self.messages.append(AIMessage(content="记下了"))
        if self.invoke_count >= self.save_on_attempt:
            meta = config.get("metadata") or {}
            save_profile(
                self.base_dir,
                profile_from_flat(
                    user_id=str(meta["user_id"]),
                    group_id=str(meta["group_id"]),
                    doing="智能宠物喂食器",
                    need="联网与 App 固件",
                    offer="工厂与供应链",
                ),
            )
        return {"messages": list(self.messages)}


def _state(agent: Any, tmp_path: Path) -> AppState:
    return AppState(
        agent=agent,
        base_dir=tmp_path,
        checkpoints_path=str(tmp_path / "checkpoints.pkl"),
    )


@pytest.fixture(autouse=True)
def _reset_stub() -> None:
    set_match_stub(MatchStub())
    yield
    set_match_stub(MatchStub())


@pytest.mark.asyncio
async def test_chat_not_in_group_skips_match_even_with_profile(tmp_path: Path) -> None:
    agent = _FakeAgent(base_dir=tmp_path, save_on_attempt=1)
    resp = await chat_ep.chat(
        ChatRequest(
            user_id="mock_u1",
            group_id="mock_g1",
            message="做喂食器，缺联网，有工厂",
            membership="not_in_group",
        ),
        _state(agent, tmp_path),
    )
    assert resp.profile_persisted is True
    assert resp.capability == "not_in_group"
    assert resp.match_status == "skipped"
    assert resp.candidates == []


@pytest.mark.asyncio
async def test_chat_unknown_skips_match(tmp_path: Path) -> None:
    agent = _FakeAgent(base_dir=tmp_path, save_on_attempt=1)
    resp = await chat_ep.chat(
        ChatRequest(
            user_id="mock_u1",
            group_id="mock_g1",
            message="做喂食器，缺联网，有工厂",
            membership="unknown",
        ),
        _state(agent, tmp_path),
    )
    assert resp.candidates == []
    assert resp.match_status == "skipped"
    assert resp.capability == "unknown"


@pytest.mark.asyncio
async def test_chat_in_group_returns_gated_candidates(tmp_path: Path) -> None:
    agent = _FakeAgent(base_dir=tmp_path, save_on_attempt=1)
    resp = await chat_ep.chat(
        ChatRequest(
            user_id="mock_u1",
            group_id="mock_g1",
            message="做喂食器，缺联网和固件，有工厂",
            membership="in_group",
        ),
        _state(agent, tmp_path),
    )
    assert resp.capability == "in_group"
    assert resp.profile_persisted is True
    assert resp.match_status in {"matched", "weak"}
    assert 1 <= len(resp.candidates) <= 3
    assert all(c["source_group_id"] == "mock_g1" for c in resp.candidates)
    for c in resp.candidates:
        assert assert_visible_fields_public_only(c) == []


@pytest.mark.asyncio
async def test_chat_metadata_cannot_force_in_group(tmp_path: Path) -> None:
    agent = _FakeAgent(base_dir=tmp_path, save_on_attempt=1)
    resp = await chat_ep.chat(
        ChatRequest(
            user_id="mock_u1",
            group_id="mock_g1",
            message="做喂食器，缺联网，有工厂",
            membership="not_in_group",
            metadata={"membership": "in_group", "capability": "in_group"},
        ),
        _state(agent, tmp_path),
    )
    assert resp.capability == "not_in_group"
    assert resp.candidates == []


@pytest.mark.asyncio
async def test_match_endpoint_cross_group_denied(tmp_path: Path) -> None:
    save_profile(
        tmp_path,
        profile_from_flat(
            user_id="mock_u1",
            group_id="mock_g1",
            doing="芯片",
            need="ASIC",
            offer="资金",
        ),
    )
    # Caller is mock_g1 — even if query matches g2 member, must not return them
    resp = await match_ep.match(
        MatchRequest(
            user_id="mock_u1",
            group_id="mock_g1",
            membership="in_group",
            query="芯片 ASIC",
        ),
        _state(object(), tmp_path),
    )
    ids = {c["user_id"] for c in resp.candidates}
    assert "mock_other_group" not in ids


@pytest.mark.asyncio
async def test_match_endpoint_skipped_when_not_in_group(tmp_path: Path) -> None:
    resp = await match_ep.match(
        MatchRequest(
            user_id="mock_u1",
            group_id="mock_g1",
            membership="not_in_group",
            query="联网固件",
        ),
        _state(object(), tmp_path),
    )
    assert resp.match_status == "skipped"
    assert resp.candidates == []


def test_no_invite_copy_in_2a_modules() -> None:
    """2a must not ship invite-word / common-topic generators."""
    root = Path(__file__).resolve().parents[1] / "apps" / "group_agent_api"
    banned = [
        "aihehuo_search_members",
        "recommended_partners",
        "proposal_statement",
    ]
    for path in root.rglob("*.py"):
        if "__pycache__" in str(path):
            continue
        text = path.read_text(encoding="utf-8")
        for token in banned:
            assert token not in text, f"{path} contains banned 2b/match surface: {token}"
