"""REQ-014 deterministic, no-network content-quality regression tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from apps.group_agent_api.agent_factory.agent import SYSTEM_PROMPT, save_group_profile
from apps.group_agent_api.agent_factory.capability import CapabilityTier
from apps.group_agent_api.agent_factory.content_quality import (
    finalize_and_guard_user_visible_reply,
    finalize_user_visible_reply,
    profile_confirmation_parts,
)
from apps.group_agent_api.agent_factory.integrations.membership_client import (
    MembershipResult,
)
from apps.group_agent_api.agent_factory.integrations.principal import SessionPrincipal
from apps.group_agent_api.agent_factory.invite_copy import generate_invite_copy
from apps.group_agent_api.agent_factory.profile_schema import profile_from_flat
from apps.group_agent_api.agent_factory.profile_store import load_profile, save_profile
from apps.group_agent_api.app.async_manager import _execute_core_agent
from apps.group_agent_api.app.endpoints import chat as chat_endpoint
from apps.group_agent_api.app.models import AsyncCallRequest, ChatRequest
from apps.group_agent_api.app.session import TrustedSession
from apps.group_agent_api.app.state import AppState


def _profile(
    *,
    doing: str = "AI / LLM Agent 创业项目",
    need: str = "精通 Python、LangChain 和 PyTorch 的技术负责人",
    offer: str = "业务拓展和客户资源",
):
    return profile_from_flat(
        user_id="u105",
        group_id="group_l1_alpha",
        doing=doing,
        need=need,
        offer=offer,
    )


def _candidates() -> list[dict]:
    return [
        {
            "user_id": "u101",
            "group_id": "group_l1_alpha",
            "source_group_id": "group_l1_alpha",
            "doing": {
                "value": "Python 与 LangChain Agent 开发",
                "disclosure": "confirmed_public",
            },
        }
    ]


def _assert_concrete_profile_confirmation(text: str, profile) -> None:
    parts = profile_confirmation_parts(profile)
    assert all(part in text for part in parts)
    assert any(marker in text for marker in ("下一步", "继续", "已经可以用于"))


def _assert_no_network_surface(text: str) -> None:
    for marker in ("本群", "候选", "推荐", "点名", "邀请", "查找", "@"):
        assert marker not in text


class _Checkpointer:
    def flush(self) -> None:
        return None


class _PersistingAgent:
    """No-network agent that exposes a deliberately conflicting raw reply."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.checkpointer = _Checkpointer()
        self.messages: list[Any] = []

    async def aget_state(self, _config: dict[str, Any]) -> Any:
        class _State:
            values = {"messages": []}

        return _State()

    async def ainvoke(
        self, payload: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        meta = config["metadata"]
        save_profile(
            self.base_dir,
            profile_from_flat(
                user_id=str(meta["user_id"]),
                group_id=str(meta["group_id"]),
                doing="AI Agent 产品",
                need="Python 技术负责人",
                offer="客户资源",
            ),
        )
        reply = AIMessage(content="不能推荐，感谢补充，随时告诉我。")
        self.messages = [payload["messages"][0], reply]
        return {"messages": list(self.messages)}


@pytest.mark.parametrize("original", ["你好！", "感谢补充，随时告诉我。", ""])
def test_profile_reply_replaces_vague_greeting_with_concrete_confirmation(
    original: str,
) -> None:
    profile = _profile()
    reply = finalize_user_visible_reply(
        original_reply=original,
        profile=profile,
        profile_persisted=True,
        match_status="skipped",
        candidate_count=0,
        delivery_kind=None,
        invite_ok=None,
        network_unlocked=True,
    )

    _assert_concrete_profile_confirmation(reply, profile)
    assert reply != original
    assert "本群查找" in reply


@pytest.mark.parametrize(
    "denial",
    ["不能直接推荐具体人选", "无法推荐", "没有人选", "不能推荐"],
)
def test_directed_reply_is_consistent_with_formal_match_result(denial: str) -> None:
    profile = _profile()
    reply = finalize_user_visible_reply(
        original_reply=f"{denial}，请自行联系。",
        profile=profile,
        profile_persisted=True,
        match_status="matched",
        candidate_count=2,
        delivery_kind="directed",
        invite_ok=True,
        network_unlocked=True,
    )

    _assert_concrete_profile_confirmation(reply, profile)
    assert "本群" in reply
    assert "2" in reply
    assert "定向邀请" in reply
    assert "聊过" in reply or "沟通" in reply
    assert denial not in reply


@pytest.mark.parametrize("status", ["empty", "weak"])
def test_no_match_reply_is_honest_and_does_not_claim_candidates(status: str) -> None:
    reply = finalize_user_visible_reply(
        original_reply="已找到三个人。",
        profile=_profile(),
        profile_persisted=True,
        match_status=status,
        candidate_count=0,
        delivery_kind="undirected",
        invite_ok=True,
        network_unlocked=True,
    )

    assert "暂未找到" in reply
    assert "不点名" in reply
    assert "已找到三个人" not in reply


@pytest.mark.parametrize(
    ("delivery_kind", "invite_ok", "required", "forbidden"),
    [
        (None, None, "当前没有生成群话题", "已准备不点名"),
        ("undirected", False, "尚未准备完成", "已准备不点名"),
        ("undirected", True, "已准备不点名", "尚未准备完成"),
    ],
)
@pytest.mark.parametrize("status", ["empty", "weak"])
def test_no_match_reply_describes_actual_topic_delivery(
    status: str,
    delivery_kind: str | None,
    invite_ok: bool | None,
    required: str,
    forbidden: str,
) -> None:
    reply = finalize_user_visible_reply(
        original_reply="已经准备好了。",
        profile=_profile(),
        profile_persisted=True,
        match_status=status,
        candidate_count=0,
        delivery_kind=delivery_kind,
        invite_ok=invite_ok,
        network_unlocked=True,
    )
    assert required in reply
    assert forbidden not in reply
    assert "已经准备好了" not in reply


@pytest.mark.parametrize(
    ("delivery_kind", "invite_ok", "required", "forbidden"),
    [
        ("directed", False, "还没有发出", "先不点名"),
        ("undirected", True, "先不点名", "定向邀请尚未准备完成"),
        (None, None, "当前没有生成邀请", "先不点名"),
    ],
)
def test_matched_reply_describes_actual_invite_outcome(
    delivery_kind: str | None,
    invite_ok: bool | None,
    required: str,
    forbidden: str,
) -> None:
    reply = finalize_user_visible_reply(
        original_reply="不能推荐。",
        profile=_profile(),
        profile_persisted=True,
        match_status="matched",
        candidate_count=2,
        delivery_kind=delivery_kind,
        invite_ok=invite_ok,
        network_unlocked=True,
    )

    assert required in reply
    assert forbidden not in reply
    assert "不能推荐" not in reply


def test_failed_persistence_does_not_claim_a_trusted_profile() -> None:
    original = "我还需要确认你的项目方向。"
    assert (
        finalize_user_visible_reply(
            original_reply=original,
            profile=None,
            profile_persisted=False,
            match_status="skipped",
            candidate_count=0,
            delivery_kind=None,
            invite_ok=None,
            network_unlocked=True,
        )
        == original
    )


@pytest.mark.parametrize(
    "tier", [CapabilityTier.not_in_group, CapabilityTier.unknown]
)
def test_non_network_capability_never_promises_people_or_invites(
    tier: CapabilityTier,
) -> None:
    guarded = finalize_and_guard_user_visible_reply(
        tier=tier,
        caller_group_id="group_l1_alpha",
        user_id="u105",
        original_reply="我已为你推荐候选人 @u101。",
        profile=_profile(),
        profile_persisted=True,
        match_status="skipped",
        candidates=[],
        delivery_kind=None,
        invite_ok=None,
    )
    _assert_no_network_surface(guarded.reply)
    assert guarded.candidates == []


def test_post_finalization_guard_fails_closed_on_future_network_regression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.content_quality."
        "finalize_user_visible_reply",
        lambda **_kwargs: "我已在本群为你推荐候选人 @u101。",
    )
    guarded = finalize_and_guard_user_visible_reply(
        tier=CapabilityTier.not_in_group,
        caller_group_id="group_l1_alpha",
        user_id="u105",
        original_reply="普通回复",
        profile=_profile(),
        profile_persisted=True,
        match_status="skipped",
        candidates=[],
        delivery_kind=None,
        invite_ok=None,
    )

    assert guarded.blocked is True
    assert guarded.violations
    _assert_no_network_surface(guarded.reply)


def test_invite_degrades_need_shaped_doing_and_preference_shaped_offer() -> None:
    result = generate_invite_copy(
        profile=_profile(
            doing="找懂 Python、LangChain 的技术负责人",
            need="全职技术负责人",
            offer="合作方式可以谈",
        ),
        candidates=_candidates(),
        match_status="matched",
        willing_to_at=True,
    )

    assert result.ok
    assert result.kind == "directed"
    assert "我在做找" not in result.text
    assert "目前手上有：合作方式可以谈" not in result.text
    assert "具体项目还没补充清楚" in result.text
    assert "具体资源或能力还没补充清楚" in result.text
    assert "@u101" in result.text


def test_normal_invite_maps_doing_and_offer_naturally() -> None:
    profile = _profile()
    result = generate_invite_copy(
        profile=profile,
        candidates=_candidates(),
        match_status="matched",
        willing_to_at=True,
    )

    assert result.ok
    assert "我在做的项目：" in result.text
    assert profile.doing.value in result.text
    assert "我能提供的资源或能力：" in result.text
    assert profile.offer.value in result.text
    assert "不一定" in result.text or "以确认" in result.text


def test_offer_with_real_resource_plus_cooperation_note_is_not_degraded() -> None:
    profile = _profile(offer="业务拓展和客户资源，合作方式可以谈")
    doing, need, offer = profile_confirmation_parts(profile)
    assert profile.doing.value in doing
    assert profile.need.value in need
    assert profile.offer.value in offer


def test_actual_preference_only_offer_with_urgency_is_degraded() -> None:
    profile = _profile(offer="合作方式可以谈，希望能快速启动")
    _, _, offer = profile_confirmation_parts(profile)
    assert "具体资源或能力还需要补充" in offer
    assert profile.offer.value not in offer


def test_save_tool_rejects_projection_without_overwriting_existing_profile(
    tmp_path: Path,
) -> None:
    original = _profile()
    save_profile(tmp_path, original)

    result = save_group_profile.invoke(
        {
            "doing": "找懂 Python、LangChain 的技术负责人",
            "need": "能独立负责后端架构的全职技术负责人",
            "offer": "合作方式可以谈，希望能快速启动",
            "doing_disclosure": "inferred_unconfirmed",
            "need_disclosure": "inferred_unconfirmed",
            "offer_disclosure": "inferred_unconfirmed",
        },
        config={
            "metadata": {
                "user_id": original.user_id,
                "group_id": original.group_id,
                "base_dir": str(tmp_path),
            }
        },
    )

    assert str(result).startswith("error: semantic_projection:")
    updated = load_profile(tmp_path, original.user_id, original.group_id)
    assert updated is not None
    assert updated.doing.value == original.doing.value
    assert updated.offer.value == original.offer.value
    assert updated.updated_at == original.updated_at


def test_explicit_offer_withdrawal_replaces_old_resource(tmp_path: Path) -> None:
    original = _profile()
    save_profile(tmp_path, original)

    result = save_group_profile.invoke(
        {
            "doing": original.doing.value,
            "need": original.need.value,
            "offer": "暂无可提供资源",
            "doing_disclosure": "confirmed_public",
            "need_disclosure": "confirmed_public",
            "offer_disclosure": "confirmed_public",
        },
        config={
            "metadata": {
                "user_id": original.user_id,
                "group_id": original.group_id,
                "base_dir": str(tmp_path),
            }
        },
    )

    assert str(result).startswith("ok:")
    updated = load_profile(tmp_path, original.user_id, original.group_id)
    assert updated is not None
    assert updated.offer.value == "暂无可提供资源"
    assert original.offer.value not in updated.offer.value


def test_legitimate_project_name_starting_with_find_is_not_rejected(
    tmp_path: Path,
) -> None:
    profile = _profile(doing="找工作招聘平台")
    result = save_group_profile.invoke(
        {
            "doing": profile.doing.value,
            "need": profile.need.value,
            "offer": profile.offer.value,
        },
        config={
            "metadata": {
                "user_id": profile.user_id,
                "group_id": profile.group_id,
                "base_dir": str(tmp_path),
            }
        },
    )

    assert str(result).startswith("ok:")
    stored = load_profile(tmp_path, profile.user_id, profile.group_id)
    assert stored is not None
    assert stored.doing.value == "找工作招聘平台"


def test_prompt_prevents_dimension_overwrite_and_pre_match_claims() -> None:
    required_semantics = (
        "具体的 doing / need / offer",
        "不得把 doing 改写成",
        "不得把 offer 改写成",
        "暂无可提供资源",
        "search_candidates",
    )
    assert all(item in SYSTEM_PROMPT for item in required_semantics)


def test_sync_and_async_final_payloads_use_shared_finalizer() -> None:
    repo = Path(__file__).resolve().parents[1]
    sources = (
        repo / "apps/group_agent_api/app/endpoints/chat.py",
        repo / "apps/group_agent_api/app/async_manager.py",
    )
    for source in sources:
        body = source.read_text(encoding="utf-8")
        assert "finalize_and_guard_user_visible_reply(" in body
        assert "original_reply=guarded.reply" in body
        assert "candidates=guarded.candidates" in body


@pytest.mark.asyncio
async def test_sync_chat_returns_finalized_reply_not_raw_model_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("GROUP_AGENT_INTEGRATION", "stub")
    monkeypatch.setenv("GROUP_AGENT_ENV", "test")
    state = AppState(agent=_PersistingAgent(tmp_path), base_dir=tmp_path)

    response = await chat_endpoint.chat(
        ChatRequest(
            user_id="u105",
            group_id="group_l1_alpha",
            conversation_id="req014_sync",
            message="我在做 AI Agent 产品。",
            membership="in_group",
            run_match=False,
            run_invite=False,
        ),
        state,
    )

    assert "AI Agent 产品" in response.reply
    assert "Python 技术负责人" in response.reply
    assert "客户资源" in response.reply
    assert "不能推荐" not in response.reply
    assert "随时告诉我" not in response.reply


@pytest.mark.asyncio
async def test_async_final_callback_returns_same_finalized_semantics(
    tmp_path: Path,
) -> None:
    state = AppState(agent=_PersistingAgent(tmp_path), base_dir=tmp_path)
    session = TrustedSession(
        principal=SessionPrincipal(
            user_id="u105",
            unionid="union_u105",
            user_token=None,
            source="stub",
        ),
        group_id="group_l1_alpha",
        group_token=None,
        membership=MembershipResult(tier=CapabilityTier.not_in_group, source="stub"),
    )
    req = AsyncCallRequest(
        run_id="req014_async_run",
        idempotency_key="req014_async_idem",
        user_id="u105",
        unionid="union_u105",
        group_id="group_l1_alpha",
        conversation_id="req014_async",
        message="我在做 AI Agent 产品。",
        callback_url="http://localhost:3009/group_agent_callbacks",
        run_match=False,
        run_invite=False,
    )
    final_payload: dict[str, Any] = {}

    async def emit_callback(event_type: str, payload: dict[str, Any]) -> bool:
        if event_type == "final":
            final_payload.update(payload)
        return True

    await _execute_core_agent(
        req=req,
        session=session,
        state=state,
        tid="ga::u105::group_l1_alpha::req014_async",
        emit_callback=emit_callback,
    )

    reply = str(final_payload["reply"])
    assert "AI Agent 产品" in reply
    assert "Python 技术负责人" in reply
    assert "客户资源" in reply
    assert "不能推荐" not in reply
    assert "随时告诉我" not in reply
    _assert_no_network_surface(reply)


@pytest.mark.asyncio
async def test_sync_not_in_group_reply_has_no_network_surface(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("GROUP_AGENT_INTEGRATION", "stub")
    monkeypatch.setenv("GROUP_AGENT_ENV", "test")
    state = AppState(agent=_PersistingAgent(tmp_path), base_dir=tmp_path)

    response = await chat_endpoint.chat(
        ChatRequest(
            user_id="u105",
            group_id="group_l1_alpha",
            conversation_id="req014_sync_not_member",
            message="帮我推荐群里的人。",
            membership="not_in_group",
            run_match=True,
            run_invite=True,
            willing_to_at=True,
        ),
        state,
    )

    _assert_no_network_surface(response.reply)
    assert response.candidates == []
    assert response.invite_text is None
    assert response.delivery_kind is None
