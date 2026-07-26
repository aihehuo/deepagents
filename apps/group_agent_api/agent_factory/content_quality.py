"""Deterministic user-visible reply finalization (REQ-014).

The dialogue model runs before matching and therefore cannot truthfully describe
the later match/invite result.  This module turns the persisted caller profile
and the formal orchestration result into the final reply without another LLM
call.  It never receives or narrates candidate profile fields.
"""

from __future__ import annotations

import re
from typing import Any

from apps.group_agent_api.agent_factory.capability import CapabilityTier, unlocks_network
from apps.group_agent_api.agent_factory.guard import GuardResult, enforce_capability_guard
from apps.group_agent_api.agent_factory.profile_schema import GroupProfile

_NEED_SHAPED_DOING = re.compile(
    r"^\s*(?:找|寻找|寻求|需要|缺少|想找|希望找)"
    r".*(?:负责人|工程师|合伙人|伙伴|人才|专家|顾问|开发者|设计师)\s*$"
)
_OFFER_PREFERENCE_CLAUSE = re.compile(
    r"(?:合作方式|合作模式)\s*(?:可以谈|可谈|均可)"
    r"|(?:全职|兼职)(?:投入)?"
    r"|(?:薪资|待遇)\s*(?:可以谈|可谈|面议)?"
    r"|(?:希望|期望)?(?:能|可以)?(?:快速|尽快)启动"
)


def is_need_shaped_doing(value: str) -> bool:
    """Return True when a supposed doing field is actually phrased as a need."""
    return bool(_NEED_SHAPED_DOING.search(value or ""))


def is_preference_shaped_offer(value: str) -> bool:
    """Return True when an offer contains only preferences, not resources.

    A model may join multiple constraints, for example
    ``合作方式可以谈，希望能快速启动``.  Removing every recognised
    preference clause and punctuation must leave no business capability or
    resource text; mixed values such as ``客户资源，合作方式可以谈`` remain
    valid offers.
    """
    text = (value or "").strip()
    if not text:
        return False
    remainder = _OFFER_PREFERENCE_CLAUSE.sub("", text)
    remainder = re.sub(r"[\s，,、；;。.!！/]+", "", remainder)
    return not remainder


def _value(profile: GroupProfile, field_name: str) -> str:
    field: Any = getattr(profile, field_name, None)
    return str(getattr(field, "value", "") or "").strip()


def profile_confirmation_parts(profile: GroupProfile) -> tuple[str, str, str]:
    """Return truthful, human-readable doing/need/offer confirmations."""
    doing = _value(profile, "doing")
    need = _value(profile, "need")
    offer = _value(profile, "offer")

    doing_part = (
        "正在推进的具体项目还需要补充"
        if not doing or is_need_shaped_doing(doing)
        else f"正在推进「{doing}」"
    )
    need_part = f"目前需要「{need}」" if need else "具体需求还需要补充"
    offer_part = (
        "能提供的具体资源或能力还需要补充"
        if not offer or is_preference_shaped_offer(offer)
        else f"能提供「{offer}」"
    )
    return doing_part, need_part, offer_part


def finalize_user_visible_reply(
    *,
    original_reply: str,
    profile: GroupProfile | None,
    profile_persisted: bool,
    match_status: str,
    candidate_count: int,
    delivery_kind: str | None,
    invite_ok: bool | None,
    network_unlocked: bool,
) -> str:
    """Build a reply consistent with the persisted profile and formal result.

    When persistence failed we retain the agent reply because there is no
    trusted structured profile to confirm.  Once persistence succeeded, the
    deterministic result becomes authoritative and replaces any pre-match
    statement from the dialogue model.
    """
    if not profile_persisted or profile is None:
        return (original_reply or "").strip()

    doing, need, offer = profile_confirmation_parts(profile)
    confirmation = f"我理解并已更新画像：你{doing}；{need}；{offer}。"

    if not network_unlocked:
        next_step = (
            "下一步：你可以继续补充项目目标、具体约束或希望如何合作，"
            "我会帮你把信息整理得更清楚。"
        )
    elif (
        match_status == "matched"
        and candidate_count > 0
        and delivery_kind == "directed"
        and invite_ok is True
    ):
        next_step = (
            f"下一步：我已按这些条件在本群找到 {candidate_count} 位值得进一步聊的人选，"
            "并生成了定向邀请。是否真正匹配还需要你们聊过后确认。"
        )
    elif (
        match_status == "matched"
        and candidate_count > 0
        and delivery_kind == "directed"
    ):
        next_step = (
            f"下一步：本群已有 {candidate_count} 位公开信息与需求有交集的人选，"
            "但定向邀请尚未准备完成，因此还没有发出；可以稍后重试，"
            "是否匹配仍需沟通确认。"
        )
    elif (
        match_status == "matched"
        and candidate_count > 0
        and delivery_kind == "undirected"
    ):
        next_step = (
            f"下一步：本群已有 {candidate_count} 位公开信息与需求有交集的人选；"
            "按你的选择先不点名，可以先用群话题了解彼此，是否匹配仍需沟通确认。"
        )
    elif match_status == "matched" and candidate_count > 0:
        next_step = (
            f"下一步：本群已有 {candidate_count} 位公开信息与需求有交集的人选；"
            "当前没有生成邀请，如需联系，可以再选择是否点名，"
            "是否匹配仍需沟通确认。"
        )
    elif (
        match_status in {"empty", "weak"}
        and delivery_kind == "undirected"
        and invite_ok is True
    ):
        next_step = (
            "下一步：这次暂未找到足够明确的本群人选，我已准备不点名的话题方向；"
            "也可以继续补充筛选条件后再查找。"
        )
    elif match_status in {"empty", "weak"} and delivery_kind == "undirected":
        next_step = (
            "下一步：这次暂未找到足够明确的本群人选，不点名话题也尚未准备完成；"
            "可以稍后重试，或继续补充筛选条件。"
        )
    elif match_status in {"empty", "weak"}:
        next_step = (
            "下一步：这次暂未找到足够明确的本群人选，当前没有生成群话题；"
            "如果需要，可以继续补充筛选条件后再查找。"
        )
    else:
        next_step = (
            "画像已经可以用于下一步；如果需要，我可以按这些条件继续在本群查找"
            "可交流的人选，并按你的意愿决定是否点名邀请。"
        )

    return f"{confirmation}{next_step}"


def finalize_and_guard_user_visible_reply(
    *,
    tier: CapabilityTier,
    caller_group_id: str,
    user_id: str,
    original_reply: str,
    profile: GroupProfile | None,
    profile_persisted: bool,
    match_status: str,
    candidates: list[dict[str, Any]],
    delivery_kind: str | None,
    invite_ok: bool | None,
) -> GuardResult:
    """Finalize from authoritative state, then apply the capability guard.

    The second guard is intentional: the dialogue reply was guarded before
    matching, while this deterministic reply is created afterwards.  If a
    future finalizer accidentally introduces a network promise for a caller
    without network capability, fail closed to a generic non-network reply.
    """
    reply = finalize_user_visible_reply(
        original_reply=original_reply,
        profile=profile,
        profile_persisted=profile_persisted,
        match_status=match_status,
        candidate_count=len(candidates),
        delivery_kind=delivery_kind,
        invite_ok=invite_ok,
        network_unlocked=unlocks_network(tier),
    )
    guarded = enforce_capability_guard(
        tier=tier,
        reply=reply,
        candidates=candidates,
        caller_group_id=caller_group_id,
        user_id=user_id,
    )
    if guarded.blocked and not unlocks_network(tier):
        safe_reply = (
            "我可以继续帮你梳理当前目标、具体需求和可提供的资源，"
            "让这些信息更清楚。"
        )
        assertion = enforce_capability_guard(
            tier=tier,
            reply=safe_reply,
            candidates=[],
            caller_group_id=caller_group_id,
            user_id=user_id,
        )
        if not assertion.ok:
            raise RuntimeError("capability_safe_reply_rejected")
        return GuardResult(
            ok=False,
            tier=tier,
            candidates=[],
            reply=safe_reply,
            violations=guarded.violations,
            blocked=True,
        )
    return guarded
