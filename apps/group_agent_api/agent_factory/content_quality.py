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
from apps.group_agent_api.agent_factory.revisit import RevisitHint, build_revisit_opener

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
# Model invents async wait because it cannot see in-request match results.
_PENDING_MATCH_WAIT = re.compile(
    r"请稍候|稍后(?:将)?返回|正在(?:基于|做)?(?:精准)?筛选|"
    r"匹配结果将|匹配流程已启动|尚未返回|后台生成后|"
    r"系统正在.*(?:匹配|筛选)|匹配结果尚未"
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
    revisit_hint: RevisitHint | None = None,
    match_reason: str | None = None,
    quality_gaps: list[str] | None = None,
    candidates: list[dict[str, Any]] | None = None,
) -> str:
    """Build a reply consistent with the persisted profile and formal result.

    When persistence failed we retain the agent reply because there is no
    trusted structured profile to confirm.  Once persistence succeeded, the
    deterministic result becomes authoritative and replaces any pre-match
    statement from the dialogue model.

    REQ-028: when Micro injects ``revisit_hint.has_prior_invite``, the first
    user-visible sentence mentions the prior recommendation and offers
    有回 / 换人换题 / 开新一轮 branches (PRC-01 §9.2).

    REQ-029: thin / unavailable match reasons drive follow-up questions instead
    of promising candidates.
    """
    opener = build_revisit_opener(revisit_hint) if network_unlocked else None

    if not profile_persisted or profile is None:
        original = (original_reply or "").strip()
        if opener and original:
            return f"{opener}\n\n{original}"
        return opener or original

    doing, need, offer = profile_confirmation_parts(profile)
    confirmation = f"我理解并已更新画像：你{doing}；{need}；{offer}。"

    gap = ""
    for item in quality_gaps or []:
        text = str(item or "").strip()
        if text:
            gap = text
            break

    if network_unlocked and match_reason == "profile_too_thin":
        ask = gap or "你在做的具体场景，以及现在最卡的点，再补一句？"
        original = (original_reply or "").strip()
        # Prefer the dialogue model's follow-up when it already asked something
        # concrete — don't drown user answers under a repeated template gap.
        # But never keep hallucinated「请稍候」as the next step.
        if (
            original
            and not original.startswith("我理解并已更新画像")
            and not _PENDING_MATCH_WAIT.search(original)
        ):
            next_step = original
            if ask and ask not in original:
                next_step = f"{original}\n\n（若还没说到：{ask}）"
        else:
            next_step = f"我还想再确认一下再帮你找人：{ask}"
    elif network_unlocked and match_reason == "profile_quality_unavailable":
        next_step = (
            "我先把需求再对齐一下："
            f"{gap or '再具体一点你在做的事和最卡的点？'}"
        )
    elif network_unlocked and match_reason == "profile_thin_degraded":
        thin_note = "你补充的信息还比较粗，结果仅供参考——"
        if not network_unlocked:
            next_step = thin_note
        elif (
            match_status == "matched"
            and candidate_count > 0
            and delivery_kind == "directed"
            and invite_ok is True
        ):
            next_step = (
                f"{thin_note}我已按现有条件在本群找到 {candidate_count} 位值得进一步聊的人选，"
                "并生成了定向邀请。是否真正匹配还需要你们聊过后确认。"
            )
        elif match_status == "matched" and candidate_count > 0:
            next_step = (
                f"{thin_note}本群已有 {candidate_count} 位公开信息与需求有交集的人选；"
                "是否匹配仍需沟通确认。"
            )
        elif match_status in {"empty", "weak"}:
            next_step = (
                f"{thin_note}这次暂未找到足够明确的本群人选，可以继续补充后再查找。"
            )
        else:
            next_step = f"{thin_note}可以继续补充后再查找，或再说「先匹配」。"
    elif not network_unlocked:
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

    unreachable_cands = [
        c for c in (candidates or [])
        if isinstance(c, dict) and c.get("is_reachable") is False
    ]
    if unreachable_cands:
        unreachable_lines = []
        for c in unreachable_cands:
            name = str(
                c.get("display_name") or c.get("name") or c.get("user_id") or "候选人"
            ).strip()
            gi = c.get("group_info")
            gname = ""
            if isinstance(gi, dict):
                gname = str(
                    gi.get("name")
                    or gi.get("group_name")
                    or gi.get("title")
                    or gi.get("id")
                    or gi.get("group_id")
                    or ""
                ).strip()
            elif isinstance(gi, str):
                gname = gi.strip()
            if not gname:
                gname = str(c.get("source_group_id") or c.get("group_id") or "对应群").strip()
            unreachable_lines.append(
                f"候选人【{name}】在【{gname}】，你目前不在该群，可以申请加入【{gname}】"
            )
        unreachable_text = "\n".join(unreachable_lines)
        if unreachable_text not in next_step:
            next_step = f"{next_step}\n\n{unreachable_text}" if next_step else unreachable_text

    original = (original_reply or "").strip()
    is_simple_fallback_stub = any(
        original.startswith(prefix)
        for prefix in ("不能推荐", "无法推荐", "没有人选", "不能直接推荐", "我理解并已更新画像")
    ) or len(original) < 15 or bool(_PENDING_MATCH_WAIT.search(original))

    has_substantive_custom_reply = (
        bool(original)
        and network_unlocked
        and not is_simple_fallback_stub
    )

    # Detect if the model already included a profile confirmation or next-step
    # summary in its reply — avoid double-stacking the template on top.
    _original_lower = original.lower()
    _already_has_confirmation = any(
        marker in _original_lower
        for marker in (
            "已落库", "已更新", "已确认", "三维齐备", "画像已更新",
            "已保存", "profile saved", "updated",
            "已按这些条件", "已按你的",
        )
    )
    _already_has_next_step = any(
        marker in _original_lower
        for marker in (
            "下一步", "接下来", "启动匹配", "帮你匹配", "找到",
            "next step", "i can help",
            "位值得", "位人选", "位候选",
        )
    )

    if has_substantive_custom_reply:
        # Empty match with no invite card: keep the model's clarifying reply.
        # Appending「暂未找到人选」onto an ongoing Q&A contradicts the dialogue.
        if match_status == "empty" and delivery_kind is None:
            body = original
        elif confirmation in original:
            body = original
        elif _already_has_confirmation and _already_has_next_step:
            # Model reply already covers both confirmation and next-step;
            # don't append the template version (which caused double-stacking).
            body = original
        elif next_step == original or (next_step and next_step in original):
            body = f"{original}\n\n{confirmation}" if confirmation else original
        elif original in next_step:
            body = f"{confirmation}{next_step}"
        elif _already_has_confirmation:
            # Model confirmed profile but didn't mention next step — append only next_step.
            body = f"{original}\n\n{next_step}" if next_step else original
        else:
            body = f"{original}\n\n{confirmation}{next_step}"
    else:
        body = f"{confirmation}{next_step}"

    if opener and not has_substantive_custom_reply:
        return f"{opener}\n\n{body}"
    return body


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
    revisit_hint: RevisitHint | None = None,
    match_reason: str | None = None,
    quality_gaps: list[str] | None = None,
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
        revisit_hint=revisit_hint,
        match_reason=match_reason,
        quality_gaps=quality_gaps,
        candidates=candidates,
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
