"""REQ-040 · 群智能体 2.0 单候选人独立 Prompt 文案生成引擎。

针对每个 Candidate 独家输出 4 个关键字段：
- invite_text: 第一人称同群破冰 @ 词（五要素完备）
- match_highlights: 双方 3 大匹配亮点数组
- forward_copy: 第三人称管理员企微预撰写转发词（格式：“Hi [被推荐人]，[发起人] 正在做… 想请教…”）
- quick_connect_copy: 站内极速联系打招呼文案
"""

from __future__ import annotations

import re
from typing import Any

from apps.group_agent_api.agent_factory.profile_schema import GroupProfile


def _extract_profile_field(profile: GroupProfile, field_name: str) -> str:
    field = getattr(profile, field_name, None)
    if field is None:
        return ""
    return str(getattr(field, "value", "") or "").strip()


def _extract_candidate_doing(candidate: dict[str, Any]) -> str:
    doing = candidate.get("doing")
    if isinstance(doing, dict):
        val = str(doing.get("value") or "").strip()
    else:
        val = str(doing or "").strip()
    if not val:
        val = str(candidate.get("headline") or "").strip()
    if not val:
        val = str(candidate.get("display_name") or candidate.get("name") or candidate.get("user_id") or "相关")
    return val


def generate_single_candidate_copy(
    profile: GroupProfile,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Generate 4 single-candidate tailored copy fields for a candidate."""
    from apps.group_agent_api.agent_factory.invite_copy import (
        compact_doing_for_invite,
    )
    raw_c_name = str(
        candidate.get("display_name")
        or candidate.get("name")
        or candidate.get("user_id")
        or "合伙人"
    ).strip()
    # Strip spaces for handles
    c_handle = candidate.get("user_id") if re.search(r"\s", raw_c_name) else raw_c_name
    c_handle = str(c_handle or "合伙人").strip()

    c_doing_raw = _extract_candidate_doing(candidate)
    c_doing_short = compact_doing_for_invite(c_doing_raw, max_chars=24) or "专业方向"

    init_doing_raw = _extract_profile_field(profile, "doing")
    init_need_raw = _extract_profile_field(profile, "need")
    init_offer_raw = _extract_profile_field(profile, "offer")

    init_doing_short = compact_doing_for_invite(init_doing_raw, max_chars=36) or "创业项目"
    init_need_short = compact_doing_for_invite(init_need_raw, max_chars=36) or "资源与对接"
    init_offer_short = compact_doing_for_invite(init_offer_raw, max_chars=36) or "行业积累"

    # 1. Topic & Icebreaker hook
    topic_hook = init_need_short or c_doing_short

    # 2. invite_text: 第一人称同群破冰 @ 词（五要素完备）
    who_line = f"我在做的项目：{init_doing_short}"
    resources_line = f"我能提供的资源或能力：{init_offer_short}"
    topic_line = f"想聊聊：想请教「{topic_hook}」相关经验"
    why_line = f"@{c_handle}，想请教几位一起对齐一下——不一定对得上，供参考，聊聊看就好"
    low_pressure_line = "聊聊就好，不耽误大家太多时间，有意向也可顺便交流"

    invite_text = "\n".join(
        [who_line, resources_line, topic_line, why_line, low_pressure_line]
    )

    # 3. match_highlights: 双方 3 大匹配亮点数组
    match_highlights = [
        f"对方优势：{c_doing_short}",
        f"您的需求：{init_need_short}",
        f"双方契合点：在「{topic_hook}」方向具备强协同优势",
    ]

    # 4. forward_copy: 第三人称管理员企微预撰写转发词
    forward_copy = (
        f"Hi {raw_c_name}，爱合伙平台上有一位创业者正在做{init_doing_short}，"
        f"想请教{topic_hook}相关经验，是否方便为您做微信对接？"
    )

    # 5. quick_connect_copy: 站内极速联系打招呼文案
    quick_connect_copy = (
        f"你好，在爱合伙看到你的{c_doing_short}背景，"
        f"我们正在做{init_doing_short}，希望能交流合作。"
    )

    return {
        "invite_text": invite_text,
        "match_highlights": match_highlights,
        "forward_copy": forward_copy,
        "quick_connect_copy": quick_connect_copy,
    }


def enrich_candidate_with_single_copy(
    candidate: dict[str, Any],
    profile: GroupProfile,
) -> dict[str, Any]:
    """Enrich a candidate dictionary with single-candidate copy fields."""
    enriched = dict(candidate)
    copy_fields = generate_single_candidate_copy(profile, candidate)
    enriched.update(copy_fields)
    return enriched


def enrich_candidates_with_single_copy(
    candidates: list[dict[str, Any]],
    profile: GroupProfile,
) -> list[dict[str, Any]]:
    """Enrich all candidate items in a candidate list."""
    return [enrich_candidate_with_single_copy(c, profile) for c in (candidates or [])]
