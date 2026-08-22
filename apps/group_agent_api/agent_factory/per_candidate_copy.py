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
    return val


def _clean_action_prefix(text: str) -> str:
    s = (text or "").strip()
    s = re.sub(r"^(?:正在做|做)\s*", "", s)
    return s


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
    c_doing_short = compact_doing_for_invite(c_doing_raw, max_chars=24) if c_doing_raw else ""
    c_doing_clean = _clean_action_prefix(c_doing_short) if c_doing_short else ""

    init_doing_raw = _extract_profile_field(profile, "doing")
    init_need_raw = _extract_profile_field(profile, "need")
    init_offer_raw = _extract_profile_field(profile, "offer")

    init_doing_short = compact_doing_for_invite(init_doing_raw, max_chars=36) or "创业项目"
    init_doing_clean = _clean_action_prefix(init_doing_short)
    init_need_short = compact_doing_for_invite(init_need_raw, max_chars=36) or "资源与对接"
    init_offer_short = compact_doing_for_invite(init_offer_raw, max_chars=36) or "行业积累"

    # 1. Topic & Icebreaker hook
    topic_hook = init_need_short or (c_doing_clean or "专业经验")

    # 2. invite_text: 第一人称同群破冰 @ 词（五要素完备）
    who_line = f"我在做的项目：{init_doing_clean}"
    resources_line = f"我能提供的资源或能力：{init_offer_short}"
    topic_line = f"想聊聊：想请教「{topic_hook}」相关经验"
    why_line = f"@{c_handle}，想请教几位一起对齐一下——不一定对得上，供参考，聊聊看就好"
    low_pressure_line = "聊聊就好，不耽误大家太多时间，有意向也可顺便交流"

    invite_text = "\n".join(
        [who_line, resources_line, topic_line, why_line, low_pressure_line]
    )

    # 3. match_highlights: 严格基于 match_evidence 映射，无证据不造句
    cand_dir_text = c_doing_clean if c_doing_clean and c_doing_clean != "相关" else "现有资料未说明"
    init_need_text = init_need_short if init_need_short else "现有资料未说明"

    ev_list = candidate.get("match_evidence") or []
    if isinstance(ev_list, list) and ev_list:
        match_highlights = []
        for ev in ev_list[:3]:
            if isinstance(ev, dict):
                summary = str(ev.get("summary") or "").strip()
                if summary:
                    match_highlights.append(summary)
            elif isinstance(ev, str) and ev.strip():
                match_highlights.append(ev.strip())
        if not match_highlights:
            match_highlights = [
                f"对方方向：{cand_dir_text}",
                f"您的需求：{init_need_text}",
                "交流重点：基于双方已有公开方向探讨合作契合点",
            ]
    else:
        match_highlights = [
            f"对方方向：{cand_dir_text}",
            f"您的需求：{init_need_text}",
            "交流重点：基于双方已有公开方向探讨合作契合点",
        ]

    # 4. forward_copy: 第三人称管理员企微预撰写转发词（无“做做”叠字）
    applicant_doing_text = init_doing_clean if init_doing_clean else "相关项目"
    forward_topic = topic_hook if topic_hook else "相关方向"
    forward_copy = (
        f"Hi {raw_c_name}，爱合伙平台上有一位创业者正在做{applicant_doing_text}，"
        f"想请教{forward_topic}相关经验，是否方便为您做微信对接？"
    )

    # 5. quick_connect_copy: 站内极速联系打招呼文案（无“做做”叠字）
    cand_doing_text = c_doing_clean if c_doing_clean != "相关" else "公开资料"
    quick_connect_copy = (
        f"你好，在爱合伙看到你的{cand_doing_text}背景，"
        f"我们正在做{applicant_doing_text}，希望能交流合作。"
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
