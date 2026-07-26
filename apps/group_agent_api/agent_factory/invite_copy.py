"""REQ-006 slice 2b · FR-04 共同话题 + FR-05/05B 邀请词生成。

只消费 2a 已双闸过滤的候选（confirmed_public）。
两类交付物互斥：定向邀请词 vs 不点名话题词。
五要素 / 零@ 用后处理断言，缺则告警重试，不靠模型自觉。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from apps.group_agent_api.agent_factory.disclosure import (
    filter_member_for_visibility,
)
from apps.group_agent_api.agent_factory.guard import _AT_PATTERN
from apps.group_agent_api.agent_factory.match_stub import MAX_CANDIDATES
from apps.group_agent_api.agent_factory.profile_schema import (
    DisclosureLevel,
    GroupProfile,
)

_logger = logging.getLogger("uvicorn.error")

DeliveryKind = Literal["directed", "undirected"]
MatchStatusLike = Literal["matched", "weak", "empty", "skipped"]

_PARTNERSHIP_BAN = re.compile(r"(合伙|股份|股权|当合伙人|一起创业搭班子)")
_PARTNERSHIP_ALLOW = ("不谈合伙", "不提合伙", "不是谈合伙")


def _has_partnership_language(text: str) -> bool:
    cleaned = text or ""
    for phrase in _PARTNERSHIP_ALLOW:
        cleaned = cleaned.replace(phrase, "")
    return bool(_PARTNERSHIP_BAN.search(cleaned))
_VAGUE_TOPIC_BAN = re.compile(r"^(想认识一下|交流交流|互相认识一下)$")
_UNCERTAINTY_HINT = re.compile(r"(不一定|不确定|值得聊|供参考|公开信息|以确认)")

MAX_INVITE_ATTEMPTS = 2


@dataclass
class TopicResult:
    topic: str
    degraded: bool = False  # True if fell back to「想请教 X」


@dataclass
class InviteResult:
    kind: DeliveryKind
    text: str
    topic: str
    match_status: str
    willing_to_at: bool
    mentioned_user_ids: list[str] = field(default_factory=list)
    elements: dict[str, str] | None = None
    honest_note: str | None = None
    ok: bool = True
    violations: list[str] = field(default_factory=list)
    assert_attempts: int = 0


def decide_delivery(
    *,
    match_status: str,
    candidates: list[dict[str, Any]],
    willing_to_at: bool,
) -> DeliveryKind:
    """两类交付物互斥判定（PRC §3 / SC-05/06/07）。"""
    if (
        willing_to_at
        and match_status == "matched"
        and candidates
    ):
        return "directed"
    return "undirected"


def _public_value(field: dict[str, Any] | None) -> str:
    if not field or not isinstance(field, dict):
        return ""
    if field.get("disclosure") not in {
        DisclosureLevel.confirmed_public.value,
        DisclosureLevel.confirmed_public,
    }:
        return ""
    return str(field.get("value") or "").strip()


def _profile_value(profile: GroupProfile, dim: str) -> str:
    """Caller first-person fields: use stored values (self-speech)."""
    field = getattr(profile, dim, None)
    if field is None:
        return ""
    return str(field.value or "").strip()


def derive_common_topic(
    profile: GroupProfile,
    candidates: list[dict[str, Any]],
) -> TopicResult:
    """FR-04 / AI-01 (REQ-007 doing-only):

    Topic = 候选 confirmed_public **doing** ∩ 发起方自己的 need/offer。
    不得再依赖候选 need/offer（new_api 契约不回这些字段）。
    """
    user_need = _profile_value(profile, "need")
    user_offer = _profile_value(profile, "offer")
    user_doing = _profile_value(profile, "doing")

    candidate_doings: list[str] = []
    for c in candidates:
        val = _public_value(c.get("doing"))
        if val:
            candidate_doings.append(val)

    if user_need and candidate_doings:
        hook = candidate_doings[0]
        topic = f"{user_need}这块，想请教下做过「{hook}」的朋友一般怎么落地最稳"
        if _VAGUE_TOPIC_BAN.match(topic.strip()):
            topic = f"想请教「{hook}」相关的选型思路"
        return TopicResult(topic=topic, degraded=False)

    if user_offer and candidate_doings:
        hook = candidate_doings[0]
        topic = f"我这边有{user_offer}，想请教「{hook}」怎么对接最稳"
        return TopicResult(topic=topic, degraded=False)

    if candidate_doings:
        hook = candidate_doings[0]
        return TopicResult(
            topic=f"想请教「{hook}」相关的实践经验",
            degraded=True,
        )

    seed = user_need or user_doing or "当前卡点"
    return TopicResult(
        topic=f"想请教群里做过类似「{seed}」的朋友怎么破局",
        degraded=True,
    )


def _display_name(c: dict[str, Any]) -> str:
    return str(c.get("display_name") or c.get("name") or c.get("user_id") or "").strip()


def _build_directed_elements(
    *,
    profile: GroupProfile,
    candidates: list[dict[str, Any]],
    topic: str,
    force_complete: bool = True,
) -> dict[str, str]:
    """Assemble five elements. Reasons only use confirmed_public + uncertainty."""
    doing = _profile_value(profile, "doing") or "一件正在推进的事"
    offer = _profile_value(profile, "offer") or "一些手头资源"
    who = f"我在做{doing}"
    resources = f"目前手上有：{offer}"
    topic_line = topic

    why_parts: list[str] = []
    for c in candidates[:MAX_CANDIDATES]:
        # Precise mention identity: use the stable, whitespace-free user_id as the @handle.
        # display_name may contain spaces (e.g. "Alice AI Dev"), which _AT_PATTERN truncates
        # at the first space — that ambiguity is exactly what a prefix allowlist must NOT paper over.
        uid = str(c.get("user_id") or "").strip()
        handle = uid or _display_name(c).replace(" ", "")
        doing_pub = _public_value(c.get("doing"))
        hook = doing_pub or "相关公开经验"
        # AI-03: worth a chat + explicit uncertainty; never「很适合当合伙人」
        # REQ-007: candidate narration = doing only
        why_parts.append(
            f"@{handle} 你公开资料里提到「{hook}」，"
            f"基于公开信息值得聊一次以确认是否对得上——不一定合适"
        )
    why = "\n".join(why_parts) if why_parts else ""
    low_pressure = "聊聊就好，不耽误大家太多时间，不谈合伙"

    elements = {
        "who_doing": who,
        "resources": resources,
        "topic": topic_line,
        "why_invite": why,
        "low_pressure": low_pressure,
    }
    if not force_complete:
        # Used only in tests to simulate incomplete drafts
        return elements
    return elements


def render_directed_text(elements: dict[str, str]) -> str:
    return "\n".join(
        [
            elements.get("who_doing", "").strip(),
            elements.get("resources", "").strip(),
            f"想聊聊：{elements.get('topic', '').strip()}",
            elements.get("why_invite", "").strip(),
            elements.get("low_pressure", "").strip(),
        ]
    ).strip()


def render_undirected_text(
    *,
    profile: GroupProfile,
    topic: str,
    honest_note: str | None,
) -> str:
    doing = _profile_value(profile, "doing") or "正在推进的方向"
    parts = [
        f"我在做{doing}。",
        f"想抛个群话题：{topic}",
        "群里有经验的朋友欢迎冒个泡，开放聊聊就好，不点名也行。",
    ]
    if honest_note:
        parts.insert(0, honest_note)
    return "\n".join(parts)


def assert_directed_invite(
    *,
    text: str,
    elements: dict[str, str],
    candidates: list[dict[str, Any]],
) -> list[str]:
    """FR-05 post-assert: five elements, @ ⊆ candidates ≤3, no partnership."""
    violations: list[str] = []
    required = ("who_doing", "resources", "topic", "why_invite", "low_pressure")
    for key in required:
        val = (elements.get(key) or "").strip()
        if not val:
            violations.append(f"missing_element:{key}")
        elif key == "topic" and _VAGUE_TOPIC_BAN.match(val):
            violations.append("vague_topic")

    if _has_partnership_language(text or ""):
        violations.append("partnership_language")

    allowed_names = set()
    for c in candidates:
        dn = _display_name(c)
        if dn:
            # Only the COMPLETE display_name counts as a valid identity credential.
            # No prefix (split()[0]) / no-space collapse: those allow @L1 to match any
            # of "L1 User 2" / "L1 User 3", destroying precise @ ⊆ candidates.
            allowed_names.add(dn)
    allowed_ids = {str(c.get("user_id")) for c in candidates}
    ats = _AT_PATTERN.findall(text or "")
    # pattern captures @Name → strip @
    names = [a[1:] for a in ats]
    if len(names) > MAX_CANDIDATES:
        violations.append(f"too_many_at:{len(names)}")
    for name in names:
        if name not in allowed_names and name not in allowed_ids:
            violations.append(f"at_not_in_candidates:{name}")

    # why_invite must contain uncertainty (AI-03)
    why = elements.get("why_invite") or ""
    if why and not _UNCERTAINTY_HINT.search(why):
        violations.append("missing_uncertainty")

    # Non-public / sensitive values must not appear in generated text
    for c in candidates:
        for dim in ("doing", "need", "offer"):
            field = c.get(dim)
            if not isinstance(field, dict):
                continue
            disclosure = field.get("disclosure")
            value = str(field.get("value") or "").strip()
            if not value:
                continue
            if disclosure in {
                DisclosureLevel.confirmed_public.value,
                DisclosureLevel.confirmed_public,
            }:
                continue
            if value and value in (text or ""):
                violations.append(
                    f"text_leaks_non_public:{c.get('user_id')}:{dim}"
                )
        for bad in ("phone", "mobile", "wechat", "weixin"):
            secret = str(c.get(bad) or "").strip()
            if secret and secret in (text or ""):
                violations.append(f"text_leaks_sensitive:{bad}")

    return violations


def assert_undirected_invite(*, text: str) -> list[str]:
    """FR-05B: zero @ / zero candidate pointing."""
    violations: list[str] = []
    if _AT_PATTERN.search(text or ""):
        violations.append("undirected_contains_at")
    if re.search(r"(推荐对象|候选人)", text or ""):
        violations.append("undirected_points_candidates")
    if _has_partnership_language(text or ""):
        violations.append("partnership_language")
    if _VAGUE_TOPIC_BAN.search((text or "").strip()):
        violations.append("vague_topic")
    return violations


def alert_invite_failure(
    *,
    user_id: str,
    group_id: str,
    kind: str,
    attempt: int,
    violations: list[str],
) -> None:
    _logger.error(
        "ALERT action=invite_assert_failed user_id=%s group_id=%s "
        "kind=%s attempt=%s violations=%s status=will_retry_or_fail",
        user_id,
        group_id,
        kind,
        attempt,
        ",".join(violations) or "none",
    )


def _honest_note_for(match_status: str, *, willing_to_at: bool) -> str | None:
    if match_status == "empty":
        return "暂时没找到特别对得上的人，先给你一段不点名的群话题："
    if match_status == "weak":
        return "关联度一般，供参考——先给你一段不点名的群话题："
    if match_status == "matched" and not willing_to_at:
        return "按你的选择，先给你一段不点名版本："
    return None


def generate_invite_copy(
    *,
    profile: GroupProfile,
    candidates: list[dict[str, Any]],
    match_status: str,
    willing_to_at: bool,
    user_id: str = "",
    group_id: str = "",
    _broken_first_draft: bool = False,  # test hook
) -> InviteResult:
    """Generate directed or undirected copy with assert → alert → retry."""
    # Re-apply disclosure filter so 2b never narrates non-public fields
    safe_candidates = [
        filter_member_for_visibility(c) | {
            "source_group_id": c.get("source_group_id") or c.get("group_id"),
            "match_confidence": c.get("match_confidence"),
            "match_score": c.get("match_score"),
            "confidence_note": c.get("confidence_note"),
        }
        for c in (candidates or [])
    ]
    kind = decide_delivery(
        match_status=match_status,
        candidates=safe_candidates,
        willing_to_at=willing_to_at,
    )
    topic_res = derive_common_topic(
        profile, safe_candidates if kind == "directed" else safe_candidates[:1]
    )
    # For undirected, topic from user need alone is fine even with empty candidates
    if kind == "undirected" and not safe_candidates:
        topic_res = derive_common_topic(profile, [])

    honest = _honest_note_for(match_status, willing_to_at=willing_to_at)

    mentioned: list[str] = []
    elements: dict[str, str] | None = None
    text = ""
    violations: list[str] = []

    for attempt in range(1, MAX_INVITE_ATTEMPTS + 1):
        if kind == "directed":
            force = not (_broken_first_draft and attempt == 1)
            elements = _build_directed_elements(
                profile=profile,
                candidates=safe_candidates,
                topic=topic_res.topic,
                force_complete=force,
            )
            if _broken_first_draft and attempt == 1:
                elements["why_invite"] = ""  # force missing element
            text = render_directed_text(elements)
            mentioned = [
                str(c.get("user_id"))
                for c in safe_candidates[:MAX_CANDIDATES]
                if c.get("user_id")
            ]
            # Pass original candidates so text-leak of non-public values is caught
            violations = assert_directed_invite(
                text=text, elements=elements, candidates=candidates or safe_candidates
            )
        else:
            elements = None
            text = render_undirected_text(
                profile=profile,
                topic=topic_res.topic,
                honest_note=honest,
            )
            if _broken_first_draft and attempt == 1:
                text = text + "\n@偷渡候选人"
            mentioned = []
            violations = assert_undirected_invite(text=text)

        if not violations:
            return InviteResult(
                kind=kind,
                text=text,
                topic=topic_res.topic,
                match_status=match_status,
                willing_to_at=willing_to_at,
                mentioned_user_ids=mentioned,
                elements=elements,
                honest_note=honest,
                ok=True,
                violations=[],
                assert_attempts=attempt,
            )

        alert_invite_failure(
            user_id=user_id or profile.user_id,
            group_id=group_id or profile.group_id,
            kind=kind,
            attempt=attempt,
            violations=violations,
        )

    # Failed after retries — withhold output (not达标)
    return InviteResult(
        kind=kind,
        text="",
        topic=topic_res.topic,
        match_status=match_status,
        willing_to_at=willing_to_at,
        mentioned_user_ids=[],
        elements=None,
        honest_note=honest,
        ok=False,
        violations=violations,
        assert_attempts=MAX_INVITE_ATTEMPTS,
    )
