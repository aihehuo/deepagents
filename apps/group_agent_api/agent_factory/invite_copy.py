"""REQ-006 slice 2b · FR-04 共同话题 + FR-05/05B 邀请词生成。

只消费 2a 已双闸过滤的候选（confirmed_public）。
两类交付物互斥：定向邀请词 vs 不点名话题词。
五要素 / 零@ 用后处理断言，缺则告警重试，不靠模型自觉。

---
mod.brain.invite_copy hang points (Brief B1 — document only; YAML not wired):

| check id | hang | gate today |
|---|---|---|
| ``chk.invite_scaffold`` | ``generate_invite_copy`` via ``invite_llm.generate_invite_with_optional_llm`` when ``should_emit_invite_artifact`` | match/candidates/`run_invite` (hard when path runs) |
| ``chk.invite_llm_polish`` | ``invite_llm.generate_invite_with_optional_llm`` | ENV ``GROUP_AGENT_LLM_POLISH`` via ``integrations.config.llm_polish_enabled`` (not YAML) |

Orchestrator callers: ``app/endpoints/chat.py``, ``app/async_manager.py``, ``app/endpoints/invite.py``.
Do not wrap behind ``is_check_enabled`` here until ENV→YAML migration is deliberate.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from apps.group_agent_api.agent_factory.disclosure import (
    filter_member_for_visibility,
    public_match_basis,
    stable_candidate_user_id,
)
from apps.group_agent_api.agent_factory.content_quality import (
    is_need_shaped_doing,
    is_preference_shaped_offer,
)
from apps.group_agent_api.agent_factory.guard import _AT_PATTERN
from apps.group_agent_api.agent_factory.match_stub import MAX_CANDIDATES
from apps.group_agent_api.agent_factory.per_candidate_copy import (
    enrich_candidate_with_single_copy,
)
from apps.group_agent_api.agent_factory.profile_schema import (
    DisclosureLevel,
    GroupProfile,
)

_logger = logging.getLogger("uvicorn.error")

DeliveryKind = Literal["directed", "undirected"]
MatchStatusLike = Literal["matched", "weak", "empty", "skipped"]

def _has_partnership_language(text: str) -> bool:
    """REQ-026: Partnership language ban removed per boss decision. Invites can express partnership intent."""
    return False

_VAGUE_TOPIC_BAN = re.compile(r"^(想认识一下|交流交流|互相认识一下)$")
_UNCERTAINTY_HINT = re.compile(r"(不一定|不确定|值得聊|供参考|公开信息|以确认)")
_FIELD_LINE = re.compile(r"^([^:：]{1,24})[:：]\s*(.+)$")
_INLINE_FIELD_LABELS = re.compile(
    r"(用户名|所在地|所在行业|细分行业|个人目标|具体介绍|合伙需求|教育和工作经历)[:：]\s*"
)
_PHONE_OR_CONTACT = re.compile(
    r"(?:\+?86[-\s]?)?1[3-9]\d[\d\s-]{8,12}"
    r"|(?:微信|wx|WeChat)[:：\s]*[A-Za-z0-9_-]{4,}"
    r"|(?:电话|手机|联系)[:：\s]*\d[\d\s-]{6,}",
    re.IGNORECASE,
)
_SOLICIT_AD = re.compile(
    r"(找合伙人|寻求合伙人|寻找投资人|招募合伙人|招商加盟|加我微信|私聊详谈|详谈请)"
)
_BRACKET_TAG = re.compile(r"【([^】]{2,20})】")

MAX_INVITE_ATTEMPTS = 2
# WeChat group paste: keep invites skimmable (PRC AI-02「一屏读完」)
MAX_HOOK_CHARS = 24
MAX_SELF_CHARS = 48
MAX_TOPIC_CHARS = 48
MAX_INVITE_CHARS = 520


def compact_doing_for_invite(raw: str, *, max_chars: int = MAX_HOOK_CHARS) -> str:
    """Turn profile dumps / ads into a short topical hook — never paste bios or contacts."""
    text = str(raw or "").replace("\\n", "\n").replace("\r", "").strip()
    if not text:
        return ""

    tag = _BRACKET_TAG.search(text)
    tag_hint = tag.group(1).strip() if tag else ""

    fields: dict[str, str] = {}
    for line in text.split("\n"):
        m = _FIELD_LINE.match(line.strip())
        if m:
            fields[m.group(1).strip()] = m.group(2).strip()

    if fields:
        prefer = (
            fields.get("细分行业")
            or fields.get("所在行业")
            or fields.get("具体介绍")
            or fields.get("个人目标")
            or fields.get("合伙需求")
        )
        if prefer:
            text = prefer
        else:
            text = max(fields.values(), key=len)

    if tag_hint and (
        not text
        or _SOLICIT_AD.search(text)
        or _PHONE_OR_CONTACT.search(text)
        or len(text) > max_chars * 2
    ):
        text = tag_hint

    text = _INLINE_FIELD_LABELS.sub("", text)
    text = _PHONE_OR_CONTACT.sub("", text)
    text = _SOLICIT_AD.sub("", text)
    text = re.sub(r"[【】\[\]]", "", text)
    text = re.sub(r"\s+", " ", text).strip(" ，,;；。.|/、")
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip(" ，,;；。.") + "…"
    return text


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
    candidates: list[dict[str, Any]] = field(default_factory=list)


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


def should_emit_invite_artifact(
    *,
    match_status: str,
    match_reason: str | None,
    candidate_count: int,
) -> bool:
    """Whether the chat/async pipeline should attach invite/topic cards.

    Empty match must NOT auto-emit an undirected invite while the dialogue is
    still clarifying — that felt like「过早跳出邀请词」. Invite artifacts are
    only for turns that actually have candidates (matched/weak).
    """
    if match_status in {"skipped", "empty"}:
        return False
    if match_reason in {"profile_too_thin", "profile_quality_unavailable"}:
        return False
    if candidate_count <= 0:
        return False
    return True


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
        val = compact_doing_for_invite(_public_value(c.get("doing")))
        if val:
            candidate_doings.append(val)

    user_need = compact_doing_for_invite(user_need, max_chars=MAX_TOPIC_CHARS)
    user_offer = compact_doing_for_invite(user_offer, max_chars=MAX_HOOK_CHARS)
    user_doing = compact_doing_for_invite(user_doing, max_chars=MAX_HOOK_CHARS)

    if user_need and candidate_doings:
        hook = candidate_doings[0]
        # Prefer initiator need as the group topic — don't paste candidate ads into topic.
        topic = f"想请教「{user_need}」怎么落地"
        if len(topic) > MAX_TOPIC_CHARS:
            topic = f"想请教「{user_need[:18].rstrip('…')}」相关经验"
        if _VAGUE_TOPIC_BAN.match(topic.strip()):
            topic = f"想请教「{hook}」相关的选型思路"
        return TopicResult(topic=topic, degraded=False)

    if user_offer and candidate_doings:
        hook = candidate_doings[0]
        topic = f"我这边有「{user_offer}」，想找人对齐一下"
        if len(topic) > MAX_TOPIC_CHARS:
            topic = f"想聊聊「{hook}」怎么协作"
        return TopicResult(topic=topic, degraded=False)

    if candidate_doings:
        hook = candidate_doings[0]
        topic = f"想请教「{hook}」相关经验"
        return TopicResult(topic=topic, degraded=True)

    # Empty match / undirected: prefer doing (product) over need (often「缺…」).
    if user_doing:
        topic = f"想请教「{user_doing}」怎么做得更稳"
        if len(topic) > MAX_TOPIC_CHARS:
            topic = f"想请教「{user_doing[:16].rstrip('…')}」落地思路"
        return TopicResult(topic=topic, degraded=True)
    if user_need:
        topic = f"想请教「{user_need}」怎么落地"
        if len(topic) > MAX_TOPIC_CHARS:
            topic = f"想请教「{user_need[:16].rstrip('…')}」相关经验"
        return TopicResult(topic=topic, degraded=True)
    return TopicResult(topic="想请教当前卡点的落地思路", degraded=True)


def _display_name(c: dict[str, Any]) -> str:
    return str(c.get("display_name") or c.get("name") or c.get("user_id") or "").strip()


def _at_handle(c: dict[str, Any]) -> str:
    """Prefer a paste-friendly display name; fall back to user_id when name has spaces or is generic."""
    uid = str(c.get("user_id") or "").strip()
    dn = str(c.get("display_name") or c.get("name") or "").strip()
    if dn and dn == f"用户{uid}":
        return uid
    if dn and not re.search(r"\s", dn):
        return dn
    return uid or dn.replace(" ", "")


def _build_directed_elements(
    *,
    profile: GroupProfile,
    candidates: list[dict[str, Any]],
    topic: str,
    force_complete: bool = True,
) -> dict[str, str]:
    """Assemble five elements. Reasons only use confirmed_public + uncertainty."""
    doing_raw = _profile_value(profile, "doing")
    offer_raw = _profile_value(profile, "offer")
    doing = compact_doing_for_invite(doing_raw, max_chars=MAX_SELF_CHARS)
    offer = compact_doing_for_invite(offer_raw, max_chars=MAX_SELF_CHARS)
    who = (
        "我在做的具体项目还没补充清楚"
        if not doing or is_need_shaped_doing(doing_raw)
        else f"我在做的项目：{doing}"
    )
    resources = (
        "我能提供的具体资源或能力还没补充清楚"
        if not offer or is_preference_shaped_offer(offer_raw)
        else f"我能提供的资源或能力：{offer}"
    )
    topic_line = topic

    # Evidence gate: only @ candidates that still have confirmed_public doing.
    # Do NOT paste their doing / ads into the group message (AI-05 + paste UX).
    handles: list[str] = []
    for c in candidates[:MAX_CANDIDATES]:
        if c.get("is_reachable") is False:
            continue
        doing_pub = _public_value(c.get("doing"))
        if not doing_pub:
            continue
        hook = compact_doing_for_invite(doing_pub, max_chars=MAX_HOOK_CHARS)
        if not hook:
            continue
        handles.append(_at_handle(c))

    if handles:
        ats = " ".join(f"@{h}" for h in handles)
        why = (
            f"{ats}，想请教几位一起对齐一下——"
            f"不一定对得上，供参考，聊聊看就好"
        )
    else:
        why = ""
    low_pressure = "聊聊就好，不耽误大家太多时间，有意向也可顺便交流"

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
    doing_raw = _profile_value(profile, "doing")
    doing = compact_doing_for_invite(doing_raw, max_chars=MAX_SELF_CHARS)
    doing_line = (
        "我在做的具体项目还没补充清楚。"
        if not doing or is_need_shaped_doing(doing_raw)
        else f"我在做的项目：{doing}。"
    )
    parts = [
        doing_line,
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
    """FR-05 post-assert: five elements, @ ⊆ candidates ≤3 (REQ-026: no partnership ban)."""
    violations: list[str] = []
    body = text or ""
    if len(body) > MAX_INVITE_CHARS:
        violations.append(f"invite_too_long:{len(body)}")
    if re.search(
        r"(用户名|所在地|所在行业|细分行业|个人目标|具体介绍|合伙需求|教育和工作经历)[:：]",
        body,
    ):
        violations.append("invite_profile_dump")
    if _PHONE_OR_CONTACT.search(body):
        violations.append("invite_leaks_contact")
    if "公开资料" in body:
        violations.append("invite_too_formal_public_cite")

    required = ("who_doing", "resources", "topic", "why_invite", "low_pressure")
    for key in required:
        val = (elements.get(key) or "").strip()
        if not val:
            violations.append(f"missing_element:{key}")
        elif key == "topic" and _VAGUE_TOPIC_BAN.match(val):
            violations.append("vague_topic")

    allowed_names = set()
    for c in candidates:
        dn = _display_name(c)
        if dn:
            allowed_names.add(dn)
        uid = str(c.get("user_id") or "").strip()
        if uid:
            allowed_names.add(f"用户{uid}")
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
    safe_candidates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    accepted_ids: set[str] = set()
    for candidate in candidates or []:
        user_id_value = stable_candidate_user_id(candidate)
        if user_id_value is None:
            _logger.warning("action=invite_candidate_gate violation=missing_candidate_id")
            continue
        if user_id_value in seen_ids:
            _logger.warning(
                "action=invite_candidate_gate violation=duplicate_candidate_id:%s",
                user_id_value,
            )
        seen_ids.add(user_id_value)
        if user_id_value in accepted_ids:
            continue
        if not public_match_basis(candidate):
            continue
        visible = filter_member_for_visibility(candidate)
        visible.update(
            {
                "source_group_id": candidate.get("source_group_id")
                or candidate.get("group_id"),
                "match_confidence": candidate.get("match_confidence"),
                "match_score": candidate.get("match_score"),
                "confidence_note": candidate.get("confidence_note"),
                "facts": candidate.get("facts", []),
                "match_evidence": candidate.get("match_evidence", []),
                "connection": candidate.get("connection", {"type": "admin_referral", "available": True}),
                "shared_group": candidate.get("shared_group"),
                "same_group": candidate.get("same_group", True),
                "wechat_reachable": candidate.get("wechat_reachable", True),
            }
        )
        visible = enrich_candidate_with_single_copy(visible, profile)
        accepted_ids.add(user_id_value)
        safe_candidates.append(visible)

    effective_match_status = (
        "empty"
        if match_status == "matched" and not safe_candidates
        else match_status
    )
    kind = decide_delivery(
        match_status=effective_match_status,
        candidates=safe_candidates,
        willing_to_at=willing_to_at,
    )
    topic_res = derive_common_topic(
        profile, safe_candidates if kind == "directed" else safe_candidates[:1]
    )
    # For undirected, topic from user need alone is fine even with empty candidates
    if kind == "undirected" and not safe_candidates:
        topic_res = derive_common_topic(profile, [])

    honest = _honest_note_for(effective_match_status, willing_to_at=willing_to_at)

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
                match_status=effective_match_status,
                willing_to_at=willing_to_at,
                mentioned_user_ids=mentioned,
                elements=elements,
                honest_note=honest,
                ok=True,
                violations=[],
                assert_attempts=attempt,
                candidates=safe_candidates,
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
        match_status=effective_match_status,
        willing_to_at=willing_to_at,
        mentioned_user_ids=[],
        elements=None,
        honest_note=honest,
        ok=False,
        violations=violations,
        assert_attempts=MAX_INVITE_ATTEMPTS,
        candidates=[],
    )
