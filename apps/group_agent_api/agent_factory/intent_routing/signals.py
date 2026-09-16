"""Linear deterministic L0 signals for group-agent business intents."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .constraint_lexicon import (
    aliases_for,
    canonicalize_constraint_token,
    match_phrases,
    phrase_in_text,
)
from .schema import Confidence, IntentConstraintsV1, IntentKind

_SPACE = re.compile(r"\s+")
_SEARCH = (
    "找",
    "搜索",
    "推荐",
    "有没有",
    "帮我查",
    "查一下",
    "看看",
    "查看",
    "浏览",
    "想认识",
    "想加入",
    "加入",
)
_AMBIGUOUS_QUERY = ("有什么合适", "还有合适", "合适的吗")
_FULL_NEGATION = (
    "先别查",
    "不要搜索",
    "暂不推荐",
    "先不找",
    "不想找",
    "不用找",
    "只是聊聊",
    "do not search",
    "don't search",
    "dont search",
    "do not find",
    "don't find",
    "dont find",
    "no searching",
    "just chat",
)
_DISCUSSION = (
    "怎么找",
    "如何找",
    "是不是很难",
    "是否难",
    "应该注意",
    "如果以后",
    "怎么看待",
)
_HELP_ACTION = ("如何", "怎么", "哪里", "为什么", "入口")
_HELP_OBJECT = ("发布", "报名", "认证", "使用", "操作", "设置", "申请", "上传")
_PEOPLE = ("合伙人", "创业者", "创始人", "技术伙伴", "运营伙伴", "人才", "成员", "的人")
_PROJECT = ("项目", "项目方", "已发布项目")
_PARTNERSHIP = ("招募", "招聘", "合作席位", "合作关系", "正在找人", "项目合作")
_OPPORTUNITY = ("idea", "创意", "机会", "赛道", "创业项目")
_COMMUNITY = ("社群", "社区", "创业群", "地域群")
_COMMUNITY_EXCLUDE = ("群智能体", "群聊功能", "聊天群功能")
_DIGEST = ("周报", "文章", "创业内容", "近期内容", "本周内容")
_PROFILE_ACTION = ("修改", "改成", "更新", "保存", "补充", "改为")
_PROFILE_OBJECT = ("我的", "本人", "我")
_PROFILE_FIELD = (
    "画像",
    "简介",
    "所在地",
    "城市",
    "行业",
    "角色",
    "技能",
    "doing",
    "need",
    "offer",
)
_LOCATIONS = aliases_for("location")
_INDUSTRIES = aliases_for("industry")
_ROLES = aliases_for("role")
_PRIORITY_CAPABILITY = re.compile(
    r"优先(?:有|具备)?\s*([^,，。！？;；]{1,32}?(?:经验|能力|资源))"
)
_FIELD_EXCLUSION_PREFIX = re.compile(
    r"(?:not\s+in|not|不要|排除|别去|别在|不含|除了)\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SignalResult:
    """L0 result containing bounded evidence rather than the original message."""

    intent: IntentKind | None
    confidence: Confidence
    signal_ids: tuple[str, ...]
    evidence_spans: tuple[str, ...]
    constraints: IntentConstraintsV1
    negations: tuple[str, ...]
    needs_classifier: bool = False


def normalize_text(message: str) -> str:
    """Normalize width, case, whitespace, and punctuation without regex backtracking."""
    normalized = unicodedata.normalize("NFKC", message or "").lower()
    return _SPACE.sub(" ", normalized).strip()


def _matches(text: str, phrases: tuple[str, ...]) -> tuple[str, ...]:
    """Match fixed phrases; Latin tokens require whole-word boundaries."""
    return match_phrases(text, phrases)


def _canonical_hits(kind: str, text: str, phrases: tuple[str, ...]) -> tuple[str, ...]:
    hits: list[str] = []
    for phrase in _matches(text, phrases):
        canonical = canonicalize_constraint_token(kind, phrase)
        if canonical and canonical not in hits:
            hits.append(canonical)
    return tuple(hits)[:10]


def _field_exclusions(
    kind: str, text: str, phrases: tuple[str, ...]
) -> tuple[str, ...]:
    """Detect explicit field exclusions such as ``not in Beijing`` / ``不要北京``."""
    excluded: list[str] = []
    for phrase in phrases:
        if not phrase_in_text(text, phrase):
            continue
        for match in re.finditer(re.escape(phrase), text, flags=re.IGNORECASE):
            prefix = text[max(0, match.start() - 24) : match.start()]
            if _FIELD_EXCLUSION_PREFIX.search(prefix) is None:
                continue
            canonical = canonicalize_constraint_token(kind, phrase)
            key = f"{kind}:{canonical}"
            if canonical and key not in excluded:
                excluded.append(key)
            break
    return tuple(excluded)[:16]


def _constraints_and_field_negations(
    text: str,
) -> tuple[IntentConstraintsV1, tuple[str, ...]]:
    location_exclusions = _field_exclusions("location", text, _LOCATIONS)
    industry_exclusions = _field_exclusions("industry", text, _INDUSTRIES)
    role_exclusions = _field_exclusions("role", text, _ROLES)
    excluded_locations = {
        item.partition(":")[2] for item in location_exclusions if ":" in item
    }
    excluded_industries = {
        item.partition(":")[2] for item in industry_exclusions if ":" in item
    }
    excluded_roles = {item.partition(":")[2] for item in role_exclusions if ":" in item}
    locations = tuple(
        value
        for value in _canonical_hits("location", text, _LOCATIONS)
        if value not in excluded_locations
    )
    industries = tuple(
        value
        for value in _canonical_hits("industry", text, _INDUSTRIES)
        if value not in excluded_industries
    )
    roles = tuple(
        value
        for value in _canonical_hits("role", text, _ROLES)
        if value not in excluded_roles
    )
    capabilities = tuple(
        dict.fromkeys(match.strip() for match in _PRIORITY_CAPABILITY.findall(text))
    )[:10]
    time_range = next(
        (value for value in ("本周", "近期", "最近", "本月") if value in text), None
    )
    constraints = IntentConstraintsV1(
        location=locations,
        industry=industries,
        role=roles,
        capability=capabilities,
        time_range=time_range,
    )
    negations = location_exclusions + industry_exclusions + role_exclusions
    return constraints, negations


def detect_signals(message: str) -> SignalResult:
    """Resolve deterministic signals in frozen conflict order.

    Runtime is linear in message length because each phrase family is fixed and
    no user-controlled expression or nested-regex alternation is evaluated.

    Args:
        message: Current user message.

    Returns:
        A bounded signal result. `intent=None` delegates to context or L2.
    """
    text = normalize_text(message)
    constraints, field_negations = _constraints_and_field_negations(text)
    profile_hits = _matches(text, _PROFILE_ACTION) + _matches(text, _PROFILE_FIELD)
    if (
        _matches(text, _PROFILE_ACTION)
        and _matches(text, _PROFILE_OBJECT)
        and _matches(text, _PROFILE_FIELD)
    ):
        return SignalResult(
            "profile_update",
            "high",
            ("explicit_write",),
            profile_hits[:16],
            constraints,
            field_negations,
        )

    cancelled = _matches(text, _FULL_NEGATION)
    if cancelled:
        return SignalResult(
            "conversation",
            "high",
            ("explicit_negation",),
            cancelled,
            constraints,
            tuple(dict.fromkeys((*cancelled, *field_negations)))[:16],
        )

    help_hits = _matches(text, _HELP_ACTION) + _matches(text, _HELP_OBJECT)
    if _matches(text, _HELP_ACTION) and _matches(text, _HELP_OBJECT):
        return SignalResult(
            "platform_help",
            "high",
            ("action_object:platform_operation",),
            help_hits[:16],
            constraints,
            field_negations,
        )

    discussion = _matches(text, _DISCUSSION)
    if discussion:
        return SignalResult(
            "conversation",
            "high",
            ("method_discussion",),
            discussion,
            constraints,
            field_negations,
        )

    search = _matches(text, _SEARCH)
    project = _matches(text, _PROJECT)
    partnership = _matches(text, _PARTNERSHIP)
    people = _matches(text, _PEOPLE)
    if search and project and partnership:
        evidence = search + project + partnership
        return SignalResult(
            "project_partnership_search",
            "high",
            ("action_object:project_seat",),
            evidence[:16],
            constraints,
            field_negations,
        )
    if search and people:
        return SignalResult(
            "people_search",
            "high",
            ("action_object:people",),
            (search + people)[:16],
            constraints,
            field_negations,
        )

    community = _matches(text, _COMMUNITY)
    if search and community and not _matches(text, _COMMUNITY_EXCLUDE):
        return SignalResult(
            "community_search",
            "high",
            ("action_object:community",),
            (search + community)[:16],
            constraints,
            field_negations,
        )

    digest = _matches(text, _DIGEST)
    if search and digest:
        return SignalResult(
            "startup_digest",
            "high",
            ("action_object:digest",),
            (search + digest)[:16],
            constraints,
            field_negations,
        )

    opportunity = _matches(text, _OPPORTUNITY)
    if search and (opportunity or project):
        return SignalResult(
            "opportunity_search",
            "high",
            ("action_object:opportunity",),
            (search + opportunity + project)[:16],
            constraints,
            field_negations,
        )

    if search:
        return SignalResult(
            "clarification",
            "low",
            ("search_missing_object",),
            search[:16],
            constraints,
            field_negations,
            True,
        )
    ambiguous = _matches(text, _AMBIGUOUS_QUERY)
    if ambiguous:
        return SignalResult(
            None,
            "low",
            ("ambiguous_object",),
            ambiguous,
            constraints,
            field_negations,
            True,
        )
    business = people + project + partnership + opportunity + community + digest
    if business:
        return SignalResult(
            None,
            "low",
            ("business_object_without_action",),
            business[:16],
            constraints,
            field_negations,
            True,
        )
    # Absence of an L0 keyword match is unknown, not proof of conversation.
    return SignalResult(
        None,
        "low",
        ("no_business_action",),
        (),
        constraints,
        field_negations,
        True,
    )
