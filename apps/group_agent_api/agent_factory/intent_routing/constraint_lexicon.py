"""Controlled constraint lexicon shared by L0 extraction and candidate projection."""

from __future__ import annotations

import re

# Canonical form first, then aliases. Prefer Chinese canonicals for public records.
_LOCATION_GROUPS: tuple[tuple[str, ...], ...] = (
    ("北京", "beijing"),
    ("上海", "shanghai"),
    ("广州", "guangzhou"),
    ("深圳", "shenzhen"),
    ("杭州", "hangzhou"),
    ("成都", "chengdu"),
    ("南京", "nanjing"),
    ("苏州", "suzhou"),
    ("武汉", "wuhan"),
    ("西安", "xian", "xi'an"),
)
_INDUSTRY_GROUPS: tuple[tuple[str, ...], ...] = (
    ("餐饮", "restaurant", "restaurants", "catering"),
    ("人工智能", "ai", "artificial intelligence"),
    ("电商", "ecommerce", "e-commerce"),
    ("教育", "education"),
    ("医疗", "healthcare", "medical"),
    ("文旅", "tourism"),
    ("新能源", "new energy"),
    ("消费", "consumer"),
)
_ROLE_GROUPS: tuple[tuple[str, ...], ...] = (
    ("技术合伙人", "technical cofounder", "tech cofounder"),
    ("运营合伙人", "operations partner"),
    ("产品合伙人", "product partner"),
    ("联合创始人", "cofounder", "co-founder"),
    ("合伙人", "partner"),
    ("创始人", "founder"),
    ("运营", "operators", "operator", "operations"),
)

_KIND_GROUPS = {
    "location": _LOCATION_GROUPS,
    "industry": _INDUSTRY_GROUPS,
    "role": _ROLE_GROUPS,
}

_LATIN_TOKEN = re.compile(r"^[a-z0-9][a-z0-9+.#'_\s-]*$", re.IGNORECASE)
FIELD_EXCLUSION_PREFIX_PATTERN = r"(?:not\s+in|not|不要|排除|别去|别在|不含|除了)"
_HARD_LOCATION_PREFIX = re.compile(
    r"(?:只要|只找|只看|只推荐|只考虑|只搜索|仅限|仅找|仅看|只能|必须)"
    r"[^，。！？,;；\n]{0,16}$",
    flags=re.IGNORECASE,
)


def aliases_for(kind: str) -> tuple[str, ...]:
    """Return every alias for a constraint kind, longest first for matching."""
    groups = _KIND_GROUPS.get(kind, ())
    aliases = [alias for group in groups for alias in group]
    return tuple(sorted(dict.fromkeys(aliases), key=len, reverse=True))


def canonicalize_constraint_token(kind: str, value: str) -> str:
    """Map an alias onto the controlled canonical token for that kind."""
    text = str(value or "").strip()
    if not text:
        return ""
    folded = text.casefold()
    for group in _KIND_GROUPS.get(kind, ()):
        canonical = group[0]
        for alias in group:
            if alias.casefold() == folded:
                return canonical
    return text[:64]


def expand_constraint_token(kind: str, value: str) -> tuple[str, ...]:
    """Expand one frozen token into all controlled aliases used for matching."""
    text = str(value or "").strip()
    if not text:
        return ()
    folded = text.casefold()
    for group in _KIND_GROUPS.get(kind, ()):
        if any(alias.casefold() == folded for alias in group):
            return tuple(dict.fromkeys(group))
    return (text,)


def phrase_in_text(text: str, phrase: str) -> bool:
    """Match a fixed phrase; Latin phrases require whole-word boundaries."""
    if not phrase:
        return False
    if _LATIN_TOKEN.fullmatch(phrase):
        return (
            re.search(
                rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])",
                text,
                flags=re.IGNORECASE,
            )
            is not None
        )
    return phrase in text


def match_phrases(text: str, phrases: tuple[str, ...]) -> tuple[str, ...]:
    """Return phrases present in text without expanding synonyms."""
    return tuple(phrase for phrase in phrases if phrase_in_text(text, phrase))


def explicit_hard_locations(text: str) -> tuple[str, ...]:
    """Extract only locations the user explicitly made an exclusive boundary.

    A city merely appearing in a semantic search query remains a relevance hint.
    Phrases such as ``只找上海`` or ``上海本地`` become hard constraints that
    downstream projection may enforce against the candidate's public city field.
    """
    source = str(text or "")
    selected: list[str] = []
    for alias in aliases_for("location"):
        for match in re.finditer(re.escape(alias), source, flags=re.IGNORECASE):
            before = source[max(0, match.start() - 24) : match.start()]
            after = source[match.end() : match.end() + 12]
            prefixed = _HARD_LOCATION_PREFIX.search(before) is not None
            suffixed = bool(after) and (
                after.startswith(("本地", "当地"))
                or re.match(
                    r"^(?:的)?(?:候选人|人选|人)?(?:就行|即可|才行)",
                    after,
                )
                is not None
            )
            if not (prefixed or suffixed):
                continue
            canonical = canonicalize_constraint_token("location", alias)
            if canonical and canonical not in selected:
                selected.append(canonical)
            break
    return tuple(selected[:10])
