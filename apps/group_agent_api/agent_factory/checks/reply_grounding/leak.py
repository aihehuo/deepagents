"""Strip / detect implementation jargon that must not reach users.

Prompt bans, L0 checks, and scrub share FORBIDDEN_INTERNAL_FIELDS so coverage
stays aligned. Prefer structured field forms over bare English words like
``ground`` / ``receipt`` that appear in normal business copy.
"""

from __future__ import annotations

import re

# Single source of truth for repair prompts + deterministic leak detection.
FORBIDDEN_INTERNAL_FIELDS: tuple[str, ...] = (
    "check_deny",
    "public_reads",
    "candidate_count",
    "repairable_by",
    "routing_snapshot",
    "routing_snapshot_hash",
    "evidence_digest",
    "plan_step_id",
    "initiator_profile",
    "receipt_id",
    "user_id",
    "unsupported_claim",
    "invented_candidate",
    "internal_context_leak",
)

_FIELD_ALT = "|".join(re.escape(name) for name in FORBIDDEN_INTERNAL_FIELDS)

# Structured / jargon forms only — not bare English "ground" or "receipt".
_INTERNAL_CONTEXT_LEAK = re.compile(
    rf"(?i)(?:"
    rf"ground\s*中|"
    rf"(?:^|\n)\s*ground\s*[=:：]|"
    rf"<check_deny\b|</check_deny>|"
    rf"\b(?:{_FIELD_ALT})\b\s*[=:：]\s*\S+|"
    rf"\b(?:{_FIELD_ALT})\b|"
    rf"\bcandidates\s*[=:：]\s*\["
    rf")"
)


def forbidden_internal_fields_prompt() -> str:
    """Comma-separated ban list for repair / judge prompts."""
    return "、".join(("ground 中", "ground:", *FORBIDDEN_INTERNAL_FIELDS))


def scrub_internal_context_leak(reply: str) -> str:
    """Remove implementation jargon that must never reach the user."""
    text = (reply or "").strip()
    if not text:
        return ""
    parts = re.split(r"(?<=[。！？!?；;\n])", text)
    kept: list[str] = []
    for part in parts:
        if not part or not part.strip():
            continue
        if _INTERNAL_CONTEXT_LEAK.search(part):
            continue
        kept.append(part)
    cleaned = "".join(kept).strip()
    cleaned = _INTERNAL_CONTEXT_LEAK.sub("", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def reply_has_internal_context_leak(reply: str) -> bool:
    return bool(_INTERNAL_CONTEXT_LEAK.search(reply or ""))
