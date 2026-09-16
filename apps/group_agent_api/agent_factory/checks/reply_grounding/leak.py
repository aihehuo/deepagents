"""Strip / detect implementation jargon that must not reach users."""

from __future__ import annotations

import re

_INTERNAL_CONTEXT_LEAK = re.compile(
    r"(?i)(?:\bground\b|ground\s*中|<check_deny\b|</check_deny>|\bcheck_deny\b|"
    r"\breceipts?\b|\bpublic_reads\b|\bcandidate_count\b|\brepairable_by\b|"
    r"\brouting_snapshot(?:_hash)?\b|\bevidence_digest\b|\bplan_step_id\b|"
    r"\bunsupported_claim\b|\binvented_candidate\b)"
)


def scrub_internal_context_leak(reply: str) -> str:
    """Remove implementation jargon that must never reach the user."""
    text = (reply or "").strip()
    if not text:
        return ""
    parts = re.split(r"(?<=[。！？!?；;\n])", text)
    kept: list[str] = []
    for part in parts:
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
