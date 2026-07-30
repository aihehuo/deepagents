"""Strip phone / WeChat / QQ contact handles from user-visible text.

Contacts may appear inside free-text bios (ES description → doing.value) even when
dedicated contact keys are already banned. Always scrub before surfacing candidates.
"""

from __future__ import annotations

import re
from typing import Any

# Labeled handles + mainland mobile numbers + QQ.
_CONTACT_PATTERN = re.compile(
    r"(?:微信|微信号|薇信|v信|V信|vx|VX|wx|WeChat|weixin)[:：\s]*[A-Za-z0-9_-]{4,}"
    r"|(?:电话|手机号?|联系方式?|联系电话)[:：\s]*\+?\d[\d\s\-()]{6,}"
    r"|(?:\+?86[-\s]?)?1[3-9]\d[\d\s-]{8,12}"
    r"|(?:QQ|qq)[:：\s]*\d{5,12}"
    r"|(?:加我微信|私聊详谈|详谈请加)",
    re.IGNORECASE,
)

# Entire display name that is clearly a WeChat-style handle, not a person name.
_HANDLE_ONLY_NAME = re.compile(
    r"^(?:wx|weixin|wechat)[_-]?[A-Za-z0-9_-]{3,}$",
    re.IGNORECASE,
)


def scrub_contact_text(text: str | None) -> str:
    """Remove contact handles from prose; collapse leftover whitespace."""
    raw = (text or "").strip()
    if not raw:
        return ""
    cleaned = _CONTACT_PATTERN.sub(" ", raw)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"[，,]{2,}", "，", cleaned)
    return cleaned.strip(" ，,;；")


def scrub_display_name(name: str | None, *, fallback_user_id: str = "") -> str:
    """Scrub contacts from a display name; replace handle-only names."""
    cleaned = scrub_contact_text(name)
    if not cleaned or _HANDLE_ONLY_NAME.match(cleaned):
        uid = (fallback_user_id or "").strip()
        return f"用户{uid}" if uid else "群友"
    return cleaned


def scrub_candidate_contacts(candidate: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy with contact handles stripped from visible fields."""
    out = dict(candidate)
    uid = str(out.get("user_id") or "").strip()
    if "display_name" in out or "name" in out:
        raw_name = out.get("display_name") or out.get("name") or ""
        out["display_name"] = scrub_display_name(str(raw_name), fallback_user_id=uid)
        if "name" in out:
            out["name"] = out["display_name"]
    for dim in ("doing", "need", "offer"):
        field = out.get(dim)
        if isinstance(field, dict) and "value" in field:
            scrubbed = scrub_contact_text(str(field.get("value") or ""))
            if scrubbed:
                out[dim] = {**field, "value": scrubbed}
            else:
                out.pop(dim, None)
        elif isinstance(field, str):
            scrubbed = scrub_contact_text(field)
            if scrubbed:
                out[dim] = scrubbed
            else:
                out.pop(dim, None)
    for key in ("reason_summary", "worth_meeting", "worthMeeting", "description", "bio"):
        if key in out and out[key]:
            scrubbed = scrub_contact_text(str(out[key]))
            if scrubbed:
                out[key] = scrubbed
            else:
                out.pop(key, None)
    return out
