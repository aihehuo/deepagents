"""SAFE-01/02 披露双闸 · 后处理闸（字段过滤）。

只把 confirmed_public 放进对当前用户可见 / 递交下游的候选面。
match_only / inferred_unconfirmed / 未分级 → 剔除。
"""

from __future__ import annotations

import re
from typing import Any

from apps.group_agent_api.agent_factory.profile_schema import DisclosureLevel


PUBLIC = DisclosureLevel.confirmed_public
_STABLE_CANDIDATE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def stable_user_id_value(raw: Any) -> str | None:
    """Return a native, already-canonical user ID; never coerce or normalize."""
    if not isinstance(raw, str) or not raw or raw != raw.strip():
        return None
    return raw if _STABLE_CANDIDATE_ID.fullmatch(raw) else None


def stable_candidate_user_id(candidate: dict[str, Any]) -> str | None:
    """Return an already-canonical native string ID; never normalize input."""
    return stable_user_id_value(candidate.get("user_id"))


def _field_public(field: dict[str, Any] | None) -> dict[str, Any] | None:
    if not field or not isinstance(field, dict):
        return None
    disclosure = field.get("disclosure")
    value = (field.get("value") or "").strip()
    if not value:
        return None
    if disclosure != PUBLIC.value and disclosure != PUBLIC:
        return None
    return {"value": value, "disclosure": PUBLIC.value}


def filter_member_for_visibility(raw: dict[str, Any]) -> dict[str, Any]:
    """Strip non-public fields from a pool member / candidate payload.

    Always drops phone / wechat / sensitive contact keys if present.
    """
    banned_keys = {
        "phone",
        "mobile",
        "wechat",
        "weixin",
        "wx",
        "email",
        "tel",
        "contact",
    }
    out: dict[str, Any] = {
        "user_id": raw.get("user_id"),
        "group_id": raw.get("group_id"),
        "display_name": raw.get("display_name") or raw.get("name") or "",
        "profile_url": raw.get("profile_url") or "",
        "bound": bool(raw.get("bound", True)),
    }
    for dim in ("doing", "need", "offer"):
        pub = _field_public(raw.get(dim) if isinstance(raw.get(dim), dict) else None)
        if pub is not None:
            out[dim] = pub
    # Drop any leaked sensitive keys
    for k in list(out.keys()):
        if k.lower() in banned_keys:
            out.pop(k, None)
    return out


def public_match_basis(candidate: dict[str, Any]) -> dict[str, str]:
    """Return the auditable basis allowed by the directed-match contract.

    REQ-015 intentionally requires confirmed-public ``doing``. Public offer
    may still be visible in a candidate payload, but it is not sufficient on
    its own to justify a directed recommendation.
    """
    doing = _field_public(
        candidate.get("doing")
        if isinstance(candidate.get("doing"), dict)
        else None
    )
    if doing is None:
        return {}
    return {"doing": str(doing["value"])}


def assert_visible_fields_public_only(candidate: dict[str, Any]) -> list[str]:
    """Return list of violation reasons if non-public fields leak into visible surface."""
    violations: list[str] = []
    for dim in ("doing", "need", "offer"):
        field = candidate.get(dim)
        if field is None:
            continue
        if not isinstance(field, dict):
            violations.append(f"{dim}:not_structured")
            continue
        disclosure = field.get("disclosure")
        if disclosure not in {PUBLIC.value, PUBLIC}:
            violations.append(f"{dim}:disclosure={disclosure}")
    for bad in ("phone", "mobile", "wechat", "weixin", "wx", "email"):
        if bad in candidate and candidate[bad]:
            violations.append(f"sensitive:{bad}")
    return violations
