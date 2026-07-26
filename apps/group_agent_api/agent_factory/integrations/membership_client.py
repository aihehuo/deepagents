"""HTTP client · aihehuomicro REQ-018 membership soft signal."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import requests

from apps.group_agent_api.agent_factory.capability import (
    CapabilityTier,
    resolve_capability,
)
from apps.group_agent_api.agent_factory.integrations.config import (
    http_timeout_s,
    micro_base,
)

_logger = logging.getLogger("uvicorn.error")


@dataclass
class MembershipResult:
    tier: CapabilityTier
    event_id: str | None = None
    reason: str = ""
    raw: dict[str, Any] | None = None
    source: str = "http"


def fetch_membership(
    *,
    unionid: str,
    group_token: str,
    base_url: str | None = None,
    timeout_s: float | None = None,
) -> MembershipResult:
    """POST /group_agent/membership — fail closed to unknown on any non-2xx / errors."""
    uid = (unionid or "").strip()
    token = (group_token or "").strip()
    if not uid or not token:
        return MembershipResult(
            tier=CapabilityTier.unknown,
            reason="blank_unionid_or_token",
            source="http",
        )

    url = f"{(base_url or micro_base()).rstrip('/')}/group_agent/membership"
    try:
        resp = requests.post(
            url,
            json={"unionid": uid, "g": token},
            headers={"Content-Type": "application/json", "User-Agent": "LLM_AGENT"},
            timeout=timeout_s or http_timeout_s(),
        )
        # Fail closed: never parse tier from non-2xx bodies (even if JSON says in_group).
        if resp.status_code < 200 or resp.status_code >= 300:
            _logger.warning(
                "action=membership_http_non_2xx url=%s status=%s → unknown",
                url,
                resp.status_code,
            )
            return MembershipResult(
                tier=CapabilityTier.unknown,
                reason=f"http_{resp.status_code}",
                source="http",
            )

        try:
            data = resp.json() if resp.content else {}
        except Exception:  # noqa: BLE001
            _logger.warning(
                "action=membership_http_bad_json url=%s → unknown", url
            )
            return MembershipResult(
                tier=CapabilityTier.unknown,
                reason="bad_json",
                source="http",
            )
        if not isinstance(data, dict):
            return MembershipResult(
                tier=CapabilityTier.unknown,
                reason="bad_json_shape",
                source="http",
            )
        tier = resolve_capability(data.get("tier"))
        event_id = data.get("event_id")
        return MembershipResult(
            tier=tier,
            event_id=str(event_id) if event_id is not None else None,
            reason=str(data.get("reason") or ""),
            raw=data,
            source="http",
        )
    except Exception as exc:  # noqa: BLE001
        _logger.error(
            "ALERT action=membership_http_error url=%s error=%s → unknown",
            url,
            exc,
        )
        return MembershipResult(
            tier=CapabilityTier.unknown,
            reason=f"exception:{type(exc).__name__}",
            source="http",
        )
