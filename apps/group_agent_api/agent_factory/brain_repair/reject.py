"""Consume Micro mouth 422 / disposition=rejected into MouthIngressRejected.

Part of bb.brain.repair (BSD-01 P2 extract). Reject-class tables live in
``ingress_repair.reject_meta`` — this module only parses the HTTP body.
"""

from __future__ import annotations

from typing import Any

_REJECT_DISPOSITIONS = frozenset({"rejected", "rejected_retry"})


class MouthIngressRejected(Exception):
    """Mouth returned disposition=rejected / rejected_retry (BSD-01 P0/P1).

    Not a transport failure — orchestrator may repair and re-emit same seq.
    """

    def __init__(
        self,
        reason_code: str | None = None,
        *,
        status_code: int = 422,
        disposition: str = "rejected",
        repairable_by: str = "llm",
        reject_class: str = "truth",
        fields: list[str] | None = None,
        message: str = "",
        hint: str = "",
        raw_body: dict[str, Any] | None = None,
    ) -> None:
        self.reason_code = (reason_code or "mouth_rejected").strip() or "mouth_rejected"
        self.status_code = status_code
        self.disposition = disposition or "rejected"
        self.repairable_by = repairable_by or "llm"
        self.reject_class = reject_class or "truth"
        self.fields = list(fields or [])
        self.message = message or ""
        self.hint = hint or ""
        self.raw_body = raw_body or {}
        super().__init__(self.reason_code)


def parse_mouth_reject_body(body: dict[str, Any] | None) -> MouthIngressRejected | None:
    """Build MouthIngressRejected from Micro 422 JSON, or None if not a reject."""
    if not isinstance(body, dict):
        return None
    disposition = str(body.get("disposition") or "").strip()
    if disposition not in _REJECT_DISPOSITIONS:
        return None
    nested = body.get("reject") if isinstance(body.get("reject"), dict) else {}
    reason = (
        nested.get("reason_code")
        or body.get("reason_code")
        or body.get("error")
        or "mouth_rejected"
    )
    from apps.group_agent_api.agent_factory.ingress_repair import reject_meta

    meta_class, meta_by, meta_hint = reject_meta(str(reason))
    return MouthIngressRejected(
        str(reason),
        status_code=422,
        disposition=disposition,
        repairable_by=str(nested.get("repairable_by") or meta_by),
        reject_class=str(nested.get("class") or meta_class),
        fields=[str(x) for x in (nested.get("fields") or []) if x],
        message=str(nested.get("message") or body.get("message") or ""),
        hint=str(nested.get("hint") or meta_hint),
        raw_body=body,
    )
