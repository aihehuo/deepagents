"""HTTP client · new_api REQ-050-A group_agent_match (doing-only candidates)."""

from __future__ import annotations

import logging
from typing import Any

import requests

from apps.group_agent_api.agent_factory.disclosure import filter_member_for_visibility
from apps.group_agent_api.agent_factory.integrations.config import (
    http_timeout_s,
    new_api_base,
    new_api_bearer,
)
from apps.group_agent_api.agent_factory.match_stub import (
    MAX_CANDIDATES,
    MatchResult,
)

_logger = logging.getLogger("uvicorn.error")


class MatchHttpError(Exception):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _normalize_candidate(raw: dict[str, Any], *, fallback_group: str) -> dict[str, Any]:
    """Normalize doing-only payload; strip need/offer if somehow present."""
    doing = raw.get("doing")
    if isinstance(doing, str):
        doing = {"value": doing, "disclosure": "confirmed_public"}
    elif not isinstance(doing, dict):
        doing = None

    normalized = {
        # Preserve the raw type/value for the downstream stable-ID gate.
        # Coercing 101/True/" u101 " here would silently canonicalize an
        # invalid external identity into a different accepted identity.
        "user_id": raw.get("user_id"),
        "group_id": str(raw.get("group_id") or fallback_group),
        "source_group_id": str(
            raw.get("source_group_id") or raw.get("group_id") or fallback_group
        ),
        "display_name": raw.get("display_name") or raw.get("name") or "",
        "profile_url": raw.get("profile_url") or "",
        "bound": bool(raw.get("bound", True)),
        "match_score": raw.get("match_score"),
        "match_confidence": raw.get("match_confidence"),
        "confidence_note": raw.get("confidence_note"),
        "is_reachable": raw.get("is_reachable") if raw.get("is_reachable") is not None else True,
        "group_info": raw.get("group_info"),
    }
    if doing:
        normalized["doing"] = doing
    # Explicitly drop need/offer (doing-only contract)
    visible = filter_member_for_visibility(normalized)
    for k in ("match_score", "match_confidence", "confidence_note", "source_group_id", "is_reachable", "group_info"):
        if normalized.get(k) is not None:
            visible[k] = normalized[k]
    return visible


def fetch_group_agent_match(
    *,
    query: str,
    group_token: str | None = None,
    excluded_ids: list[str] | None = None,
    limit: int = MAX_CANDIDATES,
    bearer: str | None = None,
    base_url: str | None = None,
    timeout_s: float | None = None,
    rank_query: str | None = None,
) -> MatchResult:
    """POST /users/group_agent_match — requires User JWT; GroupAgent ``g`` optional.

    Micro REQ-028 / full-network agent: omit ``g`` and new_api matches globally
    using the login bearer only. Optional ``g`` still preferred for entry-group
    preference when a leftover group_token is available.

    Does NOT send plaintext group_id / member_ids (REQ-050-A).
    ``query`` should be broad (recall); optional ``rank_query`` carries fine need.
    """
    token = (group_token or "").strip()
    auth = (bearer if bearer is not None else new_api_bearer()).strip()
    if not auth:
        raise MatchHttpError("missing_user_bearer", status_code=401)

    url = f"{(base_url or new_api_base()).rstrip('/')}/users/group_agent_match"
    payload: dict[str, Any] = {
        "query": query,
        "limit": max(1, min(int(limit or MAX_CANDIDATES), MAX_CANDIDATES)),
        "vector_search": True,
    }
    if token:
        payload["g"] = token
    if excluded_ids:
        payload["excluded_ids"] = list(excluded_ids)
    rq = (rank_query or "").strip()
    if rq:
        payload["rank_query"] = rq[:500]

    resp = requests.post(
        url,
        json=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {auth}",
            "User-Agent": "LLM_AGENT",
        },
        timeout=timeout_s or http_timeout_s(),
    )
    if resp.status_code >= 400:
        _logger.error(
            "ALERT action=match_http_error status=%s body=%s",
            resp.status_code,
            (resp.text or "")[:300],
        )
        raise MatchHttpError(
            f"http_{resp.status_code}", status_code=resp.status_code
        )

    data = resp.json() if resp.content else {}
    if not isinstance(data, dict):
        raise MatchHttpError("invalid_json")

    status = str(data.get("status") or "empty")
    if status not in {"matched", "weak", "empty"}:
        status = "empty"
    group_id = str(data.get("group_id") or "")
    raw_cands = data.get("candidates") or []
    candidates: list[dict[str, Any]] = []
    if isinstance(raw_cands, list):
        for item in raw_cands[:MAX_CANDIDATES]:
            if isinstance(item, dict):
                candidates.append(
                    _normalize_candidate(item, fallback_group=group_id)
                )

    return MatchResult(
        status=status,  # type: ignore[arg-type]
        candidates=candidates,
        query=str(data.get("query") or query),
        group_id=group_id,
        reason=str(data.get("reason") or ""),
    )
