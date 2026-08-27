"""HTTP client · new_api REQ-050-A group_agent_match (doing-only candidates)."""

from __future__ import annotations

import logging
from typing import Any

import requests
from pydantic import ValidationError

from apps.group_agent_api.agent_factory.disclosure import filter_member_for_visibility
from apps.group_agent_api.agent_factory.grounding_protocol import CandidateV2
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


def _is_wechat_reachable_candidate(raw: dict[str, Any]) -> bool:
    """Require positive WeChat reachability evidence from the match service."""
    bound = raw.get("bound")
    reachable = raw.get("wechat_reachable")
    if bound is False or reachable is False:
        return False
    return bound is True or reachable is True


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
        "same_group": bool(raw["same_group"]) if "same_group" in raw and raw["same_group"] is not None else None,
        "wechat_reachable": bool(raw["wechat_reachable"]) if "wechat_reachable" in raw and raw["wechat_reachable"] is not None else None,
        "app_registered": bool(raw["app_registered"]) if "app_registered" in raw and raw["app_registered"] is not None else None,
        "has_talked_with_agent": bool(raw["has_talked_with_agent"]) if "has_talked_with_agent" in raw and raw["has_talked_with_agent"] is not None else None,
        "is_masked": bool(raw["is_masked"]) if "is_masked" in raw and raw["is_masked"] is not None else None,
    }
    if doing:
        normalized["doing"] = doing
    # Explicitly drop need/offer (doing-only contract)
    visible = filter_member_for_visibility(normalized)
    for k in (
        "match_score",
        "match_confidence",
        "confidence_note",
        "source_group_id",
        "is_reachable",
        "group_info",
        "same_group",
        "wechat_reachable",
        "app_registered",
        "has_talked_with_agent",
        "is_masked",
    ):
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
    contract_version: str | None = None,
    constraints: dict[str, Any] | None = None,
    pool: str | None = None,
) -> MatchResult:
    """POST /users/group_agent_match — requires User JWT; GroupAgent ``g`` optional.

    Micro REQ-028 / full-network agent: omit ``g`` and new_api matches globally
    using the login bearer only. Optional ``g`` still preferred for entry-group
    preference when a leftover group_token is available.

    Supports contract_version="ga-match-v2", constraints, and pool
    (``all_reachable`` | ``agent_profiles``).
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
    if contract_version:
        payload["contract_version"] = contract_version
    if constraints:
        payload["constraints"] = constraints
    pool_norm = (pool or "").strip()
    if pool_norm in {"all_reachable", "agent_profiles"}:
        payload["pool"] = pool_norm
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

    is_v2 = contract_version == "ga-match-v2" or data.get("contract_version") == "ga-match-v2"
    status = str(data.get("status") or "empty")
    if is_v2:
        if status not in {"matched", "empty", "failed"}:
            status = "empty"
    else:
        if status not in {"matched", "weak", "empty"}:
            status = "empty"

    group_id = str(data.get("group_id") or "")
    raw_cands = data.get("candidates") or []
    candidates: list[dict[str, Any]] = []
    if isinstance(raw_cands, list):
        for item in raw_cands[:MAX_CANDIDATES]:
            if isinstance(item, dict):
                # Reject source_group_id == 'global'
                cand_src_group = str(item.get("source_group_id") or item.get("group_id") or "").strip().lower()
                if cand_src_group == "global":
                    continue
                # For v2, check CandidateFact and MatchEvidence
                if is_v2:
                    facts = item.get("facts")
                    evidence = item.get("match_evidence")
                    if not facts or not evidence:
                        continue

                # Product boundary: the group agent may only recommend people
                # whom Aihehuo can reach on WeChat.
                if is_v2:
                    if item.get("wechat_reachable") is not True:
                        continue
                else:
                    # In v1, only drop if explicitly False
                    if item.get("bound") is False or item.get("wechat_reachable") is False:
                        continue
                candidate = _normalize_candidate(item, fallback_group=group_id)
                # Preserve v2 facts/match_evidence/connection
                if is_v2:
                    candidate["facts"] = item.get("facts", [])
                    candidate["match_evidence"] = item.get("match_evidence", [])
                    candidate["connection"] = item.get("connection", {"type": "admin_referral", "available": True})
                    candidate["shared_group"] = item.get("shared_group")
                    from apps.group_agent_api.agent_factory.checks.match_v2_schema import (
                        match_v2_schema_enabled,
                    )

                    if match_v2_schema_enabled():
                        try:
                            # Trust boundary for new_api — every item must satisfy
                            # the exact Micro consumer contract (chk.match_v2_schema).
                            candidate = CandidateV2.model_validate(candidate).model_dump(
                                mode="json",
                                exclude_none=True,
                            )
                        except ValidationError:
                            _logger.warning(
                                "action=match_v2_candidate_rejected "
                                "reason=invalid_grounding_contract"
                            )
                            continue
                    else:
                        _logger.info(
                            "action=module_span check_id=chk.match_v2_schema "
                            "skipped=True reason=yaml_off"
                        )
                candidates.append(candidate)

    reason = str(data.get("reason") or "")
    if status in {"matched", "weak"} and not candidates:
        status = "empty"
        reason = "no_wechat_reachable_candidates"

    return MatchResult(
        status=status,  # type: ignore[arg-type]
        candidates=candidates,
        query=str(data.get("query") or query),
        group_id=group_id,
        reason=reason,
    )
