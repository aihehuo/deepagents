"""Match backend facade: stub (default) or HTTP (REQ-007)."""

from __future__ import annotations

import logging
from typing import Any

from apps.group_agent_api.agent_factory.integrations.config import integration_mode
from apps.group_agent_api.agent_factory.integrations.match_client import (
    MatchHttpError,
    fetch_group_agent_match,
)
from apps.group_agent_api.agent_factory.match_stub import (
    MatchResult,
    get_match_stub,
    rerank_candidates_by_detail,
)


_logger = logging.getLogger("uvicorn.error")


def run_match(
    *,
    query: str,
    group_id: str,
    excluded_ids: list[str] | None = None,
    group_token: str | None = None,
    user_bearer: str | None = None,
    force_mode: str | None = None,
    rank_query: str | None = None,
) -> MatchResult:
    """Run FR-03 match via stub or HTTP.

    HTTP mode requires group_token (GroupAgent JWT). Plaintext group_id is
    only used by the stub; HTTP derives group from token (REQ-050-A).

    ``query`` = broad recall text; ``rank_query`` = fine need for re-ranking.
    """
    mode = (force_mode or integration_mode()).strip().lower()
    if mode == "http":
        try:
            result = fetch_group_agent_match(
                query=query,
                group_token=group_token or "",
                excluded_ids=excluded_ids,
                bearer=user_bearer,
                rank_query=rank_query,
            )
        except MatchHttpError as exc:
            _logger.error(
                "ALERT action=match_backend_http_failed error=%s → empty",
                exc,
            )
            return MatchResult(
                status="empty",
                candidates=[],
                query=query,
                group_id=group_id,
                reason=f"http_error:{exc}",
            )
        except Exception as exc:  # noqa: BLE001
            _logger.error(
                "ALERT action=match_backend_exception error=%s → empty",
                exc,
            )
            return MatchResult(
                status="empty",
                candidates=[],
                query=query,
                group_id=group_id,
                reason=f"exception:{type(exc).__name__}",
            )
        # Local re-rank as a safety net when new_api ignores rank_query (old deploy).
        if rank_query and result.candidates:
            reranked = rerank_candidates_by_detail(
                result.candidates, rank_query=rank_query
            )
            return MatchResult(
                status=result.status,
                candidates=reranked,
                query=result.query or query,
                group_id=result.group_id or group_id,
                reason=result.reason,
            )
        return result

    stub_result = get_match_stub().search(
        query=query,
        group_id=group_id,
        excluded_ids=excluded_ids,
    )
    if rank_query and stub_result.candidates:
        return MatchResult(
            status=stub_result.status,
            candidates=rerank_candidates_by_detail(
                stub_result.candidates, rank_query=rank_query
            ),
            query=stub_result.query or query,
            group_id=stub_result.group_id or group_id,
            reason=stub_result.reason,
        )
    return stub_result
