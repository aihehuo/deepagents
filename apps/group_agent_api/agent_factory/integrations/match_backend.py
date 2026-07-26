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
) -> MatchResult:
    """Run FR-03 match via stub or HTTP.

    HTTP mode requires group_token (GroupAgent JWT). Plaintext group_id is
    only used by the stub; HTTP derives group from token (REQ-050-A).
    """
    mode = (force_mode or integration_mode()).strip().lower()
    if mode == "http":
        try:
            return fetch_group_agent_match(
                query=query,
                group_token=group_token or "",
                excluded_ids=excluded_ids,
                bearer=user_bearer,
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

    return get_match_stub().search(
        query=query,
        group_id=group_id,
        excluded_ids=excluded_ids,
    )
