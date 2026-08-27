"""BSD-01 P0: match_backend empty ≠ rejected ≠ error."""

from __future__ import annotations

import pytest

from apps.group_agent_api.agent_factory.integrations.match_backend import (
    disposition_for_http_error,
    run_match,
)
from apps.group_agent_api.agent_factory.integrations.match_client import MatchHttpError


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        (422, "rejected"),
        (400, "rejected"),
        (401, "rejected"),
        (403, "rejected"),
        (503, "error"),
        (500, "error"),
        (429, "error"),
        (None, "error"),
    ],
)
def test_disposition_for_http_error(
    status_code: int | None, expected: str
) -> None:
    status, reason = disposition_for_http_error(
        MatchHttpError("boom", status_code=status_code)
    )
    assert status == expected
    assert reason.startswith("http_error:")


def test_run_match_http_422_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(**_kwargs):  # noqa: ANN001
        raise MatchHttpError("http_422", status_code=422)

    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.integrations.match_backend.fetch_group_agent_match",
        _raise,
    )
    monkeypatch.setenv("GROUP_AGENT_INTEGRATION", "http")
    result = run_match(query="q", group_id="g1", force_mode="http")
    assert result.status == "rejected"
    assert result.candidates == []
    assert "http_error:" in result.reason


def test_run_match_http_503_is_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(**_kwargs):  # noqa: ANN001
        raise MatchHttpError("http_503", status_code=503)

    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.integrations.match_backend.fetch_group_agent_match",
        _raise,
    )
    result = run_match(query="q", group_id="g1", force_mode="http")
    assert result.status == "error"
    assert result.candidates == []
