"""Production-shaped contracts for brain neighbor fakes (ear / hand / mouth).

These types mirror what the production brain already consumes today:

* Hand search → ``MatchResult`` (``match_stub.MatchResult``) or ``MatchHttpError``
* Hand profile write → Micro ack ``dict`` or ``ProfileHttpError``
* Ear membership → ``MembershipResult``
* Mouth callback → ``bool`` (accepted / transport failure); PoC also records the
  envelope the brain *would* POST

BSD target semantics (empty ≠ rejected ≠ error) are expressed as *scripts* the
fake can play. Production ``match_backend`` maps HTTP failures via
``disposition_for_http_error``; FakeHand mirrors that mapping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from apps.group_agent_api.agent_factory.integrations.match_client import MatchHttpError
from apps.group_agent_api.agent_factory.integrations.membership_client import (
    MembershipResult,
)
from apps.group_agent_api.agent_factory.integrations.profile_client import (
    ProfileHttpError,
)
from apps.group_agent_api.agent_factory.match_stub import MatchResult

# ---------------------------------------------------------------------------
# Hand · search (production return type of run_match / fetch_group_agent_match)
# ---------------------------------------------------------------------------

HandSearchKind = Literal[
    "ok_matched",
    "ok_weak",
    "ok_empty",
    "rejected",  # contract/config illegal — must NOT be treated as empty
    "error",  # transport / 5xx
]


@dataclass(frozen=True)
class HandSearchScript:
    """One scripted hand-search outcome.

    ``ok_*`` → return a real ``MatchResult``.
    ``rejected`` / ``error`` → raise ``MatchHttpError`` (same type production
    ``fetch_group_agent_match`` raises on HTTP failure).
    """

    kind: HandSearchKind
    result: MatchResult | None = None
    http_error: MatchHttpError | None = None

    def materialize(self) -> MatchResult:
        if self.kind in {"ok_matched", "ok_weak", "ok_empty"}:
            if self.result is None:
                raise AssertionError(f"HandSearchScript {self.kind} missing MatchResult")
            return self.result
        err = self.http_error or MatchHttpError(
            f"hand_{self.kind}",
            status_code=422 if self.kind == "rejected" else 503,
        )
        raise err


def matched_result(
    *,
    query: str,
    group_id: str,
    candidates: list[dict[str, Any]],
    reason: str = "fake_hand_matched",
) -> MatchResult:
    return MatchResult(
        status="matched",
        candidates=list(candidates),
        query=query,
        group_id=group_id,
        reason=reason,
    )


def empty_result(
    *,
    query: str,
    group_id: str,
    reason: str = "fake_hand_empty",
) -> MatchResult:
    return MatchResult(
        status="empty",
        candidates=[],
        query=query,
        group_id=group_id,
        reason=reason,
    )


def production_shaped_candidate(
    *,
    user_id: str,
    group_id: str,
    display_name: str,
    doing_value: str,
) -> dict[str, Any]:
    """Candidate dict shape aligned with ``match_client._normalize_candidate``."""
    return {
        "user_id": user_id,
        "group_id": group_id,
        "source_group_id": group_id,
        "display_name": display_name,
        "profile_url": f"/users/{user_id}",
        "bound": True,
        "is_reachable": True,
        "wechat_reachable": True,
        "match_confidence": "high",
        "doing": {
            "value": doing_value,
            "disclosure": "confirmed_public",
        },
    }


# ---------------------------------------------------------------------------
# Hand · profile write (production return of persist_group_profile)
# ---------------------------------------------------------------------------

ProfileAckStatus = Literal[
    "created",
    "updated",
    "idempotent",
    "stale_ignored",
    "fence_rejected",
]


@dataclass(frozen=True)
class HandProfileScript:
    """ok → Micro ack dict; bad → ``ProfileHttpError`` (production exception)."""

    ok: bool = True
    ack: dict[str, Any] | None = None
    error: ProfileHttpError | None = None

    def materialize(self) -> dict[str, Any]:
        if self.ok:
            if self.ack is None:
                raise AssertionError("HandProfileScript ok missing ack")
            return self.ack
        raise self.error or ProfileHttpError("hand_profile_rejected", status_code=422)


# ---------------------------------------------------------------------------
# Ear · membership
# ---------------------------------------------------------------------------

EarMembershipScript = MembershipResult


# ---------------------------------------------------------------------------
# Mouth · callback
# ---------------------------------------------------------------------------

MouthDisposition = Literal["accepted", "rejected", "transport_fail"]


@dataclass
class MouthCallbackRecord:
    callback_url: str
    envelope: dict[str, Any]
    disposition: MouthDisposition


@dataclass
class MouthScript:
    """What FakeMouth returns to the brain for each callback attempt."""

    disposition: MouthDisposition = "accepted"
    reason_code: str = "unverified_fact_source"
    repairable_by: str = "orchestrator"
    message: str = ""
    hint: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
