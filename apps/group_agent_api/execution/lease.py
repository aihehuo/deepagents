"""Lease helpers re-exported for worker clarity (REQ-032-FIX1)."""

from __future__ import annotations

from apps.group_agent_api.execution.models import LeaseClaim
from apps.group_agent_api.execution.redis_store import ClaimOutcome, ExecutionStore

__all__ = ["LeaseClaim", "ExecutionStore", "ClaimOutcome", "claim_lease", "renew_lease"]


def claim_lease(
    store: ExecutionStore,
    *,
    run_id: str,
    conversation_id: str,
    owner: str,
) -> ClaimOutcome:
    """Claim a worker lease for ``run_id`` (queued → running only)."""
    return store.claim_lease(run_id=run_id, conversation_id=conversation_id, owner=owner)


def renew_lease(
    store: ExecutionStore,
    claim: LeaseClaim,
    *,
    conversation_id: str,
) -> float:
    """Renew an owned lease; returns new expiry (Redis time)."""
    return store.renew_lease(claim, conversation_id=conversation_id)
