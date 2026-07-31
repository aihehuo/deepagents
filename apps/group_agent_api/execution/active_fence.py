"""Active attempt fence with fail-closed durable mode (REQ-032-FIX3)."""

from __future__ import annotations

import contextvars
import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from apps.group_agent_api.execution.models import LeaseClaim
    from apps.group_agent_api.execution.redis_store import ExecutionStore

_logger = logging.getLogger("uvicorn.error")

_ACTIVE: contextvars.ContextVar["ActiveAttemptFence | None"] = contextvars.ContextVar(
    "ga_active_attempt_fence",
    default=None,
)
# FIX3: when True, missing fence is reject (durable worker), not no-op.
_REQUIRE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "ga_require_attempt_fence",
    default=False,
)


class FenceRejectedError(RuntimeError):
    """Raised when a write is rejected because the attempt lost the lease."""

    def __init__(self, code: str = "fence_rejected") -> None:
        self.code = code
        super().__init__(code)


@dataclass
class ActiveAttemptFence:
    """Bind a running attempt identity for write-point checks + CAS commit."""

    store: "ExecutionStore"
    claim: "LeaseClaim"
    conversation_id: str
    cancel_event: threading.Event

    def assert_write(self, action: str) -> None:
        """Soft pre-check (still required); commit uses cas_profile_write_fence."""
        if self.cancel_event.is_set():
            _logger.warning(
                "fence_write_rejected run_id=%s attempt_id=%s fencing=%s action=%s reason=cancelled",
                self.claim.run_id,
                self.claim.attempt_id,
                self.claim.fencing_token,
                action,
            )
            raise FenceRejectedError("fence_cancelled")
        record = self.store.get(self.claim.run_id)
        if record is None:
            raise FenceRejectedError("fence_missing")
        if record.status.value != "running":
            raise FenceRejectedError("fence_not_running")
        if record.current_attempt_id != self.claim.attempt_id:
            raise FenceRejectedError("fence_attempt_mismatch")
        if int(getattr(record, "fencing_token", 0) or 0) != int(self.claim.fencing_token):
            raise FenceRejectedError("fence_epoch_mismatch")
        now = self.store.redis_time()
        if record.lease_expires_at is None or record.lease_expires_at < now:
            raise FenceRejectedError("fence_lease_expired")
        from apps.group_agent_api.execution.crypto import digest_token

        expected = digest_token(self.claim.lease_token)
        if record.lease_token_digest and record.lease_token_digest != expected:
            raise FenceRejectedError("fence_token_mismatch")

    def commit_profile_write(self, *, user_id: str, group_id: str) -> None:
        """Atomic fencing commit at the profile write boundary (FIX3)."""
        self.assert_write("profile_cas")
        kind = self.store.cas_profile_write_fence(
            user_id=user_id,
            group_id=group_id,
            claim=self.claim,
        )
        if kind != "ok":
            _logger.warning(
                "fence_cas_rejected run_id=%s attempt_id=%s fencing=%s kind=%s",
                self.claim.run_id,
                self.claim.attempt_id,
                self.claim.fencing_token,
                kind,
            )
            raise FenceRejectedError(kind)

    def cancel(self) -> None:
        self.cancel_event.set()

    def metadata_fields(self) -> dict[str, str]:
        return {
            "attempt_id": self.claim.attempt_id,
            "fencing_token": str(self.claim.fencing_token),
            "lease_owner": self.claim.lease_owner,
        }


def set_active_fence(fence: ActiveAttemptFence | None, *, require: bool = False) -> tuple[contextvars.Token, contextvars.Token]:
    return _ACTIVE.set(fence), _REQUIRE.set(require)


def get_active_fence() -> ActiveAttemptFence | None:
    return _ACTIVE.get()


def clear_active_fence(
    tokens: tuple[contextvars.Token, contextvars.Token] | contextvars.Token | None = None,
) -> None:
    if isinstance(tokens, tuple):
        _ACTIVE.reset(tokens[0])
        _REQUIRE.reset(tokens[1])
    elif tokens is not None:
        _ACTIVE.reset(tokens)
        _REQUIRE.set(False)
    else:
        _ACTIVE.set(None)
        _REQUIRE.set(False)


def assert_write_allowed(action: str) -> None:
    """Fail closed when durable worker requires a fence (FIX3)."""
    fence = get_active_fence()
    if fence is None:
        if _REQUIRE.get():
            raise FenceRejectedError("fence_required")
        return
    fence.assert_write(action)


def commit_profile_write_allowed(*, user_id: str, group_id: str) -> None:
    """Atomic profile fence commit — fail closed under durable require."""
    fence = get_active_fence()
    if fence is None:
        if _REQUIRE.get():
            raise FenceRejectedError("fence_required")
        return
    fence.commit_profile_write(user_id=user_id, group_id=group_id)
