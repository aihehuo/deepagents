"""Side-effect fence for at-least-once worker execution (REQ-032-FIX1)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from apps.group_agent_api.execution.models import LeaseClaim
from apps.group_agent_api.execution.redis_store import ExecutionStore, ExecutionStoreError

_logger = logging.getLogger("uvicorn.error")

EmitFn = Callable[[str, dict[str, Any]], Awaitable[bool]]


class LeaseLostError(RuntimeError):
    """Raised when a fenced attempt lost ownership mid-execution."""

    def __init__(self, code: str = "lease_lost") -> None:
        self.code = code
        super().__init__(code)


@dataclass
class SideEffectFence:
    """Gate callbacks / profile / invite side effects on live lease ownership."""

    store: ExecutionStore
    claim: LeaseClaim
    conversation_id: str
    abort_on_heartbeat_failures: int = 3
    _heartbeat_failures: int = 0
    _aborted: bool = False
    _terminal_delivered: bool = False
    seq: int = 0
    final_callback_ok: bool | None = None
    _cancel_callbacks: list[Callable[[], None]] = field(default_factory=list)

    @property
    def aborted(self) -> bool:
        return self._aborted

    def on_abort(self, callback: Callable[[], None]) -> None:
        self._cancel_callbacks.append(callback)

    def _fire_abort(self) -> None:
        self._aborted = True
        for cb in self._cancel_callbacks:
            try:
                cb()
            except Exception:  # noqa: BLE001
                pass

    def renew_or_abort(self) -> bool:
        """Renew lease; only abort after consecutive failures reach threshold."""
        if self._aborted:
            return False
        try:
            self.store.renew_lease(self.claim, conversation_id=self.conversation_id)
            self._heartbeat_failures = 0
            return True
        except ExecutionStoreError:
            self._heartbeat_failures += 1
            _logger.warning(
                "lease_renew_failed run_id=%s attempt_id=%s failures=%d",
                self.claim.run_id,
                self.claim.attempt_id,
                self._heartbeat_failures,
            )
            if self._heartbeat_failures >= self.abort_on_heartbeat_failures:
                self._fire_abort()
                return False
            # FIX1: keep trying until threshold — do not exit heartbeat loop early.
            return True

    def assert_owner(self) -> bool:
        """True when this claim still owns a non-expired running lease."""
        if self._aborted:
            return False
        record = self.store.get(self.claim.run_id)
        if record is None:
            self._fire_abort()
            return False
        if record.status.value != "running":
            self._fire_abort()
            return False
        if record.current_attempt_id != self.claim.attempt_id:
            self._fire_abort()
            return False
        now = self.store.redis_time()
        if record.lease_expires_at is None or record.lease_expires_at < now:
            self._fire_abort()
            return False
        return True

    def check_side_effect(self, action: str = "side_effect") -> None:
        """Raise LeaseLostError if ownership lost — call before profile/invite/etc."""
        if not self.assert_owner():
            _logger.warning(
                "fence_blocked_side_effect run_id=%s attempt_id=%s action=%s",
                self.claim.run_id,
                self.claim.attempt_id,
                action,
            )
            raise LeaseLostError("lease_lost")

    async def emit(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        send: EmitFn,
    ) -> bool:
        """Emit callback only when fence still owns the lease.

        Terminal is marked delivered only when send() returns True.
        """
        if event_type in {"final", "error"} and self._terminal_delivered:
            return False
        if not self.assert_owner():
            _logger.warning(
                "fence_blocked_side_effect run_id=%s attempt_id=%s event=%s",
                self.claim.run_id,
                self.claim.attempt_id,
                event_type,
            )
            return False
        self.seq += 1
        enriched = {
            **payload,
            "attempt_id": self.claim.attempt_id,
            "lease_owner_digest": self.claim.lease_owner[:16],
        }
        ok = await send(event_type, enriched)
        if event_type in {"final", "error"}:
            if ok:
                self._terminal_delivered = True
                self.final_callback_ok = True
            else:
                self.final_callback_ok = False
        return ok
