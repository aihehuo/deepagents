"""Side-effect fence for at-least-once worker execution (REQ-032-FIX1)."""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from apps.group_agent_api.agent_factory.integrations.callback_client import (
    CallbackRunTerminal,
    MouthIngressRejected,
    serialize_callback_body,
)
from apps.group_agent_api.execution.crypto import (
    PayloadCryptoError,
    decrypt_callback_body,
    encrypt_callback_body,
)
from apps.group_agent_api.execution.models import LeaseClaim, PendingCallback
from apps.group_agent_api.execution.redis_store import ExecutionStore, ExecutionStoreError

_logger = logging.getLogger("uvicorn.error")

EmitFn = Callable[[str, dict[str, Any]], Awaitable[bool]]
EnvelopeBuilder = Callable[[int, str, dict[str, Any]], dict[str, Any]]
BodySender = Callable[[dict[str, Any], bytes], Awaitable[bool]]


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
    _emit_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __post_init__(self) -> None:
        record = self.store.get(self.claim.run_id)
        if record is None:
            raise ExecutionStoreError("missing_run")
        self.seq = record.last_callback_seq

    @property
    def aborted(self) -> bool:
        return self._aborted

    @property
    def terminal_delivered(self) -> bool:
        return self._terminal_delivered

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
        include_attempt_metadata: bool = True,
    ) -> bool:
        """Emit callback only when fence still owns the lease.

        Terminal is marked delivered only when send() returns True.

        Args:
            event_type: Callback event name.
            payload: Semantic callback payload.
            send: Transport callable.
            include_attempt_metadata: Defaults to legacy diagnostic enrichment;
                false for strict venue greeting payloads. Lease checks remain.
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
        enriched = dict(payload)
        if include_attempt_metadata:
            enriched.update({"attempt_id": self.claim.attempt_id, "lease_owner_digest": self.claim.lease_owner[:16]})
        try:
            ok = await send(event_type, enriched)
        except Exception:
            # Mouth reject / transport raise before Micro seq commit — rewind so
            # the same seq can be reused (TSD-14 §7.2 / BSD P1).
            if event_type in {"final", "error"}:
                self.seq = max(0, self.seq - 1)
                self.final_callback_ok = None
            raise
        if event_type in {"final", "error"}:
            if ok:
                self._terminal_delivered = True
                self.final_callback_ok = True
            else:
                # Transport False: rewind local seq; Micro did not accept.
                self.seq = max(0, self.seq - 1)
                self.final_callback_ok = False
        return ok

    async def recover_pending(self, *, send_body: BodySender) -> bool:
        """Adopt and retransmit the single durable callback before new work."""
        async with self._emit_lock:
            record = self.store.get(self.claim.run_id)
            if record is None:
                raise ExecutionStoreError("pending_callback_invalid")
            pending = self.store.adopt_pending_callback(
                self.claim,
                attempt_number=record.attempt_count,
            )
            if pending is None:
                self.seq = record.last_callback_seq
                return False
            keys, _ = self.store.routing_crypto_material()
            try:
                body_bytes = decrypt_callback_body(
                    pending.ciphertext,
                    keys=keys,
                    run_id=record.run_id,
                    request_fingerprint=record.request_fingerprint,
                    seq=pending.seq,
                    body_sha256=pending.body_sha256,
                )
                envelope = json.loads(body_bytes.decode("utf-8"))
            except (PayloadCryptoError, UnicodeError, ValueError) as exc:
                raise ExecutionStoreError("pending_callback_invalid") from exc
            if (
                type(envelope) is not dict
                or envelope.get("run_id") != record.run_id
                or envelope.get("seq") != pending.seq
            ):
                raise ExecutionStoreError("pending_callback_invalid")
            try:
                delivered = await send_body(envelope, body_bytes)
            except CallbackRunTerminal:
                self.store.discard_pending_callback(
                    self.claim,
                    seq=pending.seq,
                    body_sha256=pending.body_sha256,
                )
                self._terminal_delivered = True
                raise
            if not delivered:
                self.final_callback_ok = False if envelope.get("event") in {"final", "error"} else None
                return False
            self.seq = self.store.commit_callback_ack(
                self.claim,
                seq=pending.seq,
                body_sha256=pending.body_sha256,
            )
            if envelope.get("event") in {"final", "error"}:
                self._terminal_delivered = True
                self.final_callback_ok = True
            return True

    async def emit_durable(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        build_envelope: EnvelopeBuilder,
        send_body: BodySender,
    ) -> bool:
        """Persist exact bytes, deliver, then CAS seq+pending in one commit."""
        async with self._emit_lock:
            if event_type in {"final", "error"} and self._terminal_delivered:
                return False
            self.check_side_effect(f"callback:{event_type}")
            record = self.store.get(self.claim.run_id)
            if record is None or record.pending_callback is not None:
                raise ExecutionStoreError("pending_callback_invalid")
            seq = record.last_callback_seq + 1
            envelope = build_envelope(seq, event_type, payload)
            body_bytes = serialize_callback_body(envelope)
            body_sha256 = "sha256:" + hashlib.sha256(body_bytes).hexdigest()
            keys, key_version = self.store.routing_crypto_material()
            try:
                encrypted = encrypt_callback_body(
                    body_bytes,
                    key=keys[key_version],
                    key_version=key_version,
                    run_id=record.run_id,
                    request_fingerprint=record.request_fingerprint,
                    seq=seq,
                    body_sha256=body_sha256,
                )
            except (KeyError, PayloadCryptoError) as exc:
                raise ExecutionStoreError("pending_callback_invalid") from exc
            pending = PendingCallback(
                seq=seq,
                body_sha256=body_sha256,
                producer_attempt_id=self.claim.attempt_id,
                producer_attempt_number=record.attempt_count,
                routing_snapshot_hash=envelope.get("routing_snapshot_hash"),
                created_at=datetime.datetime.now(datetime.UTC)
                .isoformat(timespec="microseconds")
                .replace("+00:00", "Z"),
                ciphertext=encrypted,
            )
            self.store.create_pending_callback(
                self.claim,
                attempt_number=record.attempt_count,
                pending=pending,
            )
            try:
                delivered = await send_body(envelope, body_bytes)
            except MouthIngressRejected:
                # Micro rejected before seq commit; the repair loop may replace
                # the body while retaining the same run-global seq.
                self.store.discard_pending_callback(
                    self.claim,
                    seq=seq,
                    body_sha256=body_sha256,
                )
                raise
            except CallbackRunTerminal:
                self.store.discard_pending_callback(
                    self.claim,
                    seq=seq,
                    body_sha256=body_sha256,
                )
                self._terminal_delivered = True
                raise
            if not delivered:
                if event_type in {"final", "error"}:
                    self.final_callback_ok = False
                return False
            self.seq = self.store.commit_callback_ack(
                self.claim,
                seq=seq,
                body_sha256=body_sha256,
            )
            if event_type in {"final", "error"}:
                self._terminal_delivered = True
                self.final_callback_ok = True
            return True
