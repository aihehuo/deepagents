"""Create-once routing snapshot construction and cryptographic verification."""

from __future__ import annotations

import datetime
from collections.abc import Mapping

from apps.group_agent_api.execution.models import EncryptedPayload, ExecutionRecord

from .canonical import (
    RoutingCryptoError,
    canonical_hash,
    decrypt_canonical,
    encrypt_canonical,
)
from .contracts import RoutingSnapshotV1
from .registry import REGISTRY_VERSION, registry_entry
from .schema import IntentDecisionV1


def utc_timestamp(now: datetime.datetime | None = None) -> str:
    """Return the contract millisecond-or-second UTC timestamp."""
    value = (now or datetime.datetime.now(datetime.UTC)).astimezone(datetime.UTC)
    if value.microsecond:
        return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")


def build_routing_snapshot(
    *,
    run_id: str,
    decision: IntentDecisionV1,
    created_at: str | None = None,
) -> RoutingSnapshotV1:
    """Build and self-hash the frozen snapshot from server-owned policy state."""
    entry = registry_entry(decision.primary_intent)
    plan = entry.plan if decision.action == "execute" and entry is not None else None
    if decision.action == "execute" and (
        entry is None
        or entry.skill != decision.required_skill
        or decision.query_plan_ref is None
        or decision.query_plan_ref.plan_id != entry.plan.plan_id
    ):
        raise RoutingCryptoError("query_plan_invalid")
    body = {
        "routing_snapshot_version": 1,
        "run_id": run_id,
        "intent_decision": decision.model_dump(mode="json"),
        "query_plan": plan.model_dump(mode="json") if plan is not None else None,
        "policy_version": decision.policy_version,
        "registry_version": REGISTRY_VERSION,
        "created_at": created_at or utc_timestamp(),
    }
    body["canonical_content_hash"] = canonical_hash(body)
    return RoutingSnapshotV1.model_validate(body)


def routing_aad(
    *,
    run_id: str,
    request_fingerprint: str,
    key_version: str,
) -> dict[str, object]:
    """Return the exact frozen routing snapshot AAD object."""
    return {
        "purpose": "group-agent-routing-snapshot",
        "routing_snapshot_version": 1,
        "run_id": run_id,
        "request_fingerprint": request_fingerprint,
        "key_version": key_version,
    }


def encrypt_routing_snapshot(
    snapshot: RoutingSnapshotV1,
    *,
    key: bytes,
    key_version: str,
    request_fingerprint: str,
    nonce: bytes | None = None,
) -> EncryptedPayload:
    """Encrypt a validated snapshot with its run/fingerprint AAD binding."""
    return encrypt_canonical(
        snapshot.model_dump(mode="json"),
        key=key,
        key_version=key_version,
        aad=routing_aad(
            run_id=snapshot.run_id,
            request_fingerprint=request_fingerprint,
            key_version=key_version,
        ),
        nonce=nonce,
    )


def decrypt_routing_snapshot(
    record: ExecutionRecord,
    *,
    keys: Mapping[str, bytes],
) -> RoutingSnapshotV1:
    """Decrypt and verify every record/snapshot/policy/registry binding."""
    envelope = record.routing_snapshot_ciphertext
    if (
        record.routing_snapshot_version != 1
        or envelope is None
        or not record.routing_snapshot_hash
        or envelope.key_version != record.routing_snapshot_key_version
    ):
        raise RoutingCryptoError()
    raw = decrypt_canonical(
        envelope,
        keys=keys,
        aad=routing_aad(
            run_id=record.run_id,
            request_fingerprint=record.request_fingerprint,
            key_version=envelope.key_version,
        ),
        error_code="routing_snapshot_invalid",
    )
    try:
        snapshot = RoutingSnapshotV1.model_validate(raw)
    except Exception as exc:  # noqa: BLE001 - expose only the stable contract code.
        raise RoutingCryptoError() from exc
    if (
        snapshot.run_id != record.run_id
        or snapshot.canonical_content_hash != record.routing_snapshot_hash
        or snapshot.registry_version != REGISTRY_VERSION
    ):
        raise RoutingCryptoError()
    return snapshot
