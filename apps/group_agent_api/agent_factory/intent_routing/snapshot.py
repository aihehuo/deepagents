"""Create-once routing snapshot construction and cryptographic verification."""

from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any

from apps.group_agent_api.execution.models import EncryptedPayload, ExecutionRecord

from .canonical import (
    RoutingCryptoError,
    canonical_hash,
    decrypt_canonical,
    encrypt_canonical,
)
from .contracts import QueryPlanV1, RoutingSnapshotV1
from .registry import REGISTRY_VERSION, registry_entry
from .schema import IntentDecisionV1, ResourceQueriesV2

_INTENT_PRIMARY_QUERY_FIELD: dict[str, str | None] = {
    "people_search": "users",
    "opportunity_search": "ideas",
    "project_partnership_search": "projects",
    "community_search": "groups",
    "startup_digest": None,
    "platform_help": "faqs",
}


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


def normalize_intent_decision_payload(decision_raw: Any) -> dict[str, Any]:
    """Fill Phase-2 optional defaults for non-execute or complete execute decisions.

    Phase-1 execute snapshots that lack frozen ``resource_queries`` are rejected.
    Constraints-only synthesis is intentionally not performed: it would change
    retry query semantics relative to the original Phase-1 parameter builder.
    """
    if not isinstance(decision_raw, dict):
        raise RoutingCryptoError("routing_snapshot_invalid")
    decision = dict(decision_raw)
    decision.setdefault("resource_queries", {})
    decision.setdefault("group_member_scope", "none")
    decision.setdefault("group_event_id", None)

    if decision.get("action") != "execute":
        return decision

    primary = _INTENT_PRIMARY_QUERY_FIELD.get(str(decision.get("primary_intent") or ""))
    queries = ResourceQueriesV2.model_validate(
        decision["resource_queries"]
        if isinstance(decision.get("resource_queries"), dict)
        else {}
    )
    decision["resource_queries"] = queries.model_dump(mode="json")
    if primary is not None and queries.for_resource(primary) is None:
        # Refuse resume rather than invent a different query from constraints.
        raise RoutingCryptoError("routing_snapshot_invalid")
    return decision


def parse_routing_snapshot_payload(raw: dict[str, Any]) -> RoutingSnapshotV1:
    """Verify the stored hash, then load a Phase-2-compatible decision for execution."""
    stored_hash = raw.get("canonical_content_hash")
    if not isinstance(stored_hash, str):
        raise RoutingCryptoError()
    body = {key: value for key, value in raw.items() if key != "canonical_content_hash"}
    if canonical_hash(body) != stored_hash:
        raise RoutingCryptoError()
    try:
        decision = IntentDecisionV1.model_validate(
            normalize_intent_decision_payload(body.get("intent_decision"))
        )
        plan_raw = body.get("query_plan")
        plan = QueryPlanV1.model_validate(plan_raw) if plan_raw is not None else None
    except RoutingCryptoError:
        raise
    except Exception as exc:  # noqa: BLE001 - expose only the stable contract code.
        raise RoutingCryptoError() from exc
    if body.get("policy_version") != decision.policy_version:
        raise RoutingCryptoError()
    if decision.action == "execute":
        if plan is None or decision.query_plan_ref is None:
            raise RoutingCryptoError("query_plan_invalid")
        if (
            decision.query_plan_ref.plan_id != plan.plan_id
            or decision.query_plan_ref.plan_version != plan.plan_version
        ):
            raise RoutingCryptoError("query_plan_invalid")
    elif plan is not None:
        raise RoutingCryptoError("query_plan_invalid")
    # Keep the original content hash (verified above). model_construct avoids
    # rehashing after optional-field defaults are filled for non-execute rows.
    return RoutingSnapshotV1.model_construct(
        routing_snapshot_version=1,
        run_id=str(body.get("run_id") or ""),
        intent_decision=decision,
        query_plan=plan,
        policy_version=decision.policy_version,
        registry_version=str(body.get("registry_version") or REGISTRY_VERSION),  # type: ignore[arg-type]
        canonical_content_hash=stored_hash,
        created_at=str(body.get("created_at") or ""),
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
        snapshot = parse_routing_snapshot_payload(raw)
    except RoutingCryptoError:
        raise
    except Exception as exc:  # noqa: BLE001 - expose only the stable contract code.
        raise RoutingCryptoError() from exc
    if (
        snapshot.run_id != record.run_id
        or snapshot.canonical_content_hash != record.routing_snapshot_hash
        or snapshot.registry_version != REGISTRY_VERSION
    ):
        raise RoutingCryptoError()
    return snapshot
