"""Experimental, strict agent-execution-mode-v1 admission and routing policy."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from apps.group_agent_api.app.venue_protocol import (
    _choice,
    _equal,
    _integer,
    _object,
    _text,
    calculate_v2_fingerprint,
    canonical_json,
)

V3_FIELDS = set(
    "request_schema_version queue_schema_version mode_contract_version execution_mode executor_policy_version run_id idempotency_key request_fingerprint user_id unionid group_id conversation_id episode_id profile_scope_id memory_scope_id input_kind message event venue_context context_snapshot metadata rollout_context run_match run_invite willing_to_at membership group_token user_token callback_url".split()
)
V4_FIELDS = (V3_FIELDS - {"group_token", "user_token"}) | {
    "access_contract_version",
    "credential_binding",
    "chat_access",
}
MODE_FIELDS = {
    "mode_contract_version",
    "execution_mode",
    "executor_policy_version",
    "profile_scope_id",
    "memory_scope_id",
}
UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
LEGACY_PAID_SEARCH_INTENTS = (
    "opportunity_search",
    "people_search",
    "project_partnership_search",
)
PAID_SEARCH_INTENTS = (
    "community_search",
    "opportunity_search",
    "people_search",
    "project_partnership_search",
    "startup_digest",
)
FREE_SEARCH_INTENTS = ("platform_help",)


def _time(value: Any) -> datetime:
    _text(
        value,
        maximum=20,
        pattern=r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z",
    )
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")


def validate_v3_request(
    value: dict[str, Any], *, allow_cafe_host_without_unionid: bool = False
) -> dict[str, Any]:
    """Validate the full trusted body without coercion or fallback.

    Args:
        value: Raw parsed request or restored frozen envelope.
        allow_cafe_host_without_unionid: Only the v4 adapter may permit its
            signed App user principal to omit unionid; direct v3 stays strict.
    """
    body = _object(value, V3_FIELDS)
    for name, expected in {
        "request_schema_version": 3,
        "queue_schema_version": 3,
        "mode_contract_version": 1,
        "executor_policy_version": 1,
        "group_id": "global",
        "profile_scope_id": "global",
        "group_token": None,
    }.items():
        _equal(body[name], expected)
    _choice(body["execution_mode"], {"cafe_host", "group_agent"})
    cafe = body["execution_mode"] == "cafe_host"
    suffix = "cafe" if cafe else "group"
    _text(body["user_id"], maximum=64, pattern=r"[1-9][0-9]*")
    if not (allow_cafe_host_without_unionid and cafe and body["unionid"] is None):
        _text(body["unionid"])
    _equal(body["conversation_id"], f"ga_mode_{suffix}_{body['user_id']}")
    _equal(body["memory_scope_id"], f"community_cafe_mode_{suffix}")
    _text(body["episode_id"])
    _text(body["run_id"], pattern=UUID)
    _text(body["idempotency_key"], maximum=256)
    _text(body["request_fingerprint"], maximum=64, pattern=r"[0-9a-f]{64}")
    _text(body["callback_url"], maximum=1024, pattern=r"https://.+")
    _choice(body["membership"], {"unknown", "in_group", "not_in_group"})
    if body["user_token"] is not None:
        _text(body["user_token"], minimum=0, maximum=1024)
    for flag in ("run_match", "run_invite", "willing_to_at"):
        if type(body[flag]) is not bool:
            raise ValueError("schema_invalid: invalid boolean")
        if cafe:
            _equal(body[flag], False)
    _choice(body["input_kind"], {"user_message", "venue_event"})
    if body["input_kind"] == "venue_event":
        if not cafe:
            raise ValueError("schema_invalid: group agent cannot greet on entry")
        _equal(body["message"], "")
        event = _object(
            body["event"], {"type", "visit_id", "occurred_at", "expires_at"}
        )
        _equal(event["type"], "venue_entered")
        _text(event["visit_id"], pattern=UUID)
        delta = (
            _time(event["expires_at"]) - _time(event["occurred_at"])
        ).total_seconds()
        try:
            configured_ttl = int(
                os.environ.get("GROUP_AGENT_VENUE_ENTRY_GREETING_TTL_SECONDS", "60")
            )
        except ValueError:
            configured_ttl = 60
        ttl = configured_ttl if 10 <= configured_ttl <= 120 else 60
        if not 0 < delta <= ttl:
            raise ValueError("schema_invalid: invalid event lifetime")
        # Micro owns key derivation. Deep binds its exact opaque value through
        # the fingerprint and ledger, without rewriting synthetic golden keys.
    else:
        _text(body["message"], maximum=8192)
        if not body["message"].strip():
            raise ValueError("schema_invalid: empty user message")
        _equal(body["event"], None)
    scene = _object(
        body["venue_context"],
        set(
            "space_id space_kind owner_user_id visitor_user_id host_agent_id effective_mode timezone mode_override".split()
        ),
    )
    for key, expected in {
        "space_id": "community_cafe",
        "space_kind": "cafe_bar",
        "owner_user_id": None,
        "visitor_user_id": body["user_id"],
    }.items():
        _equal(scene[key], expected)
    _choice(scene["effective_mode"], {"day", "night"})
    _choice(scene["mode_override"], {"auto", "day", "night"})
    _equal(
        scene["host_agent_id"],
        "community_barista"
        if scene["effective_mode"] == "day"
        else "community_bartender",
    )
    if scene["mode_override"] != "auto":
        _equal(scene["effective_mode"], scene["mode_override"])
    _text(scene["timezone"], maximum=64)
    try:
        ZoneInfo(scene["timezone"])
    except (KeyError, ValueError) as exc:
        raise ValueError("schema_invalid: invalid timezone") from exc
    snapshot = _object(
        body["context_snapshot"],
        {
            "episode_id",
            "user_turn_version",
            "recent_messages",
            "previous_greeting",
            "previous_visit_at",
        },
    )
    _equal(snapshot["episode_id"], body["episode_id"])
    _integer(snapshot["user_turn_version"])
    if (
        type(snapshot["recent_messages"]) is not list
        or len(snapshot["recent_messages"]) > 20
    ):
        raise ValueError("schema_invalid: invalid recent messages")
    seen: set[int] = set()
    for message in snapshot["recent_messages"]:
        fields = {"message_id", "role", "text", "created_at", "result_kind"}
        _object(message, fields | {"visit_id"}, required=fields)
        _integer(message["message_id"], minimum=1)
        if message["message_id"] in seen:
            raise ValueError("schema_invalid: duplicate message id")
        seen.add(message["message_id"])
        _choice(message["role"], {"user", "assistant"})
        _text(message["text"], minimum=0, maximum=2000)
        _time(message["created_at"])
        _choice(message["result_kind"], {"conversation", "greeting"})
        if message.get("visit_id") is not None:
            _text(message["visit_id"], pattern=UUID)
        if not cafe and message["result_kind"] == "greeting":
            raise ValueError("schema_invalid: group snapshot cannot contain greetings")
    if snapshot["previous_greeting"] is not None:
        _text(snapshot["previous_greeting"], minimum=0, maximum=1000)
    if snapshot["previous_visit_at"] is not None:
        _time(snapshot["previous_visit_at"])
    if not cafe and (
        snapshot["previous_greeting"] is not None
        or snapshot["previous_visit_at"] is not None
    ):
        raise ValueError("schema_invalid: group snapshot cannot contain entry history")
    metadata = _object(
        body["metadata"],
        {"source", "client_surface", "source_message_id"},
        required={"source", "client_surface"},
    )
    _equal(metadata["source"], "ios_app")
    _equal(metadata["client_surface"], "ios_app")
    if "source_message_id" in metadata:
        _integer(metadata["source_message_id"], minimum=1)
        if body["input_kind"] == "venue_event":
            raise ValueError("schema_invalid: entry event is not profile evidence")
    rollout = _object(body["rollout_context"], {"protocol_mode", "rollout_version"})
    _choice(rollout["protocol_mode"], {"legacy_v1", "grounded_v2"})
    _equal(rollout["rollout_version"], "ga-v2-canary-v1")
    if len(canonical_json(body).encode("utf-8")) > 128 * 1024:
        raise ValueError("schema_invalid: request budget exceeded")
    if calculate_v2_fingerprint(body) != body["request_fingerprint"]:
        raise ValueError("schema_invalid: request fingerprint mismatch")
    return body


def validate_v4_request(value: dict[str, Any]) -> dict[str, Any]:
    """Validate the Cafe Host access overlay without weakening frozen v3.

    Args:
        value: Raw parsed request or restored encrypted request.

    Returns:
        The exact validated v4 request.

    Raises:
        ValueError: If common v3 structure or any access invariant fails.
    """
    body = _object(value, V4_FIELDS | {"trusted_search"}, required=V4_FIELDS)
    common = deepcopy(body)
    for name in ("access_contract_version", "credential_binding", "chat_access", "trusted_search"):
        common.pop(name, None)
    common.update(
        request_schema_version=3,
        queue_schema_version=3,
        group_token=None,
        user_token=None,
    )
    common["request_fingerprint"] = calculate_v2_fingerprint(common)
    validate_v3_request(common, allow_cafe_host_without_unionid=True)

    for name, expected in {
        "request_schema_version": 4,
        "queue_schema_version": 4,
        "access_contract_version": 1,
        "execution_mode": "cafe_host",
        "input_kind": "user_message",
    }.items():
        _equal(body[name], expected)
    binding = _object(
        body["credential_binding"],
        {"user_token_sha256", "group_token_sha256"},
    )
    for digest in binding.values():
        _text(digest, maximum=64, pattern=r"(?:-|[0-9a-f]{64})")
    access = _object(
        body["chat_access"],
        {
            "version",
            "mode",
            "policy_version",
            "revision",
            "drink_session_id",
            "pending_search_ref",
            "search_confirmation_id",
            "confirmed_search_intent",
            "allowed_search_intents",
            "restricted_search_intents",
        },
    )
    _equal(access["version"], 2)
    _equal(access["policy_version"], "chat-access-v2")
    _integer(access["revision"], minimum=0)
    _choice(access["mode"], {"ordinary", "active"})
    if access["drink_session_id"] is not None:
        _integer(access["drink_session_id"], minimum=1)
    for name, prefix in (
        ("pending_search_ref", "psr_"),
        ("search_confirmation_id", "scf_"),
    ):
        if access[name] is not None:
            _text(access[name], maximum=124, pattern=rf"{prefix}[A-Za-z0-9_-]{{8,120}}")
    if access["confirmed_search_intent"] is not None:
        _choice(access["confirmed_search_intent"], set(PAID_SEARCH_INTENTS))
    for name in ("allowed_search_intents", "restricted_search_intents"):
        values = access[name]
        if type(values) is not list or len(values) != len(set(values)):
            raise ValueError("schema_invalid: invalid search intent partition")
        if any(item not in PAID_SEARCH_INTENTS for item in values):
            raise ValueError("schema_invalid: invalid search intent partition")
    partition = access["allowed_search_intents"] + access["restricted_search_intents"]
    supported_partitions = {
        LEGACY_PAID_SEARCH_INTENTS,
        PAID_SEARCH_INTENTS,
    }
    if tuple(sorted(partition)) not in supported_partitions:
        raise ValueError("schema_invalid: invalid search intent partition")
    frozen_paid_intents = list(sorted(partition))

    if access["mode"] == "ordinary":
        _equal(binding, {"user_token_sha256": "-", "group_token_sha256": "-"})
        _equal(access["drink_session_id"], None)
        _equal(access["search_confirmation_id"], None)
        _equal(access["confirmed_search_intent"], None)
        _equal(access["allowed_search_intents"], [])
        _equal(access["restricted_search_intents"], frozen_paid_intents)
    else:
        if binding["user_token_sha256"] == "-":
            raise ValueError("schema_invalid: active user credential missing")
        if access["drink_session_id"] is None:
            raise ValueError("schema_invalid: active drink session missing")
        _equal(access["allowed_search_intents"], frozen_paid_intents)
        _equal(access["restricted_search_intents"], [])
        confirmed = access["search_confirmation_id"]
        intent = access["confirmed_search_intent"]
        if (confirmed is None) != (intent is None):
            raise ValueError("schema_invalid: incomplete search confirmation")
    if "trusted_search" in body:
        _validate_trusted_search(body["trusted_search"])
    if len(canonical_json(body).encode("utf-8")) > 128 * 1024:
        raise ValueError("schema_invalid: request budget exceeded")
    if calculate_v4_fingerprint(body) != body["request_fingerprint"]:
        raise ValueError("schema_invalid: request fingerprint mismatch")
    return body


def _validate_trusted_search(value: Any) -> None:
    """Accept only the server continuation object, never a client plan blob."""
    fields = {
        "version",
        "status",
        "pending_search_ref",
        "query_plan",
        "query_plan_hash",
        "routing_snapshot_hash",
        "original_message_sha256",
    }
    trusted = _object(value, fields)
    _equal(trusted["version"], 1)
    _choice(trusted["status"], {"ready", "unavailable"})
    if trusted["pending_search_ref"] is not None:
        _text(
            trusted["pending_search_ref"],
            maximum=124,
            pattern=r"psr_[A-Za-z0-9_-]{8,120}",
        )
    digest = r"sha256:[0-9a-f]{64}"
    if trusted["status"] == "unavailable":
        for name in (
            "query_plan",
            "query_plan_hash",
            "routing_snapshot_hash",
            "original_message_sha256",
        ):
            _equal(trusted[name], None)
        return
    if type(trusted["query_plan"]) is not dict or not trusted["query_plan"]:
        raise ValueError("schema_invalid: invalid trusted search plan")
    for name in ("query_plan_hash", "routing_snapshot_hash", "original_message_sha256"):
        _text(trusted[name], maximum=71, pattern=digest)


def calculate_v4_fingerprint(body: dict[str, Any]) -> str:
    """Hash the exact token-free v4 body except its fingerprint field.

    Args:
        body: Frozen Cafe Host v4 request dictionary.

    Returns:
        Lowercase SHA-256 hex digest defined by the cross-service contract.

    Unlike v2/v3, v4 already carries credential digests in
    ``credential_binding`` and must not synthesize the legacy token digest
    fields a second time.
    """
    import hashlib

    projection = {
        key: value for key, value in body.items() if key != "request_fingerprint"
    }
    return hashlib.sha256(canonical_json(projection).encode("utf-8")).hexdigest()


def execution_binding(req: Any) -> dict[str, Any]:
    """Derive output binding exclusively from an accepted frozen request.

    Args:
        req: Validated v3 request.
    """
    return {
        "schema_version": 1,
        **{
            key: getattr(req, key)
            for key in (
                "user_id",
                "execution_mode",
                "conversation_id",
                "episode_id",
                "run_id",
            )
        },
    }


def v4_queue_binding(req: Any) -> dict[str, Any]:
    """Build the frozen Cafe Host v4 queue AAD projection.

    Args:
        req: Strictly validated v4 request.

    Returns:
        Canonical access and credential fields authenticated by AES-GCM.
    """
    raw = req if isinstance(req, dict) else req.model_dump()
    body = validate_v4_request(raw)
    access = body["chat_access"]
    credential = body["credential_binding"]
    return {
        "user_id": body["user_id"],
        "execution_mode": body["execution_mode"],
        "conversation_id": body["conversation_id"],
        "episode_id": body["episode_id"],
        "access_contract_version": body["access_contract_version"],
        "chat_access_revision": access["revision"],
        "chat_access_mode": access["mode"],
        "drink_session_id": access["drink_session_id"],
        "confirmed_search_intent": access["confirmed_search_intent"],
        "user_token_sha256": credential["user_token_sha256"],
        "group_token_sha256": credential["group_token_sha256"],
    }


def validate_v3_wire(raw: bytes, req: Any) -> None:
    """Reject duplicate keys before binding the original transport to a model.

    Args:
        raw: Actual received signed UTF-8 bytes.
        req: Strict parsed v3 model.
    """

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("schema_invalid: duplicate JSON key")
            value[key] = item
        return value

    if len(raw) > 128 * 1024:
        raise ValueError("schema_invalid: request budget exceeded")
    body = json.loads(raw.decode("utf-8"), object_pairs_hook=object_pairs)
    validate_v3_request(body)
    if body != req.model_dump():
        raise ValueError("schema_invalid: model differs from signed request")


def validate_v4_wire(raw: bytes, req: Any) -> None:
    """Bind signed v4 bytes to the strict model and reject duplicate keys.

    Args:
        raw: Exact signed request bytes.
        req: Parsed `AsyncCallRequest`.
    """

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("schema_invalid: duplicate JSON key")
            result[key] = item
        return result

    if len(raw) > 128 * 1024:
        raise ValueError("schema_invalid: request budget exceeded")
    body = json.loads(raw.decode("utf-8"), object_pairs_hook=object_pairs)
    validate_v4_request(body)
    if raw != canonical_json(body).encode("utf-8"):
        raise ValueError("schema_invalid: v4 body is not canonical JSON")
    if body != req.model_dump():
        raise ValueError("schema_invalid: model differs from signed request")
