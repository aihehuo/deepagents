"""Strict, non-normalizing venue-entry-v1 durable protocol (experimental).

Validation intentionally precedes legacy Pydantic coercion. The signed request
must have exactly the same meaning and fingerprint after queue restoration.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

SAFE_INTEGER = 9007199254740991
V2_FIELDS = set("request_schema_version queue_schema_version run_id idempotency_key user_id unionid group_id conversation_id episode_id input_kind message event venue_context context_snapshot metadata callback_url rollout_context run_match run_invite willing_to_at group_token user_token request_fingerprint membership".split())
_VISIT = r"venue_[A-Za-z0-9_-]{1,96}"
_UTC = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z"


def canonical_json(value: Any) -> str:
    """Encode canonical UTF-8 JSON without normalization or float repair.

    Args:
        value: A validated protocol JSON value.
    """
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _object(value: Any, fields: set[str], *, required: set[str] | None = None) -> dict[str, Any]:
    if type(value) is not dict or set(value) - fields or (fields if required is None else required) - set(value):
        raise ValueError("schema_invalid: missing or unknown object fields")
    return value


def _text(value: Any, *, maximum: int = 128, minimum: int = 1, pattern: str | None = None) -> None:
    if type(value) is not str or not minimum <= len(value) <= maximum:
        raise ValueError("schema_invalid: invalid string")
    # Lone surrogates cannot be serialized as canonical UTF-8 bytes.
    value.encode("utf-8")
    if pattern is not None and re.fullmatch(pattern, value) is None:
        raise ValueError("schema_invalid: invalid string pattern")


def _time(value: Any) -> None:
    _text(value, maximum=27, pattern=_UTC)
    datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")


def _integer(value: Any, *, minimum: int = 0) -> None:
    if type(value) is not int or not minimum <= value <= SAFE_INTEGER:
        raise ValueError("schema_invalid: unsafe integer")


def _equal(value: Any, expected: Any) -> None:
    if type(value) is not type(expected) or value != expected:
        raise ValueError("schema_invalid: unsupported protocol value")


def _choice(value: Any, choices: set[str]) -> None:
    if type(value) is not str or value not in choices:
        raise ValueError("schema_invalid: invalid enum")


def validate_v2_request(value: dict[str, Any]) -> dict[str, Any]:
    """Validate the entire frozen body without silently changing any fields.

    Args:
        value: Raw parsed /call_async request or restored durable request.
    """
    body = _object(value, V2_FIELDS)
    _equal(body["request_schema_version"], 2)
    _equal(body["queue_schema_version"], 2)
    for key in ("run_id", "idempotency_key", "conversation_id", "episode_id"):
        _text(body[key])
    _text(body["run_id"], pattern=r"[a-zA-Z0-9_\-]+")
    _text(body["user_id"], pattern=r"[1-9][0-9]*")
    _text(body["unionid"], minimum=0)
    _equal(body["group_id"], "global")
    _equal(body["membership"], "unknown")
    _text(body["request_fingerprint"], maximum=64, pattern=r"[0-9a-f]{64}")
    _text(body["callback_url"], maximum=1024, pattern=r"https?://.+")
    for key in ("group_token", "user_token"):
        if body[key] is not None:
            _text(body[key], minimum=0, maximum=1024)
    for key in ("run_match", "run_invite", "willing_to_at"):
        if type(body[key]) is not bool:
            raise ValueError("schema_invalid: invalid boolean")
    _choice(body["input_kind"], {"venue_event", "user_message"})
    if body["input_kind"] == "venue_event":
        _equal(body["message"], None)
        event = _object(body["event"], {"name", "visit_id"})
        _equal(event["name"], "venue_entered")
        _text(event["visit_id"], maximum=102, pattern=_VISIT)
        for key in ("run_match", "run_invite", "willing_to_at"):
            _equal(body[key], False)
    else:
        _text(body["message"], maximum=8192)
        if not body["message"].strip():
            raise ValueError("schema_invalid: empty user message")
        _equal(body["event"], None)
    venue = _object(body["venue_context"], set("space_id space_kind owner_user_id host_agent_id memory_scope_id host_config_version context_policy_version visitor_user_id timezone mode_override effective_mode".split()))
    versions = (venue["host_config_version"], venue["context_policy_version"])
    if any(type(item) is not int for item in versions) or versions not in {(1, 1), (2, 2)}:
        raise ValueError("schema_invalid: unsupported host policy versions")
    if versions == (2, 2):
        for key in ("run_match", "run_invite", "willing_to_at"):
            _equal(body[key], False)
    for key, expected in {"space_id": "community_cafe", "space_kind": "cafe_bar", "owner_user_id": None, "memory_scope_id": "community_cafe_host", "visitor_user_id": body["user_id"]}.items():
        _equal(venue[key], expected)
    _choice(venue["effective_mode"], {"day", "night"})
    _choice(venue["mode_override"], {"auto", "day", "night"})
    _equal(venue["host_agent_id"], "community_barista" if venue["effective_mode"] == "day" else "community_bartender")
    if venue["mode_override"] != "auto":
        _equal(venue["effective_mode"], venue["mode_override"])
    _text(venue["timezone"], maximum=64)
    try:
        ZoneInfo(venue["timezone"])
    except (ValueError, ZoneInfoNotFoundError) as exc:
        raise ValueError("schema_invalid: timezone must already be normalized") from exc
    snapshot = _object(body["context_snapshot"], {"episode_id", "user_turn_version", "recent_messages", "previous_greeting", "previous_visit_at"})
    _equal(snapshot["episode_id"], body["episode_id"])
    _integer(snapshot["user_turn_version"])
    recent = snapshot["recent_messages"]
    if type(recent) is not list or len(recent) > 20:
        raise ValueError("schema_invalid: too many recent messages")
    ids: set[str] = set()
    total = 0
    for item in recent:
        _object(item, set("message_id role text created_at run_id episode_id result_kind visit_id".split()))
        _text(item["message_id"])
        if item["message_id"] in ids:
            raise ValueError("schema_invalid: duplicate message_id")
        ids.add(item["message_id"])
        _choice(item["role"], {"user", "assistant"})
        _text(item["text"], maximum=2000)
        total += len(item["text"])
        _time(item["created_at"])
        _equal(item["episode_id"], body["episode_id"])
        _choice(item["result_kind"], {"conversation", "greeting"})
        if item["run_id"] is not None:
            _text(item["run_id"])
        if item["visit_id"] is not None:
            _text(item["visit_id"], maximum=102, pattern=_VISIT)
    if total > 12000:
        raise ValueError("schema_invalid: recent message budget exceeded")
    previous = snapshot["previous_greeting"]
    if previous is not None:
        _object(previous, {"message_id", "text", "created_at"})
        _text(previous["message_id"])
        _text(previous["text"], maximum=1000)
        _time(previous["created_at"])
    if snapshot["previous_visit_at"] is not None:
        _time(snapshot["previous_visit_at"])
    rollout = _object(body["rollout_context"], {"protocol_mode", "rollout_version"})
    _choice(rollout["protocol_mode"], {"legacy_v1", "grounded_v2"})
    _equal(rollout["rollout_version"], "ga-v2-canary-v1")
    metadata = body["metadata"]
    if type(metadata) is not dict or len(metadata) > 20:
        raise ValueError("schema_invalid: invalid metadata")
    forbidden = {"candidates", "candidate_pool", "mock_candidates", "override_group_id", "trusted_group_id", "protocol_mode", "rollout_version", "contract_version", "rollout_context", "episode_id", "episodeId", "input_kind", "event", "venue_context", "context_snapshot"}
    for key, item in metadata.items():
        _text(key, minimum=0, maximum=64)
        if key in forbidden or key.lower() in forbidden:
            raise ValueError("schema_invalid: reserved metadata")
        if type(item) is int:
            _integer(item, minimum=-SAFE_INTEGER)
        elif type(item) is str:
            _text(item, minimum=0, maximum=1024)
        elif item is not None and type(item) is not bool:
            raise ValueError("schema_invalid: metadata must contain scalar values")
    if body["input_kind"] == "venue_event" and any(key in metadata for key in ("source_message_id", "source_message_text")):
        raise ValueError("schema_invalid: events are not profile evidence")
    if "source_message_id" in metadata:
        _integer(metadata["source_message_id"], minimum=1)
    if len(canonical_json(body).encode("utf-8")) > 128 * 1024:
        raise ValueError("schema_invalid: request budget exceeded")
    if calculate_v2_fingerprint(body) != body["request_fingerprint"]:
        raise ValueError("schema_invalid: request fingerprint mismatch")
    return body


def calculate_v2_fingerprint(body: dict[str, Any]) -> str:
    """Compute Micro-authoritative full request fingerprint.

    Args:
        body: Frozen request containing all venue-entry-v1 fields.
    """
    projection = {key: value for key, value in body.items() if key not in {"request_fingerprint", "user_token", "group_token"}}
    for key in ("group_token", "user_token"):
        token = (body.get(key) or "").strip()
        projection[f"{key}_sha256"] = hashlib.sha256(token.encode("utf-8")).hexdigest() if token else "-"
    return hashlib.sha256(canonical_json(projection).encode("utf-8")).hexdigest()


def lease_scope(req: Any) -> str:
    """Keep read-only greeting leases independent of conversation checkpoint locks.

    Args:
        req: A validated async request.
    """
    return f"ga_greeting::{req.run_id}" if req.input_kind == "venue_event" else req.conversation_id


def callback_bindings(req: Any) -> dict[str, Any]:
    """Derive v2 callback identity only from the frozen request.

    Args:
        req: A validated async request.
    """
    if req.request_schema_version != 2:
        return {}
    venue = req.venue_context
    return {"episode_id": req.episode_id, "space_id": venue["space_id"], "host_agent_id": venue["host_agent_id"], "memory_scope_id": venue["memory_scope_id"], "input_kind": req.input_kind, "event_name": req.event["name"] if req.event else None, "visit_id": req.event["visit_id"] if req.event else None}


def validate_v2_callback(value: dict[str, Any]) -> dict[str, Any]:
    """Validate exact v2 callback transport before the legacy envelope parser.

    Args:
        value: Frozen-identity callback JSON dictionary.
    """
    fields = set("version input_kind event_name visit_id run_id conversation_id episode_id space_id host_agent_id memory_scope_id user_id group_id seq event occurred_at payload".split())
    _object(value, fields)
    _equal(value["version"], "GA-CALLBACK-V1")
    for key in ("run_id", "conversation_id", "episode_id"):
        _text(value[key])
    _text(value["user_id"], pattern=r"[1-9][0-9]*")
    _equal(value["group_id"], "global")
    _integer(value["seq"], minimum=1)
    _time(value["occurred_at"])
    _choice(value["input_kind"], {"venue_event", "user_message"})
    if type(value["payload"]) is not dict:
        raise ValueError("schema_invalid: callback payload must be an object")
    return value
