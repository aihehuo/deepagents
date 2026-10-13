"""Strict AI assistant execution over the shared durable transport."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from apps.group_agent_api.app.endpoints.ai_assistant import Input

FIELDS = set(
    "request_schema_version queue_schema_version run_id idempotency_key "
    "request_fingerprint user_id unionid group_id conversation_id message "
    "membership group_token user_token callback_url metadata rollout_context "
    "run_match run_invite willing_to_at assistant_input execution_mode".split()
)


def fingerprint(value: dict[str, Any]) -> str:
    """Hash the complete frozen task excluding its digest.

    Args:
        value: Validated assistant request.
    """
    data = {key: value[key] for key in FIELDS if key != "request_fingerprint"}
    encoded = json.dumps(
        data, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate(value: dict[str, Any]) -> dict[str, Any]:
    """Refuse legacy tools, identity coercion and unexpected fields.

    Args:
        value: Parsed request body.
    """
    if set(value) != FIELDS:
        raise ValueError("assistant_schema_invalid")
    expected_fields = {
        "request_schema_version": 5,
        "queue_schema_version": 5,
        "execution_mode": "ai_assistant",
        "group_id": "global",
        "unionid": None,
        "group_token": None,
        "user_token": None,
        "membership": "unknown",
        "run_match": False,
        "run_invite": False,
        "willing_to_at": False,
        "metadata": {"source": "ai_assistant"},
        "rollout_context": None,
    }
    for key, expected in expected_fields.items():
        if value[key] != expected or type(value[key]) is not type(expected):
            raise ValueError("assistant_schema_invalid")
    uid = value["user_id"]
    if not isinstance(uid, str) or not uid.isascii() or not uid.isdigit() or int(uid) <= 0:
        raise ValueError("assistant_identity_invalid")
    if value["conversation_id"] != "ai_assistant_" + uid:
        raise ValueError("assistant_identity_invalid")
    task = Input.model_validate(value["assistant_input"])
    if task.user_id != int(uid) or value["message"] != task.kind:
        raise ValueError("assistant_identity_invalid")
    if fingerprint(value) != value["request_fingerprint"]:
        raise ValueError("assistant_fingerprint_invalid")
    return value
