"""Bind the WeCom channel adapter to the same dialogue input and executor as APP."""
from __future__ import annotations

from typing import Any
from apps.group_agent_api.app.endpoints.ai_assistant import Input


def wecom_input(value: dict[str, Any]) -> Input | None:
    """Validate server-owned channel metadata before entering the common core.

    Args:
        value: Complete principal-authenticated, fingerprint-bound request.
    """
    metadata = value.get("metadata") or {}
    if "wecom_service_shared_input" not in metadata:
        return None
    uid, turn_id = value.get("user_id"), metadata.get("wecom_service_turn_id")
    if (
        value.get("request_schema_version", 1) not in {1, 3}
        or metadata.get("source") != "wecom_service"
        or metadata.get("wecom_service_action") not in {"conversation", "greeting"}
        or value.get("group_id") != "global"
        or value.get("execution_mode") == "cafe_host"
        or type(turn_id) is not int or turn_id <= 0
        or not isinstance(uid, str) or not uid.isascii() or not uid.isdigit()
        or value.get("conversation_id") != f"ga_wecom_{uid}_{turn_id}"
    ):
        raise ValueError("shared_dialogue_binding_invalid")
    payload = Input.model_validate(metadata["wecom_service_shared_input"])
    if payload.kind != "conversation" or payload.user_id != int(uid):
        raise ValueError("shared_dialogue_identity_invalid")
    expected = "greeting" if metadata["wecom_service_action"] == "greeting" else None
    if expected and payload.purpose != expected:
        raise ValueError("shared_dialogue_purpose_invalid")
    return payload
