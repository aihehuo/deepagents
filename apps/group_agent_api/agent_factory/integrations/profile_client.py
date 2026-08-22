"""Signed Micro client for authoritative group-profile persistence."""

from __future__ import annotations

import json
import re
import hashlib
from datetime import datetime
from typing import Any

import requests

from apps.group_agent_api.agent_factory.integrations.callback_client import (
    sign_callback_payload,
)
from apps.group_agent_api.agent_factory.integrations.config import (
    callback_hmac_secret,
    http_timeout_s,
    micro_base,
)
from apps.group_agent_api.agent_factory.profile_schema import GroupProfile

PROFILE_PATH = "/group_agent/profile"
_UTC_TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$"
)
_PROFILE_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


class ProfileHttpError(Exception):
    """Raised when Micro rejects or cannot persist a group profile."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def canonical_profile_digest(profile: GroupProfile) -> str:
    """Return Micro's canonical SHA-256 digest for a validated profile.

    Args:
        profile: Validated group-scoped profile.

    Returns:
        Lowercase SHA-256 hex over Micro's fixed-order compact JSON form.
    """
    if getattr(profile, "schema_version", 1) == 2:
        canonical = {
            "schema_version": 2,
            "doing": profile.doing.model_dump(mode="json") if hasattr(profile.doing, "model_dump") else profile.doing,
            "need": profile.need.model_dump(mode="json") if hasattr(profile.need, "model_dump") else profile.need,
            "offer": profile.offer.model_dump(mode="json") if hasattr(profile.offer, "model_dump") else profile.offer,
            "match_constraints": getattr(profile, "match_constraints", []) or [],
        }
    else:
        canonical = {
            "schema_version": getattr(profile, "schema_version", 1),
            "doing": {
                "value": getattr(profile.doing, "value", ""),
                "disclosure": getattr(getattr(profile.doing, "disclosure", None), "value", str(getattr(profile.doing, "disclosure", ""))),
            },
            "need": {
                "value": getattr(profile.need, "value", ""),
                "disclosure": getattr(getattr(profile.need, "disclosure", None), "value", str(getattr(profile.need, "disclosure", ""))),
            },
            "offer": {
                "value": getattr(profile.offer, "value", ""),
                "disclosure": getattr(getattr(profile.offer, "disclosure", None), "value", str(getattr(profile.offer, "disclosure", ""))),
            },
        }
    raw = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True if getattr(profile, "schema_version", 1) == 2 else False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def persist_group_profile(
    *,
    profile: GroupProfile,
    run_id: str,
    attempt_id: str | None = None,
    fencing_token: int | None = None,
    base_url: str | None = None,
    secret: str | None = None,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    """Persist a profile through Micro's signed, run-bound upsert endpoint.

    Args:
        profile: Validated group-scoped profile.
        run_id: Micro-owned run that produced this profile.
        attempt_id: Durable attempt id (FIX3 fencing).
        fencing_token: Monotonic lease epoch (FIX3 fencing).
        base_url: Optional Micro origin override for tests.
        secret: Optional directional HMAC secret override for tests.
        timeout_s: Optional request timeout override.

    Returns:
        Validated Micro acknowledgement.

    Raises:
        ProfileHttpError: If configuration, transport, or response validation fails.
    """
    rid = (run_id or "").strip()
    if not rid:
        raise ProfileHttpError("missing_run_id")

    hmac_secret = secret if secret is not None else callback_hmac_secret()
    if not hmac_secret:
        raise ProfileHttpError("missing_profile_hmac_secret")

    profile_dict: dict[str, Any] = {
        "doing": profile.doing.model_dump(mode="json") if hasattr(profile.doing, "model_dump") else profile.doing,
        "need": profile.need.model_dump(mode="json") if hasattr(profile.need, "model_dump") else profile.need,
        "offer": profile.offer.model_dump(mode="json") if hasattr(profile.offer, "model_dump") else profile.offer,
        "schema_version": getattr(profile, "schema_version", 1),
    }
    if getattr(profile, "schema_version", 1) == 2 or getattr(profile, "match_constraints", None):
        profile_dict["match_constraints"] = getattr(profile, "match_constraints", []) or []

    payload: dict[str, Any] = {
        "run_id": rid,
        "user_id": profile.user_id,
        "group_id": profile.group_id,
        "profile": profile_dict,
    }
    if attempt_id:
        payload["attempt_id"] = str(attempt_id)
    if fencing_token is not None and int(fencing_token) > 0:
        payload["fencing_token"] = int(fencing_token)
    body = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    headers = sign_callback_payload(
        secret=hmac_secret,
        method="POST",
        path=PROFILE_PATH,
        body_bytes=body,
    )
    url = f"{(base_url or micro_base()).rstrip('/')}{PROFILE_PATH}"

    try:
        response = requests.post(
            url,
            data=body,
            headers=headers,
            timeout=timeout_s if timeout_s is not None else http_timeout_s(),
            allow_redirects=False,
        )
    except requests.RequestException as exc:
        raise ProfileHttpError(f"transport_error:{type(exc).__name__}") from exc

    if response.status_code != 200:
        raise ProfileHttpError(
            f"http_{response.status_code}",
            status_code=response.status_code,
        )

    if not response.content:
        raise ProfileHttpError("invalid_json")
    try:
        data = response.json()
    except ValueError as exc:
        raise ProfileHttpError("invalid_json") from exc
    if not isinstance(data, dict):
        raise ProfileHttpError("invalid_json")

    if str(data.get("user_id") or "") != profile.user_id:
        raise ProfileHttpError("user_id_mismatch")
    if str(data.get("group_id") or "") != profile.group_id:
        raise ProfileHttpError("group_id_mismatch")
    status = str(data.get("status") or "")
    if status not in {"created", "updated", "idempotent", "stale_ignored", "fence_rejected"}:
        raise ProfileHttpError("invalid_status")
    version = data.get("profile_version")
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version < 1
        or (status == "created" and version != 1)
    ):
        raise ProfileHttpError("invalid_profile_version")
    schema_version = data.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != profile.schema_version
    ):
        raise ProfileHttpError("schema_version_mismatch")
    updated_at = data.get("updated_at")
    if not isinstance(updated_at, str) or not _UTC_TIMESTAMP_RE.fullmatch(updated_at):
        raise ProfileHttpError("invalid_updated_at")
    try:
        datetime.strptime(updated_at, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError as exc:
        raise ProfileHttpError("invalid_updated_at") from exc
    digest = data.get("profile_digest")
    if not isinstance(digest, str) or not _PROFILE_DIGEST_RE.fullmatch(digest):
        raise ProfileHttpError("invalid_profile_digest")
    if status not in {"stale_ignored", "fence_rejected"} and digest != canonical_profile_digest(profile):
        raise ProfileHttpError("profile_digest_mismatch")

    return data
