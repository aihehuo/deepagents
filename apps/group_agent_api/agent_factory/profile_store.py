"""Forced structured profile persistence (FR-06).

Writes JSON under /users/{uid}/groups/{gid}/profile.json relative to the
FilesystemBackend root. Upsert via Path.write_text (backend.write is create-only).

Post-process assertion: missing/empty/invalid profile → alert + retry (caller).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from apps.group_agent_api.agent_factory.profile_schema import GroupProfile

_logger = logging.getLogger("uvicorn.error")

# Prevent path traversal via mock ids (no dots-only / no ..)
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class ProfileStoreError(Exception):
    """Raised when profile read/write/assert fails."""


@dataclass
class AssertResult:
    ok: bool
    path: str
    reason: str = ""
    profile: GroupProfile | None = None


def validate_id(value: str, *, field: str) -> str:
    text = (value or "").strip()
    if text in {".", ".."} or not _SAFE_ID.fullmatch(text):
        raise ProfileStoreError(
            f"invalid {field}: must match ^[A-Za-z0-9_-]{{1,64}}$ "
            "(no path dots)"
        )
    return text


def virtual_profile_path(user_id: str, group_id: str) -> str:
    uid = validate_id(user_id, field="user_id")
    gid = validate_id(group_id, field="group_id")
    return f"/users/{uid}/groups/{gid}/profile.json"


def disk_profile_path(base_dir: Path, user_id: str, group_id: str) -> Path:
    uid = validate_id(user_id, field="user_id")
    gid = validate_id(group_id, field="group_id")
    root = base_dir.resolve()
    path = (root / "users" / uid / "groups" / gid / "profile.json").resolve()
    try:
        path.relative_to(root / "users")
    except ValueError as exc:
        raise ProfileStoreError(f"path escape blocked for {uid}/{gid}") from exc
    return path


def save_profile(base_dir: Path, profile: GroupProfile) -> Path:
    """Upsert structured profile JSON. Returns absolute disk path."""
    if not profile.is_complete():
        raise ProfileStoreError("profile incomplete: doing/need/offer required")
    # Ensure ids in payload match path keys
    path = disk_profile_path(base_dir, profile.user_id, profile.group_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = profile.to_storage_dict()
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _logger.info(
        "action=profile_saved user_id=%s group_id=%s path=%s",
        profile.user_id,
        profile.group_id,
        path,
    )
    return path


def load_profile(base_dir: Path, user_id: str, group_id: str) -> GroupProfile | None:
    path = disk_profile_path(base_dir, user_id, group_id)
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not raw:
        return None
    try:
        data: Any = json.loads(raw)
        return GroupProfile.model_validate(data)
    except Exception:  # noqa: BLE001
        return None


def assert_profile_persisted(
    base_dir: Path, user_id: str, group_id: str
) -> AssertResult:
    """Assert non-empty structured profile exists for user×group."""
    vpath = virtual_profile_path(user_id, group_id)
    path = disk_profile_path(base_dir, user_id, group_id)
    if not path.exists():
        return AssertResult(ok=False, path=vpath, reason="missing_file")
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        return AssertResult(ok=False, path=vpath, reason=f"read_error:{exc}")
    if not raw:
        return AssertResult(ok=False, path=vpath, reason="empty_file")
    try:
        profile = GroupProfile.model_validate(json.loads(raw))
    except Exception as exc:  # noqa: BLE001
        return AssertResult(ok=False, path=vpath, reason=f"invalid_schema:{exc}")
    if not profile.is_complete():
        return AssertResult(ok=False, path=vpath, reason="incomplete_fields")
    if profile.user_id != user_id or profile.group_id != group_id:
        return AssertResult(ok=False, path=vpath, reason="id_mismatch")
    return AssertResult(ok=True, path=vpath, profile=profile)


def alert_persist_failure(
    *,
    user_id: str,
    group_id: str,
    attempt: int,
    reason: str,
) -> None:
    """Log a loud alert — never silently drop a missing profile."""
    _logger.error(
        "ALERT action=profile_persist_failed user_id=%s group_id=%s "
        "attempt=%s reason=%s status=will_retry_or_fail",
        user_id,
        group_id,
        attempt,
        reason,
    )
