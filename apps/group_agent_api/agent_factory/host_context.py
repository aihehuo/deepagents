"""Authorized, run-scoped café host context; never a shared visitor memory."""

from __future__ import annotations

import json
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from apps.group_agent_api.app.venue_protocol import validate_v2_request
from apps.group_agent_api.agent_factory.profile_store import load_profile
from apps.group_agent_api.agent_factory.contact_scrub import scrub_contact_text

_ACTIVE_HOST: ContextVar[HostContext | None] = ContextVar("group_agent_venue_host", default=None)
_SECRET = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{8,}|eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+)\b|(?:api[_ -]?key|authorization|bearer|jwt|token|secret|签名|密钥)\s*[:=：]?\s*[A-Za-z0-9_./+\-=]{8,}", re.IGNORECASE)
_EMAIL = re.compile(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")


def safe_context_text(value: str, *, limit: int = 2000) -> str:
    """Bound quoted content and redact credential-like source text.

    Args:
        value: Authorized visitor text, not trusted instructions.
        limit: Maximum Unicode scalar length.
    """
    cleaned = _SECRET.sub("[credential redacted]", value.replace("\x00", ""))
    cleaned = _EMAIL.sub("[contact redacted]", cleaned)
    return scrub_contact_text(cleaned)[:limit]


@dataclass(frozen=True)
class HostContext:
    """Experimental host/visitor policy binding; contains no transport secrets."""

    visitor_user_id: str
    host_agent_id: str
    effective_mode: str
    space_id: str
    memory_scope_id: str
    episode_id: str
    profile: dict[str, str]
    snapshot: dict[str, Any]

    @classmethod
    def from_request(cls, req: Any, *, principal: Any, base_dir: Path) -> HostContext:
        """Resolve only built-in café policy and this visitor's own profile.

        Args:
            req: Validated and authenticated frozen v2 request.
            principal: Trusted OAuth principal, never model-selected.
            base_dir: Existing local visitor-profile root.
        """
        validate_v2_request(req.model_dump())
        venue = req.venue_context
        if req.user_id != principal.user_id or req.unionid != principal.unionid or venue["visitor_user_id"] != principal.user_id:
            raise ValueError("binding_mismatch: unauthorized host context")
        if str(req.metadata.get("source", "")).startswith("group_agent_admin") or req.metadata.get("admin_mode"):
            raise ValueError("security_guard_failed: café is not an admin bucket")
        profile = load_profile(base_dir, req.user_id, req.group_id)
        values: dict[str, str] = {}
        if profile is not None:
            if profile.user_id != req.user_id or profile.group_id != req.group_id:
                raise ValueError("security_guard_failed: profile identity mismatch")
            for name in ("doing", "need", "offer"):
                value = str(getattr(getattr(profile, name), "value", "") or "")
                if value.strip():
                    values[name] = safe_context_text(value)
        # Defensive copy: neither a model nor another request can mutate the
        # authenticated snapshot or turn quoted facts into policy overrides.
        snapshot = json.loads(json.dumps(req.context_snapshot, ensure_ascii=False))
        for message in snapshot["recent_messages"]:
            message["text"] = safe_context_text(message["text"])
        if snapshot["previous_greeting"] is not None:
            snapshot["previous_greeting"]["text"] = safe_context_text(snapshot["previous_greeting"]["text"], limit=1000)
        return cls(visitor_user_id=req.user_id, host_agent_id=venue["host_agent_id"], effective_mode=venue["effective_mode"], space_id=venue["space_id"], memory_scope_id=venue["memory_scope_id"], episode_id=req.episode_id, profile=values, snapshot=snapshot)

    def quoted_data(self, *, greeting: bool, existing_messages: list[Any] | None = None) -> str:
        """Render minimized evidence as JSON, never executable prompt fragments.

        Args:
            greeting: True for full read-only visit context; ordinary turns
                carry only committed greetings, avoiding repeated user messages.
            existing_messages: Checkpoint messages for greeting deduplication.
        """
        recent = self.snapshot["recent_messages"]
        if not greeting:
            recent = [item for item in recent if item["role"] == "assistant" and item["result_kind"] == "greeting"]
        seen = {item["message_id"] for item in recent}
        previous = self.snapshot["previous_greeting"]
        if previous and previous["message_id"] in seen:
            previous = None
        existing = existing_messages or []
        existing_text = {str(item.content) for item in existing if getattr(item, "type", "") == "ai"}
        recent = [item for item in recent if item["text"] not in existing_text]
        if previous and previous["text"] in existing_text:
            previous = None
        minimized = [{"message_id": item["message_id"], "role": item["role"], "text": item["text"], "created_at": item["created_at"], "result_kind": item["result_kind"]} for item in recent]
        value = {"profile": self.profile, "recent_messages": minimized, "previous_greeting": previous, "previous_visit_at": self.snapshot["previous_visit_at"], "has_previous_visit": bool(self.snapshot["previous_visit_at"] or self.snapshot["previous_greeting"] or self.snapshot["recent_messages"])}
        # Escape delimiters so quoted user strings cannot close the data block.
        return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c").replace(">", "\\u003e")


def current_host_context() -> HostContext | None:
    """Return this task's host binding, not checkpoint/client metadata."""
    return _ACTIVE_HOST.get()


@contextmanager
def active_host_context(host: HostContext | None) -> Iterator[None]:
    """Scope a host to one execution, restoring it across errors/cancellation.

    Args:
        host: Authorized frozen host, or None for unchanged legacy execution.
    """
    token = _ACTIVE_HOST.set(host)
    try:
        yield
    finally:
        _ACTIVE_HOST.reset(token)
