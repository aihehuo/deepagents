"""Opt-in private café review records, never public transcripts or model inputs."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from apps.group_agent_api.agent_factory.host_context import HostContext, _SECRET

_EVENTS: ContextVar[list[dict[str, Any]] | None] = ContextVar("cafe_review_audit", default=None)
_LOGGER = logging.getLogger(__name__)
_KEY = re.compile(r"token|secret|password|authorization|api[_ -]?key|cookie", re.I)
_QUOTED = re.compile(r'("(?:token|secret|password|authorization|api[_ -]?key|cookie)"\s*:\s*)"[^"\n]*"', re.I)
_PASSWORD = re.compile(r"\bpassword\s*[:=]\s*\S+", re.I)


def _redact(value: Any) -> Any:
    if isinstance(value, str):
        text = _SECRET.sub("[credential redacted]", value.replace("\x00", ""))
        text = _QUOTED.sub(r'\1"[credential redacted]"', text)
        text = _PASSWORD.sub("password=[credential redacted]", text)
        return text if len(text) <= 20000 else text[:20000] + "[truncated]"
    if isinstance(value, dict):
        return {str(key): "[credential redacted]" if _KEY.search(str(key)) else _redact(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_redact(item) for item in value[:100]]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _redact(str(value))


def record_review_event(stage: str, **detail: Any) -> None:
    """Capture explicit verdicts/replies, not hidden reasoning or transport metadata.

    Args:
        stage: Fixed review stage chosen by application code.
        detail: Authorized turn data; credentials are redacted before persistence.
    """
    events = _EVENTS.get()
    if events is not None:
        events.append({"stage": stage, **detail})


def _persist(host: HostContext, events: list[dict[str, Any]]) -> None:
    from apps.group_agent_api.agent_factory.agent import default_runtime_dir
    from apps.group_agent_api.execution.active_fence import get_active_fence

    runtime = Path(os.environ.get("GROUP_AGENT_RUNTIME_DIR") or default_runtime_dir())
    directory = runtime / "cafe_review_audits"
    if directory.is_symlink():
        raise OSError("audit directory must not be a symlink")
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory.chmod(0o700)
    now = time.time()
    files = sorted(directory.glob("review-*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    for index, path in enumerate(files):
        if index >= 999 or now - path.stat().st_mtime > 30 * 86400:
            path.unlink(missing_ok=True)
    fence = get_active_fence()
    payload = {"schema_version": 1, "recorded_at": now,
               "visitor_user_id": host.visitor_user_id, "episode_id": host.episode_id,
               "host_agent_id": host.host_agent_id, "mode": host.effective_mode,
               "run_id": fence.claim.run_id if fence else None,
               "attempt_id": fence.claim.attempt_id if fence else None,
               "events": events}
    encoded = json.dumps(_redact(payload), ensure_ascii=False).encode("utf-8")
    if len(encoded) > 1024 * 1024:
        raise ValueError("audit exceeds size limit")
    path = directory / f"review-{int(now)}-{uuid4().hex}.json"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)


@contextmanager
def review_audit(host: HostContext, text: str, messages: list[Any]) -> Iterator[None]:
    """Record one authorized review only when the operator enables private auditing.

    Args:
        host: Authenticated visitor binding, not client-selected storage paths.
        text: Original model draft.
        messages: This visitor's conversation supplied to the reviewer.
    """
    enabled = os.environ.get("GROUP_AGENT_CAFE_REVIEW_AUDIT", "").lower() in {"1", "true", "yes", "on"}
    if not enabled:
        yield
        return
    events: list[dict[str, Any]] = []
    token = _EVENTS.set(events)
    record_review_event("input", draft=text, conversation=[{"role": item.type, "text": item.content} for item in messages[-12:] if item.type in {"human", "ai"} and not getattr(item, "tool_calls", None)])
    try:
        yield
    except Exception as exc:
        record_review_event("execution_exception", exception_type=type(exc).__name__)
        raise
    finally:
        _EVENTS.reset(token)
        try:
            _persist(host, events)
        except Exception as exc:
            # Audit failure must not alter delivery or expose private content.
            _LOGGER.warning("cafe_review_audit_write_failed exception_type=%s", type(exc).__name__)
