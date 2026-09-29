"""Best-effort semantic observation that never participates in Cafe Host delivery."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from typing import Any

from apps.group_agent_api.agent_factory.cafe_output import HostOutput, _review_messages
from apps.group_agent_api.agent_factory.host_context import HostContext

_LOGGER = logging.getLogger(__name__)
_INFLIGHT = 0
_LOCK = threading.Lock()


def shadow_review_enabled() -> bool:
    """Return whether the operator explicitly enabled private shadow review."""
    return os.environ.get("GROUP_AGENT_CAFE_SHADOW_REVIEW_ENABLED", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def shadow_review_sample_rate() -> float:
    """Return the bounded sampling fraction; invalid values disable sampling."""
    try:
        value = float(os.environ.get("GROUP_AGENT_CAFE_SHADOW_REVIEW_SAMPLE_RATE", "0.05"))
    except ValueError:
        return 0.0
    return value if 0.0 <= value <= 1.0 else 0.0


def shadow_review_max_inflight() -> int:
    """Return the process-local concurrency cap for disposable observer threads."""
    try:
        value = int(os.environ.get("GROUP_AGENT_CAFE_SHADOW_REVIEW_MAX_INFLIGHT", "2"))
    except ValueError:
        return 0
    return value if 0 <= value <= 8 else 0


def _sampled(host: HostContext, reply: str, rate: float) -> bool:
    """Select a stable sample without storing visitor identifiers in telemetry."""
    digest = hashlib.sha256(
        f"{host.episode_id}\0{reply}".encode("utf-8", errors="replace")
    ).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    return bucket < rate


def _run_review(*, model: Any, review_messages: list[Any], started: float) -> None:
    """Run one observer call and emit metadata only; never return a replacement."""
    global _INFLIGHT  # noqa: PLW0603
    result = "invalid"
    try:
        response = model.invoke(
            review_messages,
            tools=[],
            tool_choice="none",
            response_format={"type": "json_object"},
        )
        data = json.loads(getattr(response, "content", ""))
        verdict = data.get("verdict") if isinstance(data, dict) else None
        result = verdict if verdict in {"pass", "fail"} else "invalid"
    except Exception as exc:  # noqa: BLE001
        result = "error"
        _LOGGER.info(
            "cafe_shadow_review result=error exception_type=%s elapsed_ms=%d",
            type(exc).__name__,
            int((time.monotonic() - started) * 1000),
        )
    else:
        _LOGGER.info(
            "cafe_shadow_review result=%s elapsed_ms=%d",
            result,
            int((time.monotonic() - started) * 1000),
        )
    finally:
        with _LOCK:
            _INFLIGHT -= 1


def schedule_cafe_shadow_review(
    *,
    model: Any | None,
    output: HostOutput,
    host: HostContext,
    messages: list[Any],
) -> bool:
    """Schedule one sampled observer call without awaiting or mutating its result.

    Returns whether a daemon observer thread was started. Capacity exhaustion,
    disabled configuration, sampling misses, and thread failures all degrade to
    ``False`` and cannot affect the authoritative reply.
    """
    global _INFLIGHT  # noqa: PLW0603
    rate = shadow_review_sample_rate()
    limit = shadow_review_max_inflight()
    if (
        model is None
        or not shadow_review_enabled()
        or rate <= 0
        or limit <= 0
        or not _sampled(host, output.reply, rate)
    ):
        return False
    with _LOCK:
        if _INFLIGHT >= limit:
            _LOGGER.info("cafe_shadow_review result=dropped reason=capacity")
            return False
        _INFLIGHT += 1
    try:
        review_messages = _review_messages(
            output,
            host=host,
            messages=messages,
        )
        thread = threading.Thread(
            target=_run_review,
            kwargs={
                "model": model,
                "review_messages": review_messages,
                "started": time.monotonic(),
            },
            name="cafe-shadow-review",
            daemon=True,
        )
        thread.start()
    except Exception as exc:  # noqa: BLE001
        with _LOCK:
            _INFLIGHT -= 1
        _LOGGER.info(
            "cafe_shadow_review result=dropped reason=schedule_error exception_type=%s",
            type(exc).__name__,
        )
        return False
    return True
