"""Opt-in per-turn debug traces for group_agent (ops / debugging only).

TSD-03 forbids persisting hidden model chains in Micro session memory.
These dumps stay on the Deep Agents host under runtime/debug_traces/ and are
gated by GROUP_AGENT_DEBUG_TRACE=1.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

_logger = logging.getLogger("uvicorn.error")

_SECRET_KEY_RE = re.compile(
    r"(token|secret|password|authorization|api[_-]?key|hmac|cookie)",
    re.IGNORECASE,
)
_MAX_TEXT = 4000
_MAX_TOOL_ARG = 1500
_MAX_TRACES = 200


def debug_trace_enabled() -> bool:
    return os.environ.get("GROUP_AGENT_DEBUG_TRACE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _truncate(text: str, limit: int = _MAX_TEXT) -> str:
    t = (text or "").strip()
    if len(t) <= limit:
        return t
    return f"{t[:limit]}…[truncated {len(t) - limit} chars]"


def _scrub_mapping(data: Any, depth: int = 0) -> Any:
    if depth > 6:
        return "[max_depth]"
    if isinstance(data, dict):
        out: dict[str, Any] = {}
        for key, value in data.items():
            sk = str(key)
            if _SECRET_KEY_RE.search(sk):
                out[sk] = "[redacted]"
            else:
                out[sk] = _scrub_mapping(value, depth + 1)
        return out
    if isinstance(data, list):
        return [_scrub_mapping(v, depth + 1) for v in data[:40]]
    if isinstance(data, str):
        return _truncate(data, _MAX_TOOL_ARG)
    if isinstance(data, (int, float, bool)) or data is None:
        return data
    return _truncate(str(data), _MAX_TOOL_ARG)


def serialize_messages_delta(messages: list[Any], start: int) -> list[dict[str, Any]]:
    """Serialize Human / AI / Tool messages from this turn for debugging."""
    out: list[dict[str, Any]] = []
    for msg in messages[start:]:
        if isinstance(msg, HumanMessage):
            out.append(
                {
                    "role": "human",
                    "content": _truncate(str(msg.content or "")),
                }
            )
            continue
        if isinstance(msg, AIMessage):
            item: dict[str, Any] = {
                "role": "ai",
                "content": _truncate(str(msg.content or "")),
            }
            tool_calls = getattr(msg, "tool_calls", None) or []
            if tool_calls:
                serialized_calls = []
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        name = tc.get("name")
                        args = tc.get("args")
                        tc_id = tc.get("id")
                    else:
                        name = getattr(tc, "name", None)
                        args = getattr(tc, "args", None)
                        tc_id = getattr(tc, "id", None)
                    serialized_calls.append(
                        {
                            "id": tc_id,
                            "name": name,
                            "args": _scrub_mapping(args),
                        }
                    )
                item["tool_calls"] = serialized_calls
            out.append(item)
            continue
        if isinstance(msg, ToolMessage):
            out.append(
                {
                    "role": "tool",
                    "name": getattr(msg, "name", None),
                    "tool_call_id": getattr(msg, "tool_call_id", None),
                    "content": _truncate(str(getattr(msg, "content", "") or "")),
                }
            )
            continue
        out.append(
            {
                "role": type(msg).__name__,
                "content": _truncate(str(getattr(msg, "content", "") or "")),
            }
        )
    return out


def _prune_old_traces(dir_path: Path) -> None:
    files = sorted(
        dir_path.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for stale in files[_MAX_TRACES:]:
        try:
            stale.unlink(missing_ok=True)
        except OSError:
            pass


def write_turn_trace(
    *,
    base_dir: str | Path,
    run_id: str | None,
    thread_id: str,
    user_id: str,
    group_id: str,
    conversation_id: str | None,
    episode_id: str | None,
    user_message: str,
    messages: list[Any],
    msg_count_before: int,
    reply: str,
    profile_status: str | None = None,
    match_status: str | None = None,
    match_reason: str | None = None,
    extra: dict[str, Any] | None = None,
) -> str | None:
    """Write one turn trace JSON. Returns path when written, else None."""
    if not debug_trace_enabled():
        return None

    runtime = Path(base_dir)
    out_dir = runtime / "debug_traces"
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    safe_run = (run_id or "norun").replace("/", "_")[:80]
    path = out_dir / f"{stamp}_{safe_run}.json"

    payload = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "thread_id": thread_id,
        "user_id": user_id,
        "group_id": group_id,
        "conversation_id": conversation_id,
        "episode_id": episode_id,
        "user_message": _truncate(user_message),
        "reply": _truncate(reply),
        "profile_status": profile_status,
        "match_status": match_status,
        "match_reason": match_reason,
        "msg_count_before": msg_count_before,
        "msg_count_after": len(messages),
        "turn_messages": serialize_messages_delta(messages, msg_count_before),
        "extra": _scrub_mapping(extra or {}),
        "note": (
            "Ops debug dump only. Not stored in Micro session memory (TSD-03). "
            "Enable with GROUP_AGENT_DEBUG_TRACE=1."
        ),
    }
    try:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _prune_old_traces(out_dir)
        _logger.info(
            "action=debug_trace_written path=%s run_id=%s thread_id=%s",
            path,
            run_id,
            thread_id,
        )
        return str(path)
    except OSError as exc:
        _logger.warning("action=debug_trace_write_failed err=%s", exc)
        return None
