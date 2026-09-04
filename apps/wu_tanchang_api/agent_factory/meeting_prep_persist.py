"""Ensure meeting-prep text shown in chat is persisted via save_meeting_prep.

Models often print 【会议准备】 into the AI reply and skip the tool. Booking
email and Micro drawers read the Micro落库 column, so a forgotten tool leaves
「暂无落库」 even though the user already saw the material. This module is the
hard safety net after each front-agent turn.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from apps.wu_tanchang_api.agent_factory.prompt_modules import meeting_prep_policy_for_run

_logger = logging.getLogger("uvicorn.error")

# thread_id -> True after a successful save_meeting_prep in this process.
_saved_threads: dict[str, bool] = {}
_saved_lock = threading.Lock()


def mark_thread_prep_saved(thread_id: str) -> None:
    tid = (thread_id or "").strip()
    if not tid:
        return
    with _saved_lock:
        _saved_threads[tid] = True


def thread_prep_saved(thread_id: str) -> bool:
    tid = (thread_id or "").strip()
    if not tid:
        return False
    with _saved_lock:
        return bool(_saved_threads.get(tid))


def clear_thread_prep_saved(thread_id: str) -> None:
    tid = (thread_id or "").strip()
    if not tid:
        return
    with _saved_lock:
        _saved_threads.pop(tid, None)


def turn_has_successful_save(messages: list[Any]) -> bool:
    for msg in messages or []:
        if not isinstance(msg, ToolMessage):
            continue
        name = str(getattr(msg, "name", "") or "")
        content = str(getattr(msg, "content", "") or "")
        if name == "save_meeting_prep" and "meeting_prep_saved" in content:
            return True
        if content.strip() == "meeting_prep_saved":
            return True
    return False


def extract_prep_body_from_messages(messages: list[Any]) -> str | None:
    """Longest AI span starting at 【会议准备】 in the given messages."""
    from apps.wu_tanchang_api.app.stream_history import normalize_message_text

    best: str | None = None
    for msg in messages or []:
        if not isinstance(msg, AIMessage):
            continue
        text = normalize_message_text(msg)
        idx = text.find("【会议准备")
        if idx < 0:
            continue
        body = text[idx:].strip()
        if not body:
            continue
        if best is None or len(body) > len(best):
            best = body
    return best


def body_looks_complete(body: str, config: RunnableConfig | None = None) -> bool:
    text = (body or "").strip()
    if not text or "【会议准备" not in text:
        return False
    required, forbidden = meeting_prep_policy_for_run(config)
    if any(h not in text for h in required):
        return False
    if any(h in text for h in forbidden):
        return False
    min_len = max(80, 80 * max(len(required), 1))
    return len(text) >= min_len


def ensure_meeting_prep_persisted(
    *,
    new_messages: list[Any],
    config: RunnableConfig,
) -> str | None:
    """If this turn showed complete prep but did not save, invoke save_meeting_prep.

    Returns the tool result string when auto-save ran, else None.
    """
    if turn_has_successful_save(new_messages):
        configurable = config.get("configurable") or {}
        mark_thread_prep_saved(str(configurable.get("thread_id") or ""))
        return None

    body = extract_prep_body_from_messages(new_messages)
    if not body or not body_looks_complete(body, config):
        return None

    from apps.wu_tanchang_api.agent_factory.agent import save_meeting_prep

    metadata = config.get("metadata") or {}
    configurable = config.get("configurable") or {}
    user_id = metadata.get("user_id", "unknown")
    thread_id = configurable.get("thread_id", "unknown")
    _logger.warning(
        "[MeetingPrepPersist] auto-saving prep omitted by model "
        "user_id=%s thread_id=%s body_len=%s",
        user_id,
        thread_id,
        len(body),
    )
    try:
        result = save_meeting_prep.invoke({"body": body}, config=config)
    except Exception:  # noqa: BLE001
        _logger.exception(
            "[MeetingPrepPersist] auto-save raised user_id=%s thread_id=%s",
            user_id,
            thread_id,
        )
        return None

    result_text = str(result or "")
    if "meeting_prep_saved" in result_text:
        mark_thread_prep_saved(str(thread_id))
        _logger.warning(
            "[MeetingPrepPersist] auto-save ok user_id=%s thread_id=%s",
            user_id,
            thread_id,
        )
    else:
        _logger.error(
            "[MeetingPrepPersist] auto-save failed user_id=%s thread_id=%s result=%s",
            user_id,
            thread_id,
            result_text[:240],
        )
    return result_text
