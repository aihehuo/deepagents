"""Filter LangGraph message-stream replays of checkpoint history.

``astream(..., stream_mode=["messages"])`` on a resumed thread can re-emit
completed AIMessages from earlier turns as ``AIMessageChunk``s. Each chunk
carries a distinct LangChain id, so Micro treats them as new chat bubbles.

Calendar export 2026-09-03 (wu_agent_1): after the user answered 店面面积,
five prior assistant replies were posted again as new AI messages before the
real next question. Skip those by pre-turn id and by exact full-text match.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk

# Short tokens like "好的" can legitimately repeat. Only fingerprint full replies.
_MIN_REPLAY_TEXT_CHARS = 40


@dataclass(frozen=True)
class PreTurnAiSnapshot:
    ids: frozenset[str]
    texts: frozenset[str]


def normalize_message_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""


def message_id_from_chunk(message: Any) -> str | None:
    if hasattr(message, "id"):
        mid = getattr(message, "id", None)
        return mid if isinstance(mid, str) and mid else None
    if isinstance(message, dict):
        mid = message.get("id")
        return mid if isinstance(mid, str) and mid else None
    return None


def snapshot_pre_turn_ai(messages: list[Any] | None) -> PreTurnAiSnapshot:
    ids: set[str] = set()
    texts: set[str] = set()
    for msg in messages or []:
        if not isinstance(msg, (AIMessage, AIMessageChunk)):
            continue
        mid = message_id_from_chunk(msg)
        if mid:
            ids.add(mid)
        text = normalize_message_text(msg).strip()
        if len(text) >= _MIN_REPLAY_TEXT_CHARS:
            texts.add(text)
    return PreTurnAiSnapshot(ids=frozenset(ids), texts=frozenset(texts))


def should_emit_streamed_ai_text(
    message: Any, snapshot: PreTurnAiSnapshot
) -> bool:
    """Return True only for this-turn assistant text, not checkpoint replay."""
    if not isinstance(message, (AIMessage, AIMessageChunk)):
        return False
    text = normalize_message_text(message)
    if not text:
        return False
    mid = message_id_from_chunk(message)
    if mid and mid in snapshot.ids:
        return False
    stripped = text.strip()
    if len(stripped) >= _MIN_REPLAY_TEXT_CHARS and stripped in snapshot.texts:
        return False
    return True


def extract_main_agent_ai_delta(
    subgraph_path: Any,
    stream_mode: str,
    data: Any,
    snapshot: PreTurnAiSnapshot,
) -> tuple[str, str | None] | None:
    """Parse a subgraphs=True astream item into emit-able main-agent text."""
    if stream_mode != "messages" or subgraph_path:
        return None
    if not isinstance(data, tuple) or not data:
        return None
    msg_chunk = data[0]
    if not isinstance(msg_chunk, AIMessageChunk):
        return None
    if not should_emit_streamed_ai_text(msg_chunk, snapshot):
        return None
    text = normalize_message_text(msg_chunk)
    if not text:
        return None
    return text, message_id_from_chunk(msg_chunk)
