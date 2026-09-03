"""Ground meeting-prep section 四 in this turn's kb_semantic_search hits.

Search can succeed (search_log shows 5 tea notes) while the front agent still
saves a skeleton 「四、可参考案例」 and never names the brands in chat.
Keep the last hits per conversation and splice 1–2 real citations into the
saved body when section 四 is missing, placeholder, or ungrounded.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from typing import Any

_logger = logging.getLogger("uvicorn.error")

_LOCK = threading.Lock()
_HITS_BY_KEY: dict[str, tuple["StoredKbHit", ...]] = {}

_SECTION_SPLIT = re.compile(r"(?=^\*{0,2}[一二三四]、)", re.MULTILINE)
_SECTION_FOUR = re.compile(r"^\*{0,2}四、", re.MULTILINE)
_PLACEHOLDER_MARKERS = (
    "来自 kb_analyst",
    "完整URL",
    "完整短链",
    "1–2 个品牌",
    "1-2 个品牌",
    "必须换成",
    "禁止把上一行模板",
)
_MISS_PHRASE = "本轮知识库检索未命中可直接引用的探店笔记"
_MAX_CASES = 2


@dataclass(frozen=True)
class StoredKbHit:
    note_id: str
    title: str
    brand: str | None
    xhs_url: str | None
    insight: str | None = None


def remember_kb_hits(config: dict[str, Any] | None, hits: list[Any]) -> None:
    """Store this search's hits under conversation keys the parent turn can find."""
    stored = tuple(_coerce_hit(h) for h in hits)
    keys = config_hit_keys(config)
    if not keys:
        return
    with _LOCK:
        for key in keys:
            _HITS_BY_KEY[key] = stored


def last_kb_hits_for_config(config: dict[str, Any] | None) -> tuple[StoredKbHit, ...]:
    with _LOCK:
        for key in config_hit_keys(config):
            found = _HITS_BY_KEY.get(key)
            if found is not None:
                return found
    return ()


def clear_kb_hits() -> None:
    with _LOCK:
        _HITS_BY_KEY.clear()


def config_hit_keys(config: dict[str, Any] | None) -> list[str]:
    if not config:
        return []
    metadata = config.get("metadata") or {}
    configurable = config.get("configurable") or {}
    keys: list[str] = []
    user = str(metadata.get("user_id") or "").strip()
    conv = str(metadata.get("conversation_id") or "").strip()
    if user and conv:
        keys.append(f"conv::{user}::{conv}")
    session = str(
        metadata.get("session_id") or configurable.get("thread_id") or ""
    ).strip()
    if session:
        keys.append(f"thread::{session}")
        if session.startswith("wt::"):
            parts = session.split("::")
            if len(parts) >= 4:
                parent = "::".join(parts[:4])
                if parent != session:
                    keys.append(f"thread::{parent}")
    # dedupe, keep order
    out: list[str] = []
    for key in keys:
        if key not in out:
            out.append(key)
    return out


def format_kb_cases_markdown(hits: tuple[StoredKbHit, ...] | list[StoredKbHit]) -> str:
    chosen = list(hits)[:_MAX_CASES]
    if not chosen:
        return _MISS_PHRASE + "，面谈时由吴探长按品类口头补案例。"
    lines = ["知识库里和你这店最贴近的探店笔记：", ""]
    for hit in chosen:
        lines.append(f"- {_case_link_line(hit)}")
        if hit.insight:
            lines.append(f"  {hit.insight}")
    return "\n".join(lines).strip()


def format_section_four(hits: tuple[StoredKbHit, ...] | list[StoredKbHit]) -> str:
    chosen = list(hits)[:_MAX_CASES]
    lines = ["四、可参考案例", ""]
    if not chosen:
        lines.append(_MISS_PHRASE + "，面谈时由吴探长按品类口头补案例。")
        return "\n".join(lines)
    for hit in chosen:
        lines.append(f"- {_case_link_line(hit)}")
        if hit.insight:
            lines.append(f"  {hit.insight}")
    return "\n".join(lines)


def ground_meeting_prep_kb_cases(
    body: str,
    hits: tuple[StoredKbHit, ...] | list[StoredKbHit],
) -> tuple[str, str | None]:
    """Return (body, chat_markdown_if_filled).

    chat_markdown is set only when we had to splice real hits (or a miss
    phrase) into section 四, so the calendar transcript can show the cases.
    """
    text = (body or "").strip()
    if not text:
        return body, None
    hit_tuple = tuple(hits)
    section = _section_four_text(text)
    if hit_tuple:
        if section and _section_cites_hits(section, hit_tuple) and not _is_placeholder(section):
            return text, None
        filled = _replace_or_append_section_four(text, format_section_four(hit_tuple))
        _logger.info(
            "[kb_cases] spliced %d search hits into meeting-prep section 四",
            min(len(hit_tuple), _MAX_CASES),
        )
        return filled, format_kb_cases_markdown(hit_tuple)
    if section and not _is_placeholder(section) and _MISS_PHRASE in section:
        return text, None
    if _is_placeholder(section or "") or not section:
        filled = _replace_or_append_section_four(text, format_section_four(()))
        return filled, None
    return text, None


def emit_kb_cases_chat(config: dict[str, Any] | None, markdown: str) -> None:
    if not config or not (markdown or "").strip():
        return
    metadata = config.get("metadata") or {}
    callback_url = metadata.get("callback_url")
    if not callback_url:
        return
    configurable = config.get("configurable") or {}
    thread_id = configurable.get("thread_id") or metadata.get("session_id")
    if not thread_id:
        return
    from datetime import datetime
    import uuid

    from apps.wu_tanchang_api.app.callbacks import invoke_callback

    payload = {
        "session_id": thread_id,
        "user_id": metadata.get("user_id"),
        "conversation_id": metadata.get("conversation_id"),
        "agent_name": metadata.get("agent_name") or "default",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "type": "message",
        "message_id": f"wu_kb_cases_{uuid.uuid4().hex[:16]}",
        "message": markdown.strip(),
    }
    try:
        invoke_callback(str(callback_url), payload)
    except Exception as exc:  # noqa: BLE001
        _logger.warning("[kb_cases] emit chat cases failed: %s", exc)


def _coerce_hit(hit: Any) -> StoredKbHit:
    if isinstance(hit, StoredKbHit):
        return hit
    insights = getattr(hit, "matched_insights", None) or []
    insight = ""
    if insights:
        insight = str(insights[0]).strip()[:80]
    return StoredKbHit(
        note_id=str(getattr(hit, "note_id", "") or ""),
        title=str(getattr(hit, "title", "") or ""),
        brand=getattr(hit, "brand", None),
        xhs_url=getattr(hit, "xhs_url", None),
        insight=insight or None,
    )


def _case_link_line(hit: StoredKbHit) -> str:
    label = _case_label(hit)
    if hit.xhs_url:
        return f"[{label} · 看探店笔记]({hit.xhs_url})"
    return label


def _case_label(hit: StoredKbHit) -> str:
    brand = (hit.brand or "").strip()
    if brand and brand.lower() not in {"n/a", "未知", "none"}:
        return brand
    title = (hit.title or "").strip()
    if not title:
        return hit.note_id or "探店笔记"
    return re.split(r"[，。？?！!——：:]", title, maxsplit=1)[0].strip() or title


def _cite_tokens(hit: StoredKbHit) -> list[str]:
    tokens: list[str] = []
    label = _case_label(hit)
    if len(label) >= 2:
        tokens.append(label)
    brand = (hit.brand or "").strip()
    if len(brand) >= 2:
        tokens.append(brand)
    return tokens


def _section_cites_hits(section: str, hits: tuple[StoredKbHit, ...]) -> bool:
    for hit in hits:
        for token in _cite_tokens(hit):
            if token and token in section:
                return True
    return False


def _is_placeholder(section: str) -> bool:
    text = (section or "").strip()
    if len(text) < 24:
        return True
    if any(m in text for m in _PLACEHOLDER_MARKERS) and "xhslink" not in text and "http" not in text:
        return True
    return False


def _section_four_text(body: str) -> str | None:
    parts = _SECTION_SPLIT.split(body)
    for part in parts:
        if _SECTION_FOUR.match(part.lstrip()):
            lines = part.splitlines()
            content = lines[1:] if len(lines) > 1 else []
            return "\n".join(content).strip() or part.strip()
    return None


def _replace_or_append_section_four(body: str, new_section: str) -> str:
    parts = _SECTION_SPLIT.split(body)
    out: list[str] = []
    replaced = False
    for part in parts:
        if not replaced and _SECTION_FOUR.match(part.lstrip()):
            out.append(new_section.rstrip() + "\n")
            replaced = True
        else:
            out.append(part)
    if not replaced:
        return body.rstrip() + "\n\n" + new_section.rstrip() + "\n"
    return "".join(out)
