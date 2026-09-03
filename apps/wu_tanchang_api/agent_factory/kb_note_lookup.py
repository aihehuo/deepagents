"""Look up Wu Tanchang KB note body + Xiaohongshu share口令 by xhslink URL."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Any

from apps.wu_tanchang_api.agent_factory.utils import default_workspace_path

_logger = logging.getLogger("uvicorn.error")

_XHS_URL_RE = re.compile(r"https://xhslink\.cn/[A-Za-z0-9/_-]+", re.IGNORECASE)


def normalize_xhs_url(value: str | None) -> str | None:
    """Extract a canonical xhslink URL from a URL or share口令 blob."""
    text = (value or "").strip()
    if not text:
        return None
    match = _XHS_URL_RE.search(text)
    if not match:
        return None
    return match.group(0).rstrip("/")


def _workspace_kb_dir(workspace: Path) -> Path:
    return workspace / "kb"


def _load_xhs_notes(kb_dir: Path) -> list[dict[str, Any]]:
    links_path = kb_dir / "xhs_links.json"
    if not links_path.exists():
        return []
    try:
        payload = json.loads(links_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        _logger.warning("[kb_note_lookup] failed to load xhs_links.json: %s", exc)
        return []
    notes = payload.get("notes") if isinstance(payload, dict) else None
    if not isinstance(notes, list):
        return []
    return [n for n in notes if isinstance(n, dict)]


def _load_index_by_id(kb_dir: Path) -> dict[str, dict[str, Any]]:
    index_path = kb_dir / "index.json"
    if not index_path.exists():
        return {}
    try:
        entries = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        _logger.warning("[kb_note_lookup] failed to load index.json: %s", exc)
        return {}
    if not isinstance(entries, list):
        return {}
    by_id: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        note_id = entry.get("id")
        if note_id:
            by_id[str(note_id)] = entry
    return by_id


def _read_raw_markdown(workspace: Path, raw_file: str | None) -> str:
    if not raw_file:
        return ""
    rel = Path(raw_file)
    candidates = [
        workspace / rel,
        workspace / "kb" / "raw" / rel.name,
    ]
    for path in candidates:
        if path.is_file():
            try:
                return path.read_text(encoding="utf-8")
            except OSError as exc:
                _logger.warning("[kb_note_lookup] failed to read %s: %s", path, exc)
    return ""


def _read_sqlite_content(kb_dir: Path, note_id: str) -> tuple[str, str]:
    """Return (title, content) from kb.db when raw markdown is missing."""
    db_path = kb_dir / "kb.db"
    if not db_path.is_file():
        return "", ""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT title, content FROM notes WHERE id = ?;", (note_id,))
        row = cursor.fetchone()
        conn.close()
    except sqlite3.Error as exc:
        _logger.warning("[kb_note_lookup] sqlite read failed: %s", exc)
        return "", ""
    if not row:
        return "", ""
    return str(row["title"] or ""), str(row["content"] or "")


@lru_cache(maxsize=8)
def _url_index(workspace_key: str) -> dict[str, dict[str, Any]]:
    workspace = Path(workspace_key)
    index: dict[str, dict[str, Any]] = {}
    for note in _load_xhs_notes(_workspace_kb_dir(workspace)):
        url = normalize_xhs_url(note.get("url") or note.get("share_text"))
        if not url:
            continue
        index[url.lower()] = note
    return index


def lookup_note_by_xhs_url(
    url: str,
    *,
    workspace: Path | None = None,
) -> dict[str, Any] | None:
    """Return note payload for a Xiaohongshu short link, or None if unknown."""
    canonical = normalize_xhs_url(url)
    if not canonical:
        return None

    ws = (workspace or default_workspace_path()).resolve()
    note = _url_index(str(ws)).get(canonical.lower())
    if not note:
        return None

    note_id = str(note.get("id") or "")
    kb_dir = _workspace_kb_dir(ws)
    index_entry = _load_index_by_id(kb_dir).get(note_id) or {}
    title = str(index_entry.get("title") or note.get("raw_stem") or "").strip()
    body = _read_raw_markdown(ws, index_entry.get("raw_file"))
    if not body.strip():
        db_title, db_body = _read_sqlite_content(kb_dir, note_id)
        if db_body.strip():
            body = db_body
        if not title and db_title:
            title = db_title

    share_text = str(note.get("share_text") or "").strip()
    status = str(note.get("status") or "ok")
    note_url = note.get("url")
    note_url_s = str(note_url).strip() if note_url else None

    return {
        "id": note_id,
        "title": title or note_id,
        "body": body.strip(),
        "share_text": share_text,
        "url": note_url_s,
        "status": status,
    }


def clear_lookup_cache() -> None:
    """Test helper: drop the URL index cache."""
    _url_index.cache_clear()
