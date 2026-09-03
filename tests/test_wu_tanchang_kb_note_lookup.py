"""Lookup Xiaohongshu share口令 + KB note body by xhslink URL."""
# ruff: noqa: S101, RUF001

from __future__ import annotations

import json
from pathlib import Path

import pytest

from apps.wu_tanchang_api.agent_factory.kb_note_lookup import (
    clear_lookup_cache,
    lookup_note_by_xhs_url,
    normalize_xhs_url,
)


def test_normalize_xhs_url_from_share_text() -> None:
    blob = (
        "吴探长从3方面拆解 ⚡️ https://xhslink.cn/o/4J5VnH4DmGh\n"
        "复制这段话，然后去【小红书】就能找到笔记。"
    )
    assert normalize_xhs_url(blob) == "https://xhslink.cn/o/4J5VnH4DmGh"
    assert normalize_xhs_url("https://xhslink.cn/o/4J5VnH4DmGh/") == (
        "https://xhslink.cn/o/4J5VnH4DmGh"
    )
    assert normalize_xhs_url("视频已下架，暂无连接") is None


def test_lookup_note_by_xhs_url_returns_body_and_share_text(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    kb_dir = workspace / "kb" / "raw"
    kb_dir.mkdir(parents=True)
    raw_name = "吴探长商业探店笔记：Punch Monday.md"
    (kb_dir / raw_name).write_text(
        "# Punch Monday\n\n正文：6.9元锚定低价。\n",
        encoding="utf-8",
    )
    share = (
        "吴探长拆解Punch Monday ⚡️ https://xhslink.cn/o/4J5VnH4DmGh\n"
        "复制这段话，然后去【小红书】就能找到笔记。"
    )
    (workspace / "kb" / "xhs_links.json").write_text(
        json.dumps(
            {
                "notes": [
                    {
                        "id": "wu-punch-monday",
                        "raw_stem": "Punch Monday",
                        "url": "https://xhslink.cn/o/4J5VnH4DmGh",
                        "share_text": share,
                        "status": "ok",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (workspace / "kb" / "index.json").write_text(
        json.dumps(
            [
                {
                    "id": "wu-punch-monday",
                    "title": "Punch Monday——6.9元杀出一条血路",
                    "raw_file": f"kb/raw/{raw_name}",
                }
            ]
        ),
        encoding="utf-8",
    )

    clear_lookup_cache()
    note = lookup_note_by_xhs_url(
        "https://xhslink.cn/o/4J5VnH4DmGh",
        workspace=workspace,
    )
    assert note is not None
    assert note["id"] == "wu-punch-monday"
    assert note["title"].startswith("Punch Monday")
    assert "6.9元" in note["body"]
    assert note["share_text"] == share
    assert note["url"] == "https://xhslink.cn/o/4J5VnH4DmGh"
    assert note["status"] == "ok"

    from_share = lookup_note_by_xhs_url(share, workspace=workspace)
    assert from_share is not None
    assert from_share["id"] == "wu-punch-monday"

    missing = lookup_note_by_xhs_url(
        "https://xhslink.cn/o/does-not-exist",
        workspace=workspace,
    )
    assert missing is None
    clear_lookup_cache()


def test_lookup_falls_back_to_sqlite_when_raw_missing(tmp_path: Path) -> None:
    import sqlite3

    workspace = tmp_path / "workspace"
    kb_dir = workspace / "kb"
    kb_dir.mkdir(parents=True)
    db_path = kb_dir / "kb.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE notes (id TEXT PRIMARY KEY, title TEXT, content TEXT);")
    conn.execute(
        "INSERT INTO notes VALUES ('wu-test', '库内标题', '来自 sqlite 的正文');"
    )
    conn.commit()
    conn.close()
    (kb_dir / "xhs_links.json").write_text(
        json.dumps(
            {
                "notes": [
                    {
                        "id": "wu-test",
                        "raw_stem": "stem",
                        "url": "https://xhslink.cn/o/sqlitebody",
                        "share_text": "口令 https://xhslink.cn/o/sqlitebody 打开【小红书】",
                        "status": "ok",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (kb_dir / "index.json").write_text(
        json.dumps([{"id": "wu-test", "title": "库内标题", "raw_file": "kb/raw/missing.md"}]),
        encoding="utf-8",
    )

    clear_lookup_cache()
    note = lookup_note_by_xhs_url(
        "https://xhslink.cn/o/sqlitebody",
        workspace=workspace,
    )
    assert note is not None
    assert note["body"] == "来自 sqlite 的正文"
    assert "小红书" in note["share_text"]
    clear_lookup_cache()


def test_lookup_punch_monday_from_default_workspace() -> None:
    workspace = Path(__file__).resolve().parents[1] / "apps/wu_tanchang_api/workspace"
    links_path = workspace / "kb/xhs_links.json"
    if not links_path.exists():
        pytest.skip("xhs_links.json not present")
    payload = json.loads(links_path.read_text(encoding="utf-8"))
    punch = next(
        (n for n in payload.get("notes", []) if n.get("id") == "wu-punch-monday"),
        None,
    )
    if not punch or not punch.get("url"):
        pytest.skip("punch monday link missing")
    if not punch.get("share_text"):
        pytest.skip("share_text not imported yet")

    clear_lookup_cache()
    note = lookup_note_by_xhs_url(punch["url"], workspace=workspace)
    assert note is not None
    assert note["id"] == "wu-punch-monday"
    assert punch["url"] in note["share_text"]
    assert "小红书" in note["share_text"]
    raw_exists = any((workspace / "kb/raw").glob("*Punch Monday*"))
    if raw_exists:
        assert len(note["body"]) > 40
    clear_lookup_cache()
