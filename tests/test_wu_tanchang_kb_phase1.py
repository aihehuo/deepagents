"""Unit tests for Wu Tanchang KB Phase 1 logic."""
# ruff: noqa: S101, RUF001, TC003

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from apps.wu_tanchang_api.agent_factory.kb_search import (
    NoteHit,
    get_note_content,
    kb_semantic_search,
    semantic_search,
)
from apps.wu_tanchang_api.scripts.kb_build_db import validate_db_constraints
from apps.wu_tanchang_api.scripts.kb_extract import validate_extraction
from apps.wu_tanchang_api.scripts.kb_import_xhs_links import (
    build_xhs_links,
    extract_xhs_url,
)


def test_validate_extraction_valid() -> None:
    """Test validate_extraction with fully valid data."""
    valid_data = {
        "keywords": ["咖啡", "AOKKA", "日咖夜酒", "低价锚定", "品牌学", "社群运营"],
        "insights": [
            "明档厨房将现场制作过程转化为可感知的品牌信任资产。",
            "反选址降低租金并卡位高线城市，对传统渠道降维打击。",
            "聚焦垂直品类建立长效壁垒，突破同质化竞争红海。",
            "依靠社群裂变进行精准获客，降低单店私域获客成本。",
            "利用联名活动与时令菜单持续更新场景内容对抗审美疲劳。",
        ],
    }
    ok, err = validate_extraction(valid_data)
    assert ok, f"Expected validation to succeed, got error: {err}"


def test_validate_extraction_invalid_keywords_count() -> None:
    """Test validate_extraction fails when keywords count is out of bounds (6-12)."""
    # Too few keywords (5)
    data_too_few = {"keywords": ["咖啡", "AOKKA", "日咖夜酒", "低价锚定", "品牌学"], "insights": ["这是一条长度超过十个字符的有效核心洞察语句。"] * 5}
    ok, err = validate_extraction(data_too_few)
    assert not ok
    assert "keywords count must be between 6 and 12" in err

    # Too many keywords (13)
    data_too_many = {"keywords": [f"词{i}" for i in range(13)], "insights": ["这是一条长度超过十个字符的有效核心洞察语句。"] * 5}
    ok, err = validate_extraction(data_too_many)
    assert not ok
    assert "keywords count must be between 6 and 12" in err


def test_validate_extraction_invalid_keyword_chinese_length() -> None:
    """Test validate_extraction fails when a keyword has more than 8 Chinese characters."""
    data = {
        "keywords": ["一个超级无敌长长长的关键词", "AOKKA", "日咖夜酒", "低价锚定", "品牌学", "社群运营"],
        "insights": ["这是一条长度超过十个字符的有效核心洞察语句。"] * 5,
    }
    ok, err = validate_extraction(data)
    assert not ok
    assert "exceeds the limit of 8" in err


def test_validate_extraction_invalid_insights_count() -> None:
    """Test validate_extraction fails when insights count is out of bounds (5-8)."""
    # Too few insights (4)
    data_too_few = {
        "keywords": ["咖啡", "AOKKA", "日咖夜酒", "低价锚定", "品牌学", "社群运营"],
        "insights": ["这是一条长度超过十个字符的有效核心洞察语句。"] * 4,
    }
    ok, err = validate_extraction(data_too_few)
    assert not ok
    assert "insights count must be between 5 and 8" in err


def test_validate_extraction_invalid_insight_length() -> None:
    """Test validate_extraction fails when an insight is too short or too long."""
    # Too short (< 10 Chinese characters)
    data_too_short = {
        "keywords": ["咖啡", "AOKKA", "日咖夜酒", "低价锚定", "品牌学", "社群运营"],
        "insights": [
            "太短了",  # Only 3 Chinese characters
            "反选址降低租金并卡位高线城市，对传统渠道降维打击。",
            "聚焦垂直品类建立长效壁垒，突破同质化竞争红海。",
            "依靠社群裂变进行精准获客，降低单店私域获客成本。",
            "利用联名活动与时令菜单持续更新场景内容对抗审美疲劳。",
        ],
    }
    ok, err = validate_extraction(data_too_short)
    assert not ok
    assert "is not within the limit of 10-50" in err


def test_validate_extraction_insight_contains_path_or_newlines() -> None:
    """Test validate_extraction fails when insight contains path characters or newlines."""
    # Contains .md
    data_path = {
        "keywords": ["咖啡", "AOKKA", "日咖夜酒", "低价锚定", "品牌学", "社群运营"],
        "insights": [
            "明档厨房将现场制作过程转化为可感知的品牌信任.md资产",
            "反选址降低租金并卡位高线城市，对传统渠道降维打击。",
            "聚焦垂直品类建立长效壁垒，突破同质化竞争红海。",
            "依靠社群裂变进行精准获客，降低单店私域获客成本。",
            "利用联名活动与时令菜单持续更新场景内容对抗审美疲劳。",
        ],
    }
    ok, err = validate_extraction(data_path)
    assert not ok
    assert "contains path or file extension characters" in err

    # Contains newlines
    data_newline = {
        "keywords": ["咖啡", "AOKKA", "日咖夜酒", "低价锚定", "品牌学", "社群运营"],
        "insights": [
            "明档厨房将现场制作过程\n转化为可感知的品牌信任资产",
            "反选址降低租金并卡位高线城市，对传统渠道降维打击。",
            "聚焦垂直品类建立长效壁垒，突破同质化竞争红海。",
            "依靠社群裂变进行精准获客，降低单店私域获客成本。",
            "利用联名活动与时令菜单持续更新场景内容对抗审美疲劳。",
        ],
    }
    ok, err = validate_extraction(data_newline)
    assert not ok
    assert "contains newlines" in err


def test_validate_db_constraints() -> None:
    """Test SQLite DB constraints validator."""
    # Valid should not raise error
    validate_db_constraints(["咖啡", "AOKKA", "不超过八字关键词"], ["这是一条长度超过十个字符的有效核心洞察语句。"])

    # Invalid keyword (> 8 Chinese characters)
    with pytest.raises(ValueError, match="exceeds 8 Chinese characters limit"):
        validate_db_constraints(["一个包含超级超级超级无敌长长长的关键词"], ["这是一条长度超过十个字符的有效核心洞察语句。"])

    # Invalid insight (< 10 Chinese characters)
    with pytest.raises(ValueError, match="must contain 10-50 Chinese characters"):
        validate_db_constraints(["咖啡"], ["太短了"])

    # Invalid insight contains newlines
    with pytest.raises(ValueError, match="contains newlines"):
        validate_db_constraints(["咖啡"], ["这是一条有效的洞察语句\n但是有换行"])


@pytest.fixture
def mock_db_and_chroma(tmp_path: Path) -> tuple[Path, MagicMock]:
    """Set up an in-memory SQLite DB and mock Chroma collection for search testing."""
    # 1. Create SQLite DB schema and insert mock data
    db_path = tmp_path / "test_kb.db"
    conn = sqlite3.connect(db_path)
    conn.execute("""
    CREATE TABLE notes (
      id            TEXT PRIMARY KEY,
      title         TEXT NOT NULL,
      series        TEXT,
      brand         TEXT,
      city          TEXT,
      categories    TEXT,
      topics        TEXT,
      raw_path      TEXT NOT NULL,
      content_sha   TEXT NOT NULL,
      content       TEXT NOT NULL,
      created_at    TEXT NOT NULL,
      updated_at    TEXT NOT NULL
    );
    """)
    conn.execute("""
    CREATE TABLE note_keywords (
      note_id  TEXT NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
      position INTEGER NOT NULL,
      keyword  TEXT NOT NULL,
      PRIMARY KEY (note_id, position)
    );
    """)
    conn.execute("""
    CREATE TABLE note_insights (
      note_id  TEXT NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
      position INTEGER NOT NULL,
      insight  TEXT NOT NULL,
      PRIMARY KEY (note_id, position)
    );
    """)

    # Insert mock note
    conn.execute("""
    INSERT INTO notes VALUES (
        'wu-test', '测试笔记标题', '商业探店笔记', '测试品牌', '上海', '["烘焙", "咖啡"]', '[]',
        'kb/raw/test.md', 'sha256', 'content', '2026-06-13', '2026-06-13'
    );
    """)
    conn.execute("INSERT INTO note_keywords VALUES ('wu-test', 0, '关键词一');")
    conn.execute("INSERT INTO note_keywords VALUES ('wu-test', 1, '关键词二');")
    conn.execute("INSERT INTO note_insights VALUES ('wu-test', 0, '这是一条测试洞察语句一。');")
    conn.execute("INSERT INTO note_insights VALUES ('wu-test', 1, '这是一条测试洞察语句二。');")
    conn.commit()
    conn.close()

    # 2. Mock Chroma collection query results
    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "ids": [["wu-test-note"]],
        "distances": [[0.35]],
        "metadatas": [[{"note_id": "wu-test", "vec_type": "note", "categories": '["烘焙", "咖啡"]'}]],
        "documents": [["测试笔记标题\n\n关键词：关键词一、关键词二\n\n核心洞察：这是一条测试洞察语句一。；这是一条测试洞察语句二。"]],
    }

    return db_path, mock_collection


@patch("apps.wu_tanchang_api.agent_factory.kb_search.get_embeddings")
@patch("apps.wu_tanchang_api.agent_factory.kb_search._get_kb_connection")
def test_semantic_search_success(
    mock_get_conn: MagicMock,
    mock_get_embs: MagicMock,
    mock_db_and_chroma: tuple[Path, MagicMock],
) -> None:
    """Test semantic_search successfully queries Chroma and SQLite and maps objects."""
    db_path, mock_coll = mock_db_and_chroma

    # Mock _get_kb_connection to return our test db and collection
    mock_get_conn.return_value = (MagicMock(), mock_coll, db_path)

    mock_get_embs.return_value = [[0.1] * 1024]

    # Test basic search
    hits = semantic_search("测试查询", k=1)

    assert len(hits) == 1
    hit = hits[0]
    assert isinstance(hit, NoteHit)
    assert hit.note_id == "wu-test"
    assert hit.title == "测试笔记标题"
    assert hit.brand == "测试品牌"
    assert hit.score == pytest.approx(0.65)  # 1.0 - 0.35
    assert hit.matched_keywords == ["关键词一", "关键词二"]
    assert hit.matched_insights == ["这是一条测试洞察语句一。", "这是一条测试洞察语句二。"]
    assert hit.raw_path == "kb/raw/test.md"


@patch("apps.wu_tanchang_api.agent_factory.kb_search.semantic_search")
def test_kb_semantic_search_tool(mock_search: MagicMock) -> None:
    """Test the kb_semantic_search @tool formats output as markdown properly."""
    # Mock search response
    mock_search.return_value = [
        NoteHit(
            note_id="wu-test",
            title="测试笔记标题",
            brand="测试品牌",
            score=0.85,
            matched_keywords=["关键一", "关键二"],
            matched_insights=["洞察一", "洞察二"],
            raw_path="kb/raw/test.md",
            xhs_url="https://xhslink.cn/o/test123",
        )
    ]

    res = kb_semantic_search.invoke({"query": "测试需求", "k": 1})

    assert "### 1. 测试笔记标题 (ID: `wu-test`)" in res
    assert "- **品牌**: 测试品牌" in res
    assert "- **相似度分数**: 0.8500" in res
    assert "- **关键词**: 关键一, 关键二" in res
    assert "  * 洞察一" in res
    assert "  * 洞察二" in res
    assert "- **探店笔记网页**: https://xhslink.cn/o/test123" in res
    assert "- **路径**: `kb/raw/test.md`" in res


@patch("apps.wu_tanchang_api.agent_factory.kb_search.semantic_search")
def test_kb_semantic_search_tool_omits_xhs_line_when_missing(mock_search: MagicMock) -> None:
    """Do not emit 暂无 — other tenants have no xhs_links.json."""
    mock_search.return_value = [
        NoteHit(
            note_id="wu-test",
            title="测试笔记标题",
            brand="测试品牌",
            score=0.85,
            matched_keywords=["关键一"],
            matched_insights=["洞察一"],
            raw_path="kb/raw/test.md",
            xhs_url=None,
        )
    ]
    res = kb_semantic_search.invoke({"query": "测试需求", "k": 1})
    assert "探店笔记网页" not in res


@patch("apps.wu_tanchang_api.agent_factory.kb_search._get_kb_connection")
def test_kb_semantic_search_tool_handles_missing_db_error(mock_get_conn: MagicMock) -> None:
    """Test the @tool handles FileNotFoundError gracefully and returns fallback string."""
    mock_get_conn.side_effect = FileNotFoundError("SQLite database file not found")

    res = kb_semantic_search.invoke({"query": "测试", "k": 1})
    assert "错误: 知识库向量索引未构建" in res


def test_extract_xhs_url_from_share_text() -> None:
    """Extract short link from Xiaohongshu share口令."""
    text = "复制这段话 https://xhslink.cn/o/4J5VnH4DmGh 打开小红书"
    assert extract_xhs_url(text) == "https://xhslink.cn/o/4J5VnH4DmGh"
    glued = "https://xhslink.cn/o/4J5VnH4DmGh，复制本条信息，打开【小红书】App"
    assert extract_xhs_url(glued) == "https://xhslink.cn/o/4J5VnH4DmGh"
    assert extract_xhs_url("视频已下架，暂无连接") is None


def test_kb_import_xhs_links_from_workspace_excel() -> None:
    """Import script aligns Excel stems with index.json and writes expected counts."""
    workspace = Path(__file__).resolve().parents[1] / "apps/wu_tanchang_api/workspace"
    xlsx_path = workspace / "kb/raw/短视频连接.xlsx"
    index_path = workspace / "kb/index.json"
    if not xlsx_path.exists():
        pytest.skip("短视频连接.xlsx not present locally (gitignored raw/)")

    payload = build_xhs_links(
        xlsx_path=xlsx_path,
        index_path=index_path,
        generated_from="kb/raw/短视频连接.xlsx",
    )

    assert len(payload["notes"]) == 70
    assert sum(1 for n in payload["notes"] if n["status"] == "ok") == 69
    assert sum(1 for n in payload["notes"] if n["status"] == "taken_down") == 1
    assert len(payload["unmatched_excel_rows"]) == 1
    assert any("3又1/2" in row for row in payload["unmatched_excel_rows"])

    bake = next(n for n in payload["notes"] if n["id"] == "wu-bake-no-title")
    assert bake["url"] is None
    assert bake["status"] == "taken_down"

    punch = next(n for n in payload["notes"] if n["id"] == "wu-punch-monday")
    assert punch["url"] and punch["url"].startswith("https://xhslink.cn/")


@patch("apps.wu_tanchang_api.agent_factory.kb_search._get_kb_connection")
def test_get_note_content_includes_xhs_url(mock_get_conn: MagicMock, tmp_path: Path) -> None:
    """get_note_content prepends探店笔记网页 when xhs_links.json has a URL."""
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    db_path = kb_dir / "kb.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE notes (id TEXT PRIMARY KEY, title TEXT, content TEXT);"
    )
    conn.execute(
        "INSERT INTO notes VALUES ('wu-test', '测试标题', '正文内容');"
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
                        "url": "https://xhslink.cn/o/notecontent",
                        "status": "ok",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    mock_get_conn.return_value = (MagicMock(), MagicMock(), db_path)
    res = get_note_content.invoke({"note_id": "wu-test"})
    assert "**探店笔记网页**: https://xhslink.cn/o/notecontent" in res
    assert "正文内容" in res

