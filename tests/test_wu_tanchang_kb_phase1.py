"""Unit tests for Wu Tanchang KB Phase 1 logic."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from apps.wu_tanchang_api.scripts.kb_extract import validate_extraction
from apps.wu_tanchang_api.scripts.kb_build_db import validate_db_constraints
from apps.wu_tanchang_api.agent_factory.kb_search import NoteHit, semantic_search, kb_semantic_search


def test_validate_extraction_valid() -> None:
    """Test validate_extraction with fully valid data."""
    valid_data = {
        "keywords": ["咖啡", "AOKKA", "日咖夜酒", "低价锚定", "品牌学", "社群运营"],
        "insights": [
            "明档厨房将现场制作过程转化为可感知的品牌信任资产。",
            "反选址降低租金并卡位高线城市，对传统渠道降维打击。",
            "聚焦垂直品类建立长效壁垒，突破同质化竞争红海。",
            "依靠社群裂变进行精准获客，降低单店私域获客成本。",
            "利用联名活动与时令菜单持续更新场景内容对抗审美疲劳。"
        ]
    }
    ok, err = validate_extraction(valid_data)
    assert ok, f"Expected validation to succeed, got error: {err}"


def test_validate_extraction_invalid_keywords_count() -> None:
    """Test validate_extraction fails when keywords count is out of bounds (6-12)."""
    # Too few keywords (5)
    data_too_few = {
        "keywords": ["咖啡", "AOKKA", "日咖夜酒", "低价锚定", "品牌学"],
        "insights": ["这是一条长度超过十个字符的有效核心洞察语句。"] * 5
    }
    ok, err = validate_extraction(data_too_few)
    assert not ok
    assert "keywords count must be between 6 and 12" in err

    # Too many keywords (13)
    data_too_many = {
        "keywords": [f"词{i}" for i in range(13)],
        "insights": ["这是一条长度超过十个字符的有效核心洞察语句。"] * 5
    }
    ok, err = validate_extraction(data_too_many)
    assert not ok
    assert "keywords count must be between 6 and 12" in err


def test_validate_extraction_invalid_keyword_chinese_length() -> None:
    """Test validate_extraction fails when a keyword has more than 8 Chinese characters."""
    data = {
        "keywords": ["一个超级无敌长长长的关键词", "AOKKA", "日咖夜酒", "低价锚定", "品牌学", "社群运营"],
        "insights": ["这是一条长度超过十个字符的有效核心洞察语句。"] * 5
    }
    ok, err = validate_extraction(data)
    assert not ok
    assert "exceeds the limit of 8" in err


def test_validate_extraction_invalid_insights_count() -> None:
    """Test validate_extraction fails when insights count is out of bounds (5-8)."""
    # Too few insights (4)
    data_too_few = {
        "keywords": ["咖啡", "AOKKA", "日咖夜酒", "低价锚定", "品牌学", "社群运营"],
        "insights": ["这是一条长度超过十个字符的有效核心洞察语句。"] * 4
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
            "利用联名活动与时令菜单持续更新场景内容对抗审美疲劳。"
        ]
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
            "利用联名活动与时令菜单持续更新场景内容对抗审美疲劳。"
        ]
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
            "利用联名活动与时令菜单持续更新场景内容对抗审美疲劳。"
        ]
    }
    ok, err = validate_extraction(data_newline)
    assert not ok
    assert "contains newlines" in err


def test_validate_db_constraints() -> None:
    """Test SQLite DB constraints validator."""
    # Valid should not raise error
    validate_db_constraints(
        ["咖啡", "AOKKA", "不超过八字关键词"],
        ["这是一条长度超过十个字符的有效核心洞察语句。"]
    )


    # Invalid keyword (> 8 Chinese characters)
    with pytest.raises(ValueError, match="exceeds 8 Chinese characters limit"):
        validate_db_constraints(
            ["一个包含超级超级超级无敌长长长的关键词"],
            ["这是一条长度超过十个字符的有效核心洞察语句。"]
        )

    # Invalid insight (< 10 Chinese characters)
    with pytest.raises(ValueError, match="must contain 10-50 Chinese characters"):
        validate_db_constraints(
            ["咖啡"],
            ["太短了"]
        )

    # Invalid insight contains newlines
    with pytest.raises(ValueError, match="contains newlines"):
        validate_db_constraints(
            ["咖啡"],
            ["这是一条有效的洞察语句\n但是有换行"]
        )


@pytest.fixture
def mock_db_and_chroma(tmp_path: Path):
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
        "metadatas": [[{
            "note_id": "wu-test",
            "vec_type": "note",
            "categories": '["烘焙", "咖啡"]'
        }]],
        "documents": [["测试笔记标题\n\n关键词：关键词一、关键词二\n\n核心洞察：这是一条测试洞察语句一。；这是一条测试洞察语句二。"]]
    }
    
    return db_path, mock_collection


@patch("apps.wu_tanchang_api.agent_factory.kb_search.get_embeddings")
@patch("apps.wu_tanchang_api.agent_factory.kb_search.resolve_model_config")
@patch("apps.wu_tanchang_api.agent_factory.kb_search._collection")
@patch("apps.wu_tanchang_api.agent_factory.kb_search._db_path")
@patch("apps.wu_tanchang_api.agent_factory.kb_search._api_key", "fake-api-key")
@patch("apps.wu_tanchang_api.agent_factory.kb_search._chroma_client")
def test_semantic_search_success(
    mock_chroma_client,
    mock_db_path_var,
    mock_collection_var,
    mock_model_config,
    mock_get_embs,
    mock_db_and_chroma
) -> None:
    """Test semantic_search successfully queries Chroma and SQLite and maps objects."""
    db_path, mock_coll = mock_db_and_chroma
    
    # Bind mocks to internal kb_search variables
    mock_collection_var.__get__ = MagicMock(return_value=mock_coll)
    mock_collection_var.query = mock_coll.query
    
    # We patch _ensure_loaded to do nothing, and set the internal variables manually
    with patch("apps.wu_tanchang_api.agent_factory.kb_search._db_path", db_path), \
         patch("apps.wu_tanchang_api.agent_factory.kb_search._collection", mock_coll), \
         patch("apps.wu_tanchang_api.agent_factory.kb_search._api_key", "fake-key"), \
         patch("apps.wu_tanchang_api.agent_factory.kb_search._ensure_loaded") as mock_ensure:
        
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


@patch("apps.wu_tanchang_api.agent_factory.kb_search.get_embeddings")
@patch("apps.wu_tanchang_api.agent_factory.kb_search._ensure_loaded")
@patch("apps.wu_tanchang_api.agent_factory.kb_search.semantic_search")
def test_kb_semantic_search_tool(
    mock_search,
    mock_ensure,
    mock_get_embs
) -> None:
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
            raw_path="kb/raw/test.md"
        )
    ]
    
    res = kb_semantic_search.invoke({"query": "测试需求", "k": 1})
    
    assert "### 1. 测试笔记标题 (ID: `wu-test`)" in res
    assert "- **品牌**: 测试品牌" in res
    assert "- **相似度分数**: 0.8500" in res
    assert "- **关键词**: 关键一, 关键二" in res
    assert "  * 洞察一" in res
    assert "  * 洞察二" in res
    assert "- **路径**: `kb/raw/test.md`" in res


@patch("apps.wu_tanchang_api.agent_factory.kb_search._ensure_loaded")
def test_kb_semantic_search_tool_handles_missing_db_error(mock_ensure) -> None:
    """Test the @tool handles FileNotFoundError gracefully and returns fallback string."""
    mock_ensure.side_effect = FileNotFoundError("SQLite database file not found")
    
    res = kb_semantic_search.invoke({"query": "测试", "k": 1})
    assert "错误: 知识库向量索引未构建" in res
