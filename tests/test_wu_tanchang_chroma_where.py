"""Chroma 0.5 where-clause shaping for wu-tanchang KB search."""
# ruff: noqa: S101, RUF001

from __future__ import annotations

from unittest.mock import MagicMock, patch

from apps.wu_tanchang_api.agent_factory.kb_search import _chroma_where, semantic_search


def test_chroma_where_single_field_stays_flat() -> None:
    assert _chroma_where(vec_type="note", series=None) == {"vec_type": "note"}
    assert _chroma_where(vec_type="all", series=None) is None


def test_chroma_where_vec_type_and_series_uses_and_not_two_keys() -> None:
    where = _chroma_where(vec_type="note", series=["案例说"])
    assert where == {
        "$and": [
            {"vec_type": "note"},
            {"series": "案例说"},
        ]
    }
    assert "vec_type" not in where
    assert "series" not in where


def test_chroma_where_aliases_shop_note_series_names() -> None:
    shop = _chroma_where(vec_type="note", series=["商业探店笔记"])
    alt = _chroma_where(vec_type="note", series=["探店商业笔记"])
    expected = {
        "$and": [
            {"vec_type": "note"},
            {"series": {"$in": ["商业探店笔记", "探店商业笔记"]}},
        ]
    }
    assert shop == expected
    assert alt == expected


@patch("apps.wu_tanchang_api.agent_factory.kb_search.get_embeddings")
@patch("apps.wu_tanchang_api.agent_factory.kb_search._get_kb_connection")
def test_semantic_search_retries_without_series_on_where_error(
    mock_get_conn: MagicMock,
    mock_get_embs: MagicMock,
    tmp_path,
) -> None:
    db_path = tmp_path / "kb.db"
    db_path.write_bytes(b"")
    collection = MagicMock()
    collection.query.side_effect = [
        ValueError("Expected where to have exactly one operator, got two keys"),
        {
            "ids": [[]],
            "distances": [[]],
            "metadatas": [[]],
        },
    ]
    mock_get_conn.return_value = (MagicMock(), collection, db_path)
    mock_get_embs.return_value = [[0.1] * 8]

    hits = semantic_search("茶饮", k=5, series=["商业探店笔记"])
    assert hits == []
    assert collection.query.call_count == 2
    first_where = collection.query.call_args_list[0].kwargs["where"]
    second_where = collection.query.call_args_list[1].kwargs["where"]
    assert first_where["$and"][0] == {"vec_type": "note"}
    assert second_where == {"vec_type": "note"}
