"""Semantic search interface for Wu Tanchang Knowledge Base."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from langchain_core.tools import tool

from apps.wu_tanchang_api.config import load_env_file, resolve_model_config

_logger = logging.getLogger("uvicorn.error")


@dataclass
class NoteHit:
    """Dataclass representing a single search hit from the KB."""

    note_id: str
    title: str
    brand: str | None
    score: float
    matched_keywords: list[str]
    matched_insights: list[str]
    raw_path: str


_chroma_client = None
_collection = None
_api_key = None
_embedding_model = None
_db_path = None


def get_embeddings(
    texts: list[str], api_key: str, model: str = "text-embedding-v3"
) -> list[list[float]]:
    """Fetch embeddings from DashScope API."""
    import dashscope
    from dashscope.embeddings import TextEmbedding

    dashscope.api_key = api_key
    batch_size = 25
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = TextEmbedding.call(model=model, input=batch)
        if resp.status_code != 200:
            raise ValueError(
                f"DashScope embedding failed: code={resp.status_code}, message={resp.message}"
            )

        output = resp.output
        if isinstance(output, dict):
            embs = output.get("embeddings", [])
        else:
            embs = getattr(output, "embeddings", [])

        for item in embs:
            if isinstance(item, dict):
                embeddings.append(item["embedding"])
            else:
                embeddings.append(item.embedding)

    return embeddings


def _ensure_loaded() -> None:
    """Lazily load SQLite database path, ChromaDB client, and DashScope API key."""
    global _chroma_client, _collection, _api_key, _embedding_model, _db_path
    if _chroma_client is not None:
        return

    load_env_file()
    db_env_path = os.environ.get(
        "WU_KB_DB_PATH", "apps/wu_tanchang_api/workspace/kb/kb.db"
    )
    vector_env_dir = os.environ.get(
        "WU_KB_VECTOR_DIR", "apps/wu_tanchang_api/workspace/kb/chroma"
    )
    _embedding_model = os.environ.get("WU_KB_EMBEDDING_MODEL", "text-embedding-v3")

    _db_path = Path(db_env_path)
    vector_dir = Path(vector_env_dir)

    if not _db_path.exists():
        msg = f"SQLite database file not found at {_db_path}. Please run kb_build_db.py first."
        raise FileNotFoundError(msg)

    if not vector_dir.exists() or not list(vector_dir.glob("*")):
        msg = f"ChromaDB directory not found or empty at {vector_dir}. Please run kb_build_vectors.py first."
        raise FileNotFoundError(msg)

    import chromadb

    _chroma_client = chromadb.PersistentClient(path=str(vector_dir))
    _collection = _chroma_client.get_collection("wu_notes")

    try:
        model_config = resolve_model_config(
            provider="qwen", model_name_suffix="MAIN_AGENT_MODEL"
        )
        _api_key = model_config.api_key
    except Exception as e:
        msg = f"Failed to load DashScope API config for search: {e}"
        raise ValueError(msg) from e


def semantic_search(
    query: str,
    *,
    k: int = 5,
    series: list[str] | None = None,
    categories: list[str] | None = None,
    vec_type: Literal["note", "insight", "all"] = "note",
) -> list[NoteHit]:
    """Perform semantic search on the ChromaDB collection and query details from SQLite.

    Args:
        query: User semantic query.
        k: Top-k results.
        series: Optional list of series to filter.
        categories: Optional list of categories to filter.
        vec_type: Filter by vector type ("note", "insight", or "all").

    Returns:
        List of NoteHit items.
    """
    _ensure_loaded()

    # Generate query embedding
    query_emb = get_embeddings([query], _api_key, model=_embedding_model)[0]

    # Build where clause
    where_clause = {}
    if vec_type != "all":
        where_clause["vec_type"] = vec_type

    if series:
        if len(series) == 1:
            where_clause["series"] = series[0]
        else:
            where_clause["series"] = {"$in": series}

    # If filtering categories, retrieve more results from vector db to allow post-filtering
    query_k = k * 4 if categories else k

    results = _collection.query(
        query_embeddings=[query_emb],
        n_results=query_k,
        where=where_clause if where_clause else None,
    )

    hits = []
    seen_notes = set()

    ids = results["ids"][0] if results["ids"] else []
    distances = results["distances"][0] if results["distances"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []

    # Retrieve metadata details from SQLite
    conn = sqlite3.connect(_db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    for idx, doc_id in enumerate(ids):
        metadata = metadatas[idx]
        distance = distances[idx]
        note_id = metadata["note_id"]

        # Calculate similarity score: cosine similarity = 1 - distance
        score = 1.0 - distance

        # Post-filter categories in Python
        if categories:
            note_categories_str = metadata.get("categories", "[]")
            try:
                note_cats = json.loads(note_categories_str)
            except Exception:
                note_cats = []
            if not any(cat in note_cats for cat in categories):
                continue

        # Avoid duplicate notes if vec_type == "all"
        if note_id in seen_notes and vec_type == "all":
            continue

        cursor.execute(
            "SELECT title, brand, raw_path FROM notes WHERE id = ?;", (note_id,)
        )
        note_row = cursor.fetchone()
        if not note_row:
            continue

        cursor.execute(
            "SELECT keyword FROM note_keywords WHERE note_id = ? ORDER BY position;",
            (note_id,),
        )
        keywords = [row["keyword"] for row in cursor.fetchall()]

        cursor.execute(
            "SELECT insight FROM note_insights WHERE note_id = ? ORDER BY position;",
            (note_id,),
        )
        insights = [row["insight"] for row in cursor.fetchall()]

        hits.append(
            NoteHit(
                note_id=note_id,
                title=note_row["title"],
                brand=note_row["brand"],
                score=score,
                matched_keywords=keywords,
                matched_insights=insights,
                raw_path=note_row["raw_path"],
            )
        )
        seen_notes.add(note_id)

        if len(hits) >= k:
            break

    conn.close()
    return hits


@tool
def kb_semantic_search(
    query: str,
    k: int = 5,
    series: str | None = None,
) -> str:
    """语义检索吴探长知识库。query 用用户原始诉求，
    返回 top-k 候选笔记及命中的关键词/洞察、note_id、相似度分数。
    series 可选过滤："商业探店笔记" / "探店记" / "案例说" / "市场调查"。"""
    try:
        series_list = [series] if series else None
        hits = semantic_search(query, k=k, series=series_list, vec_type="note")
        if not hits:
            return "未找到相关笔记。"

        lines = []
        for i, hit in enumerate(hits, 1):
            lines.append(f"### {i}. {hit.title} (ID: `{hit.note_id}`)")
            lines.append(f"- **品牌**: {hit.brand or 'N/A'}")
            lines.append(f"- **相似度分数**: {hit.score:.4f}")
            lines.append(f"- **关键词**: {', '.join(hit.matched_keywords)}")
            lines.append("- **核心洞察**:")
            for ins in hit.matched_insights:
                lines.append(f"  * {ins}")
            lines.append(f"- **路径**: `{hit.raw_path}`")
            lines.append("")
        return "\n".join(lines)
    except FileNotFoundError as e:
        return f"错误: 知识库向量索引未构建，请先运行 kb_build_vectors.py。({e})"
    except Exception as e:
        return f"错误: 语义检索失败 ({e})"


@tool
def get_note_content(note_id: str) -> str:
    """获取指定 note_id 的完整笔记内容。此工具直接从数据库中读取并返回笔记的全文。"""
    try:
        _ensure_loaded()
        conn = sqlite3.connect(_db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT title, content FROM notes WHERE id = ?;", (note_id,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            return f"错误: 未找到 ID 为 '{note_id}' 的笔记。"
        return f"### {row['title']}\n\n{row['content']}"
    except Exception as e:
        return f"错误: 无法获取笔记内容 ({e})"

