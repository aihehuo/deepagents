#!/usr/bin/env python3
"""Generate embeddings using DashScope and save to ChromaDB."""

import argparse
import json
import logging
import os
import sqlite3
import sys
from pathlib import Path

# Ensure we can import from apps.wu_tanchang_api
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from apps.wu_tanchang_api.config import load_env_file, resolve_model_config

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("kb_build_vectors")

load_env_file()


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
        logger.info("Requesting embedding batch of size %d...", len(batch))
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


def main() -> None:
    db_env_path = os.environ.get(
        "WU_KB_DB_PATH", "apps/wu_tanchang_api/workspace/kb/kb.db"
    )
    vector_env_dir = os.environ.get(
        "WU_KB_VECTOR_DIR", "apps/wu_tanchang_api/workspace/kb/chroma"
    )
    embedding_model = os.environ.get("WU_KB_EMBEDDING_MODEL", "text-embedding-v3")

    parser = argparse.ArgumentParser(
        description="Generate embeddings and build ChromaDB vectors."
    )
    parser.add_argument(
        "--db-path", type=str, default=db_env_path, help="Path to SQLite database"
    )
    parser.add_argument(
        "--vector-dir",
        type=str,
        default=vector_env_dir,
        help="Directory for ChromaDB storage",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=embedding_model,
        help="DashScope embedding model name",
    )
    parser.add_argument(
        "--force", action="store_true", help="Force rebuilding all vectors"
    )
    args = parser.parse_args()

    db_path = Path(args.db_path)
    vector_dir = Path(args.vector_dir)

    if not db_path.exists():
        logger.error(
            "SQLite database not found at %s. Please run kb_build_db.py first.", db_path
        )
        sys.exit(1)

    # Get DashScope API key from configuration
    try:
        model_config = resolve_model_config(
            provider="qwen", model_name_suffix="MAIN_AGENT_MODEL"
        )
        api_key = model_config.api_key
        if not api_key:
            raise ValueError("DashScope API key not found in config or environment")
    except Exception as e:
        logger.error("Failed to resolve DashScope API config: %s", e)
        sys.exit(1)

    import chromadb

    # Initialize ChromaDB client
    vector_dir.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(vector_dir))
    collection = chroma_client.get_or_create_collection(
        name="wu_notes", metadata={"hnsw:space": "cosine"}
    )

    # Connect to SQLite
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Query all notes
    cursor.execute(
        "SELECT id, title, series, brand, city, categories, topics, content_sha FROM notes;"
    )
    notes = cursor.fetchall()

    logger.info("Found %d notes in SQLite database", len(notes))

    updated_notes = 0
    skipped_notes = 0

    for note in notes:
        note_id = note["id"]
        title = note["title"]
        series = note["series"]
        brand = note["brand"]
        city = note["city"]
        categories_raw = note["categories"]
        content_sha = note["content_sha"]

        # Check if already up-to-date in ChromaDB
        if not args.force:
            existing = collection.get(where={"note_id": note_id}, limit=1)
            if existing and existing["metadatas"]:
                # If first item has the same content_sha, assume it's up to date
                existing_sha = existing["metadatas"][0].get("content_sha")
                if existing_sha == content_sha:
                    skipped_notes += 1
                    continue

        logger.info("Generating embeddings for note %s (%s)...", note_id, title)

        # Retrieve keywords and insights from SQLite for this note
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

        # Construct document strings
        keywords_str = "、".join(keywords)
        insights_str = "；".join(insights)

        # Convert categories JSON string to human-readable string for the doc
        try:
            cats = json.loads(categories_raw)
            cats_str = "/".join(cats)
        except Exception:
            cats_str = ""
            cats = []

        # Documents to embed
        # 1. Note summary document
        note_doc = f"{title}\n\n关键词：{keywords_str}\n\n核心洞察：{insights_str}"
        if cats_str or city:
            note_doc += f"\n\n品类/城市：{cats_str}/{city}"

        docs = [note_doc]
        # 2. Insights documents
        docs.extend(insights)

        # Generate embeddings
        try:
            embeddings = get_embeddings(docs, api_key, model=args.model)
        except Exception as e:
            logger.error("Failed to get embeddings for note %s: %s", note_id, e)
            continue

        # Delete existing vectors for this note first
        collection.delete(where={"note_id": note_id})

        # Prep metadata and ids
        ids = [f"{note_id}-note"]
        metadatas = [
            {
                "note_id": note_id,
                "vec_type": "note",
                "series": series or "",
                "brand": brand or "",
                "city": city or "",
                "categories": categories_raw,
                "content_sha": content_sha,
            }
        ]

        for idx, insight in enumerate(insights):
            ids.append(f"{note_id}-insight-{idx}")
            metadatas.append(
                {
                    "note_id": note_id,
                    "vec_type": "insight",
                    "series": series or "",
                    "brand": brand or "",
                    "city": city or "",
                    "categories": categories_raw,
                    "position": idx,
                    "content_sha": content_sha,
                }
            )

        # Add to ChromaDB
        collection.add(
            ids=ids, embeddings=embeddings, metadatas=metadatas, documents=docs
        )
        updated_notes += 1
        logger.info("Saved %d vectors for note %s to ChromaDB", len(ids), note_id)

    conn.close()
    logger.info(
        "ChromaDB build complete. Updated/Rebuilt: %d, Skipped: %d",
        updated_notes,
        skipped_notes,
    )


if __name__ == "__main__":
    main()
