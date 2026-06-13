#!/usr/bin/env python3
"""Build the SQLite database for Wu Tanchang KB."""

import argparse
import hashlib
import json
import logging
import os
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path


# Ensure we can import from apps.wu_tanchang_api
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from apps.wu_tanchang_api.config import load_env_file

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("kb_build_db")

load_env_file()


def get_sha256(content: str) -> str:
    """Calculate SHA-256 of the content string."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def init_db(db_path: Path) -> sqlite3.Connection:
    """Initialize SQLite database schema."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    
    # Create notes table
    conn.execute("""
    CREATE TABLE IF NOT EXISTS notes (
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
    
    # Create note_keywords table
    conn.execute("""
    CREATE TABLE IF NOT EXISTS note_keywords (
      note_id  TEXT NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
      position INTEGER NOT NULL,
      keyword  TEXT NOT NULL,
      PRIMARY KEY (note_id, position)
    );
    """)
    
    # Create note_insights table
    conn.execute("""
    CREATE TABLE IF NOT EXISTS note_insights (
      note_id  TEXT NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
      position INTEGER NOT NULL,
      insight  TEXT NOT NULL,
      PRIMARY KEY (note_id, position)
    );
    """)
    
    # Create indexes
    conn.execute("CREATE INDEX IF NOT EXISTS idx_notes_brand    ON notes(brand);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_notes_series   ON notes(series);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_keywords_kw    ON note_keywords(keyword);")
    
    conn.commit()
    return conn


def validate_db_constraints(keyword_list: list[str], insight_list: list[str]) -> None:
    """Hard validation before database insertion."""
    for kw in keyword_list:
        # Check keyword Chinese characters count <= 8
        han_count = len(re.findall(r'[\u4e00-\u9fa5]', kw))
        if han_count > 8:
            raise ValueError(f"Keyword '{kw}' exceeds 8 Chinese characters limit")
            
    for ins in insight_list:
        # Check insight Chinese characters count 10-50
        han_count = len(re.findall(r'[\u4e00-\u9fa5]', ins))
        if not (10 <= han_count <= 50):
            raise ValueError(f"Insight '{ins}' must contain 10-50 Chinese characters (got {han_count})")

        if '\n' in ins or '\r' in ins:
            raise ValueError(f"Insight '{ins}' contains newlines")



def main() -> None:
    db_env_path = os.environ.get("WU_KB_DB_PATH", "apps/wu_tanchang_api/workspace/kb/kb.db")
    parser = argparse.ArgumentParser(description="Build/update Wu Tanchang SQLite database.")
    parser.add_argument("--db-path", type=str, default=db_env_path, help="Path to SQLite database")
    parser.add_argument("--index-json", type=str, default="apps/wu_tanchang_api/workspace/kb/index.json", help="Path to index.json")
    parser.add_argument("--extracted-dir", type=str, default="apps/wu_tanchang_api/workspace/kb/extracted", help="Path to extracted JSON files")
    args = parser.parse_args()
    
    db_path = Path(args.db_path)
    index_json_path = Path(args.index_json)
    extracted_dir = Path(args.extracted_dir)
    
    if not index_json_path.exists():
        logger.error("index.json not found at %s", index_json_path)
        sys.exit(1)
        
    workspace_dir = index_json_path.parents[1]
    
    conn = init_db(db_path)
    logger.info("Database initialized at %s", db_path)
    
    with index_json_path.open("r", encoding="utf-8") as f:
        index_data = json.load(f)
        
    inserted_notes = 0
    skipped_notes = 0
    
    for item in index_data:
        note_id = item.get("id")
        raw_file_rel = item.get("raw_file")
        title = item.get("title")
        series = item.get("series")
        brand = item.get("brand")
        city = item.get("city")
        categories = json.dumps(item.get("categories", []), ensure_ascii=False)
        topics = json.dumps(item.get("topics", []), ensure_ascii=False)
        
        if not raw_file_rel:
            continue
            
        raw_file_path = workspace_dir / raw_file_rel
        if not raw_file_path.exists():
            logger.warning("Raw file not found at %s for note %s. Skipping...", raw_file_path, note_id)
            continue
            
        # Read raw content
        with raw_file_path.open("r", encoding="utf-8") as f:
            content = f.read()
        content_sha = get_sha256(content)
        
        # Load extracted JSON
        ext_file = extracted_dir / f"{note_id}.json"
        if not ext_file.exists():
            logger.warning("Extracted metadata file %s does not exist. Please run kb_extract.py first.", ext_file)
            continue
            
        with ext_file.open("r", encoding="utf-8") as f:
            extracted_data = json.load(f)
            
        keywords = extracted_data.get("keywords", [])
        insights = extracted_data.get("insights", [])
        
        # Check idempotency: does the DB already contain this note with the same SHA?
        cursor = conn.cursor()
        cursor.execute("SELECT content_sha FROM notes WHERE id = ?", (note_id,))
        row = cursor.fetchone()
        if row and row[0] == content_sha:
            skipped_notes += 1
            continue
            
        logger.info("Writing note %s to SQLite...", note_id)
        
        try:
            validate_db_constraints(keywords, insights)
        except ValueError as e:
            logger.error("Data validation failed for note %s: %s. Skipping SQLite update.", note_id, e)
            continue
            
        # Insert or update note (delete first to cascade delete existing keywords/insights)
        now_str = datetime.now().isoformat()
        
        try:
            conn.execute("BEGIN TRANSACTION;")
            conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
            
            conn.execute("""
            INSERT INTO notes (
                id, title, series, brand, city, categories, topics, raw_path, content_sha, content, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                note_id, title, series, brand, city, categories, topics, str(raw_file_rel), content_sha, content, now_str, now_str
            ))
            
            # Insert keywords
            for idx, kw in enumerate(keywords):
                conn.execute("""
                INSERT INTO note_keywords (note_id, position, keyword) VALUES (?, ?, ?);
                """, (note_id, idx, kw))
                
            # Insert insights
            for idx, ins in enumerate(insights):
                conn.execute("""
                INSERT INTO note_insights (note_id, position, insight) VALUES (?, ?, ?);
                """, (note_id, idx, ins))
                
            conn.execute("COMMIT;")
            inserted_notes += 1
        except Exception as e:
            conn.execute("ROLLBACK;")
            logger.error("Failed to insert note %s: %s", note_id, e)
            
    conn.close()
    logger.info("SQLite database build complete. Inserted/Updated: %d, Skipped: %d", inserted_notes, skipped_notes)


if __name__ == "__main__":
    main()
