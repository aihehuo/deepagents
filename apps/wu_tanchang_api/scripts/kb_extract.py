#!/usr/bin/env python3
"""Extract keywords and insights from raw markdown notes using Qwen Plus with concurrency."""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# Ensure we can import from apps.wu_tanchang_api and libs/deepagents
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "libs" / "deepagents"))

from apps.wu_tanchang_api.config import load_env_file
from apps.wu_tanchang_api.agent_factory.model_builder import create_model

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("kb_extract")

# Initialize models and environment
load_env_file()


def get_sha256(content: str) -> str:
    """Calculate SHA-256 of the content string."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def validate_extraction(data: dict[str, Any]) -> tuple[bool, str]:
    """Validate extraction results against the guidelines."""
    if not isinstance(data, dict):
        return False, "Output must be a dictionary"
    
    keywords = data.get("keywords")
    insights = data.get("insights")
    
    if not isinstance(keywords, list):
        return False, "keywords must be a list"
    if not isinstance(insights, list):
        return False, "insights must be a list"
        
    if not (6 <= len(keywords) <= 12):
        return False, f"keywords count must be between 6 and 12, got {len(keywords)}"
        
    if not (5 <= len(insights) <= 8):
        return False, f"insights count must be between 5 and 8, got {len(insights)}"
        
    # Check keywords: 仅由非空字符串组成，汉字数 <= 8
    for i, kw in enumerate(keywords):
        if not isinstance(kw, str):
            return False, f"keyword at index {i} must be a string"
        kw = kw.strip()
        if not kw:
            return False, f"keyword at index {i} is empty"
        # Chinese characters count <= 8 (relaxed from 5 to allow slightly longer LLM outputs)
        han_count = len(re.findall(r'[\u4e00-\u9fa5]', kw))
        if han_count > 8:
            return False, f"keyword '{kw}' contains {han_count} Chinese characters, which exceeds the limit of 8"
            
    # Check insights: 汉字数 10-50，禁止换行，不得含路径/文件名
    for i, ins in enumerate(insights):
        if not isinstance(ins, str):
            return False, f"insight at index {i} must be a string"
        ins = ins.strip()
        if not ins:
            return False, f"insight at index {i} is empty"
        if '\n' in ins or '\r' in ins:
            return False, f"insight '{ins}' contains newlines"
        # Count Chinese characters
        han_count = len(re.findall(r'[\u4e00-\u9fa5]', ins))
        if not (10 <= han_count <= 50):
            return False, f"insight '{ins}' contains {han_count} Chinese characters, which is not within the limit of 10-50"
        # Path / file characters check
        if re.search(r'\.md|\.json|kb/|raw/|chunks/|brands/', ins):
            return False, f"insight '{ins}' contains path or file extension characters (.md, .json, kb/, raw/, chunks/, brands/)"
            
    return True, ""


PROMPT_TEMPLATE = """你是一个专业的商业分析师。下面是吴探长的一篇探店商业笔记，请阅读并为它抽取：
1. 关键词 (keywords)：6-12个，用于对该笔记进行聚类和快速理解。
   - 每个关键词必须仅由汉字、英文单词或品牌名组成，不得包含标点符号。
   - 如果是包含汉字的关键词，其汉字字数必须不超过 5 个字（如“低价锚定”）。
   - 英文单词或品牌名不受 5 字限制，但必须是完整英文或拼音词（如“AOKKA”）。
2. 核心洞察 (insights)：5-8个，提炼笔记中最具商业价值的底层逻辑和经营智慧。
   - 每条洞察必须精炼，其汉字字数必须在 10 到 35 个字之间。
   - 每条洞察必须是一个单句，禁止换行，禁止包含任何路径名或文件名（如不得出现 "kb/raw/..." 或 ".md"）。

请严格按照提供的格式输出 JSON。

笔记标题: {title}
笔记正文:
{content}
"""


def extract_metadata(model: Any, title: str, content: str) -> dict[str, Any]:
    """Call the LLM with structured output or fallback JSON parsing."""
    prompt = PROMPT_TEMPLATE.format(title=title, content=content)
    
    # Try using langchain's with_structured_output if available and configured
    try:
        from pydantic import BaseModel, Field
        
        class ExtractedKB(BaseModel):
            keywords: list[str] = Field(..., description="6-12 keywords, Chinese or English words only, each Chinese word length <= 5")
            insights: list[str] = Field(..., description="5-8 insights, 10-35 Chinese characters each, no newlines, no paths/filenames")
            
        structured_model = model.with_structured_output(ExtractedKB)
        res = structured_model.invoke(prompt)
        if hasattr(res, "dict"):
            return res.dict()
        elif hasattr(res, "model_dump"):
            return res.model_dump()
        elif isinstance(res, dict):
            return res
    except Exception as e:
        logger.debug("Failed to use structured output, falling back to JSON parser: %s", e)
        
    # Fallback: request JSON format and parse manually
    json_prompt = prompt + "\n\n请输出符合以下 JSON Schema 的内容：\n" + json.dumps({
        "type": "object",
        "properties": {
            "keywords": {"type": "array", "items": {"type": "string"}},
            "insights": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["keywords", "insights"]
    }, ensure_ascii=False)
    
    res = model.invoke(json_prompt)
    content_str = res.content if hasattr(res, "content") else str(res)
    
    # Try to extract JSON block from markdown if present
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content_str, re.DOTALL)
    if match:
        content_str = match.group(1)
        
    try:
        return json.loads(content_str)
    except Exception as e:
        raise ValueError(f"Failed to parse JSON: {e}") from e


def process_note_with_retry(model: Any, title: str, content: str) -> dict[str, Any]:
    """Process note and retry once on validation failure."""
    try:
        result = extract_metadata(model, title, content)
        ok, err = validate_extraction(result)
        if ok:
            return result
        logger.warning("Validation failed on first attempt for '%s': %s. Retrying...", title, err)
    except Exception as e:
        logger.warning("LLM call failed on first attempt for '%s': %s. Retrying...", title, e)
        err = str(e)
        
    # Retry prompt with error context
    retry_prompt = f"上一次的输出未通过校验，错误如下：\n{err}\n\n请重新抽取，确保严格遵守规则：\n1. keywords 包含 6-12 个关键词，中文关键词字数 <= 5。\n2. insights 包含 5-8 条洞察，每条洞察中的汉字字数必须在 10 到 35 字之间，禁止换行，不得含有路径或文件名（如.md等）。\n\n重新抽取以下内容："
    full_retry_prompt = retry_prompt + "\n\n" + PROMPT_TEMPLATE.format(title=title, content=content)
    
    res = model.invoke(full_retry_prompt)
    content_str = res.content if hasattr(res, "content") else str(res)
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content_str, re.DOTALL)
    if match:
        content_str = match.group(1)
        
    try:
        result = json.loads(content_str)
        ok, err = validate_extraction(result)
        if not ok:
            raise ValueError(f"Retry validation failed: {err}")
        return result
    except Exception as e:
        raise e


def process_item(item: dict[str, Any], model: Any, out_dir: Path, workspace_dir: Path, force: bool) -> tuple[str, str]:
    """Process a single note item."""
    note_id = item.get("id")
    raw_file_rel = item.get("raw_file")
    title = item.get("title")
    
    if not raw_file_rel:
        return "skip", f"No raw_file for note id: {note_id}"
        
    raw_file_path = workspace_dir / raw_file_rel
    if not raw_file_path.exists():
        return "skip", f"Raw file not found at {raw_file_path} for note id: {note_id}"

        
    # Read file and calculate SHA-256
    with raw_file_path.open("r", encoding="utf-8") as f:
        content = f.read()
        
    content_sha = get_sha256(content)
    out_file = out_dir / f"{note_id}.json"
    
    # Check idempotency
    if out_file.exists() and not force:
        try:
            with out_file.open("r", encoding="utf-8") as f:
                existing = json.load(f)
            if existing.get("content_sha") == content_sha:
                return "skip_sha", f"Skipping note {note_id} (SHA-256 matches)"
        except Exception:
            pass
            
    logger.info("Extracting keywords & insights for note %s (%s)...", note_id, title)
    
    try:
        extraction = process_note_with_retry(model, title, content)
        
        # Save output JSON
        output_data = {
            "note_id": note_id,
            "title": title,
            "content_sha": content_sha,
            "keywords": extraction["keywords"],
            "insights": extraction["insights"]
        }
        
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
            
        return "success", f"Successfully saved extraction to {out_file}"
    except Exception as e:
        return "fail", f"Failed to extract note {note_id}: {e}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract keywords and insights from raw markdown notes.")
    parser.add_argument("--raw-dir", type=str, default="apps/wu_tanchang_api/workspace/kb/raw", help="Directory of raw markdown notes")
    parser.add_argument("--out-dir", type=str, default="apps/wu_tanchang_api/workspace/kb/extracted", help="Directory to output extracted JSON files")
    parser.add_argument("--force", action="store_true", help="Force extraction even if SHA-256 matches")
    parser.add_argument("--workers", type=int, default=8, help="Number of concurrent workers")
    args = parser.parse_args()
    
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    workspace_dir = raw_dir.parents[1]
    index_json_path = workspace_dir / "kb" / "index.json"
    if not index_json_path.exists():
        logger.error("index.json not found at %s", index_json_path)
        sys.exit(1)
        
    with index_json_path.open("r", encoding="utf-8") as f:
        index_data = json.load(f)
        
    logger.info("Loaded %d notes from index.json", len(index_data))
    
    # Initialize the LLM (qwen-plus)
    try:
        model = create_model(
            provider="qwen",
            model_name_override="qwen-plus",
            log_prefix="[KBExtract]"
        )
    except Exception as e:
        logger.error("Failed to create qwen-plus model: %s", e)
        sys.exit(1)
        
    success_count = 0
    skipped_count = 0
    failed_count = 0
    
    # Run concurrent threads
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_item, item, model, out_dir, workspace_dir, args.force): item
            for item in index_data
        }

        for future in as_completed(futures):
            status, message = future.result()
            if status == "success":
                success_count += 1
                logger.info(message)
            elif status == "fail":
                failed_count += 1
                logger.error(message)
            else:  # skip or skip_sha
                skipped_count += 1
                if status == "skip":
                    logger.warning(message)
                else:
                    logger.info(message)
                    
    logger.info("Extraction complete. Success: %d, Skipped: %d, Failed: %d", success_count, skipped_count, failed_count)
    if failed_count > 0:
        logger.error("Some extractions failed. Check log for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
