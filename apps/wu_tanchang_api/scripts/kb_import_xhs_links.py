#!/usr/bin/env python3
"""Import Xiaohongshu share links from Excel into workspace/kb/xhs_links.json."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

logger = logging.getLogger("kb_import_xhs_links")

_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
_XHS_URL_RE = re.compile(r"https://xhslink\.cn/[A-Za-z0-9/_-]+")
_TAKEN_DOWN_MARKERS = ("视频已下架", "暂无连接")


def _col_letter(cell_ref: str) -> str:
    match = re.match(r"([A-Z]+)", cell_ref)
    return match.group(1) if match else ""


def _read_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    strings: list[str] = []
    for si in root.findall("m:si", _NS):
        parts = [node.text or "" for node in si.findall(".//m:t", _NS)]
        strings.append("".join(parts))
    return strings


def _read_sheet_rows(zf: zipfile.ZipFile, sheet_path: str, shared_strings: list[str]) -> list[dict[str, str]]:
    root = ET.fromstring(zf.read(sheet_path))
    rows: list[dict[str, str]] = []
    for row in root.findall("m:sheetData/m:row", _NS):
        cells: dict[str, str] = {}
        for cell in row.findall("m:c", _NS):
            ref = cell.attrib.get("r", "")
            col = _col_letter(ref)
            value_node = cell.find("m:v", _NS)
            if value_node is None or value_node.text is None:
                continue
            value = value_node.text
            if cell.attrib.get("t") == "s":
                value = shared_strings[int(value)]
            cells[col] = value
        if cells:
            rows.append(cells)
    return rows


def read_xlsx_rows(xlsx_path: Path) -> list[tuple[str, str]]:
    """Return (column_a_stem, column_b_text) rows from the first worksheet."""
    with zipfile.ZipFile(xlsx_path) as zf:
        shared_strings = _read_shared_strings(zf)
        sheet_path = "xl/worksheets/sheet1.xml"
        if sheet_path not in zf.namelist():
            sheets = sorted(name for name in zf.namelist() if name.startswith("xl/worksheets/sheet"))
            if not sheets:
                msg = f"No worksheet found in {xlsx_path}"
                raise ValueError(msg)
            sheet_path = sheets[0]
        sheet_rows = _read_sheet_rows(zf, sheet_path, shared_strings)

    parsed: list[tuple[str, str]] = []
    for cells in sheet_rows:
        stem = (cells.get("A") or "").strip()
        text = (cells.get("B") or "").strip()
        if stem:
            parsed.append((stem, text))
    return parsed


def extract_xhs_url(text: str) -> str | None:
    """Extract xhslink URL from share口令 text; None if taken down or missing."""
    if any(marker in text for marker in _TAKEN_DOWN_MARKERS):
        return None
    match = _XHS_URL_RE.search(text)
    if not match:
        return None
    return match.group(0)


def load_index_stem_map(index_path: Path) -> dict[str, str]:
    entries = json.loads(index_path.read_text(encoding="utf-8"))
    stem_map: dict[str, str] = {}
    for entry in entries:
        raw_file = entry.get("raw_file")
        note_id = entry.get("id")
        if not raw_file or not note_id:
            continue
        stem = Path(raw_file).stem
        stem_map[stem] = note_id
    return stem_map


def build_xhs_links(
    *,
    xlsx_path: Path,
    index_path: Path,
    generated_from: str,
) -> dict:
    stem_to_id = load_index_stem_map(index_path)
    notes: list[dict] = []
    unmatched_excel_rows: list[str] = []

    for raw_stem, share_text in read_xlsx_rows(xlsx_path):
        note_id = stem_to_id.get(raw_stem)
        if not note_id:
            logger.warning("Excel row has no matching index.json raw_file stem: %s", raw_stem)
            unmatched_excel_rows.append(raw_stem)
            continue

        url = extract_xhs_url(share_text)
        status = "taken_down" if url is None else "ok"
        notes.append(
            {
                "id": note_id,
                "raw_stem": raw_stem,
                "url": url,
                "share_text": share_text,
                "status": status,
            }
        )

    return {
        "generated_from": generated_from,
        "notes": notes,
        "unmatched_excel_rows": unmatched_excel_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Import Xiaohongshu links from Excel into xhs_links.json")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "workspace",
        help="Wu Tanchang workspace directory (default: apps/wu_tanchang_api/workspace)",
    )
    parser.add_argument(
        "--xlsx",
        type=Path,
        default=None,
        help="Path to 短视频连接.xlsx (default: <workspace>/kb/raw/短视频连接.xlsx)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: <workspace>/kb/xhs_links.json)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    workspace = args.workspace.resolve()
    kb_dir = workspace / "kb"
    xlsx_path = (args.xlsx or kb_dir / "raw" / "短视频连接.xlsx").resolve()
    index_path = kb_dir / "index.json"
    output_path = (args.output or kb_dir / "xhs_links.json").resolve()

    if not xlsx_path.exists():
        msg = f"Excel file not found: {xlsx_path}"
        raise SystemExit(msg)
    if not index_path.exists():
        msg = f"index.json not found: {index_path}"
        raise SystemExit(msg)

    rel_generated = str(xlsx_path.relative_to(workspace)) if xlsx_path.is_relative_to(workspace) else str(xlsx_path)
    payload = build_xhs_links(
        xlsx_path=xlsx_path,
        index_path=index_path,
        generated_from=f"kb/raw/{Path(xlsx_path).name}" if "raw" in xlsx_path.parts else rel_generated,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    ok_count = sum(1 for n in payload["notes"] if n["status"] == "ok")
    taken_down = sum(1 for n in payload["notes"] if n["status"] == "taken_down")
    logger.info(
        "Wrote %s — ok=%d taken_down=%d unmatched_excel_rows=%d",
        output_path,
        ok_count,
        taken_down,
        len(payload["unmatched_excel_rows"]),
    )


if __name__ == "__main__":
    main()
