---
name: yc-kb
description: Search and retrieve YC startup method notes for consulting and coaching replies
---

# YC Startup Knowledge Base Search

Search and retrieve YC's startup method notes for consulting and coaching replies.

## When to use

- User asks about 创业开店、精益创业、设计思维、0-1去风险、核心支柱、优势与问题陈述、黄金60秒。
- Before answering, search KB — do not invent methodology details.

## Files / Data Sources

> **Multi-tenant note**: All knowledge-base paths (e.g. `kb/...`) in this file
> are **templates**. At runtime, `ensure_runtime_workspace` rewrites them to
> tenant-scoped absolute paths. Do **not** hard-code workspace names here.

- Index: `kb/index.json` (for verification)
- Database: SQLite `kb/kb.db` (accessed via tools, contains note content)
- Method: `kb/METHOD.md`
- Playbook: `kb/PLAYBOOK.md`


Note: Chunks are not used. Do not attempt to read `kb/chunks/` or raw files.

## Search & Retrieve workflow

1. Extract keywords from user question (阶段、痛点、优势、目标听众、黄金60秒等).
2. **Verify item exists** in index — mandatory before citing:

```bash
grep -i '黄金60秒\|天使客户' kb/index.json
```

- **Exit code / output empty** → 该条目**未收录**，不得引用。

3. **Read content from DB**:
   - Use the `get_note_content` tool with the verified `note_id` to retrieve the complete content of the note directly from the SQLite database.
   - Do NOT use `cat` or any file-reading command on `kb/raw/` or `kb/chunks/` files for note details.

4. Compose reply per `kb/PLAYBOOK.md` — **only cite concepts/cases that matched in step 2**

## Anti-hallucination

| Allowed | Forbidden |
|---------|-----------|
| Concepts in `kb/index.json` | Citing models or formulas not present in index.json |
| Facts retrieved from `get_note_content` | Fabricating YC resume details or course facts |
| 「库内暂无相关内容，可参考以下建议」 | Citing non-existing notes |
