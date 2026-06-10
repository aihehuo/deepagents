# Wu Tanchang Knowledge Base Search

Search and retrieve 吴探长探店笔记 for F&B consulting replies.

## When to use

- User asks about 餐饮开店、品牌落地、选址、选品、定价、连锁、赛道趋势
- Before answering, search KB — do not invent brand facts

## Files

- Index: `kb/index.json`
- Chunks: `kb/chunks/brands/{id}/`
- Method: `kb/METHOD.md`
- Playbook: `kb/PLAYBOOK.md`

## Search workflow

1. Extract keywords from user question (品类、城市、预算、品牌名等)
2. **Verify brand exists** — mandatory before citing as 探店笔记:

```bash
grep -i '阿德\|Punch Monday' kb/index.json
```

- **Exit code / output empty** → 该品牌**无探店笔记**，不得说「吴探长探店 XX」
- 笔记原文已归档为 chunks，不提供原文文件查询

3. Search index for related entries:

```bash
ls kb/chunks/brands/wu-punch-monday/
```

4. Read relevant files:

```bash
cat kb/chunks/brands/wu-punch-monday/motivation.md
cat kb/chunks/brands/wu-punch-monday/dimension-*.md
cat kb/chunks/brands/wu-punch-monday/insight.md
```

5. Compose reply per `kb/PLAYBOOK.md` — **only cite brands that matched in step 2**

## Anti-hallucination

| Allowed | Forbidden |
|---------|-----------|
| Brands in `kb/index.json` | 「吴探长探店阿德生煎」when grep returns nothing |
| Facts from read chunks | Specific numbers for brands not in index |
| 「库内暂无 XX，可参考 YY（库内）」 | 虚构未收录品牌的笔记内容 |

## Index fields

| Field | Use for |
|-------|---------|
| `categories` | 烘焙、茶饮、轻食… |
| `topics` | 定价锚定、单品店、品牌焕新… |
| `consult_for` | 用户意图匹配 |
| `series` | 商业探店笔记 / 探店记 / 案例说 / 市场调查 |
| `keywords` | 模糊匹配 |

## Update chunks after adding new raw files

Run the rebuild script from repo root:
```bash
python3 claws/agent01/scripts/build_wu_kb.py
```
