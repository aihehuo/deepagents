---
name: get-ai-ideas
description: 获取爱合伙公开创业想法列表，支持通过关键词和语义进行检索。
---

# Get AI Ideas Skill

获取爱合伙公开创业想法列表，支持通过关键词和语义进行检索。

## When to use

- 用户询问商业模式、创业点子、或者寻找相关的合伙人合作构想时。
- 需要通过特定品类或技术热词（例如 "大模型", "出海"）筛选创意时。

## Files

- Script: `skills/local/get-ai-ideas/run.py`

## Usage

使用 python 脚本执行接口请求：

```bash
python workspace_aihehuo/skills/local/get-ai-ideas/run.py --keyword "SaaS" --q "语义查询" --page 1 --per 20
```

参数说明：
- `--keyword` (可选): 检索关键词，支持多个词。
- `--q` (可选): 语义查询内容。
- `--page` (可选): 页码，默认 1。
- `--per` (可选): 每页限制数量，默认 20。
- `--format` (可选): 输出格式，支持 "json" 或 "md"，默认 "json"。
