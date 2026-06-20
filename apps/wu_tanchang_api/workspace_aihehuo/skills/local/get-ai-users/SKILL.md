---
name: get-ai-users
description: 获取爱合伙公开用户列表，支持通过关键词和语义进行检索。
---

# Get AI Users Skill

获取爱合伙公开用户列表，支持通过关键词和语义进行检索。

## When to use

- 用户询问平台上的合伙人、创始人、开发者等人才信息时。
- 需要通过特定领域、技术技能关键词（例如 "AI Agent", "Python"）查找人才时。

## Files

- Script: `skills/local/get-ai-users/run.py`

## Usage

使用 python 脚本执行接口请求：

```bash
python workspace_aihehuo/skills/local/get-ai-users/run.py --keyword "AI Agent" --q "语义查询" --page 1 --per 20
```

参数说明：
- `--keyword` (可选): 检索关键词，支持多个词。
- `--q` (可选): 语义查询内容。
- `--page` (可选): 页码，默认 1。
- `--per` (可选): 每页限制数量，默认 20。
- `--created-since` (可选): 筛选在此日期之后注册的用户，例如 "2026-06-01"。
- `--format` (可选): 输出格式，支持 "json" 或 "md"，默认 "json"。
