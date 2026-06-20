---
name: get-ai-faqs
description: 获取爱合伙公开常见问题 (FAQ) 列表与智库分享文章。
---

# Get AI FAQs Skill

获取爱合伙公开常见问题 (FAQ) 列表与智库分享文章。

## When to use

- 用户询问关于平台使用问题、产品机制、或是想看行业通用的精选问答和智库干货时。

## Files

- Script: `skills/local/get-ai-faqs/run.py`

## Usage

使用 python 脚本执行接口请求：

```bash
python workspace_aihehuo/skills/local/get-ai-faqs/run.py --keyword "合伙协议" --page 1 --per 20
```

参数说明：
- `--keyword` (可选): 检索关键词，支持多个词。
- `--page` (可选): 页码，默认 1。
- `--per` (可选): 每页限制数量，默认 20。
- `--format` (可选): 输出格式，支持 "json" 或 "md"，默认 "json"。
