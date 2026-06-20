---
name: get-ai-blog
description: 获取爱合伙官方博客最新的文章列表。
---

# Get AI Blog Skill

获取爱合伙官方博客最新的文章列表。

## When to use

- 用户询问平台最近的新闻、官方通告、活动预告或者精选行业分享文章时。

## Files

- Script: `skills/local/get-ai-blog/run.py`

## Usage

使用 python 脚本执行接口请求：

```bash
python workspace_aihehuo/skills/local/get-ai-blog/run.py --page 1 --per 20
```

参数说明：
- `--page` (可选): 页码，默认 1。
- `--per` (可选): 每页限制数量，默认 20。
- `--format` (可选): 输出格式，支持 "json" 或 "md"，默认 "json"。
