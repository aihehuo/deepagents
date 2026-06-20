---
name: get-ai-projects
description: 获取爱合伙公开创业项目列表，支持通过关键词进行过滤检索。
---

# Get AI Projects Skill

获取爱合伙公开创业项目列表，支持通过关键词进行过滤检索。

## When to use

- 用户询问具体的落地项目、已经启动的合伙团队或者具体的招聘合伙人岗位时。
- 需要了解各个细分领域的公开合伙项目进展时。

## Files

- Script: `skills/local/get-ai-projects/run.py`

## Usage

使用 python 脚本执行接口请求：

```bash
python workspace_aihehuo/skills/local/get-ai-projects/run.py --keyword "餐饮" --page 1 --per 20
```

参数说明：
- `--keyword` (可选): 检索关键词，支持多个词。
- `--page` (可选): 页码，默认 1。
- `--per` (可选): 每页限制数量，默认 20。
- `--format` (可选): 输出格式，支持 "json" 或 "md"，默认 "json"。
