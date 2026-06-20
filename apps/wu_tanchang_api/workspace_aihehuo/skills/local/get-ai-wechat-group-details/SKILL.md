---
name: get-ai-wechat-group-details
description: 获取指定 ID 的微信社群详情及成员列表 (支持分页)。
---

# Get AI WeChat Group Details Skill

获取指定 ID 的微信社群详情及成员列表 (支持分页)。

## When to use

- 用户获取了社群列表后，希望深入了解某个具体微信社群的详情、群主信息或社群成员列表时。

## Files

- Script: `skills/local/get-ai-wechat-group-details/run.py`

## Usage

使用 python 脚本执行接口请求：

```bash
python workspace_aihehuo/skills/local/get-ai-wechat-group-details/run.py --id 12345 --page 1 --per 20
```

参数说明：
- `--id` (必选): 社群的 event_id。
- `--page` (可选): 成员列表的分页页码，默认 1。
- `--per` (可选): 每页限制数量，默认 20。
- `--format` (可选): 输出格式，支持 "json" 或 "md"，默认 "json"。
