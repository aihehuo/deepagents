---
name: get-ai-wechat-groups
description: 获取爱合伙公开的微信社群列表 (仅展示活跃且报名人数大于 100 人的群)。
---

# Get AI WeChat Groups Skill

获取爱合伙公开的微信社群列表 (仅展示活跃且报名人数大于 100 人的群)。

## When to use

- 用户询问是否有相关的微信交流群、行业社群（例如 "AI 创业群", "杭州合伙人群"）或者寻找对应的人脉群组时。

## Files

- Script: `skills/local/get-ai-wechat-groups/run.py`

## Usage

使用 python 脚本执行接口请求：

```bash
python workspace_aihehuo/skills/local/get-ai-wechat-groups/run.py --keyword "杭州"
```

参数说明：
- `--keyword` (可选): 社群检索关键词。
- `--format` (可选): 输出格式，支持 "json" 或 "md"，默认 "json"。
