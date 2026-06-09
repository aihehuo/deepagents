# MEMORY.md - Long-Term Memory（仅 Agent 级）

全 workspace 共用。**只放与「吴探长」agent 本身相关**的事实，不放某位飞书用户的称呼或私人偏好。

## Agent profile

- Agent id: andy01
- Persona: 吴探长（餐饮商业探店顾问）
- Primary channel: Feishu（多人私聊，见 `USERS.md`）
- Model: DeepSeek V4 Flash
- Knowledge base: `kb/`（探店笔记索引见 `kb/INDEX.md`）

## 多人飞书

- 使用者名册：`USERS.md`（吴探长本人 = Andy 兆龙；YC = 两个飞书号同一人）
- 每人偏好：`memory/feishu-<open_id>.md`
- 当前消息以 **open_id + USERS.md** 识别真人，不以单一 `USER.md` 姓名猜人

## Notes（agent 级）

在此记录产品级、知识库级、渠道级决策——例如 KB 重建流程、禁止库外品牌冒充探店笔记等。  
**不要**在此写「用户叫 YC」「用户叫章宇辰」。
