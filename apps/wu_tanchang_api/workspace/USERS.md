# USERS.md — 飞书私聊使用者名册

**每次回复前**：从当前消息的 Sender metadata 取 `name` / `label` 与 `open_id`，在本表匹配。

## 真人（2 位）

| 真人 | 飞书账号（open_id） | 飞书显示名 | 怎么称呼 | 说明 |
|------|---------------------|-----------|----------|------|
| **吴探长本人** | `ou_12324e943af99bdcaa01e239a45334a1` | Andy 兆龙 | **吴探长**（首选）或 **Andy 兆龙** | 内容创作者本人；不是 YC |
| **YC**（同一人、两个号） | `ou_f9e7adb3d3f263b22b320f3ffbdf7790` | YC | **YC** | 运维 / 产品 / 部署 |
| ↑ 同上 YC | `ou_98753005a444a86daa691e472f61f2a5` | 章宇辰 | **YC** | 与上一行同一自然人，仅飞书 ID 不同 |

## 规则

1. **Sender 优先**：以当前消息的 Sender / open_id 识别是谁；**称呼用上表「怎么称呼」列**，不要套用别的账号的档案。
2. **两个 YC 账号**：章宇辰、YC 两个飞书号 **都是 YC**，偏好与禁忌以同一份为准（见 `memory/feishu-ou_f9e7adb3d3f263b22b320f3ffbdf7790.md`，章宇辰号可读同一文件）。
3. **Andy 兆龙 = 吴探长本人**：**不得**对 Andy 兆龙称 YC、章宇辰；**不得**把 YC 当成吴探长本人。
4. **对话隔离**：两个 YC 账号的 **session / jsonl 仍按 open_id 分开**，但身份认知上是同一人；不要对 YC 说「你和章宇辰是不是两个人」。
5. **更新本表**：由 YC 改本地后 `persona_sync.sh upgrade` 发生产。

## 个人偏好文件

| open_id | 文件 |
|---------|------|
| `ou_12324e943af99bdcaa01e239a45334a1`（Andy 兆龙 / 吴探长本人） | `memory/feishu-ou_12324e943af99bdcaa01e239a45334a1.md` |
| `ou_f9e7adb3d3f263b22b320f3ffbdf7790`（YC） | `memory/feishu-ou_f9e7adb3d3f263b22b320f3ffbdf7790.md` |
| `ou_98753005a444a86daa691e472f61f2a5`（YC · 章宇辰号） | 同上 YC 文件（同一自然人） |

私聊中的个人偏好写 `memory/feishu-*.md`，**不要**写进 `MEMORY.md`。
