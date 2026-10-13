# AI助手内部模型接口

`POST /ai_assistant/run`，仅接受 `X-AI-Assistant-Secret`，配置 `AI_ASSISTANT_SERVICE_SECRET` 至少 32 字符。无用户 token 回退，不给模型搜索、画像写入、群消息发送或成员权限工具。

四类任务：conversation 产出 reply、profile、圈子 proposals；suggestion/compose 产出私人 text；match 产出已有 offer/need 圈子之间的互补关系与合作推演。输入源消息带 ID、作者 ID、role 和 text，圈子带确认版本、owner_id、kind、description、conditions。单次输入最多 64KB，输出最多 24KB，模型总预算 60 秒，SDK 55 秒、零自动重试。主机继续负责证据、版本、权限和最终落库。

结果采用禁止未知字段的严格 schema，拒绝虚构证据键、修改他人圈子、同人自匹配、错误提供需求方向、重复匹配对和 tool_calls。生成内容不代表执行任何业务动作。

无工具模型调用复用现有 create_model，但独立组装提示，不修改 Cafe Host 或企微客服执行图。不加入新的模型依赖。真实模型配置沿用 Group Agent 环境，因此仍共享供应商预算；需在内部灰度验证并发与费用。

对应 Micro 的 docs/contracts/ai-assistant-v1.md 为完整前端接入和发布契约。当前功能为实验实现，未部署；真实模型输出质量与跨服务联调待验收。
