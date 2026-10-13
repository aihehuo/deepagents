# AI助手共享模型链路（2026-10-13）

AI助手复用现有 `POST /call_async`、Principal HMAC、Redis durable queue、group-agent-worker、严格签名回调及恢复机制。原独立 `/ai_assistant/run` 路由和专属服务密钥已移除；不新增容器。Docker 内网可以沿用现有 HTTP 地址，身份签名和回调校验仍保留。

新增请求/队列 schema `(5,5)` 和 execution_mode `ai_assistant`，不会改变 Cafe Host v4 或企微现有模式。完整冻结输入参与指纹和请求体签名，不携带个人登录 token、unionid 或群凭证。客户端不能通过旧 schema 携带 AI助手模式。

四类任务：conversation 提取 reply、profile、圈子 proposals；suggestion/compose 生成私人 text；match 生成已确认 offer/need 圈子的互补关系与合作展望。执行复用 create_model，但不进入旧工具执行图，不执行搜索、成员变更、画像写入或群消息发送。最终结果由 Micro 再检查证据、当前版本和权限后提交。

v5 强制 durable 严格回调，即使旧模式的严格回调开关关闭也不降级。回调携带 callback_version、Deep attempt、递增 seq，结果放在 payload.assistant_result。复用加密待发送回调、准确字节重试和 ACK；旧回调不广播到 APP 群聊。任务租约与 epoch 按 AI助手 run 隔离，不清空旧智能体历史或缓存。

输入限制、模型超时、严格输出 schema 沿用 AI助手生成模块；生成结果不代表已执行业务动作。邀请的48小时有效期、用户确认和当前版本禁止自动重复邀请由 Micro 执行，不由模型自行决定。

部署顺序：先更新现有 Deep API 和 worker，再更新 Micro 与新增迁移；保持入口及匹配开关关闭，完成真实内部联调后灰度开启。沿用现有 Principal HMAC、callback HMAC、durable queue 加密配置；无需新增 AI助手专属密钥。回退先关闭两个业务开关，保留新数据，不自动向下迁移。

本地测试验证了 Ruby/Python 签名一致、实际 Redis 入队与加密恢复、身份防篡改和现有 Cafe Host 模式。真实模型与生产完整链路尚未验收。完整用户接口见 Micro 的 docs/contracts/ai-assistant-v1.md。
