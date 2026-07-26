# Group Agent API (UC-34 / REQ-004 + REQ-005 slice 2a)

独立 app：对话挖需求 + 画像强制落库 + **能力分级闸门 + stub 匹配 + 披露双闸**。

## 本地启动

```bash
cd agents/deepagents && source .venv/bin/activate
export PYTHONPATH=libs/deepagents:.
uvicorn apps.group_agent_api.app:app --host 0.0.0.0 --port 8003 --reload
```

## Mock 调用

```bash
# 在群：落库后返回经双闸过滤的候选（≤3）
curl -s localhost:8003/chat -H 'content-type: application/json' -d '{
  "user_id": "mock_u1",
  "group_id": "mock_g1",
  "membership": "in_group",
  "message": "我在做智能宠物喂食器，缺联网和固件，有工厂和供应链"
}'

# 不在群：可对话落库，candidates 必空
curl -s localhost:8003/chat -H 'content-type: application/json' -d '{
  "user_id": "mock_u1",
  "group_id": "mock_g1",
  "membership": "not_in_group",
  "message": "随便聊聊"
}'

curl -s localhost:8003/match -H 'content-type: application/json' -d '{
  "user_id": "mock_u1",
  "group_id": "mock_g1",
  "membership": "in_group"
}'
```

## 红线

- 非 `in_group` → 零候选人 / 零 @（后处理拦截）
- 候选人必须来自本群可触达池；跨 `group_id` 套不到他群人脉
- 可见字段仅 `confirmed_public`；本切片不产邀请词（2b）

## Docker 构建与暗部署 (REQ-008)

### 镜像构建
```bash
# 本机镜像构建（建议指定完整 commit SHA 作为 tag）
docker build -t group-agent-api:3adaae88 -f apps/group_agent_api/Dockerfile .
```

### 暗部署拓扑 (Dark Deployment Topology)
- **监听端口**：宿主机只绑定 `127.0.0.1:8003:8001`（禁止暴露 `0.0.0.0`）。
- **运行用户**：非 root 用户 `appuser` (UID `1000:1000`)。
- **持久化挂载**：宿主机数据目录挂载到 `/home/appuser/.deepagents/group_agent_api`（具体宿主路径由部署环境决定，见 compose 示例）。
- **环境变量配置**：部署环境私有的 `.docker.env` 文件（见 `docker.env.example`）。

### 安全与运行约束
1. **签名契约 (`GA-PRINCIPAL-V1`)**：
   - HTTP 模式下要求带 HMAC-SHA256 签名请求头（`X-GA-Signature`、`X-GA-Ts`、`X-GA-Nonce` 等）。
   - canonical 必须包含 `group_token_sha256`。
   - `GROUP_AGENT_PRINCIPAL_HMAC_SECRET` 仅存在于 Calendar/BFF 与 deepagents 服务端，严禁泄漏。
2. **单 Worker 限制**：
   - Dockerfile / Uvicorn 明确 `--workers 1`。由于当前 nonce 防重放为进程内内存缓存，多 worker / 多副本可能产生伪重放判定。
3. **Fail-Closed 启动**：
   - 当 `GROUP_AGENT_ENV=production` 或 `GROUP_AGENT_INTEGRATION=http` 时，缺失 HMAC 密钥或必需配置将导致启动失败。

### 4. 异步接口与 Callback 机制 (REQ-009 `GA-ASYNC-V1` / `GA-CALLBACK-V1`)
- **前端物理隔离**：前端只连 micro 服务的 WebSocket/ActionCable，禁止直连 `group_agent_api`；浏览器永远不持有 `X-GA-*` 头部、HMAC 密钥或内网 Callback 地址。
- **异步入口 (`POST /call_async`)**：Micro 发起内网签名调用后，API 校验 `GA-PRINCIPAL-V1` 头部与 Body 身份一致性，成功后返回 `202 Accepted` ACK，后台异步执行任务。
- **Callback 安全 (`GA-CALLBACK-V1`)**：
  - **SSRF 允许列表**：`callback_url` 必须精确匹配 `GROUP_AGENT_CALLBACK_ALLOWED_BASE_URLS`，禁止公网/错 Host/userinfo/重定向逃逸。
  - **方向性 HMAC 签名**：头部携带 `X-GA-Callback-Signature`（采用 `GROUP_AGENT_CALLBACK_HMAC_SECRET` 计算）。
  - **单调 Seq 与终态**：单次运行 `seq` 递增；且只产生一个终态（`final` 或 `error`）；错误只回传安全错误码，严禁泄漏堆栈/密钥。

### 健康检查 (Healthcheck)
- 接口：`GET /health`（端口 8001），返回 HTTP 200。
- Compose 配置内置 HTTP 健康检查。

### 回滚与生产放行 (HOLD-DEPLOY)
- 当前处于 **HOLD-DEPLOY** 状态，禁止推远端镜像与生产上架。
- 生产联调前尚需完成：micro 服务 WebSocket 异步接入、三库非造样端到端、ES 阈值校准与环境指纹核验。
- 如需撤销暗部署容器（compose 文件路径由部署环境决定）：`docker compose -f <your-compose-file> stop group-agent-api && docker compose -f <your-compose-file> rm -f group-agent-api`

---

## 5. 三级 Mock Fixture + 本地容器完整对话 Scenario 测试体系 (REQ-010)

本体系为 `group_agent_api` 提供独立、分层、完全可重复运行的模块验收能力。

### 明确声明：多群测试不等于跨群匹配
- 测试中的多群设计仅用于验证数据隔离与多群并发负载。
- **跨群匹配绝对禁用**：`candidate.group_id` 必须严格等于 `trusted_session.group_id`（`match_scope = current_group_only`）。外群候选即使语义匹配度再高也必须在进入模型或输出前被过滤，跨群泄漏数必须严格为 0。

### 三级 Mock 规模与模式

| Level | 用途与定位 | 数据规模 | 默认 Model | 运行耗时预期 |
| :--- | :--- | :--- | :--- | :--- |
| **Level 1** | 每次提交门禁 (Smoke/Contract) | 2 群 / 10 成员 / 4 scenarios | `stub` | < 30 秒 |
| **Level 2** | 功能验收门禁 (Scenario/Behavior) | 8 群 / 200 成员 / 24 scenarios | `stub` (可选 opt-in `real`) | < 5 分钟 |
| **Level 3** | 暗部署前门禁 (Scale/Resilience) | 30 群 / 10,000 成员 (Seed 20260725) | `stub` | 确定性生成 (抗并发/抗注入) |

### 运行命令

```bash
# 1. 运行 REQ-010 单元与 Fixture 测试 (126 passed)
PYTHONPATH=. .venv/bin/pytest tests/test_group_agent_req010.py -v

# 2. 运行本地 Docker 完整容器对话 Runner
bash apps/group_agent_api/scripts/run_req010_e2e.sh
```

### Callback Simulator 与 Fail-Closed 保护
- **Callback Simulator** (`apps/group_agent_api/fixtures/callback_simulator.py`) 绑定 loopback/隔离 Docker 网络，校验 HMAC 签名、nonce 防重放、Strict Sequence 单调递增与单一终态（`final`/`error`）。
- **Fail-Closed 保护**：在 `GROUP_AGENT_INTEGRATION=http` 或 `GROUP_AGENT_ENV=production` 模式下，若存在任何 Mock 配置（如 `GROUP_AGENT_TEST_LEVEL`、`GROUP_AGENT_MOCK_SEED` 等），系统将 **Fail-Closed 拒绝启动**。
- **候选注入防护**：`/call_async` Body 及 `metadata` 严禁传递 `candidates`、`candidate_pool` 或 `override_group_id`。

### 职责边界说明
本套测试属于 **Deep Agents 独立模块验收**，验证 API 契约、Callback 协议、HMAC/seq/终态、群隔离与安全边界；不替代后续配合 micro 服务的真实双端端到端联调。
