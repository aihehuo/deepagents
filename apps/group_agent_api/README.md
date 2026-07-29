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

---

## 6. Real-LLM + L1 Mock Scenario 验收 (REQ-012)

在**不依赖 Micro / New API / WebSocket / callback / Docker** 的前提下，使用**真实 Qwen LLM** +
**L1 Mock Fixture**，进程内完整跑通 3 轮 Group Agent 对话，验证真实模型能理解 doing/need/offer、
发出正式 `save_group_profile` tool call、随多轮演进画像、仅在本群匹配候选、生成受安全约束的邀请词，
且不泄漏跨群候选或敏感字段。

### ⚠️ 费用 / 额度警告
- 本测试**消耗真实 Qwen/DashScope LLM 额度**（每次约 8–10 次 LLM 调用、~5–7 万 token）。
- 默认门禁**不运行**：普通 `pytest` 与 CI 不会消耗额度（`real_llm` marker + `GROUP_AGENT_REAL_LLM_TEST` 门）。
- 请在确认额度后手动运行。

### 运行命令

```bash
# 仅当显式 opt-in 时运行。密钥只从进程环境读取，脚本不会 echo 密钥值。
export GROUP_AGENT_REAL_LLM_TEST=1
export GROUP_AGENT_PROVIDER=qwen
export GROUP_AGENT_MODEL=qwen-turbo
export GROUP_AGENT_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export DASHSCOPE_API_KEY="sk-..."

bash apps/group_agent_api/scripts/run_req012_real_llm.sh
```

Runner 行为：仅当 `GROUP_AGENT_REAL_LLM_TEST=1` 时运行，否则 skip；用 `mktemp -d` 创建全新 runtime 并在退出时清理；
显式设置 `GROUP_AGENT_MODEL_MODE=real / INTEGRATION=stub / ENV=test / TEST_LEVEL=L1`；对 membership / match / callback
三个生产 HTTP client 设置「调用即失败」guard；退出码非 0 表示验收失败。

### 硬预算
- Scenario 1 个 / 对话 3 轮；单轮 `GROUP_AGENT_MAX_TOKENS<=800`；单请求 timeout `<=60s`；整体 `<=240s`；最大真实 LLM 调用 `<=12`。

### Fail-Closed 模式校验
`GROUP_AGENT_MODEL_MODE` 仅接受 `stub` / `real`（留空 `≡ real`，走同一套 real 前置校验，不再有更宽松的隐式路径）。
未知值一律 fail-closed，**禁止「只要不是 stub 就当 real」**；`real`（含留空）+ `provider=qwen` 在发起网络请求前强制要求
`GROUP_AGENT_MODEL` / `GROUP_AGENT_BASE_URL` / `DASHSCOPE_API_KEY` 均非空，缺任一项即失败，
禁止 real 模式静默降级为 stub。

### 人工可读的对话内容审计报告（REQ-013）

REQ-012 的机器 Oracle 负责判断画像保存、群隔离、匹配和邀请契约；如果还需要由人类审核回复是否自然、
准确和有帮助，可在一次获批的真实 Qwen 运行中显式开启旁路报告：

```bash
export GROUP_AGENT_REAL_LLM_TEST=1
export GROUP_AGENT_HUMAN_AUDIT_REPORT=1
# 可选；仓库内路径必须位于已忽略的 .local-artifacts 下。
export GROUP_AGENT_HUMAN_AUDIT_OUTPUT_DIR=".local-artifacts/group-agent-audit"

bash apps/group_agent_api/scripts/run_req012_real_llm.sh
```

默认不启用，也不会读取或保存用户可见回复片段。启用后不会增加任何 LLM 调用；报告使用同一次三轮
Scenario 的实际 `reply` / `invite_text`，以确定性规则选择原文高价值片段，并输出：

- `.local-artifacts/group-agent-audit/req013-audit-<run_id>/req013-audit-<run_id>.md`
- `.local-artifacts/group-agent-audit/req013-audit-<run_id>/req013-audit-<run_id>.json`
- `.local-artifacts/group-agent-audit/req013-audit-<run_id>/READY.json`

报告包含固定 L1 Mock 用户输入、原文片段及原文哈希、每轮 LLM/tool delta、画像 before/after、
公开候选依据、逐位 mentioned 候选的可核验依据、自动内容检查和留空的人工评分栏。定向候选必须
至少具有非空 `confirmed_public doing`；无依据者会在计数、回复、点名和邀请前被剔除，全部被剔除时
诚实降级为不点名结果，禁止用“相关公开经验”伪造推荐理由。可选 LLM 润色不得删除、新增或重复任何
`@`。候选还必须具有非空稳定 `user_id`，同一身份只保留首个通过群、披露和依据检查的安全记录，
禁止一人占多个候选名额。ID 必须是无需修剪的原生 ASCII 字符串；带首尾空白、数字/布尔类型、
Unicode 或非法标点不会被静默转换，而是直接拒绝；Human Audit 会从实际 invite 正文提取 `@` 并与 mentioned/逐人依据交叉校验。它不会保存完整模型回复、系统指令、隐藏推理、
tool schema、checkpoint、密钥或 fixture 敏感值；脱敏失败会以
`FAILED:HUMAN_AUDIT_REDACTION` 终止整个验收。Markdown、JSON 与 READY 先在同一隐藏 staging 目录完整
生成并校验，再通过单次目录 rename 发布；读取方只承认带有效 READY 的完整文件对。历史 run_id 禁止
覆盖，文件权限为 `0600`，且所在目录被 Git 忽略。

每类片段最多保留 4 个完整句或换行语义段（不按逗号切断），总覆盖率不超过原响应的 50%，报告会记录实际 coverage。JSON 保留
选中片段原文作为证据；Markdown 展示层会统一中和 heading、table、raw HTML、图片、链接与 code fence，
避免模型文本伪造审计结构或触发外部资源请求。Alpha L1 Scenario 还会阻断所有非公开画像值及全部外群
身份/画像值，即使这些值没有使用 phone/email 等显式敏感字段名。

该报告只是人工内容审核证据，不替代 REQ-012 的机器 Oracle。仓库不包含真实报告；必须在获得真实
Qwen 调用授权后重新运行 Scenario，才能产生真实对话片段。本轮 REQ-013 开发和无网测试不会调用 LLM。

## 调试导出（前端包 + 后端 turn trace）

排障时用户会从日历暗入口点「导出调试」粘贴前端 JSON。完整 Human/AI/Tool 链需在 prod3 打开
`GROUP_AGENT_DEBUG_TRACE=1` 后从 `debug_traces/` **压缩拉取**。

操作手册（含 prod3 路径、gzip 流程、Micro 只读核对、复盘顺序）：

→ [`docs/runbooks/group-agent-debug-export.md`](../../docs/runbooks/group-agent-debug-export.md)

同内容亦镜像在工作区 `aihehuo_total/docs/runbooks/group-agent-debug-export.md`。
