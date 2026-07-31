# Group Agent Durable Queue Runbook (REQ-032 / TSD-04 WP-02)

> **HOLD-DEPLOY / HOLD-LIVE**：本文件仅覆盖本地与隔离 Redis/Celery 操作。  
> 禁止连接生产 Redis、禁止 push、禁止部署、禁止真实 callback。

## 1. 架构一句话

```text
Micro Run Ledger（用户可见权威）
  → Deep Agents API durable admission（Redis execution ledger + Celery enqueue）
  → group_agent_worker（claim lease → decrypt envelope → shared run core → callback）
```

- Celery broker 消息只含 `run_id / idempotency_key / request_fingerprint / delivery_id`
- 完整请求 envelope 以 AES-256-GCM 密文存 Redis ledger
- `GROUP_AGENT_DURABLE_QUEUE_ENABLED=0`（默认）走 legacy `asyncio.create_task`
- `=1` 时禁止 `create_task` 承载权威 Run；配置非法 fail closed

## 2. 本地启动（隔离 Redis DB）

```bash
# 1) 专用 DB（示例 15），勿用生产
redis-cli -n 15 PING

# 2) 生成 32-byte 测试 key（勿提交真实 key）
python - <<'PY'
import base64, os
print(base64.b64encode(os.urandom(32)).decode())
PY

# 3) API env（节选）
export GROUP_AGENT_DURABLE_QUEUE_ENABLED=1
export GROUP_AGENT_REDIS_URL=redis://127.0.0.1:6379/15
export GROUP_AGENT_REDIS_PREFIX=ga:exec:v1
export GROUP_AGENT_CELERY_QUEUE=group_agent.runs
export GROUP_AGENT_DLQ_QUEUE=group_agent.dlq
export GROUP_AGENT_QUEUE_PAYLOAD_KEYS=v1:<base64-32-byte-key>
export GROUP_AGENT_QUEUE_PAYLOAD_CURRENT_VERSION=v1
export GROUP_AGENT_LEASE_TTL_S=30
export GROUP_AGENT_HEARTBEAT_INTERVAL_S=10
export GROUP_AGENT_VISIBILITY_TIMEOUT_S=240
export GROUP_AGENT_TASK_SOFT_LIMIT_S=150
export GROUP_AGENT_TASK_HARD_LIMIT_S=180

# 4) Worker replicas (no embedded beat — FIX2)
export GROUP_AGENT_WORKER_INSTANCE_ID=ga-worker-local-1
export GROUP_AGENT_WORKER_BEAT=0
celery -A apps.group_agent_worker.celery_app worker --loglevel=info --pool=solo --concurrency=1

# 4b) Dedicated beat (exactly one process)
export GROUP_AGENT_WORKER_BEAT=1
celery -A apps.group_agent_worker.celery_app beat --loglevel=info
# 或者单机调试：同一进程 worker -B，且仅此副本设置 GROUP_AGENT_WORKER_BEAT=1

# 5) API
uvicorn apps.group_agent_api.app:app --host 127.0.0.1 --port 8001 --workers 1
```

Health：

- `GET /health` — liveness（不依赖 Redis）
- `GET /ready` — durable 开启时校验 execution store + broker（不回传 URL/密钥）

## 3. 默认阈值

| 配置 | 默认 |
|---|---|
| `GROUP_AGENT_QUEUE_MAX_DEPTH` | 500 |
| `GROUP_AGENT_MAX_RUNNING` | 20 |
| `GROUP_AGENT_PROVIDER_MAX_RUNNING` | 10 |
| `GROUP_AGENT_USER_MAX_QUEUED` | 5 |
| `GROUP_AGENT_GROUP_MAX_QUEUED` | 50 |
| `GROUP_AGENT_MAX_ATTEMPTS` | 5 |
| `GROUP_AGENT_RETRY_BASE_S` / `MAX_S` | 2 / 120 |
| lease / heartbeat / visibility | 30s / 10s / 240s |

HTTP：

- user/conversation 配额 → `429 queue_limit_exceeded` + `Retry-After`
- 全局/provider/group 饱和 → `503 queue_saturated` + `Retry-After`
- Redis/broker 不可用 → `503 queue_unavailable` / `enqueue_failed`
- 同 key 异 fingerprint → `409 idempotency_conflict`

## 4. 切换与回滚（本地）

### 停 admission

1. 将 API 的 `GROUP_AGENT_DURABLE_QUEUE_ENABLED` 保持 `1`，但从负载均衡摘掉 API；或
2. 临时把 `GROUP_AGENT_QUEUE_MAX_DEPTH=0` 等价拒绝新入队（返回 503）。

**禁止**「关 durable 后遗弃已入队任务」。

### Drain worker

1. 停止领取新任务：`celery control cancel_consumer group_agent.runs`
2. 等待 running lease 自然结束（grace ≤ hard_time_limit）
3. 检查：

```bash
redis-cli -n 15 --scan --pattern 'ga:exec:v1:run:*' | head
python -m apps.group_agent_worker.dlq_cli list
```

### 回滚 API 到 legacy

仅当 **queued/running/DLQ 均为空** 后：

1. 部署仍可消费旧队列的 worker 版本直到 drain 完成
2. 再将 API `GROUP_AGENT_DURABLE_QUEUE_ENABLED=0`
3. 旧队列任务必须仍有兼容 worker 消费完毕

## 5. DLQ 本地命令

```bash
python -m apps.group_agent_worker.dlq_cli list
python -m apps.group_agent_worker.dlq_cli inspect <run_id>
python -m apps.group_agent_worker.dlq_cli replay <run_id> --operator <id> --reason <text>
python -m apps.group_agent_worker.dlq_cli cancel <run_id> --operator <id> --reason <text>
```

- inspect 只返回 safe metadata（无 ciphertext / token / message）
- replay 不改变 `run_id` / `idempotency_key` / `request_fingerprint`
- **禁止**公网 HTTP 暴露 DLQ

## 6. Recovery

Worker 侧 `group_agent.recovery_tick`（leaderless）：

- 补入队 `accepted` / `enqueue_failed`
- 过期 `running` lease → `queued` 并重投
- 到期 `retry_wait` → `queued` 并重投

## 7. 已知剩余窗口（WP-03）

Callback 发送成功后、terminal fence 提交前进程崩溃时，Micro 可能已收到 final，
但执行侧尚未落 terminal。该不确定窗口由 **WP-03 Outbox** 最终消除；本 REQ 不宣称 exactly-once。

## 8. Fencing epoch 存储约束（FIX5）

- `profile_epoch:{user_digest}:{group_digest}` 是 Deep 侧全局单调序列，不设置
  TTL；`profile_fence` 与 `profile_fence_meta` 只是短期审计 key，按 execution
  record TTL 自动过期。
- durable Run 缺少 `user_id_digest` 或 `group_id_digest` 时必须拒绝 claim，
  禁止回退到 conversation 维度。
- 隔离 Redis 必须启用持久化、备份和 no-eviction；禁止清空、重建或复用
  `GROUP_AGENT_REDIS_PREFIX`。epoch namespace 丢失后必须停止画像写入，不能
  让序列从 1 静默重启。
- 这些 Deep Redis 约束只能阻止 HTTP 发出前已经失权的 attempt，不能把
  Redis CAS 与 Micro 数据库更新合并成一个事务。Micro 原子拒绝旧 epoch
  完成前，画像 write-point fencing 状态为
  **BLOCKED-ON-MICRO-FENCING / HOLD-DEPLOY**。

## 9. Micro REQ-030 对接契约（预告）

Durable `/call_async` 请求必填：

```json
{
  "request_schema_version": 1,
  "request_fingerprint": "<64-char lowercase sha256>",
  "queue_schema_version": 1,
  "run_id": "ga_...",
  "idempotency_key": "...",
  "...": "既有字段"
}
```

202 ACK：

```json
{
  "success": true,
  "accepted": true,
  "run_id": "ga_...",
  "session_id": "...",
  "idempotency_key": "...",
  "execution_status": "queued",
  "queue_schema_version": 1,
  "message": "accepted"
}
```

规则：

- fingerprint 以 Micro 权威值为准；Deep Agents 不再把自身旧指纹当权威
- 同 key + 同 schema/fp/run_id → 返回原 job
- 同 key 异绑定 → `409 idempotency_conflict`
- 同 run_id 异 key → `409 run_binding_conflict`
