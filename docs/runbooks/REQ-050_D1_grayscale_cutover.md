# REQ-050 / REQ-051 · D-1 灰度切档操作手册 (跨仓 P0)

**适用范围**: 微信公众号客服智能体 (UC-35) 从 stub 模式切到 DashScope 模式 + 灰度放量
**拍板人**: 老板 2026-08-11
**前置 commit**: `673ef761` (A+B+C) + `e06cc179` (档 A 4 NIT) + D-1 commit
**回滚 SLA**: ≤ 5 分钟 (本仓 + 跨仓开关 + worker 重启)

---

## 🎯 切档目标

| 阶段 | model_mode | WECHAT_GREETER_ENABLED (new_api 端) | dry_run (本仓) | 流量 |
|---|---|---|---|---|
| **T-1 准备** | stub → dashscope | false (new_api 不发请求) | true | 0% (切档前 dry-run 验证链路) |
| **T0 切档** | dashscope | true (new_api 开始发请求) | false | 1% → 10% → 50% → 100% |

---

## 🚀 切档前 5 步 (T-1 准备, 必须按顺序)

### Step 1: 配 DASHSCOPE_API_KEY (生产密钥, 不入仓)

```bash
# 仅在 deep agents 部署环境配, 不入仓不入 .env 真实值
export DASHSCOPE_API_KEY="sk-real-..."  # 从现有 DashScope App 的生产密钥安全注入，绝不提交
export WECHAT_GREETER_MODEL_MODE="dashscope"
export WECHAT_GREETER_LLM_MODEL="qwen-plus"
export WECHAT_GREETER_LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
```

**回滚**: new_API 设置 `WECHAT_GREETER_FORCE_OFF=true`；模型配置保持 DashScope，不让 stub 接生产流量。

### Step 2: 启 API + worker (dry-run 模式)

```bash
# 两个 App 分别走标准链路: 本地构建 → 阿里云 Registry → prod3 pull/recreate
# 部署前确认 prod3 两个 .docker.env 均为 WECHAT_GREETER_DRY_RUN=true
SHA="$(git rev-parse HEAD)"
./apps/deploy_to_prod3.sh wechat_greeter_api "$SHA"
./apps/deploy_to_prod3.sh wechat_greeter_worker "$SHA"

# 验证 healthz 存活 + /ready 就绪 (REQ-065 P1-1: liveness/readiness 拆分)
curl http://localhost:8005/healthz
# 期望: {"status": "ok", "service": "wechat_greeter_api", ...}

curl http://localhost:8005/ready
# 期望: {"status": "ready", "checks": {"model_mode_is_dashscope": {"ok": true}, "dashscope_api_key": {"ok": true}, ...}}
# 注: dry_run=true 不阻塞 /ready (P0-3)
```

**回滚**: new_API 保持 `WECHAT_GREETER_FORCE_OFF=true`，必要时停止这两个新容器。

### Step 3: 跑 50 条负向评测 (真实 DashScope, dry-run 模式)

```bash
# 部署环境跑 (需 DASHSCOPE_API_KEY 已配)
.venv/bin/python tests/wechat_greeter/eval/run_negative_eval.py --mode real
# 期望: 100% pass + ≤ 120s + 0 越界
# 注: --mode real 走真 DashScope, 区别于 --mode stub (CI smoke 用)
```

**回滚**: 无副作用 (评测只读, 不改生产数据)

### Step 4: Dry-run 流量验证 (1% 真实流量, dry_run=true)

```bash
# new_api 端 (跨仓) 临时配 WECHAT_GREETER_ENABLED=true + 灰度 1% 流量
# 本仓仍开 dry_run=true → worker 不真打 callback, 仅 log
# 由 new_api 方负责, deep agents 仓不动
# 监控 1 小时: 0 越界 + 0 callback_skipped 异常 + UCObserver 日志正常
```

**回滚**: new_api 端 `WECHAT_GREETER_ENABLED=false` (或本仓 dry_run=true 保持, 不需要回滚)

### Step 5: 检查监控埋点 (灰度决策依据)

```bash
# 必查 5 埋点 (UCObserver uc_35_wechat_greeter.*):
#   1. wechat_msg_24h_expired_worker  ← 死信 (应保持低)
#   2. wechat_msg_thread_migrated      ← 迁移 (用户从 openid 升级到 user_id 时, 应保持低)
#   3. wechat_msg_callback_retry_count ← callback 重试 (期望 0, 偶尔 1-2 正常)
#   4. wechat_msg_callback_failed      ← callback 失败 (期望 0, > 5 立刻回滚)
#   5. DRY_RUN callback_skipped        ← dry-run 跳过 callback 计数 (T-1 阶段应等于流量)

# 跨仓协调埋点 (new_api 端):
#   6. wechat_msg_callback_received    ← new_api 收到 callback (T0 后应等于 1+2+3+4 失败数)
```

**决策依据**:
- ✅ 0 越界 + 0 callback_failed + callback 状态码全 2xx → 进 Step 6
- ❌ 任意失败率 > 1% → 立即走回滚 3 步

---

## 🚀 切档 Step 6 (T0 切档, 老板拍板后执行)

```bash
# 1. prod3: 关闭 worker dry_run (必须 force-recreate: restart 不重新注入 env)
sed -i 's/^WECHAT_GREETER_DRY_RUN=.*/WECHAT_GREETER_DRY_RUN=false/' /mnt/wechat-greeter-worker/.docker.env
WORKER_IMAGE="$(docker inspect wechat-greeter-worker --format '{{.Config.Image}}')"
WECHAT_GREETER_WORKER_IMAGE="$WORKER_IMAGE" docker compose \
  -f /mnt/deepagents/docker-compose.prod3.yml up -d --force-recreate wechat-greeter-worker

# 验证 worker 内 dry_run 已关闭
docker exec wechat-greeter-worker python -c "import os; print(os.environ.get('WECHAT_GREETER_DRY_RUN',''))"

# 2. 跨仓: new_api 端 灰度放量 (您手操作)
#   - WECHAT_GREETER_ENABLED=true (已配)
#   - 流量 1% → 10% → 50% → 100%, 每档 ≥ 30 分钟
#   - 每档必查监控 6 埋点 (见 Step 5)
```

---

## 🔄 回滚 3 步 (≤ 5 分钟 SLA)

### 回滚 1: 开 dry_run (本仓, 30 秒)

```bash
sed -i 's/^WECHAT_GREETER_DRY_RUN=.*/WECHAT_GREETER_DRY_RUN=true/' /mnt/wechat-greeter-worker/.docker.env
WORKER_IMAGE="$(docker inspect wechat-greeter-worker --format '{{.Config.Image}}')"
WECHAT_GREETER_WORKER_IMAGE="$WORKER_IMAGE" docker compose \
  -f /mnt/deepagents/docker-compose.prod3.yml up -d --force-recreate wechat-greeter-worker
# 效果: worker 走完流程但不真打 callback, 立即停止污染生产
# 验证: docker exec wechat-greeter-worker python -c "import os; print(os.environ.get('WECHAT_GREETER_DRY_RUN',''))"
```

### 回滚 2: 关 new_api 灰度 (跨仓, 1 分钟)

```bash
# new_api 端: WECHAT_GREETER_ENABLED=false
# 由 new_api 方负责, deep agents 仓不动
# 期望效果: new_api 停止发请求到 wechat_greeter, 本仓 worker 立即 idle
```

### 回滚 3: 保持停流并清空模型密钥 (本仓, 3 分钟)

```bash
sed -i 's/^DASHSCOPE_API_KEY=.*/DASHSCOPE_API_KEY=/' /mnt/wechat-greeter-api/.docker.env
sed -i 's/^DASHSCOPE_API_KEY=.*/DASHSCOPE_API_KEY=/' /mnt/wechat-greeter-worker/.docker.env
# model/key 变更影响 API + worker, 两个都必须 force-recreate
API_IMAGE="$(docker inspect wechat-greeter-api --format '{{.Config.Image}}')"
WORKER_IMAGE="$(docker inspect wechat-greeter-worker --format '{{.Config.Image}}')"
WECHAT_GREETER_API_IMAGE="$API_IMAGE" WECHAT_GREETER_WORKER_IMAGE="$WORKER_IMAGE" docker compose \
  -f /mnt/deepagents/docker-compose.prod3.yml up -d --force-recreate wechat-greeter-api wechat-greeter-worker

# 验证: curl /healthz → status=ok; curl /ready → dashscope_api_key (not_ready)
curl http://localhost:8005/healthz
curl http://localhost:8005/ready
```

**回滚决策树**:
- 失败率 < 1% → 保留 DashScope 配置, 仅关灰度 (回滚 2)
- 失败率 1-5% → 关灰度 + 清空模型密钥 (回滚 2+3)
- 失败率 > 5% 或 callback 持续失败 → 开 dry-run + 关灰度 + 清空密钥并查日志 (回滚 1+2+3)

---

## 🚨 红线 (任何阶段都不能违反)

1. **DASHSCOPE_API_KEY 绝不入仓**, 统一 `os.environ` 占位 (CLAUDE.md 铁律)
2. **dry_run 切真流量前必开**, 避免污染生产 callback
3. **失败率 > 1% 立即回滚**, 不观望
4. **跨仓协调留痕**: 切档/回滚前必在 new_api 群同步, 不单边操作
5. **TSD-09 v0.1 DRAFT 未签 v1.0 前不切真流量**, 仅 dry-run 灰度

---

## 📊 切档完成标准

| 项 | 期望 |
|---|---|
| 50 条负向评测 (real mode) | 100% pass / 0 越界 |
| Dry-run 1% 流量 1 小时 | 0 越界 + 0 callback_failed |
| 6 监控埋点 7 天 | 死信 < 0.1% / 迁移 < 0.5% / 重试 < 1% / 失败 < 0.1% / 跳过 (T-1 only) = 流量 |
| 灰度放量 1%→100% | 每档 ≥ 30 分钟, 失败率 < 1% |
| 端到端 RESP 联调 | new_api 端 callback_received 数 = deep agents 端 callback 发出数 |

---

## 📞 跨仓协调 checklist (P0 必填)

- [ ] **aihehuomicro 2 端点就绪** (跨仓: user_by_openid / user_full_profile, HMAC from=wechat_greeter 锁死) — D-2 范围, 切档前必完
- [ ] **new_api REQ-061 灰度开关 WECHAT_GREETER_ENABLED 实施** (跨仓: 默认 false, 切档时设 true) — D-1 必查
- [ ] **DASHSCOPE_API_KEY 配齐** (本仓 + 生产环境) — D-1 Step 1
- [ ] **TSD-09 v0.1 DRAFT → v1.0 签发** (docs 仓) — NIT-S1 修订, D-1 切真流量前必签
- [ ] **3 工具 E2E 真实 HMAC** (本仓: 2× micro HMAC + 1× 本地 FAQ) — D-2 范围, D-1 dry-run 阶段 stub 即可
- [ ] **回滚 3 步骤过一遍演练** (本仓 + 跨仓) — D-1 切档前必演

---

**Ref**: TSD-09 v0.1 DRAFT §3.2/3.3/3.4 (待签 v1.0)
**Lock**: 老板 2026-08-11 拍板 (UC-35 / REQ-050 / REQ-051)
