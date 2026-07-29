# 群助手调试：前端导出包 + 后端日志/思维链拉取

> 给后续接手的智能体 / 研发：用户用日历暗入口「导出调试」给你前端包后，如何到 **prod3** 拉齐后端证据做复盘。  
> 约束：TSD-03 **禁止**把模型隐藏链写入 Micro 会话库；完整 Human/AI/Tool 链只在 Deep Agents 主机落盘（需开关）。  
> 生产日志：**压缩传输 + 本地分析 + 用完即删**（见仓库 `.cursor/rules/prod-log-transfer-compress.mdc`）。

---

## 0. 两套证据，缺一不可

| 来源 | 是什么 | 怎么拿到 |
|---|---|---|
| **前端调试包** | 用户可见气泡、`conversation_id` / `episode_id` / `current_run_id`、本地 fields、邀请词预览 | 群助手顶栏 **「导出调试」** → 剪贴板 JSON |
| **后端 turn trace** | 每轮 Human / AI / Tool（含 `save_group_profile` args，已脱敏） | prod3 `group-agent-api` 的 `debug_traces/`（需 `GROUP_AGENT_DEBUG_TRACE=1`） |
| **后端运维日志** | UC34Observer / uvicorn 生命周期（非完整 CoT） | 容器 stdout +（若存在）`uc_34_group_agent.log` |
| **Micro 会话权威** | 落库 transcript / invite / profile / run | `aihehuomicro` DB 或 `GET /group_agent/conversation`（需登录） |

关联键（从前端包抄）：

- `conversation_id`（如 `ga_763_1`）
- `episode_id`（开新一轮后会变；缺省时注意）
- `current_run_id`（如 `ga_e82d2e89…`）→ **查后端 trace 的主键**

---

## 1. 前端调试包（用户侧）

入口：`?mode=group-agent` 日历暗入口 → 顶栏 **导出调试**。

期望字段：

```json
{
  "exported_at": "...",
  "conversation_id": "ga_{eventId}_{userId}",
  "episode_id": "ep_…",
  "current_run_id": "ga_…",
  "messages": [ { "sender", "kind", "text", "time" } ],
  "profile": { "short_name", "project", "has_bound_profile" },
  "fields": [],
  "invite_text": "…",
  "note": "Client-visible transcript only…"
}
```

**局限**：不含模型 tool 轨迹；`fields` / `episode_id` 有时为空（前端未同步时）。  
复盘时先据此定位 `run_id`，再去后端拉同 run 的 trace。

---

## 2. 打开后端思维链落盘（一次性 / 排障窗口）

主机：`ssh root@prod3`  
容器：`group-agent-api`  
运行时数据卷（host → 容器）：

```text
/mnt/group-agent-api/data  →  /home/appuser/.deepagents/group_agent_api
```

环境文件：`/mnt/group-agent-api/.docker.env`

```bash
ssh root@prod3
grep -n GROUP_AGENT_DEBUG_TRACE /mnt/group-agent-api/.docker.env || true
# 排障窗口打开（改完需重建/重启对应 compose 服务使 env 生效）：
# GROUP_AGENT_DEBUG_TRACE=1
cd /path/to/deepagents/apps   # 线上 compose 目录以现网为准
docker compose -f docker-compose.prod3.yml up -d group-agent-api
```

打开后，每次 `call_async` final 会在运行时目录写：

```text
{runtime}/debug_traces/{UTC时间}_{run_id}.json
# 宿主机通常为：
/mnt/group-agent-api/data/debug_traces/
```

排障结束后务必关掉开关，避免磁盘堆积（模块内保留最近约 200 个文件）。

---

## 3. 从 prod3 拉 turn trace（必须压缩）

### 3.1 先确认文件在不在

```bash
ssh root@prod3 'ls -lt /mnt/group-agent-api/data/debug_traces | head -20'
# 或按前端包里的 run_id：
ssh root@prod3 'ls -lt /mnt/group-agent-api/data/debug_traces/*ga_e82d2e89* 2>/dev/null | head'
```

容器内等价路径：

```bash
ssh root@prod3 'docker exec group-agent-api ls -lt /home/appuser/.deepagents/group_agent_api/debug_traces | head'
```

### 3.2 压缩后 scp（禁止直接拉未压缩大日志）

```bash
# ✅ 正确
ssh root@prod3 'gzip -c /mnt/group-agent-api/data/debug_traces/YYYYmmddT…_ga_xxx.json > /tmp/ga_trace_ga_xxx.json.gz'
mkdir -p /tmp/ga_debug && scp root@prod3:/tmp/ga_trace_ga_xxx.json.gz /tmp/ga_debug/
gunzip -c /tmp/ga_debug/ga_trace_ga_xxx.json.gz | less
# 分析完立刻删
rm -f /tmp/ga_debug/ga_trace_ga_xxx.json.gz
ssh root@prod3 'rm -f /tmp/ga_trace_ga_xxx.json.gz'

# ❌ 错误
scp root@prod3:/mnt/group-agent-api/data/debug_traces/*.json /tmp/
```

批量打包最近 N 个：

```bash
ssh root@prod3 'cd /mnt/group-agent-api/data && tar czf /tmp/ga_debug_traces_recent.tgz $(ls -t debug_traces/*.json | head -30)'
scp root@prod3:/tmp/ga_debug_traces_recent.tgz /tmp/ga_debug/
# 本地解压分析 → rm 本地与远端临时包
```

### 3.3 本地用仓库脚本列目录 / 按 run_id 导出

在 **deepagents** 仓库（有 runtime 挂载或已把 gz 解到本地目录时）：

```bash
cd /Users/yc/workspace/deepagents   # 或 agents/deepagents
GROUP_AGENT_RUNTIME_DIR=/path/to/runtime_or_extracted \
  .venv/bin/python apps/group_agent_api/scripts/export_debug_trace.py --list

GROUP_AGENT_RUNTIME_DIR=/path/to/runtime_or_extracted \
  .venv/bin/python apps/group_agent_api/scripts/export_debug_trace.py --run-id ga_e82d2e89210fb23c2b5e240dff4cd389
```

单文件 JSON 关键字段：`user_message`、`reply`、`turn_messages[]`（role=human|ai|tool）、`profile_status`、`match_status`、`match_reason`、`episode_id`、`thread_id`。

---

## 4. 容器 stdout / UC34 观察者日志

```bash
# 按 run_id / user 过滤（示例）
ssh root@prod3 'docker logs --tail 5000 group-agent-api 2>&1 | gzip -c > /tmp/ga_docker.log.gz'
scp root@prod3:/tmp/ga_docker.log.gz /tmp/ga_debug/
gunzip -c /tmp/ga_debug/ga_docker.log.gz | rg 'ga_e82d2e89|profile_persistence|debug_trace|save_group_profile' | head -100
rm -f /tmp/ga_debug/ga_docker.log.gz
ssh root@prod3 'rm -f /tmp/ga_docker.log.gz'
```

UCObserver 文件日志默认在容器内 `~/.deepagents/logs/uc_34_group_agent.log`。  
**注意**：现网 compose **只挂了** `…/group_agent_api` 数据目录，**logs 目录未必持久化**；优先以 `docker logs` + `debug_traces/` 为准。若日志在容器可写层，用：

```bash
ssh root@prod3 'docker exec group-agent-api ls -la /home/appuser/.deepagents/logs/ 2>/dev/null || true'
```

有文件再 `gzip -c` 拉出，用完即删。

---

## 5. Micro 侧只读核对（会话 / 画像 / run）

SSH + 容器（见 `backend/aihehuomicro/.claude/skills/prod-access/SKILL.md`）：

```bash
ssh root@prod3
# cmicro → rails console production（只读查询）
```

常用核对（示例，按前端包里的 id 替换）：

- `GroupAgentConversation`：`conversation_id` / `current_episode_id`
- `GroupAgentConversationMessage`：该 episode 的 user/assistant 气泡
- `GroupAgentInviteArtifact`：邀请词
- `GroupAgentProfile`：user×group 的 doing/need/offer
- `GroupAgentRun`：`run_id`、`source_message`、`metadata`（含 trusted `episode_id`）

**写库禁止**临时 `rails runner` 粘贴长脚本；变更走已验证 Rake（见 `prod-no-adhoc-runner` 规则）。

浏览器可读画像（需登录）：`GET /group_agent/profile?event_id=`  
会话恢复：`GET /group_agent/conversation?event_id=`

---

## 6. 推荐复盘顺序（给接手智能体）

1. 读用户粘贴的 **前端调试包**，记下 `current_run_id` / `conversation_id` / `episode_id`。  
2. 确认 prod3 上 `GROUP_AGENT_DEBUG_TRACE` 是否在问题时已打开；若未开，说明本期只有前端包 + docker logs + Micro DB。  
3. 按 §3 **压缩拉取**对应 `*_run_id.json`，对照 `turn_messages` 看是否调用了 `save_group_profile`、画像是否被覆盖、match gate 原因。  
4. 需要时再拉 docker logs / Micro 只读行。  
5. 本地临时文件与远端 `/tmp/*.gz` **分析完立刻删除**。  
6. 结论写回 `docs/notebooks/deepagents.md` 或相关 REQ RESP（不要把密钥、完整手机号写进文档）。

---

## 7. 代码锚点

| 能力 | 位置 |
|---|---|
| 前端导出调试 | `calendar` · `src/pages/GroupAgentApp.tsx` → `handleCopyDebugPack` |
| 后端 turn dump | `deepagents` · `apps/group_agent_api/agent_factory/debug_trace.py` |
| 写入点 | `apps/group_agent_api/app/async_manager.py`（final 前 `write_turn_trace`） |
| CLI | `apps/group_agent_api/scripts/export_debug_trace.py` |
| 部署 | `apps/deploy_to_prod3.sh group_agent_api <40-char-SHA>` |
| 会话记忆规格 | `docs/tsd/TSD-03-group-agent-session-memory.md` |

---

## 8. 常见坑

- **只有前端包、没有 trace**：开关未开，或 run 发生在开关打开之前。  
- **`episode_id: null`**：前端未拿到 episode；后端仍可能在 metadata / DB 有值，以 Micro `current_episode_id` 与 run metadata 为准。  
- **「信息不对」只改了 UI**：旧 bug；更正必须经 ChatChannel 触发 `save_group_profile`（见近期 calendar 提交）。  
- **直接 scp 未压缩日志**：违反生产日志拉取规则，改用 gzip/tgz。
