# wu_tanchang_api 多轮对话测试报告

- **日期**：2026-06-20
- **被测分支**：`main`
- **被测应用**：`apps/wu_tanchang_api`
- **测试人**：yc
- **测试方式**：本地 Docker + curl 多轮对话

---

## 1. 测试环境

| 项 | 值 |
|---|---|
| 主机 OS | macOS Darwin 25.5.0 |
| 容器名 | `wu-tanchang-api-local` |
| 镜像 | `aihehuo/wu-tanchang-api-local:local` |
| 容器 ID | `19a29940436d` |
| 端口映射 | `0.0.0.0:8001 -> 8001/tcp` |
| 容器内 HOME | `/home/appuser` |
| 数据卷挂载 | 宿主 `/Users/yc/.deepagents/wu_tanchang_api` ↔ 容器 `/home/appuser/.deepagents/wu_tanchang_api` |
| 模型 | provider=`qwen`, model=`deepseek-v4-flash`, max_input_tokens=131072 |
| Agent profiles | `default` / `owner` / `aihehuo`（共 3 个，本次仅测 default） |

构建与启动命令：

```bash
./apps/build_and_run_local.sh wu_tanchang_api 8001
```

健康检查：

```bash
$ curl -s http://127.0.0.1:8001/health
{"status":"ok","service":"wu_tanchang_api",
 "checkpoints_path":"/Users/yc/.deepagents/wu_tanchang_api/checkpoints.pkl",
 "backend_root":"/Users/yc/.deepagents/wu_tanchang_api"}
```

OpenAPI 暴露的端点：

| Method | Path |
|---|---|
| GET | `/health` |
| POST | `/chat` |
| POST | `/reset` |

`/chat` 请求体（`ChatRequest`）：`user_id`(必填), `message`(必填), `conversation_id`(默认 `default`), `agent_name`(默认 `""`)。

---

## 2. 测试路径

测试通过 curl 直接打 `POST /chat`、`POST /reset`，覆盖以下场景：

### 路径 A — 创业者咨询主流程（会话 `multi_turn_001`，user `test_user_1`）

| 轮 | 用户消息要点 | 预期 |
|---|---|---|
| 1 | 自我介绍 + 创业方向（小李 / 露营品牌） | agent 引导提供阶段、城市、预算等 |
| 2 | 城市（杭州）+ 预算（80 万）+ 渠道（线上电商 + 小红书） | 继续追问阶段 + 困惑 |
| 3 | 已接触代工厂 + 核心困惑（产品定位） | 进入"调知识库 / 生成材料"环节 |
| 4 | 追问杭州 25–35 岁人群定位思路 | 引导转向吴探长 1v1 深聊 |
| 5 | 反问前面提到的姓名/城市/预算 | 验证上下文记忆 |

### 路径 B — 上下文记忆 + 工作流 + reset 验证（会话 `mem_test_001`，user `test_user_2`）

| 轮 | 用户消息要点 | 预期 |
|---|---|---|
| 1 | 王晓峰 / 社区团购 / 成都 | 引导补全阶段 + 困惑 |
| 2 | 6 个月 / 月 GMV 30 万 / 生鲜 | 追问瓶颈 + 预算 |
| 3 | **不答**，反问前 4 个细节 | 验证多轮上下文记忆 |
| 4 | 损耗 12% + 预算 50 万（冷链 + 选品） | 进入"生成材料"环节 |
| 5 | 索要案例 | 验证内容护栏（拒绝在此处展开） |
| 6 | `POST /reset` | 清空当前 thread 状态 |
| 7 | reset 后问"你还记得我叫什么" | 应已遗忘 |

---

## 3. 测试结果汇总

### 路径 A

| 轮 | 状态 | 耗时 | 说明 |
|---|---|---|---|
| 1 | ✅ | 4.5 s | 引导提问城市/预算 |
| 2 | ✅ | 3.3 s | 引导提问阶段+困惑 |
| 3 | ❌ → ✅ | 4.9 s + 2.6 s | **首次 500**，重试通过 |
| 4 | ✅ | 9.7 s | 进入"交付"节点 |
| 5 | ⚠️ | **0.02 s** | 模板化短回复，未调 LLM（疑似 workflow 设计行为） |

### 路径 B

| 轮 | 状态 | 耗时 | 说明 |
|---|---|---|---|
| 1 | ✅ | 3.1 s | 引导阶段+困惑 |
| 2 | ✅ | 3.9 s | 引导瓶颈+预算 |
| 3 | ✅ | 4.0 s | **记忆全对**：成都/生鲜/30 万/6 月 |
| 4 | ❌ | 8.3 s | 同一类型 path 错误，命中"生成材料"环节 |
| 5 | ✅ | 5.7 s | 拒绝展开案例，引导预约 |
| 6 | ✅ | <1 s | reset OK |
| 7 | ✅ | 2.9 s | 正确遗忘姓名 |

---

## 4. 发现的问题

### Issue #1 — `FilesystemMiddleware` 路径越界，间歇性 500

**严重度**：High（影响主流程"生成会议准备材料"环节）
**首次发现**：2026-06-20
**状态**：🔴 Open

#### 现象

在用户回答完一系列引导问题、agent 即将进入"调知识库 → 生成会议准备材料"那一步时，`POST /chat` 偶发返回 500：

```json
{
  "detail": {
    "error_type": "ValueError",
    "error_message": "Path:/Users/yc/workspace/deepagents/apps/wu_tanchang_api/workspace/skills outside root directory: /Users/yc/.deepagents/wu_tanchang_api",
    "thread_id": "wt::default::test_user_1::multi_turn_001"
  }
}
```

#### 关键观察

1. **路径来自宿主机视角**：报错里那个 `/Users/yc/workspace/deepagents/...` 是**宿主机**的仓库源码路径，容器内根本没有这条路径。`/Users/yc/.deepagents/...` 是宿主侧的数据卷，容器内对应 `/home/appuser/.deepagents/...`。
2. **是间歇性的**：同一条用户消息重试一次就成功（路径 A turn 3）；但相同情形在新会话里照样出现（路径 B turn 4）。说明 agent 工作流里有多条分支，其中至少一条分支没有走 `Formatted skill path for tenant ...` 的路径重写。
3. **traceback 被吞**：容器 stdout 里没有 `Traceback`，上层 handler 把异常直接转成 JSON 返回，没 `logger.exception`，根因定位需要更详细的日志。

#### 复现方式

```bash
# 准备
./apps/build_and_run_local.sh wu_tanchang_api 8001

# 触发流程：让 agent 进入"生成会议材料"环节
UID_=repro_user; CID=repro_001
curl -s -X POST http://127.0.0.1:8001/reset \
  -H "Content-Type: application/json" \
  -d "{\"user_id\":\"$UID_\",\"conversation_id\":\"$CID\"}"

# 顺序补齐 agent 提问的所有信息
for MSG in \
  "你好,我叫王晓峰,做的是社区团购,基地在成都。" \
  "我现在做了 6 个月,GMV 大概 30 万一个月,主要做生鲜。" \
  "主要瓶颈是损耗高,生鲜损耗到了 12%,利润被吃光。预算 50 万,想用在冷链和选品上。"; do
  curl -s -X POST http://127.0.0.1:8001/chat \
    -H "Content-Type: application/json" \
    -d "{\"user_id\":\"$UID_\",\"conversation_id\":\"$CID\",\"message\":\"$MSG\"}" \
    | python3 -m json.tool --no-ensure-ascii
done
```

最后一条消息是触发点 — 不一定每次必现，需重试 1–3 次。

#### 怀疑根因

- 可能是 `agent_factory` / `SkillsMiddleware` 在某条懒加载路径上把启动期（在宿主进程或本地 dev 环境）的绝对路径写进了状态/缓存，运行期未做 host→container 路径重写。
- 启动日志里多数 skill 都正常打了 `[WuTanchang] Formatted skill path for tenant ...`，但触发错误的那一段路径不在打印里 — 说明它不是从 manifest 走的。
- 也可能是子 agent / general-purpose subagent 用了独立的 backend root，但传入的 skill 路径仍是 host 侧的。

#### 建议修复方向（仅记录，未实施）

1. 在抛 `ValueError` 的位置（`FilesystemMiddleware`）补 `logger.exception`，并把上层 handler 改为 `logger.exception` + JSON 返回，让 traceback 进 stdout。
2. 在 `apps/wu_tanchang_api/agent_factory/` 中确认所有传给 SkillsMiddleware/FilesystemMiddleware 的 path 是**容器侧**绝对路径，禁止使用 `os.getcwd()` / 仓库源码绝对路径。
3. 加一条断言：所有 skill 路径必须 `startswith(backend_root)`，否则启动期就报错而不是请求期。

#### 修复记录

> 待补充。
>
> - [ ] 修复 commit：`<hash>`
> - [ ] 修复人：
> - [ ] 修复日期：
> - [ ] 验证日期：
> - [ ] 验证人：
> - [ ] 验证方式（请直接贴重跑后的 curl 输出）：

---

### Issue #2 — "交付后"短回复未走 LLM（显式设计行为，无需修复）

**严重度**：Low
**首次发现**：2026-06-20
**状态**：✅ Confirmed — by design

#### 现象

路径 A turn 5（"刚才我说我叫什么名字？在哪个城市？预算多少？"）返回耗时 **0.02 s**，明显没有发起 LLM 调用，回复内容也是模板化的"建议你预约吴探长一对一深聊"。

但这条 message 在路径 B turn 3（同样的"反问"形式）下能正确召回 4 个细节，耗时 4.0 s。

差异：路径 A 在 turn 4 已经走过"生成材料"环节（agent 进入了"交付完成"状态），路径 B turn 3 时还在收集信息阶段。

#### 复现方式

```bash
# 走完路径 A 的全部 5 轮（见上文测试路径 A）
# 观察 turn 5 的耗时和 reply 内容
```

#### 代码级调查结论

通过代码审查确认这是**显式设计行为**，不是 bug：

1. **前置助手 prompt 明确规定了行为**（`agent.py:133-135`）：
   ```
   当你生成会议准备材料后，必须先将材料以文字完整呈现给用户，
   然后调用 `save_meeting_prep` 工具保存材料，最后调用
   `mark_material_delivered` 工具标记完成。顺序不可颠倒。
   材料交付后，只引导预约，不再深入探讨
   ```

2. **服务端有显式的截流逻辑**（`chat.py:175-198` `_has_delivered_material()` 函数）：检查 checkpoint 中 `mark_material_delivered` 工具调用历史，如果有 = 说明已交付，不再请求 LLM。

3. **截流后的回复是固定模板**（`chat.py:22-24` `_GUIDE_MESSAGE` 常量），不调 LLM，因此 0.02 秒返回。

4. **截流逻辑在收集回复之后**（`chat.py:280-297`）：即使是截流状态也会把 AI 回复内容拼出来返回，只是 LLM 不再被调用。

#### 修复记录

> 无需修复，确认是前端咨询助手的工作流终态设计。
>
> - [x] 是否为设计行为：是
> - [x] 决策人：代码审查确认（yc，2026-06-20）
> - [x] 决策日期：2026-06-20
> - [x] 相关代码位置：`chat.py:22-24(_GUIDE_MESSAGE)`, `chat.py:175-198(_has_delivered_material)`, `agent.py:133-135(FRONTEND_SYSTEM_PROMPT_TEMPLATE)`

---

### Issue #3 — 错误响应缺少 traceback（辅助性问题）

**严重度**：Low（DX 问题，不影响功能）
**首次发现**：2026-06-20
**状态**：🔴 Open — 需验证日志输出

#### 现象

`POST /chat` 异常时，响应体只有 `error_type` + `error_message`，**容器 stdout 里也没有 traceback**。这让 Issue #1 这种需要看堆栈才能定位的问题排查困难。

#### 代码审查结果

`POST /chat` 端点（`chat.py:262-276`）已经存在 `_logger.exception(...)` 调用：

```python
except Exception as exc:
    _logger.exception(
        "POST /chat failed thread_id=%s user_id=%s conversation_id=%s",
        tid, req.user_id, req.conversation_id,
    )
    raise HTTPException(status_code=502, detail={...}) from exc
```

测试报告中观察到 traceback 不可见可能有以下原因：

1. **uvicorn 日志 handler 配置问题**：`_logger = logging.getLogger("uvicorn.error")` 依赖 uvicorn 的日志初始化正常绑定到 stdout。
2. **测试时日志刷出时机**：异常发生后日志可能还在 buffer 中，`docker logs` 查询时未正确 dump。
3. **FastAPI 最外层有额外的 exception_handler** 在 `_logger.exception()` 之前就拦截了异常。

#### 建议验证和修复

1. 构建容器 → 复现 Issue #1 → 立即 `docker logs`，确认 `Traceback` 是否出现。
2. 如果仍不可见，在建一个 FastAPI 级别的 `@app.exception_handler(Exception)`，兜底记录所有逃逸异常，避免被中间件吞掉。

#### 修复记录

> 待补充。
>
> - [ ] 修复 commit：`<hash>`
> - [ ] 修复人：
> - [ ] 修复日期：
> - [ ] 验证日期：

---

## 5. 通过的部分（回归基线）

以下行为在本次测试中**符合预期**，可作为回归基线：

- ✅ `GET /health` 返回 200，路径回显正确
- ✅ `POST /reset` 清空 thread 状态（路径 B turn 6+7 验证）
- ✅ 多轮上下文记忆：跨 3 轮记得姓名/城市/品类/月 GMV/经营时长（路径 B turn 3）
- ✅ 引导式提问：未给齐信息时主动追问阶段/预算/困惑
- ✅ 内容护栏：在"交付"前后都拒绝直接展开案例（路径 B turn 5）
- ✅ 平均响应时延 3–6 s（不含交付环节的 LLM 长 thinking）

---

## 6. 修复记录（总览）

| Issue | 状态 | 修复 commit | 修复人 | 日期 | 验证 |
|---|---|---|---|---|---|
| #1 FilesystemMiddleware 路径越界 | 🔴 Open | - | - | - | - |
| #2 交付后短回复 | ✅ By design | — | yc | 2026-06-20 | 代码审查确认（见 Issue #2 修复记录） |
| #3 错误响应缺少 traceback | 🟡 Open（需验证日志输出） | - | - | - | - |

> 修复后请在本表更新状态、贴 commit hash、并在对应 Issue 的"修复记录"小节补全验证内容。
