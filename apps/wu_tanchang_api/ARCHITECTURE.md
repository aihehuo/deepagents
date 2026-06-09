# Wu Tanchang Pre-Consultation API

吴探长一对一咨询的**前置沟通**服务：用户在 10 轮对话内说清楚情况，系统结合知识库产出**咨询提纲**，供吴探长面谈使用。

## 与 business_cofounder_api 的关系

参考其双智能体架构，但只保留必要部分：

| 保留 | 省略 |
|------|------|
| FastAPI `/chat`, `/health`, `/reset` | `/chat/stream`, `/deep_agent/call_async` |
| 双智能体：前置助手 + 分析师 | 模拟用户、合伙人搜索、里程碑 middleware |
| LangGraph checkpoint + 线程锁 | 多 expertise 切换、canvas 语言修复 |
| 知识库 skill + expertise 模板 | Aihehuo、Artifacts、AssetUpload |

## 智能体分工

### 前置助手（Intake Agent）

- **不是**吴探长本人，是团队接待同事
- 每轮 1–2 个关键问题，最多 10 轮
- 接收分析师 `expert_guidance` 决定追问方向
- 完成时向用户呈现咨询提纲

### 分析师（Analyst Agent）

- 后台运行，不直接对用户说话
- 使用 `wu-tanchang-kb` skill 检索 `kb/index.json` + chunks
- 每 3 轮同步一次，第 10 轮强制终局同步
- 输出 `canvas`（提纲）+ `expert_guidance` + `brief_summary`

## API

```bash
# 启动（需配置 DEEPSEEK_API_KEY 等）
PYTHONPATH=libs/deepagents:libs/cli uvicorn apps.wu_tanchang_api.app:app --reload --port 8001
```

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/chat` | 前置对话 |
| POST | `/brief` | 获取当前咨询提纲 |
| POST | `/reset` | 重置会话 |

### POST /chat

```json
{
  "user_id": "user123",
  "conversation_id": "default",
  "message": "我想在上海开一家烘焙店，预算50万左右"
}
```

响应含 `conversation_round`、`rounds_remaining`、`intake_complete`。

### POST /brief

返回完整 `canvas`：

- `conversation_points` — 交谈要点
- `main_challenges` — 主要挑战
- `solution_directions` — 吴探长可能探讨的方向
- `relevant_cases` — 知识库可参考案例

## 目录

```
apps/wu_tanchang_api/
├── workspace/          # OpenClaw 人格 + 知识库（部署源）
│   ├── kb/
│   ├── skills/local/wu-tanchang-kb/
│   └── intake/INTAKE_PLAYBOOK.md
├── expertise/consult_intake.md
├── agent_factory/      # intake + analyst 工厂
├── analyst_sync.py     # 分析师同步（精简版 expert_sync）
└── app/                # FastAPI
```

## 配置

默认读取：

- `apps/wu_tanchang_api/config.json` — 模型供应商结构，使用 `env:...` 引用环境变量
- `apps/wu_tanchang_api/.env` — 本地密钥与模型名（从 `.env.example` 复制）

内置两个 OpenAI-compatible provider：

| Provider key | Base URL | LangChain client |
|------|------|------|
| `qwen` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `ChatOpenAI` |
| `deepseek` | `https://api.deepseek.com` | `ChatOpenAI` |

`config.json` 内含模型候选清单、上下文长度、计价和能力标签。API key 只从 `.env` 获取，不写入 JSON 配置。

常用变量：

| 变量 | 说明 |
|------|------|
| `WU_API_MODEL_PROVIDER` | `qwen` 或 `deepseek` |
| `WU_API_CONFIG` | 可选，自定义 `config.json` 路径 |
| `WU_API_ENV_FILE` | 可选，自定义 `.env` 路径 |
| `WU_API_WORKSPACE` | workspace 源路径（默认 `apps/wu_tanchang_api/workspace`） |

运行时数据：`~/.deepagents/wu_tanchang_api/`
