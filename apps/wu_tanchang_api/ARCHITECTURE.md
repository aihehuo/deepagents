# Wu Tanchang Pre-Consultation API

吴探长一对一咨询的**前置沟通**服务。用户在对话中提供基本信息后，系统产出一份**会议准备材料**，供吴探长面谈使用。

## 架构

```
用户 ←→ 前置助手 (Front-end Agent)
            │ 无 KB 访问权限，人格文件启动时预注入
            │
            ├── task() ──→ kb_analyst (Sub-Agent)
            │                   │ 有 SkillsMiddleware + FilesystemMiddleware
            │                   │ 读取 kb/ 下的案例和方法论
            │                   │ 返回吴探长口吻的分析
            │
            ↓ 信息足够时
           产出会议准备材料 → 调用 mark_material_delivered
            → 引导预约吴探长

材料交付后 → 任何用户消息直接返回引导话术
（通过 checkpoint 中检测 mark_material_delivered tool call）
```

## 单次会话流程

```
收集信息（多轮对话，每轮 1-2 个问题）
    ↓ 信息足够
调用 kb_analyst 子代理（仅一次）
    ↓
生成会议准备材料 → 调用 mark_material_delivered 标记完成
    ↓
引导预约吴探长一对一深聊（不再深入探讨）
```

## 智能体

### 前置助手（Front-end Agent）

- **不是**吴探长本人，是吴探长团队的前置咨询助手
- **无 FilesystemMiddleware** — 无法直接访问工作区文件
- 人格文件（`IDENTITY.md`、`SOUL.md`、`AGENTS.md`）在 Python 层预读后注入 system prompt
- 每轮 1–2 个关键问题，信息足够时调用 `kb_analyst` 子代理
- 产出材料后调用 `mark_material_delivered` 工具标记完成
- 工具：`mark_material_delivered`

### KB 子代理（kb_analyst）

- 通过 `task()` 工具调用，由前置助手触发（仅一次）
- 有 `SkillsMiddleware` + `FilesystemMiddleware`，可读取知识库
- 以吴探长口吻返回结构化分析
- 读取 `kb/METHOD.md`、`kb/PLAYBOOK.md` 了解方法论
- 检索 `kb/index.json` + `kb/chunks/` 获取案例

### 记忆与持久化设计 (Memory & Persistence Design)

- **仅会话级 Checkpoint**：前置咨询服务本质上是短周期的信息收集与一次性材料生成过程。当前的前置助手均是按知识库进行短咨询并生成会面材料，**无需长期记忆（Long-term Memory）**需求，因此未启用 `MemoryMiddleware` 或任何长期记忆文件写操作。
- **短期会话隔离**：系统通过磁盘持久化 checkpointer（`checkpoints.pkl`）仅保留单次会话状态（`thread_id` 级别），以维持多轮对话信息收集的上下文连贯，一旦材料交付，会话流程即告结束。

## API

```bash
# 启动
PYTHONPATH=libs/deepagents uvicorn apps.wu_tanchang_api.app:app --port 8001
```

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/chat` | 前置对话 |
| POST | `/reset` | 重置会话 |

### POST /chat

```json
{
  "user_id": "user123",
  "conversation_id": "default",
  "message": "我想在上海开一家烘焙店，预算30万左右"
}
```

响应：

```json
{
  "user_id": "user123",
  "conversation_id": "default",
  "thread_id": "wt::default::user123::default",
  "reply": "你好！很高兴为你服务..."
}
```

材料交付后再发消息给同一 `user_id` + `conversation_id`，API 会检测 `mark_material_delivered` 标记并直接返回引导预约话术，不再调用智能体。

## 目录

```
apps/wu_tanchang_api/
├── workspace/                # 人格文件 + 知识库
│   ├── IDENTITY.md           # 前置助手身份
│   ├── SOUL.md               # 核心定位与流程
│   ├── AGENTS.md             # 工作流程与规则
│   ├── kb/                   # 知识库（仅子代理访问）
│   │   ├── index.json        # 案例索引
│   │   ├── METHOD.md         # 商业拆解方法论
│   │   ├── PLAYBOOK.md       # 咨询回答 Playbook
│   │   └── chunks/brands/    # 按品牌切片的案例
│   └── skills/kb_analyst/wu-tanchang-kb/
├── agent_factory/            # 智能体工厂
│   ├── agent.py              # 前置助手 + kb_analyst 子代理
│   ├── model_builder.py      # 模型初始化
│   └── utils.py              # 工作区部署工具
├── checkpointer.py           # 磁盘持久化 checkpoint
├── config.py                 # 配置解析
├── config.json               # 模型供应商与模型定义
├── env.example               # 环境变量模板
└── app/                      # FastAPI
    ├── __init__.py            # 应用入口
    ├── state.py              # AppState
    ├── startup.py            # 启动初始化
    ├── utils.py              # 工具函数
    ├── models.py             # Pydantic 模型
    └── endpoints/
        ├── chat.py           # 对话端点（含材料交付检测）
        ├── health.py         # 健康检查
        └── reset.py          # 会话重置
```

## 配置

默认读取：

- `apps/wu_tanchang_api/config.json` — 模型供应商结构，使用 `env:...` 引用环境变量
- `apps/wu_tanchang_api/.env` — 本地密钥与模型名（从 `.env.example` 复制）

内置两个 OpenAI-compatible provider：

| Provider | Base URL | LangChain client |
|----------|----------|------------------|
| `qwen` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `ChatOpenAI` |
| `deepseek` | `https://api.deepseek.com` | `ChatOpenAI` |

常用变量：

| 变量 | 说明 |
|------|------|
| `WU_API_MODEL_PROVIDER` | `qwen` 或 `deepseek`；通过 `agents.defaults.provider = "env:WU_API_MODEL_PROVIDER"` 控制默认 agent 的供应商，未设置时回退 `qwen` |
| `WU_API_CONFIG` | 可选，自定义 `config.json` 路径 |
| `WU_API_ENV_FILE` | 可选，自定义 `.env` 路径 |
| `WU_API_WORKSPACE` | workspace 源路径（默认 `apps/wu_tanchang_api/workspace`） |

模型名在 `config.json` 中配置（`agents.list[].model` 或 `providers.*.default_model`），不通过环境变量设置。

运行时数据：`~/.deepagents/wu_tanchang_api/`
