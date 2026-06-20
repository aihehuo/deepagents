# 双Agent架构使用指南

## 概述

Business Co-Founder API 现在支持**前后端双Agent架构**，将对话和分析职责分离：

- **Facilitator Agent (前端)**: 专注于自然、流畅的对话，帮助用户探索想法
- **Expert Agent (后端)**: 负责结构化分析、方法论应用和战略指导生成

### 可插拔专业知识系统

Expert Agent 使用**可插拔的专业知识模板系统**：
- 专业知识定义存储在文件系统中（类似 Skills）
- 支持多种专业领域（商业、教育、健康等）
- 每个对话可以使用不同的专业知识类型
- Canvas 数据结构由专业知识模板定义

## 架构特点

### 前端 Agent (Facilitator)
- ✅ 轻量级middleware配置
- ✅ 自然对话导向的提示词
- ✅ 无刚性workflow限制
- ✅ 接收后端战略指导
- ✅ 保留memory和语言检测功能

### 后端 Agent (Expert)
- ✅ 完整的middleware stack
- ✅ 所有7个创业技能(skills)
- ✅ 商业想法跟踪和milestone管理
- ✅ 看板(kanban)生成
- ✅ 战略指导生成

### 同步机制
- ✅ 每10轮对话自动触发后端分析
- ✅ 异步执行，不阻塞前端响应
- ✅ 通过共享state传递分析结果
- ✅ 动态更新前端提示词

---

## 启用双Agent模式

### 环境变量配置

设置以下环境变量来启用双Agent架构：

```bash
# 启用双Agent模式
export BC_API_USE_DUAL_AGENT=1

# 可选：设置默认专业知识类型
export DEFAULT_EXPERTISE_TYPE=business_cofounder

# 可选：配置expert sync间隔（默认10轮）
# export EXPERT_SYNC_INTERVAL=10

# 可选：为expert使用更强大的模型
# export EXPERT_AGENT_MODEL_SUFFIX=EXPERT_AGENT_MODEL
# export QWEN_EXPERT_AGENT_MODEL=qwen-max
```

---

## 专业知识模板系统

### 概述

Expert Agent 使用**可插拔的专业知识模板**来定义其专业领域和分析方法。每个专业知识模板包含：

1. **专业知识定义**: Expert 的角色、方法论和分析任务
2. **Canvas 模板**: 结构化评估的 JSON 格式定义
3. **指导原则**: 如何生成战略指导

### 专业知识文件格式

专业知识模板使用 Markdown 文件 + YAML frontmatter 格式（类似 Skills）：

**文件位置**: `~/.deepagents/business_cofounder_api/expertise/{expertise_name}.md`

**示例** (`business_cofounder.md`):

```markdown
---
name: business_cofounder
description: Business co-founder expertise for entrepreneurial guidance
canvas_template: |
  {
    "current_stage": "idea_exploration",
    "completeness": {
      "idea_description": 0,
      "target_customer": 0,
      "pain_point": 0,
      "solution": 0,
      "value_proposition": 0,
      "business_model": 0
    },
    "next_milestones": [],
    "insights": [],
    "gaps": [],
    "strengths": []
  }
---

# Business Co-Founder Expertise

You are an expert business mentor analyzing conversations...

## Core Analysis Tasks

### 1. Conversation Analysis
Extract:
- Business idea clarity
- Customer understanding
- Pain point articulation
...

### 2. Canvas Structure
Generate a canvas with:
- current_stage: Current phase
- completeness: Scores (0-100)
- next_milestones: Upcoming goals
...
```

### YAML Frontmatter 字段

- **name** (required): 专业知识标识符（如 `business_cofounder`）
- **description** (required): 专业知识描述
- **canvas_template** (required): Canvas 数据的 JSON 模板/schema

### 创建新的专业知识模板

1. 在 `~/.deepagents/business_cofounder_api/expertise/` 创建新的 `.md` 文件
2. 添加 YAML frontmatter 定义元数据和 canvas 模板
3. 编写专业知识内容（Markdown body）
4. 在 API 请求中通过 `expertise_type` 参数使用

**示例**: 教育导师专业知识 (`education_mentor.md`)

```markdown
---
name: education_mentor
description: Education mentoring expertise for learning guidance
canvas_template: |
  {
    "learning_level": "beginner",
    "subject_area": "python",
    "topics_mastered": [],
    "topics_in_progress": [],
    "learning_style": "visual",
    "pace": "moderate",
    "insights": [],
    "gaps": [],
    "strengths": []
  }
---

# Education Mentor Expertise

You are an expert education mentor analyzing learning conversations...

## Core Analysis Tasks

### 1. Learning Progress Analysis
Assess:
- Current learning level and pace
- Topics mastered vs. in progress
- Learning style preferences
- Knowledge gaps

### 2. Canvas Structure
- learning_level: Current proficiency
- subject_area: Focus area
- topics_mastered: Completed topics
- topics_in_progress: Active learning
- insights: Key observations
- gaps: Areas needing attention
- strengths: Learning advantages
```

### 使用不同的专业知识

**方式1: 设置默认专业知识（环境变量）**

```bash
export DEFAULT_EXPERTISE_TYPE=education_mentor
uvicorn app:app --reload
```

**方式2: Per-conversation 专业知识（API 请求）**

```json
POST /chat
{
  "user_id": "user_123",
  "conversation_id": "conv_456",
  "message": "I want to learn machine learning",
  "expertise_type": "education_mentor"
}
```

每个对话可以使用不同的专业知识类型，Expert Agent 会自动加载相应的模板。

### 启动API

```bash
cd apps/business_cofounder_api
uvicorn app:app --reload
```

启动时会看到日志：

```
================================================================================
API Startup - Agent Configuration
================================================================================
  Dual-Agent Mode: ENABLED
  Initializing DUAL-AGENT architecture...
  - Frontend: Facilitator Agent (natural conversation)
  - Expert: Analyzer Agent (methodology & analysis)
  ✓ Facilitator Agent initialized
    Checkpoints: ~/.deepagents/business_cofounder_api/facilitator_checkpoints.pkl
  ✓ Expert Agent initialized
    Checkpoints: ~/.deepagents/business_cofounder_api/expert_checkpoints.pkl
================================================================================
Dual-Agent Architecture: READY
================================================================================
```

---

## API端点

### 1. `/chat` - 对话端点

**功能**: 与facilitator agent对话

**请求**:
```json
POST /chat
{
  "user_id": "user_123",
  "conversation_id": "conv_456",
  "message": "我想创建一个AI驱动的教育平台"
}
```

**响应**:
```json
{
  "user_id": "user_123",
  "conversation_id": "conv_456",
  "thread_id": "bc::user_123::conv_456",
  "reply": "很高兴听到你的想法！AI驱动的教育平台是一个很有潜力的方向。..."
}
```

**Expert同步**:
- 每10轮对话自动触发expert分析
- 异步执行，不影响响应速度
- 分析结果会在下一轮对话中通过guidance影响facilitator

### 2. `/kanban` - 看板查询端点

**功能**: 查询当前的商业看板和分析结果

**请求**:
```json
POST /kanban
{
  "user_id": "user_123",
  "conversation_id": "conv_456"
}
```

**响应**:
```json
{
  "user_id": "user_123",
  "conversation_id": "conv_456",
  "thread_id": "bc::user_123::conv_456",
  "kanban": {
    "current_stage": "customer_discovery",
    "completeness": {
      "idea_description": 75,
      "target_customer": 45,
      "pain_point": 60,
      "solution": 50,
      "value_proposition": 40,
      "business_model": 20
    },
    "next_milestones": [
      "Define specific customer segments",
      "Validate pain point intensity"
    ],
    "insights": [
      "Strong technical background in AI/ML",
      "Clear vision but needs customer validation"
    ],
    "gaps": [
      "Limited understanding of target market size"
    ],
    "strengths": [
      "Technical expertise in the domain"
    ]
  },
  "insights": [
    "User has deep technical expertise",
    "Pain point is well-articulated"
  ],
  "next_steps": [
    "Explore target customer segments",
    "Validate market size assumptions"
  ],
  "current_round": 12,
  "last_sync_round": 10,
  "analysis_timestamp": "2026-01-20T10:30:00Z"
}
```

---

## 工作流程

### 典型对话流程

```
Round 1-10: 前端facilitator与用户自然对话
           ↓
Round 10:  触发后端分析
           ├─ 提取最近10轮对话
           ├─ 后端agent分析
           ├─ 生成看板和guidance
           └─ 更新共享state
           ↓
Round 11:  前端接收guidance，调整对话方向
           ↓
Round 11-20: 继续对话，受guidance引导
           ↓
Round 20:  再次触发后端分析
           ...
```

### Expert分析内容

Expert agent会：

1. **分析对话**: 提取商业洞察和关键信息
2. **评估阶段**: 判断当前所处的创业阶段
3. **打分**: 对各个方面进行0-100的completeness评分
4. **生成看板**: 创建结构化的业务评估
5. **生成指导**: 为facilitator提供2-4句话的战略指导

### Frontend接收指导

前端agent在下一轮对话时会看到：

```markdown
## Strategic Guidance

Focus on helping them narrow down their target customer.
Ask about specific use cases and who would benefit most.
```

这会引导facilitator在对话中强调特定方向。

---

## 文件结构

```
apps/business_cofounder_api/
├── app.py                          # API主文件（已更新）
├── agent_factory.py                # Agent工厂（已添加create_facilitator_agent和create_expert_agent）
├── expert_sync.py                 # Expert同步逻辑（新建）
├── expertise_loader.py             # 专业知识加载器（新建）
├── expertise/                      # 专业知识模板（新建）
│   └── business_cofounder.md      # 默认商业创业专业知识
└── DUAL_AGENT_GUIDE.md            # 本文档

libs/deepagents/deepagents/
├── middleware/
│   └── expert_guidance.py         # Expert指导middleware（新建）
└── state/
    ├── __init__.py                 # State导出
    └── dual_agent_state.py         # 共享state定义（新建）

~/.deepagents/business_cofounder_api/
├── facilitator_checkpoints.pkl     # 前端agent checkpoints
├── expert_checkpoints.pkl         # Expert agent checkpoints
├── expertise/                      # 专业知识模板目录（运行时）
│   └── business_cofounder.md      # 默认模板（自动复制）
├── skills/                         # 创业技能库（Expert使用）
└── docs/                          # 生成的文档
```

---

## 专业知识模板系统详解

### 什么是专业知识模板？

专业知识模板定义了 Expert Agent 的专业领域、分析方法和 Canvas 数据结构。它使双Agent架构可以支持不同的应用场景：

- **Business Co-Founder**: 创业指导和商业分析
- **Education Mentor**: 学习指导和进度跟踪
- **Health Coach**: 健康目标和习惯养成
- **...**: 任何需要专家分析的领域

### 专业知识文件结构

每个专业知识模板是一个 Markdown 文件（类似 Skills）：

**文件位置**: `~/.deepagents/business_cofounder_api/expertise/{expertise_name}.md`

**文件格式**:

```markdown
---
name: expertise_identifier
description: Brief description of this expertise
canvas_template: |
  {
    "field1": "value",
    "field2": 0,
    "nested": {...}
  }
---

# Expertise Title

## Your Role
Define what this expert does...

## Core Analysis Tasks
List the analysis tasks...

## Canvas Structure
Define the canvas fields...

## Guidance Generation
How to generate strategic guidance...
```

### Canvas 模板

Canvas 模板是一个 JSON 结构，定义了这个专业知识下的评估维度：

**商业创业 Canvas**:
```json
{
  "current_stage": "idea_exploration",
  "completeness": {
    "idea_description": 0,
    "target_customer": 0,
    "pain_point": 0
  },
  "next_milestones": [],
  "insights": [],
  "gaps": [],
  "strengths": []
}
```

**教育学习 Canvas**:
```json
{
  "learning_level": "beginner",
  "subject_area": "python",
  "topics_mastered": [],
  "topics_in_progress": [],
  "learning_style": "visual",
  "insights": [],
  "gaps": [],
  "strengths": []
}
```

**健康目标 Canvas**:
```json
{
  "goals": [],
  "progress": {},
  "habits": [],
  "metrics": {},
  "insights": [],
  "gaps": [],
  "strengths": []
}
```

### 如何创建新专业知识

1. **创建 .md 文件**:
   ```bash
   cd ~/.deepagents/business_cofounder_api/expertise/
   touch my_expertise.md
   ```

2. **编写 YAML frontmatter**:
   ```markdown
   ---
   name: my_expertise
   description: My custom expertise description
   canvas_template: |
     {
       "custom_field": "value",
       "insights": [],
       "gaps": [],
       "strengths": []
     }
   ---
   ```

3. **编写专业知识内容**:
   - 定义 Expert 的角色和分析任务
   - 说明 Canvas 各字段的含义
   - 提供指导生成的原则

4. **使用新专业知识**:
   ```json
   POST /chat
   {
     "user_id": "user_123",
     "message": "...",
     "expertise_type": "my_expertise"
   }
   ```

### 专业知识加载机制

```
┌─────────────────────────────────────────┐
│  1. API Startup                         │
│  ───────────────                        │
│  expertise_dir = ~/.../expertise/       │
│  create_expert_agent(                   │
│    expertise_type="business_cofounder"  │
│  )                                      │
│  ↓                                      │
│  Load business_cofounder.md             │
│  ↓                                      │
│  Build system prompt with expertise     │
└─────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│  2. Runtime (Every 10 rounds)           │
│  ──────────────────────                 │
│  trigger_expert_analysis()              │
│  ↓                                      │
│  Get expertise_type from state          │
│  ↓                                      │
│  Load expertise template                │
│  ↓                                      │
│  Build analysis prompt with template    │
│  ↓                                      │
│  Expert analyzes → Returns canvas       │
└─────────────────────────────────────────┘
```

---

## 与单Agent模式对比

### 单Agent模式（默认）

```bash
# 不设置BC_API_USE_DUAL_AGENT或设置为0
export BC_API_USE_DUAL_AGENT=0
uvicorn app:app
```

特点：
- 单个综合agent
- 包含所有功能和workflow
- 更多的提示词限制
- 适合完整的结构化流程

### 双Agent模式

```bash
export BC_API_USE_DUAL_AGENT=1
uvicorn app:app
```

特点：
- 前后端分离
- 前端对话更自然、灵活
- 后端提供专业分析和指导
- 适合探索性对话 + 深度分析

---

## 调试和监控

### 日志

启用详细日志：

```bash
# 日志对话输入输出
export BC_API_LOG_CHAT_IO=1

# 日志state变化
export BC_API_LOG_STATE=1
```

### 关键日志点

**前端对话**:
```
POST /chat - received request (user_id=..., conversation_id=...)
[DualAgent] Expert sync needed for thread bc::...
[DualAgent] Expert sync task created (async)
```

**Expert分析**:
```
[ExpertSync] Triggering expert analysis...
[ExpertSync] Extracted 20 messages from last 10 rounds
[ExpertSync] Invoking expert agent...
[ExpertSync] Analysis parsed successfully
[ExpertSync] Expert sync completed
```

**看板查询**:
```
POST /kanban - received request (user_id=..., conversation_id=...)
[Kanban] Retrieved for thread ...: round=12, last_sync=10, has_kanban=True
```

---

## 性能考虑

### Expert同步性能

- **异步执行**: Expert分析不阻塞前端响应
- **触发频率**: 默认每10轮，可通过环境变量调整
- **分析时长**: 通常15-30秒（取决于模型和对话长度）
- **并发处理**: 每个conversation独立分析

### 优化建议

1. **使用更快的模型做expert**: `QWEN_EXPERT_AGENT_MODEL=qwen-plus`
2. **调整sync间隔**: 增加间隔减少分析频率
3. **缓存机制**: 未来可以添加分析结果缓存

---

## 故障排查

### 问题：Expert sync没有触发

**检查**:
1. 是否启用了双Agent模式？`BC_API_USE_DUAL_AGENT=1`
2. 是否达到10轮对话？查看`conversation_round`
3. 查看日志是否有`[ExpertSync]`相关信息

### 问题：Kanban endpoint返回空

**原因**: Expert还没有进行首次分析

**解决**:
- 至少对话10轮后查询
- 或者手动调整sync interval为更小值

### 问题：Facilitator guidance没有更新

**检查**:
1. Expert分析是否成功？查看`[ExpertSync]`日志
2. State是否更新？查看`expert_guidance`字段
3. ExpertGuidanceMiddleware是否正常工作？

---

## 未来扩展

### 计划中的功能

- [ ] 动态sync频率（基于对话质量自适应）
- [ ] 多个专业expert agents（市场分析、技术评估等）
- [ ] 实时协作（expert可以中途插入建议）
- [ ] Web UI看板可视化
- [ ] A/B测试框架

---

## 总结

双Agent架构实现了：

✅ **前端自由度最大化**: 自然对话，无刚性流程
✅ **后端专业化**: 完整方法论和深度分析
✅ **职责分离**: 清晰的架构边界
✅ **异步协作**: 不影响用户体验的后台分析
✅ **向后兼容**: 可以随时切换回单Agent模式

通过这个架构，用户可以享受流畅的对话体验，同时获得专业的创业指导和结构化分析。
