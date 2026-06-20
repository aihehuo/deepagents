# 路由中间件提示词 Review

## 工作原理

`SubagentRoutingMiddleware` 是一个**非强制性的路由提示中间件**：
- **不强制**工具调用，只会在检测到匹配条件时，在 system prompt 中**追加路由提示**
- 主 agent 仍然可以决定是否遵循这些提示
- 如果路由检测失败，不会影响 agent 的正常运行（有异常捕获）

## 触发机制

### 1. Coder Subagent 路由

#### 触发条件检测函数：`_looks_like_coding_task()`

**关键词匹配**（不区分大小写）：
```
code, coding, implement, implementation, bug, fix, refactor,
function, class, python, typescript, javascript, react, node,
dockerfile, sql, api, endpoint, html, css
```

**正则表达式匹配**：
- ```` ``` ```` - 代码块标记
- `<(html|div|span|body|head|script|style)\b` - HTML标签
- `\b(CSS|HTML|JS|TS|TSX|JSX)\b` - 技术栈缩写

#### 路由提示词：
```
## Routing hint (code/HTML)

If the user is asking for **code** (including **HTML/CSS/JS**, scripts, Dockerfiles, config changes, refactors, or debugging), you **MUST** delegate the heavy lifting to the `coder` subagent (unless it is truly trivial, e.g. a <5-line snippet with no repo edits):
- Use the `task` tool with `subagent_type="coder"`.
- In the task description, include: the goal, relevant constraints, file paths, and the exact expected output.
- The subagent can use the same tools (files/execute/etc.) to implement changes; then you summarize results to the user.
```

---

### 2. AI He Huo Subagent 路由

#### 触发条件检测函数：`_looks_like_aihehuo_search_task()`

**关键词匹配**（不区分大小写）：
```
co-founder, cofounder, founder, partner, investor, investment,
funding, ai he huo, aihehuo, 爱合伙, search members, find people,
find partners, find investors, business partner, technical co-founder,
business co-founder, domain expert, similar ideas, related projects,
business idea, startup partner, team member, collaborator
```

**正则表达式匹配**：
- `\b(co-?founder|cofounder)\b` - 联合创始人
- `\b(find|search|look for).*(partner|investor|co-founder|founder)` - 查找合伙人/投资者
- `\b(ai he huo|aihehuo|爱合伙)\b` - 平台名称
- `\b(technical|business).*(co-?founder|partner)` - 技术/商业合伙人
- `\b(similar|related).*(idea|project|business)` - 相似/相关项目

#### 路由提示词：
```
## Routing hint (AI He Huo search)

If the user is asking to **find co-founders, partners, investors, or search the AI He Huo (爱合伙) platform**, you **MUST** delegate the search to the `aihehuo` subagent:
- Use the `task` tool with `subagent_type="aihehuo"`.
- In the task description, include:
  - The business idea or requirements
  - What types of people are needed (technical co-founder, business co-founder, investors, domain experts)
  - Any specific criteria or constraints
- The subagent has specialized AI He Huo search tools and will perform multiple targeted searches.
- After the subagent completes the search, summarize the findings and recommendations for the user.
```

---

## 潜在问题分析

### 1. **误触发风险**

#### Coder 路由：
- ❌ **关键词过于宽泛**：`api`, `endpoint`, `function`, `class` 这些词在很多非代码场景也会出现
  - 例如："Can you help me understand the API of this service?" 可能被误判为代码任务
- ❌ **HTML/CSS 关键词**：在讨论网页设计、UI/UX 时可能误触发
  - 例如："The HTML structure needs to be more semantic" 可能触发，但可能只是讨论而非实现

#### AI He Huo 路由：
- ❌ **关键词重叠**：`partner`, `business idea`, `startup` 在一般商业讨论中很常见
  - 例如："I need a business partner for my startup" 会触发，但如果是讨论而非搜索，可能不需要路由
- ⚠️ **中英文混合**：`爱合伙` 关键词可能不够全面，用户可能用其他中文表达

### 2. **漏触发风险**

#### Coder 路由：
- ❌ **缺少常见编程术语**：`algorithm`, `data structure`, `debug`, `test`, `deploy`, `build`, `compile`
- ❌ **缺少框架/库名称**：`django`, `flask`, `express`, `vue`, `angular`
- ❌ **缺少文件扩展名检测**：`.py`, `.js`, `.ts`, `.java`, `.go` 等

#### AI He Huo 路由：
- ❌ **缺少中文变体**：用户可能用"找合伙人"、"寻找投资人"、"匹配合作伙伴"等表达
- ❌ **缺少间接表达**：用户可能说"我需要一个懂技术的合伙人"而不是直接说"co-founder"

### 3. **提示词可靠性问题**

#### 两个提示词都使用了 `**MUST**`：
- ⚠️ **强制性语言但非强制机制**：提示词说"MUST"，但中间件本身不强制，可能导致不一致
- ⚠️ **边界情况不明确**：Coder 路由提到"truly trivial"（<5行代码），但这个判断标准可能不够清晰

#### Coder 路由提示词：
- ✅ **优点**：明确说明了何时使用（非trivial代码任务）
- ❌ **缺点**：没有说明何时**不应该**使用（可能导致过度路由）

#### AI He Huo 路由提示词：
- ✅ **优点**：详细说明了任务描述应该包含的内容
- ⚠️ **缺点**：没有说明如果用户只是想了解平台功能而非实际搜索，是否应该路由

---

## 建议改进

### 1. **增强触发条件**

#### Coder 路由：
```python
# 建议添加的关键词
"algorithm", "debug", "test", "deploy", "build", "compile",
"repository", "repo", "git", "commit", "pull request", "merge",
"framework", "library", "package", "module", "import", "export"

# 建议添加的正则
re.compile(r"\.(py|js|ts|jsx|tsx|java|go|rs|cpp|h|hpp)\b"),  # 文件扩展名
re.compile(r"\b(def|function|class|interface|type|const|let|var)\s+\w+"),  # 代码结构
```

#### AI He Huo 路由：
```python
# 建议添加的中文关键词
"找合伙人", "寻找投资人", "匹配合作伙伴", "找技术合伙人",
"找商业合伙人", "寻找投资", "找团队", "招募合伙人"

# 建议添加的正则
re.compile(r"(找|寻找|匹配|招募).*(合伙人|投资人|合作伙伴|团队)"),
re.compile(r"(需要|想要|希望).*(合伙人|投资人|合作伙伴)"),
```

### 2. **改进提示词**

#### 建议的 Coder 路由提示词：
```
## Routing hint (code/HTML)

**When to delegate to `coder` subagent:**
- User explicitly asks for code implementation, debugging, or refactoring
- Task involves multiple files or complex logic (>5 lines)
- User provides code snippets or asks to modify existing code
- Task requires understanding repository structure

**When NOT to delegate:**
- Simple questions about code concepts (use general-purpose agent)
- Trivial snippets (<5 lines) with no file edits
- Code review or explanation requests (unless user asks for fixes)

**How to delegate:**
- Use the `task` tool with `subagent_type="coder"`
- Include in task description: goal, constraints, file paths, expected output format
- The subagent will implement changes; you summarize results to the user
```

#### 建议的 AI He Huo 路由提示词：
```
## Routing hint (AI He Huo search)

**When to delegate to `aihehuo` subagent:**
- User wants to **actively search** for co-founders, partners, investors, or domain experts
- User mentions the AI He Huo (爱合伙) platform
- User provides business idea and asks to find matching people/projects

**When NOT to delegate:**
- User is just asking about the platform or how it works (use general-purpose agent)
- User wants general advice about finding partners (use general-purpose agent)
- User already has specific people in mind (no search needed)

**How to delegate:**
- Use the `task` tool with `subagent_type="aihehuo"`
- In the task description, include:
  - Business idea or requirements
  - Types of people needed (technical co-founder, business co-founder, investors, domain experts)
  - Specific criteria or constraints (industry, location, experience, etc.)
- The subagent will perform multiple targeted searches
- After completion, summarize findings and provide recommendations
```

### 3. **增加置信度机制**

建议添加置信度评分，而不是简单的布尔判断：

```python
def _coding_task_confidence(text: str) -> float:
    """Return confidence score 0.0-1.0 for coding task."""
    score = 0.0
    # 强信号（代码块、文件扩展名等）
    if re.search(r"```", text):
        score += 0.5
    if re.search(r"\.(py|js|ts|jsx|tsx)\b", text):
        score += 0.3
    # 中等信号（关键词）
    keyword_count = sum(1 for k in _DEFAULT_CODE_KEYWORDS if k in text.lower())
    score += min(0.3, keyword_count * 0.1)
    # 弱信号（一般编程术语）
    # ...
    return min(1.0, score)

# 只有置信度 > 0.5 时才触发路由
if _coding_task_confidence(text) > 0.5:
    # 触发路由
```

---

## 当前实现位置

- **路由中间件代码**：`libs/deepagents/deepagents/middleware/routing.py`
- **实际使用位置**：`apps/business_cofounder_api/agent_factory.py` (第984-1022行)
- **触发函数**：`_looks_like_coding_task()` 和 `_looks_like_aihehuo_search_task()`
