# Available Middleware in DeepAgents

DeepAgents provides several middleware options that you can combine with `TodoListMiddleware` to build custom agents. Middleware is composable - you can add as many or as few as needed for your use case.

## Middleware Categories

### 1. **Core DeepAgents Middleware** (from `deepagents` package)

#### `TodoListMiddleware`
- **Source**: `langchain.agents.middleware.todo.TodoListMiddleware`
- **Purpose**: Task planning and progress tracking
- **Tools Provided**: `write_todos`
- **State Schema**: `PlanningState` with `todos` field
- **Usage**:
  ```python
  from langchain.agents.middleware import TodoListMiddleware
  
  agent = create_agent(
      middleware=[TodoListMiddleware()],
  )
  ```

#### `FilesystemMiddleware`
- **Source**: `deepagents.middleware.filesystem.FilesystemMiddleware`
- **Purpose**: File operations and context offloading
- **Tools Provided**: `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `execute`*
- **State Schema**: `FilesystemState` with `files` field
- **Features**:
  - Auto-evicts large tool results to filesystem (configurable token limit)
  - Supports ephemeral (StateBackend) or persistent (StoreBackend) storage
  - Optional `execute` tool if backend implements `SandboxBackendProtocol`
- **Usage**:
  ```python
  from deepagents.middleware.filesystem import FilesystemMiddleware
  from deepagents.backends import StateBackend, CompositeBackend, StoreBackend
  
  # Ephemeral storage (default)
  agent = create_agent(middleware=[FilesystemMiddleware()])
  
  # Hybrid storage (ephemeral + persistent)
  backend = CompositeBackend(
      default=StateBackend(),
      routes={"/memories/": StoreBackend(store=InMemoryStore())}
  )
  agent = create_agent(middleware=[FilesystemMiddleware(backend=backend)])
  
  # With sandbox (supports execute tool)
  agent = create_agent(middleware=[FilesystemMiddleware(backend=sandbox_backend)])
  ```

#### `SubAgentMiddleware`
- **Source**: `deepagents.middleware.subagents.SubAgentMiddleware`
- **Purpose**: Delegate tasks to isolated sub-agents
- **Tools Provided**: `task` (for delegating to subagents)
- **Features**:
  - Creates isolated subagents with their own context windows
  - Subagents can have custom models, tools, and middleware
  - General-purpose subagent available by default
- **Usage**:
  ```python
  from deepagents.middleware.subagents import SubAgentMiddleware
  
  agent = create_agent(
      middleware=[
          SubAgentMiddleware(
              default_model=model,
              default_tools=tools,
              subagents=[
                  {
                      "name": "research_agent",
                      "description": "Specialized in research tasks",
                      "system_prompt": "You are a research specialist...",
                      "tools": [research_tool],
                  }
              ],
          )
      ],
  )
  ```

#### `PatchToolCallsMiddleware`
- **Source**: `deepagents.middleware.patch_tool_calls.PatchToolCallsMiddleware`
- **Purpose**: Fixes dangling tool calls from interruptions
- **Tools Provided**: None (utility middleware)
- **Usage**: Automatically included in `create_deep_agent()`

#### `DateTimeMiddleware`
- **Source**: `deepagents.middleware.datetime.DateTimeMiddleware`
- **Purpose**: Provides current date and time information
- **Tools Provided**: `get_current_datetime`
- **Features**:
  - Returns current date/time in various formats (ISO, readable, or custom strftime)
  - Useful for timestamping, scheduling, and time-based context
- **Usage**:
  ```python
  from deepagents.middleware.datetime import DateTimeMiddleware
  from langchain.agents import create_agent
  
  agent = create_agent(
      model="anthropic:claude-sonnet-4-20250514",
      middleware=[DateTimeMiddleware()],
  )
  
  # Agent can now call get_current_datetime tool
  # Format options: 'iso', 'readable', or custom strftime format
  ```

### 2. **LangChain Middleware** (from `langchain` package)

#### `SummarizationMiddleware`
- **Source**: `langchain.agents.middleware.summarization.SummarizationMiddleware`
- **Purpose**: Auto-summarizes conversation when context exceeds token limits
- **Tools Provided**: None (utility middleware)
- **Features**:
  - Triggers at configurable token/fraction thresholds
  - Keeps recent messages, summarizes older ones
  - Automatically included in `create_deep_agent()` with smart defaults
- **Usage**:
  ```python
  from langchain.agents.middleware.summarization import SummarizationMiddleware
  
  agent = create_agent(
      middleware=[
          SummarizationMiddleware(
              model=model,
              trigger=("tokens", 170000),  # or ("fraction", 0.85)
              keep=("messages", 6),  # or ("fraction", 0.10)
          )
      ],
  )
  ```

#### `HumanInTheLoopMiddleware`
- **Source**: `langchain.agents.middleware.HumanInTheLoopMiddleware`
- **Purpose**: Pauses execution for human approval
- **Tools Provided**: None (interrupt middleware)
- **Features**:
  - Requires `interrupt_on` configuration
  - Can interrupt on specific tools or all tools
- **Usage**:
  ```python
  from langchain.agents.middleware import HumanInTheLoopMiddleware
  
  agent = create_agent(
      middleware=[HumanInTheLoopMiddleware()],
      interrupt_on={"write_file": True, "execute": True},  # Pause before these tools
  )
  ```

### 3. **Anthropic-Specific Middleware** (from `langchain_anthropic` package)

#### `AnthropicPromptCachingMiddleware`
- **Source**: `langchain_anthropic.middleware.AnthropicPromptCachingMiddleware`
- **Purpose**: Caches system prompts to reduce costs (Anthropic models only)
- **Tools Provided**: None (optimization middleware)
- **Usage**: Automatically included in `create_deep_agent()` for Anthropic models

### 4. **DeepAgents CLI Middleware** (from `deepagents-cli` package)

#### `SkillsMiddleware`
- **Source**: `deepagents_cli.skills.middleware.SkillsMiddleware`
- **Purpose**: Loads and exposes agent skills (specialized capabilities)
- **Tools Provided**: None (adds skills list to system prompt)
- **Features**:
  - Progressive disclosure pattern (skills list in prompt, full instructions in files)
  - Supports user-level and project-level skills
  - Skills are self-documenting (SKILL.md files)
- **Usage**:
  ```python
  from deepagents_cli.skills.middleware import SkillsMiddleware
  
  agent = create_agent(
      middleware=[
          SkillsMiddleware(
              skills_dir=Path("~/.deepagents/my_agent/skills"),
              assistant_id="my_agent",
              project_skills_dir=Path(".deepagents/skills"),
          )
      ],
  )
  ```

## Default Middleware in `create_deep_agent()`

When you use `create_deep_agent()`, these middleware are included by default:

```python
deepagent_middleware = [
    TodoListMiddleware(),           # Planning tools
    FilesystemMiddleware(backend),   # File operations
    SubAgentMiddleware(...),        # Subagent delegation
    SummarizationMiddleware(...),    # Auto-summarization
    AnthropicPromptCachingMiddleware(...),  # Cost optimization
    PatchToolCallsMiddleware(),     # Fix dangling tool calls
]
# Plus HumanInTheLoopMiddleware if interrupt_on is provided
```

## Building Custom Agent Configurations

### Example 1: Minimal Agent (Only Planning)
```python
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware

agent = create_agent(
    model="anthropic:claude-sonnet-4-20250514",
    middleware=[TodoListMiddleware()],
    tools=[],  # No other tools
)
```

### Example 2: Planning + Filesystem
```python
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from deepagents.middleware.filesystem import FilesystemMiddleware

agent = create_agent(
    model="anthropic:claude-sonnet-4-20250514",
    middleware=[
        TodoListMiddleware(),
        FilesystemMiddleware(),
    ],
)
```

### Example 3: Planning + Filesystem + SubAgents
```python
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.subagents import SubAgentMiddleware

agent = create_agent(
    model="anthropic:claude-sonnet-4-20250514",
    middleware=[
        TodoListMiddleware(),
        FilesystemMiddleware(),
        SubAgentMiddleware(
            default_model=model,
            default_tools=[],
            subagents=[],
        ),
    ],
)
```

### Example 4: Custom with Additional Middleware
```python
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware, HumanInTheLoopMiddleware
from deepagents.middleware.filesystem import FilesystemMiddleware

agent = create_agent(
    model="anthropic:claude-sonnet-4-20250514",
    middleware=[
        TodoListMiddleware(),
        FilesystemMiddleware(),
        HumanInTheLoopMiddleware(),
    ],
    interrupt_on={
        "write_file": True,  # Pause before writing files
        "execute": True,     # Pause before executing commands
    },
)
```

## Middleware Order Matters

Middleware is executed in the order provided. Generally:
1. **Tool-providing middleware** first (TodoListMiddleware, FilesystemMiddleware, SubAgentMiddleware)
2. **Utility middleware** next (SummarizationMiddleware, PatchToolCallsMiddleware)
3. **Optimization middleware** (AnthropicPromptCachingMiddleware)
4. **Interrupt middleware** last (HumanInTheLoopMiddleware)

## Summary Table

| Middleware | Source Package | Tools | Purpose | Default in `create_deep_agent()` |
|------------|---------------|-------|---------|--------------------------------|
| `TodoListMiddleware` | `langchain` | `write_todos` | Task planning | ✅ Yes |
| `FilesystemMiddleware` | `deepagents` | `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `execute`* | File operations | ✅ Yes |
| `SubAgentMiddleware` | `deepagents` | `task` | Subagent delegation | ✅ Yes |
| `SummarizationMiddleware` | `langchain` | None | Auto-summarization | ✅ Yes |
| `AnthropicPromptCachingMiddleware` | `langchain_anthropic` | None | Cost optimization | ✅ Yes (Anthropic only) |
| `PatchToolCallsMiddleware` | `deepagents` | None | Fix dangling tool calls | ✅ Yes |
| `DateTimeMiddleware` | `deepagents` | `get_current_datetime` | Current date/time | ❌ No |
| `HumanInTheLoopMiddleware` | `langchain` | None | Human approval | ⚠️ If `interrupt_on` provided |
| `SkillsMiddleware` | `deepagents-cli` | None | Skills management | ❌ No (CLI only) |

## Key Takeaways

1. **Middleware is composable**: Mix and match as needed
2. **Order matters**: Tool-providing middleware should come first
3. **Default setup**: `create_deep_agent()` includes most middleware automatically
4. **Custom agents**: Use `create_agent()` with specific middleware for fine-grained control
5. **State schemas**: Each middleware can extend agent state with its own schema
6. **System prompts**: Middleware can inject instructions into the system prompt

## Further Reading

- [DeepAgents Documentation](https://docs.langchain.com/oss/python/deepagents/overview)
- [LangChain Middleware](https://docs.langchain.com/oss/python/langchain/middleware)
- [Agent Harness Documentation](https://docs.langchain.com/oss/python/deepagents/harness)

