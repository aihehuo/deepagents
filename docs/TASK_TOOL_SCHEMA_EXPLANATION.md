# Task Tool Schema: How the Model Knows How to Call It

## The Question

When looking at the `task` tool in `subagents.py`, you might wonder: **How does the LLM know the input/output schema?** There's no explicit JSON schema visible in the code!

The answer is: **`StructuredTool.from_function()` automatically generates the schema from Python type hints.**

---

## The Code

Here's the relevant code from `subagents.py`:

```python
def task(
    description: str,
    subagent_type: str,
    runtime: ToolRuntime,
) -> str | Command:
    # ... implementation

return StructuredTool.from_function(
    name="task",
    func=task,
    coroutine=atask,
    description=task_description,
)
```

---

## How `StructuredTool.from_function()` Works

### 1. **Schema Inference from Type Hints**

`StructuredTool.from_function()` uses Python's type annotations to automatically generate a JSON schema:

- **Function parameters** → Input schema properties
- **Type hints** → Schema types (string, integer, object, etc.)
- **Return type** → Output schema (though LLMs typically don't see return types directly)

### 2. **Generated Schema**

For the `task` function, the generated schema looks something like this:

```json
{
  "name": "task",
  "description": "Launch an ephemeral subagent to handle complex, multi-step independent tasks...",
  "parameters": {
    "type": "object",
    "properties": {
      "description": {
        "type": "string",
        "description": "The task description to send to the subagent"
      },
      "subagent_type": {
        "type": "string",
        "description": "The type of subagent to use (e.g., 'general-purpose', 'content-reviewer')"
      }
    },
    "required": ["description", "subagent_type"]
  }
}
```

**Note:** `runtime: ToolRuntime` is **NOT** in the schema! (See below)

### 3. **How the Model Sees It**

When LangChain sends tools to the LLM (e.g., via OpenAI function calling), it converts this to the model's tool format:

**OpenAI Function Calling Format:**
```json
{
  "type": "function",
  "function": {
    "name": "task",
    "description": "Launch an ephemeral subagent...",
    "parameters": {
      "type": "object",
      "properties": {
        "description": {"type": "string"},
        "subagent_type": {"type": "string"}
      },
      "required": ["description", "subagent_type"]
    }
  }
}
```

**Anthropic Tool Use Format:**
```json
{
  "name": "task",
  "description": "Launch an ephemeral subagent...",
  "input_schema": {
    "type": "object",
    "properties": {
      "description": {"type": "string"},
      "subagent_type": {"type": "string"}
    },
    "required": ["description", "subagent_type"]
  }
}
```

---

## Special Parameter: `runtime: ToolRuntime`

### The Problem

The `task` function has a third parameter:
```python
def task(
    description: str,
    subagent_type: str,
    runtime: ToolRuntime,  # <-- This one!
) -> str | Command:
```

But this parameter is **NOT** shown to the LLM in the schema!

### How It's Handled

LangChain/LangGraph has special handling for `ToolRuntime` parameters:

1. **Automatic Injection**: When the tool is called, LangGraph automatically injects the `runtime` parameter
2. **Schema Exclusion**: `ToolRuntime` parameters are automatically excluded from the JSON schema
3. **Runtime Access**: The function can access state, tool_call_id, etc. via `runtime`

### Similar Pattern: `InjectedToolCallId`

This is the same pattern used for `tool_call_id` in other tools:

```python
from langchain_core.tools import InjectedToolCallId

def write_todos(
    todos: list[Todo],
    tool_call_id: Annotated[str, InjectedToolCallId],  # <-- Auto-injected!
) -> Command:
    ...
```

Both `ToolRuntime` and `InjectedToolCallId` are automatically injected and excluded from schemas.

---

## Complete Flow

```
┌─────────────────────────────────────────────────────────────┐
│ Python Function Signature                                    │
│                                                              │
│ def task(                                                     │
│     description: str,                                        │
│     subagent_type: str,                                      │
│     runtime: ToolRuntime,  # ← Auto-injected                │
│ ) -> str | Command:                                          │
│                                                              │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ StructuredTool.from_function()                              │
│                                                              │
│ 1. Analyzes type hints                                       │
│ 2. Generates Pydantic schema                                 │
│ 3. Excludes ToolRuntime parameters                           │
│ 4. Converts to JSON schema                                   │
│                                                              │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ JSON Schema (What LLM Sees)                                 │
│                                                              │
│ {                                                            │
│   "name": "task",                                            │
│   "parameters": {                                            │
│     "properties": {                                          │
│       "description": {"type": "string"},                     │
│       "subagent_type": {"type": "string"}                    │
│     },                                                       │
│     "required": ["description", "subagent_type"]             │
│   }                                                          │
│ }                                                            │
│                                                              │
│ ❌ runtime is NOT in schema (auto-injected)                 │
│                                                              │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ LLM Makes Tool Call                                          │
│                                                              │
│ {                                                            │
│   "name": "task",                                            │
│   "arguments": {                                             │
│     "description": "Research Lebron James' accomplishments", │
│     "subagent_type": "general-purpose"                       │
│   }                                                          │
│ }                                                            │
│                                                              │
│ ❌ LLM doesn't provide runtime (doesn't know about it)      │
│                                                              │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ LangGraph Tool Execution                                     │
│                                                              │
│ 1. Receives tool call from LLM                               │
│ 2. Looks up tool function                                    │
│ 3. **Automatically injects runtime**                         │
│ 4. Calls: task(                                              │
│      description="Research Lebron...",                       │
│      subagent_type="general-purpose",                        │
│      runtime=<injected ToolRuntime>  ← Auto-injected!       │
│    )                                                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Inspecting the Actual Schema

You can inspect the generated schema programmatically:

```python
from langchain_core.tools import StructuredTool
from langchain.tools import ToolRuntime
from langgraph.types import Command

def task(description: str, subagent_type: str, runtime: ToolRuntime) -> str | Command:
    return Command(update={})

tool = StructuredTool.from_function(
    name="task",
    func=task,
    description="Test"
)

# Print the schema
print(tool.args_schema.schema_json(indent=2))

# Output will show:
# {
#   "properties": {
#     "description": {"type": "string"},
#     "subagent_type": {"type": "string"}
#   },
#   "required": ["description", "subagent_type"]
# }
# Note: runtime is NOT in the schema!
```

---

## Comparison with Explicit Schema Definition

### Implicit (from_function):

```python
def task(description: str, subagent_type: str, runtime: ToolRuntime) -> str | Command:
    ...

tool = StructuredTool.from_function(
    name="task",
    func=task,
    description="..."
)
```

**Pros:**
- ✅ Less code
- ✅ Type-safe (Python type hints)
- ✅ Single source of truth (function signature)
- ✅ Auto-excludes runtime parameters

### Explicit (manual schema):

```python
from pydantic import BaseModel

class TaskInput(BaseModel):
    description: str
    subagent_type: str

def task(description: str, subagent_type: str, runtime: ToolRuntime) -> str | Command:
    ...

tool = StructuredTool(
    name="task",
    args_schema=TaskInput,
    func=task,
    description="..."
)
```

**Pros:**
- ✅ More control over schema
- ✅ Can add field descriptions, validators, etc.

**Cons:**
- ❌ More boilerplate
- ❌ Need to remember to exclude runtime manually

The `from_function` approach is preferred for most cases!

---

## Key Takeaways

1. **Schema is Auto-Generated**: `StructuredTool.from_function()` infers the JSON schema from Python type hints
2. **Runtime is Injected**: `ToolRuntime` parameters are automatically injected by LangGraph and excluded from schemas
3. **LLM Only Sees User Parameters**: The LLM only sees `description` and `subagent_type` in the schema
4. **Type Hints Matter**: The type annotations (`str`, `list`, etc.) determine the schema types
5. **Description Helps**: The `description` parameter provides natural language guidance to the LLM

---

## Related Patterns

### Pattern 1: InjectedToolCallId
```python
def write_todos(
    todos: list[Todo],
    tool_call_id: Annotated[str, InjectedToolCallId],  # Auto-injected
) -> Command:
    ...
```

### Pattern 2: ToolRuntime
```python
def task(
    description: str,
    subagent_type: str,
    runtime: ToolRuntime,  # Auto-injected
) -> str | Command:
    ...
```

### Pattern 3: Optional Parameters
```python
def grep(
    pattern: str,
    path: str | None = None,  # Optional - not in required array
    glob: str | None = None,
) -> str:
    ...
```

All three are automatically handled by `StructuredTool.from_function()`!

