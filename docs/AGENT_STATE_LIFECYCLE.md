# Agent State Lifecycle: Complete Explanation

This document explains how agent state is initialized, organized, updated, and retrieved throughout the agent's lifecycle.

---

## Table of Contents

1. [Overview: What is Agent State?](#overview)
2. [Phase 1: Agent Creation and Schema Merging](#phase-1-agent-creation)
3. [Phase 2: State Initialization via `before_agent`](#phase-2-state-initialization)
4. [Phase 3: State Updates During Execution](#phase-3-state-updates)
5. [Phase 4: State Persistence and Retrieval](#phase-4-persistence)
6. [Complete Example Flow](#complete-example)

---

## Overview: What is Agent State?

**Agent State** is a TypedDict that holds all conversation data and middleware-managed information throughout an agent's execution. It's managed by LangGraph and persisted through checkpoints.

**Key Characteristics:**
- **TypedDict-based**: Each middleware can extend the base `AgentState` with its own fields
- **Merged at Runtime**: LangGraph automatically merges all middleware state schemas
- **Immutable Updates**: State can only be updated through `Command` objects or `before_agent` returns
- **Persistent**: State persists through checkpoints (if checkpointer is configured)

**Base Structure:**
```python
class AgentState(TypedDict):
    messages: list[Message]  # Core conversation history (always present)
    # ... other fields added by middleware
```

---

## Phase 1: Agent Creation and Schema Merging

### Step 1: Middleware Registration

When you create an agent, middleware is registered:

```python
agent = create_agent(
    model=model,
    middleware=[
        BusinessIdeaTrackerMiddleware(),  # Adds business_idea_complete, materialized_business_idea
        LanguageDetectionMiddleware(),    # Adds detected_language
        SkillsMiddleware(...),            # Adds skills_metadata
        FilesystemMiddleware(...),        # Adds files
        TodoListMiddleware(),             # Adds todos
    ],
)
```

### Step 2: State Schema Merging

**LangGraph automatically merges all middleware state schemas** into a single unified state type.

Each middleware defines its state schema:

```python
# BusinessIdeaTrackerMiddleware
class BusinessIdeaState(AgentState):
    business_idea_complete: NotRequired[bool]
    materialized_business_idea: NotRequired[str | None]

# LanguageDetectionMiddleware
class LanguageDetectionState(AgentState):
    detected_language: NotRequired[str]

# SkillsMiddleware
class SkillsState(AgentState):
    skills_metadata: NotRequired[list[SkillMetadata]]

# FilesystemMiddleware
class FilesystemState(AgentState):
    files: NotRequired[dict[str, FileData]]

# TodoListMiddleware (from LangChain)
class PlanningState(AgentState):
    todos: NotRequired[list[Todo]]
```

**LangGraph merges these into the final state schema:**

```python
# Final merged state (conceptual - not actual code)
class MergedAgentState(TypedDict):
    messages: list[Message]  # Base field
    
    # From BusinessIdeaTrackerMiddleware
    business_idea_complete: NotRequired[bool]
    materialized_business_idea: NotRequired[str | None]
    
    # From LanguageDetectionMiddleware
    detected_language: NotRequired[str]
    
    # From SkillsMiddleware
    skills_metadata: NotRequired[list[SkillMetadata]]
    
    # From FilesystemMiddleware
    files: NotRequired[dict[str, FileData]]
    
    # From TodoListMiddleware
    todos: NotRequired[list[Todo]]
```

**How it works:**
- LangGraph collects `middleware.state_schema` from all middleware
- It merges all TypedDict classes (Python's TypedDict supports multiple inheritance via merging)
- The merged schema becomes the agent's state type
- All fields are `NotRequired`, meaning they're optional

---

## Phase 2: State Initialization via `before_agent`

### When `before_agent` is Called

**`before_agent` is called at the start of each agent interaction** (before the first model call), not just once at agent creation.

### Initial State Structure

When you invoke the agent:

```python
result = agent.invoke({
    "messages": [HumanMessage("Hello")]
})
```

**The initial state passed to `before_agent` contains:**
- `messages`: The user's message(s)
- **Empty/missing fields** for all middleware state (since they're `NotRequired`)

```python
# Initial state (before any before_agent calls)
{
    "messages": [HumanMessage("Hello")],
    # All other fields are absent (NotRequired)
}
```

### `before_agent` Execution Order

**Middleware `before_agent` methods are called in registration order:**

```python
# Middleware registration order
middleware = [
    BusinessIdeaTrackerMiddleware(),   # before_agent called 1st
    LanguageDetectionMiddleware(),     # before_agent called 2nd
    SkillsMiddleware(...),             # before_agent called 3rd
    FilesystemMiddleware(...),         # before_agent called 4th
    TodoListMiddleware(),              # before_agent called 5th
]
```

### How `before_agent` Initializes State

Each middleware's `before_agent` can return a `StateUpdate` to initialize its fields:

```python
# BusinessIdeaTrackerMiddleware.before_agent
def before_agent(self, state: BusinessIdeaState, runtime: Runtime) -> BusinessIdeaStateUpdate | None:
    if "business_idea_complete" not in state:
        return BusinessIdeaStateUpdate(
            business_idea_complete=False,
            materialized_business_idea=None,
        )
    return None

# LanguageDetectionMiddleware.before_agent
def before_agent(self, state: LanguageDetectionState, runtime: Runtime) -> LanguageDetectionStateUpdate | None:
    messages = state.get("messages", [])
    if not messages:
        return None
    
    detected_lang = self._detect_language_from_messages(messages, state)
    if detected_lang:
        return LanguageDetectionStateUpdate(detected_language=detected_lang)
    return None

# SkillsMiddleware.before_agent
def before_agent(self, state: SkillsState, runtime: Runtime) -> SkillsStateUpdate | None:
    skills = list_skills(
        user_skills_dir=self.skills_dir,
        project_skills_dir=self.project_skills_dir,
    )
    return SkillsStateUpdate(skills_metadata=skills)
```

**LangGraph merges all `before_agent` returns into the state:**

```python
# After all before_agent calls complete
{
    "messages": [HumanMessage("Hello")],
    "business_idea_complete": False,              # From BusinessIdeaTrackerMiddleware
    "materialized_business_idea": None,            # From BusinessIdeaTrackerMiddleware
    "detected_language": "en",                     # From LanguageDetectionMiddleware
    "skills_metadata": [...],                      # From SkillsMiddleware
    "files": {},                                   # From FilesystemMiddleware (if initialized)
    "todos": [],                                   # From TodoListMiddleware (if initialized)
}
```

**Key Points:**
- `before_agent` can check if fields already exist before initializing (avoiding overwrites)
- Returns `None` if no update needed
- Returns `StateUpdate` TypedDict with fields to set
- LangGraph merges all updates into a single state update

---

## Phase 3: State Updates During Execution

### How State is Updated

**State updates happen in two ways:**

1. **Via `Command` objects returned by tools**
2. **Via `before_agent` returns** (already covered)

### Tool-Based State Updates

When a tool returns a `Command` object, LangGraph applies the update:

```python
# mark_business_idea_complete tool
def mark_business_idea_complete(idea_summary: str, tool_call_id: str) -> Command:
    return Command(
        update={
            "business_idea_complete": True,
            "materialized_business_idea": idea_summary,
            "messages": [
                ToolMessage(
                    content=f"Marked business idea as complete: {idea_summary}",
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )
```

**LangGraph applies the Command:**
1. Merges `update` fields into state
2. Appends `ToolMessage` to `messages` list
3. State is now updated for subsequent operations

### State Access During Execution

**Middleware can access state in several lifecycle methods:**

```python
# In wrap_model_call (before each LLM call)
def wrap_model_call(self, request: ModelRequest, handler: Callable) -> ModelResponse:
    state = cast("BusinessIdeaState", request.state)
    business_idea_complete = state.get("business_idea_complete", False)
    # Use state to modify behavior
    
# In wrap_tool_call (during tool execution)
def wrap_tool_call(self, request: ToolCallRequest, handler: Callable) -> ToolMessage | Command:
    state = request.state
    # Access state during tool execution
```

**State is always available through `request.state` or `runtime.state`.**

### Important: LLM Cannot Directly Access State

**The LLM does NOT have direct access to state fields.** State fields like `business_idea_complete` or `todos` are marked with `OmitFromInput` and are not included in the LLM's prompt.

**Instead, middleware reads state and injects context-aware instructions into the system prompt.**

#### Example: BusinessIdeaTrackerMiddleware

The `BusinessIdeaTrackerMiddleware` demonstrates this pattern:

```python
def wrap_model_call(self, request: ModelRequest, handler: Callable) -> ModelResponse:
    # 1. MIDDLEWARE READS STATE (LLM cannot see this directly)
    state = cast("BusinessIdeaState", request.state)
    business_idea_complete = state.get("business_idea_complete", False)
    materialized_idea = state.get("materialized_business_idea")
    
    # 2. MIDDLEWARE BUILDS CONTEXT-AWARE INSTRUCTION BASED ON STATE
    if business_idea_complete and materialized_idea:
        idea_context = (
            f"\n\n**Current Status**: A complete business idea has been identified:\n"
            f"{materialized_idea}\n\n"
            f"**Action**: Do NOT use the business-idea-evaluation skill. "
            f"The idea is already materialized."
        )
    elif business_idea_complete:
        idea_context = (
            "\n\n**Current Status**: A complete business idea has been identified in this conversation.\n\n"
            "**Action**: Do NOT use the business-idea-evaluation skill. The idea is already materialized."
        )
    else:
        idea_context = (
            "\n\n**Current Status**: No complete business idea has been identified yet.\n\n"
            "**Action**: You may use the business-idea-evaluation skill to evaluate the user's input."
        )
    
    # 3. MIDDLEWARE INJECTS INSTRUCTION INTO SYSTEM PROMPT
    full_prompt = self.system_prompt_template + idea_context
    
    # 4. LLM SEES THE INSTRUCTION (not the raw state value)
    return handler(request.override(system_prompt=new_system_prompt))
```

**What the LLM sees:**
- ✅ A clear instruction: "**Current Status**: A complete business idea has been identified: [summary]"
- ✅ An action directive: "**Action**: Do NOT use the business-idea-evaluation skill"
- ❌ NOT the raw state field `business_idea_complete: True`

**How Skills Reference State Checking:**

The SKILL.md file mentions checking state:

```markdown
1. **Check if idea is already complete**: Before using this skill, check if 
   `business_idea_complete` is already `true` in the agent state (via the 
   BusinessIdeaTrackerMiddleware).
```

**But the LLM doesn't actually "check" state directly.** Instead:

1. **The middleware checks state** in `wrap_model_call`
2. **The middleware injects an instruction** into the system prompt based on the state value
3. **The LLM follows the instruction** from the system prompt
4. **The skill documentation** uses the language "check state" as a conceptual explanation, but the actual mechanism is system prompt injection

**This is why both the middleware AND the skill need to work together:**
- **Middleware** (`BusinessIdeaTrackerMiddleware`): Reads state, injects status instructions
- **Skill** (`business-idea-evaluation` SKILL.md): Instructs LLM to follow the middleware's status guidance
- **Tool** (`mark_business_idea_complete`): Allows LLM to update state when needed

**Flow Diagram:**

```
┌─────────────────────────────────────────────────────────────┐
│ Agent State (in LangGraph)                                  │
│ {                                                           │
│   "business_idea_complete": True,                          │
│   "materialized_business_idea": "App for students...",     │
│   "messages": [...]                                         │
│ }                                                           │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ READ (middleware only)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ BusinessIdeaTrackerMiddleware.wrap_model_call()             │
│                                                             │
│ 1. Read state: business_idea_complete = True               │
│ 2. Build instruction: "Status: Idea complete, DO NOT use   │
│    business-idea-evaluation skill"                         │
│ 3. Inject into system prompt                               │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ INJECT
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ System Prompt (what LLM sees)                               │
│                                                             │
│ ...                                                         │
│ **Current Status**: A complete business idea has been      │
│ identified: App for students to manage study schedules...   │
│                                                             │
│ **Action**: Do NOT use the business-idea-evaluation skill. │
│ The idea is already materialized.                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ FOLLOW INSTRUCTIONS
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ LLM Decision                                                │
│                                                             │
│ "The status says the idea is complete, so I should NOT     │
│  use the business-idea-evaluation skill. I'll respond      │
│  to the user's request directly."                          │
└─────────────────────────────────────────────────────────────┘
```

**Key Takeaway:** State fields are **omitted from LLM input** (`OmitFromInput`). Middleware acts as a **translator** that reads state and provides LLM-friendly instructions in the system prompt.

---

## Phase 4: State Persistence and Retrieval

### State Storage Mechanisms

**State is stored in different ways depending on configuration:**

#### 1. In-Memory (Default - `InMemorySaver`)

```python
agent = create_agent(
    model=model,
    middleware=[...],
    # No checkpointer = InMemorySaver by default
)
```

**Characteristics:**
- State exists only in memory during execution
- **Lost when agent execution completes**
- Not persisted across agent restarts
- Fast, no I/O overhead

#### 2. Checkpointed (Persistent)

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
agent = create_agent(
    model=model,
    middleware=[...],
    checkpointer=checkpointer,  # Enables checkpointing
)
```

**Characteristics:**
- State is checkpointed after each agent step
- **Persists across agent invocations** (same thread_id)
- Can resume from checkpoints
- Supports undo/redo in some implementations

#### 3. Database-Backed (Postgres, etc.)

```python
from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = PostgresSaver.from_conn_string("postgresql://...")
agent = create_agent(
    model=model,
    middleware=[...],
    checkpointer=checkpointer,
)
```

**Characteristics:**
- State persisted to database
- **Survives agent restarts**
- Supports multi-instance deployments
- Long-term persistence

### Retrieving State After Execution

**After `agent.invoke()` completes, state is returned as a dictionary:**

```python
result = agent.invoke({
    "messages": [HumanMessage("Hello")]
})

# State fields are at the top level of result
business_idea_complete = result.get("business_idea_complete", False)
materialized_idea = result.get("materialized_business_idea")
detected_language = result.get("detected_language")
skills_metadata = result.get("skills_metadata", [])
files = result.get("files", {})
todos = result.get("todos", [])
messages = result.get("messages", [])
```

**Important:** State fields are **at the top level**, not nested under `result["state"]`.

### State in Subsequent Invocations

**If using a checkpointer, state persists across invocations:**

```python
config = {"configurable": {"thread_id": "conversation-1"}}

# First invocation
result1 = agent.invoke({
    "messages": [HumanMessage("Mark idea as complete")]
}, config)
# State: business_idea_complete=True

# Second invocation (same thread_id)
result2 = agent.invoke({
    "messages": [HumanMessage("What was the idea?")]
}, config)
# State: business_idea_complete=True (persisted from previous call)
```

**Without a checkpointer:**
- Each `invoke()` starts with fresh state
- State from previous invocations is lost
- `before_agent` re-initializes all fields

---

## Complete Example Flow

### Step-by-Step: Creating and Using an Agent

```python
# 1. CREATE AGENT
agent = create_agent(
    model=ChatAnthropic(...),
    middleware=[
        BusinessIdeaTrackerMiddleware(),
        LanguageDetectionMiddleware(),
        SkillsMiddleware(...),
    ],
    checkpointer=MemorySaver(),  # Enable persistence
)

# 2. FIRST INVOCATION
config = {"configurable": {"thread_id": "thread-1"}}
result1 = agent.invoke({
    "messages": [HumanMessage("I have a business idea: an app for students")]
}, config)

# What happens internally:
# a) LangGraph merges state schemas (all middleware fields)
# b) Initial state: {"messages": [HumanMessage(...)]}
# c) before_agent calls (in order):
#    - BusinessIdeaTrackerMiddleware: returns {"business_idea_complete": False, "materialized_business_idea": None}
#    - LanguageDetectionMiddleware: detects "en", returns {"detected_language": "en"}
#    - SkillsMiddleware: loads skills, returns {"skills_metadata": [...]}
# d) Merged state after before_agent:
#    {
#        "messages": [HumanMessage(...)],
#        "business_idea_complete": False,
#        "materialized_business_idea": None,
#        "detected_language": "en",
#        "skills_metadata": [...]
#    }
# e) Agent processes, LLM calls mark_business_idea_complete tool
# f) Tool returns Command(update={"business_idea_complete": True, "materialized_business_idea": "..."})
# g) LangGraph applies Command to state
# h) Final state:
#    {
#        "messages": [HumanMessage(...), AIMessage(...), ToolMessage(...)],
#        "business_idea_complete": True,
#        "materialized_business_idea": "An app for students to manage study schedules",
#        "detected_language": "en",
#        "skills_metadata": [...]
#    }
# i) State is checkpointed (because checkpointer is configured)
# j) Result returned with all state fields at top level

# 3. RETRIEVE STATE
print(result1.get("business_idea_complete"))  # True
print(result1.get("materialized_business_idea"))  # "An app for students..."
print(result1.get("detected_language"))  # "en"

# 4. SECOND INVOCATION (same thread_id)
result2 = agent.invoke({
    "messages": [HumanMessage("Can you evaluate another idea?")]
}, config)

# What happens internally:
# a) LangGraph loads checkpointed state from thread_id "thread-1"
# b) Initial state (from checkpoint):
#    {
#        "messages": [...previous messages...],
#        "business_idea_complete": True,  # ← Persisted from previous call!
#        "materialized_business_idea": "...",  # ← Persisted!
#        "detected_language": "en",
#        "skills_metadata": [...]
#    }
# c) New message appended: {"messages": [...previous..., HumanMessage("Can you...")]}
# d) before_agent calls (fields already exist, most return None)
# e) Agent sees business_idea_complete=True, knows not to use business-idea-evaluation skill
# f) Agent responds appropriately
```

---

## Key Takeaways

1. **State Schema Merging**: LangGraph automatically merges all middleware state schemas into a unified type
2. **Initialization**: `before_agent` is called at the start of each interaction to initialize middleware fields
3. **Updates**: State is updated via `Command` objects from tools or `before_agent` returns
4. **Persistence**: State persists across invocations only if a checkpointer is configured
5. **Retrieval**: After `invoke()`, state fields are at the top level of the result dictionary
6. **Thread Isolation**: Different `thread_id` values maintain separate state instances

---

## State Organization Summary

```
Agent State (TypedDict)
├── messages: list[Message]                    # Base field (always present)
├── business_idea_complete: bool               # From BusinessIdeaTrackerMiddleware
├── materialized_business_idea: str | None     # From BusinessIdeaTrackerMiddleware
├── detected_language: str                     # From LanguageDetectionMiddleware
├── skills_metadata: list[SkillMetadata]       # From SkillsMiddleware
├── files: dict[str, FileData]                 # From FilesystemMiddleware
├── todos: list[Todo]                          # From TodoListMiddleware
└── ... (other middleware fields)
```

**All fields are `NotRequired`**, meaning they're optional and may be absent in the state.

---

## Related Documentation

- [StateBackend vs StoreBackend](./STATE_VS_STORE_BACKEND.md) - File storage in state
- [TodoList Middleware Explained](./TODOLIST_MIDDLEWARE_EXPLAINED.md) - Example state usage
- [Planning Tools Design](./PLANNING_TOOLS_DESIGN.md) - Command objects and state updates

