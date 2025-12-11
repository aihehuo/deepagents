# TodoListMiddleware Explained

## Your Understanding is Correct! ✅

Yes, `TodoListMiddleware` is essentially a combination of:
1. **The `write_todos` tool** - A tool that updates the todo list in state
2. **The system prompt** - Instructions appended to each LLM call via `wrap_model_call()`
3. **State schema definition** - Defines the `PlanningState` with `todos` field

## Structure of TodoListMiddleware

Based on inspection of the middleware:

```python
from langchain.agents.middleware.todo import TodoListMiddleware

middleware = TodoListMiddleware()

# Attributes:
middleware.tools           # List containing the 'write_todos' tool
middleware.system_prompt   # The prompt injected via wrap_model_call() (1074 chars)
middleware.state_schema    # PlanningState class (extends AgentState)
middleware.name            # "TodoListMiddleware"
middleware.tool_description # Tool description text
```

## The Three Components

### 1. The `write_todos` Tool

**What it is**: A LangChain tool that the LLM can call to update the todo list.

**What it does**:
- Takes a list of todos as input
- Returns a `Command` that:
  - Updates `state["todos"]` with the new list
  - Adds a `ToolMessage` to the conversation with the todos

**Location**: `middleware.tools[0]` (the only tool in the list)

**Tool Description**: 3654 characters of detailed instructions about when/how to use the tool

### 2. The System Prompt

**What it is**: Text instructions appended to the system message on every LLM call.

**What it does**:
- Informs the LLM about the `write_todos` tool
- Provides high-level guidance on when to use it
- Reminds about best practices (mark completed immediately, don't batch, etc.)

**How it's injected**: Via `wrap_model_call()` method

**Content** (1074 characters):
```python
WRITE_TODOS_SYSTEM_PROMPT = """## `write_todos`

You have access to the `write_todos` tool to help you manage and plan complex objectives.
Use this tool for complex objectives to ensure that you are tracking each necessary step and giving the user visibility into your progress.
This tool is very helpful for planning complex objectives, and for breaking down these larger complex objectives into smaller steps.

It is critical that you mark todos as completed as soon as you are done with a step. Do not batch up multiple steps before marking them as completed.
For simple objectives that only require a few steps, it is better to just complete the objective directly and NOT use this tool.
Writing todos takes time and tokens, use it when it is helpful for managing complex many-step problems! But not for simple few-step requests.

## Important To-Do List Usage Notes to Remember
- The `write_todos` tool should never be called multiple times in parallel.
- Don't be afraid to revise the To-Do list as you go. New information may reveal new tasks that need to be done, or old tasks that are irrelevant."""
```

### 3. The State Schema

**What it is**: A class definition that extends `AgentState` to include a `todos` field.

**What it does**:
- Defines the structure of the agent state
- Adds `todos: Annotated[NotRequired[list[Todo]], OmitFromInput]` to the state
- Ensures type safety and integration with LangGraph

**Definition**:
```python
class PlanningState(AgentState):
    """State schema for the todo middleware."""
    todos: Annotated[NotRequired[list[Todo]], OmitFromInput]
```

Where `Todo` is:
```python
class Todo(TypedDict):
    content: str
    status: Literal["pending", "in_progress", "completed"]
```

## How It Works Together

### Initialization
```python
middleware = TodoListMiddleware()
```

This creates:
- The `write_todos` tool (with its 3654-char description)
- The system prompt (1074 chars)
- Registers the `PlanningState` schema

### During Agent Execution

1. **State Schema Registration**: 
   - LangGraph uses `middleware.state_schema` to extend the agent state
   - Adds `todos` field to the state (initially empty)

2. **Tool Registration**:
   - `middleware.tools` is added to the agent's tool list
   - The LLM can now see and call `write_todos`

3. **System Prompt Injection** (on every LLM call):
   ```python
   def wrap_model_call(self, request: ModelRequest, handler):
       # Append system_prompt to existing system message
       new_system_message = SystemMessage(
           content=[..., self.system_prompt]
       )
       return handler(request.override(system_message=new_system_message))
   ```

4. **Tool Execution** (when LLM calls `write_todos`):
   - Tool updates `state["todos"]`
   - Tool adds `ToolMessage` to conversation
   - Next LLM call sees the todos in the ToolMessage

## Middleware Lifecycle Methods

`TodoListMiddleware` inherits from `AgentMiddleware`, which provides these lifecycle hooks:

- `before_agent()` - Called before agent execution starts
- `after_agent()` - Called after agent execution completes
- `before_model()` - Called before each model call
- `after_model()` - Called after each model response
- `wrap_model_call()` - **Used by TodoListMiddleware** to inject system prompt
- `wrap_tool_call()` - Can intercept tool calls (not used by TodoListMiddleware)

Plus async versions: `abefore_agent()`, `aafter_agent()`, etc.

## Comparison with Other Middleware

### FilesystemMiddleware
- **Tools**: `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `execute`
- **System Prompt**: Instructions about filesystem tools
- **State Schema**: `FilesystemState` with `files` field
- **Additional Logic**: Auto-evicts large tool results to filesystem

### SubAgentMiddleware
- **Tools**: `task` tool for delegating to subagents
- **System Prompt**: Instructions about using subagents
- **State Schema**: Uses base `AgentState`
- **Additional Logic**: Creates and manages subagent graphs

### TodoListMiddleware (Simpler!)
- **Tools**: Just `write_todos` (1 tool)
- **System Prompt**: Instructions about todo management
- **State Schema**: `PlanningState` with `todos` field
- **Additional Logic**: None - it's just tool + prompt + schema!

## Summary

**Yes, you're exactly right!** `TodoListMiddleware` is essentially:

1. ✅ **The `write_todos` tool** - Provides the capability to update todos
2. ✅ **The system prompt** - Injected via `wrap_model_call()` to guide usage
3. ✅ **The state schema** - Defines where todos are stored (`PlanningState`)

It's a **simple, focused middleware** that:
- Registers one tool
- Injects one system prompt
- Defines one state field

No complex logic, no additional processing - just these three components working together to enable todo list functionality in the agent.

## Code Reference

**Location**: `langchain.agents.middleware.todo.TodoListMiddleware`

**In DeepAgents**: Used by default in `create_deep_agent()` at:
- `libs/deepagents/deepagents/graph.py:114` (main agent)
- `libs/deepagents/deepagents/graph.py:121` (subagents)

