# DeepAgents Planning Tools Design Document

## Overview

The planning tools in DeepAgents provide task management capabilities for AI agents to handle complex, multi-step tasks. The system is built around the `TodoListMiddleware` from LangChain, which provides a `write_todos` tool and state management for tracking task progress.

## Architecture

### Core Components

1. **TodoListMiddleware** - LangChain middleware that provides todo management
2. **PlanningState** - State schema for storing todo items
3. **write_todos Tool** - The primary interface for agents to manage task lists
4. **State Management** - Integrated with LangGraph's state system

### State Schema

```python
class Todo(TypedDict):
    """A single todo item with content and status."""
    content: str  # The content/description of the todo item
    status: Literal["pending", "in_progress", "completed"]  # Current status

class PlanningState(AgentState):
    """State schema for the todo middleware."""
    todos: Annotated[NotRequired[list[Todo]], OmitFromInput]  # List of todo items
```

### Tool Implementation

The `write_todos` tool is implemented as a LangChain tool that returns a `Command` object to update the agent state:

```python
@tool(description=WRITE_TODOS_TOOL_DESCRIPTION)
def write_todos(todos: list[Todo], tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Create and manage a structured task list for your current work session."""
    return Command(
        update={
            "todos": todos,  # Updates the entire todos list in state
            "messages": [ToolMessage(f"Updated todo list to {todos}", tool_call_id=tool_call_id)],
        }
    )
```

## Detailed Step-by-Step Execution Flow

### Phase 1: Agent Initialization

**Step 1: Agent Creation with TodoListMiddleware**
```python
# User code creates agent with TodoListMiddleware
from langchain.agents.middleware.todo import TodoListMiddleware
from langchain.agents import create_agent

agent = create_agent(
    model="anthropic:claude-sonnet-4-5-20250929",
    middleware=[TodoListMiddleware()],
    # ... other configuration
)
```

**Data Structures at Initialization:**
- **Agent Graph**: LangGraph StateGraph with middleware nodes
- **State Schema**: `PlanningState` extends `AgentState` with `todos: list[Todo]`
- **Middleware Tools**: `TodoListMiddleware` provides `write_todos` tool
- **System Prompt**: Base system prompt + `WRITE_TODOS_SYSTEM_PROMPT`

### Phase 2: User Request Processing

**Step 2: User Makes Complex Request**
```python
# User invokes agent with complex task
result = agent.invoke({
    "messages": [{
        "role": "user", 
        "content": "Help me refactor this large codebase with 10+ modules"
    }]
})
```

**Initial State Data Structure:**
```python
{
    "messages": [
        HumanMessage("Help me refactor this large codebase with 10+ modules")
    ],
    "todos": [],  # Empty initially (NotRequired, OmitFromInput)
    "jump_to": None,  # LangGraph internal state
    # ... other state fields
}
```

### Phase 3: Agent Planning Decision

**Step 3: Model Call with Injected Prompt**

The `TodoListMiddleware.wrap_model_call()` is invoked:
```python
def wrap_model_call(self, request: ModelRequest, handler):
    # 1. Check existing system message
    if request.system_message is not None:
        new_system_content = [
            *request.system_message.content_blocks,
            {"type": "text", "text": f"\n\n{self.system_prompt}"},
        ]
    else:
        new_system_content = [{"type": "text", "text": self.system_prompt}]
    
    # 2. Create new system message with todo instructions
    new_system_message = SystemMessage(content=new_system_content)
    
    # 3. Pass modified request to next handler
    return handler(request.override(system_message=new_system_message))
```

**ModelRequest Data Structure:**
```python
ModelRequest(
    model=BaseChatModel(...),
    messages=[HumanMessage("Help me refactor...")],
    system_message=SystemMessage([
        {"type": "text", "text": "Base agent instructions..."},
        {"type": "text", "text": "\n\n## write_todos\n\nYou have access to the write_todos tool..."}
    ]),
    tools=[write_todos, ...],  # write_todos tool included
    state={
        "messages": [...],
        "todos": []
    },
    # ... other fields
)
```

**Step 4: LLM Reasoning with Tool Guidance**

The LLM receives this prompt structure:
```
System: [Base agent instructions]
        [WRITE_TODOS_SYSTEM_PROMPT - guidelines on when/how to use write_todos]

Human: Help me refactor this large codebase with 10+ modules

Available Tools:
- write_todos: [WRITE_TODOS_TOOL_DESCRIPTION - 100+ lines of detailed instructions]
- ... other tools

The LLM reasons: "This is a complex multi-step task (10+ modules). 
I should use write_todos to create a structured plan."
```

### Phase 4: Todo List Creation

**Step 5: LLM Generates Tool Call**

The LLM outputs an AIMessage with tool call:
```python
AIMessage(
    content="I'll help you refactor the codebase. Let me start by creating a plan.",
    tool_calls=[{
        "name": "write_todos",
        "args": {
            "todos": [
                {"content": "Analyze current codebase structure", "status": "in_progress"},
                {"content": "Identify refactoring opportunities in each module", "status": "pending"},
                {"content": "Prioritize refactoring tasks by impact", "status": "pending"},
                {"content": "Create refactoring plan for first module", "status": "pending"},
                {"content": "Execute refactoring with tests", "status": "pending"},
                {"content": "Repeat for remaining modules", "status": "pending"},
                {"content": "Document changes and update documentation", "status": "pending"}
            ]
        },
        "id": "call_123"
    }]
)
```

**Step 6: Tool Execution**

The `write_todos` tool is executed:
```python
@tool(description=WRITE_TODOS_TOOL_DESCRIPTION)
def write_todos(todos: list[Todo], tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    return Command(
        update={
            "todos": todos,  # Update state
            "messages": [
                ToolMessage(
                    content=f"Updated todo list to {todos}",
                    tool_call_id=tool_call_id
                )
            ],
        }
    )
```

**Command Data Structure:**
```python
Command(
    update={
        "todos": [
            {"content": "Analyze current codebase structure", "status": "in_progress"},
            {"content": "Identify refactoring opportunities...", "status": "pending"},
            # ... 6 more items
        ],
        "messages": [
            ToolMessage(
                content="Updated todo list to [{'content': 'Analyze...', 'status': 'in_progress'}, ...]",
                tool_call_id="call_123"
            )
        ]
    }
)
```

### Phase 5: State Update and Feedback

**Step 7: LangGraph Processes Command**

LangGraph applies the Command to update state:
```python
# Updated State:
{
    "messages": [
        HumanMessage("Help me refactor this large codebase with 10+ modules"),
        AIMessage(..., tool_calls=[...]),  # Original AI message with tool call
        ToolMessage("Updated todo list to [...]", tool_call_id="call_123")
    ],
    "todos": [
        {"content": "Analyze current codebase structure", "status": "in_progress"},
        {"content": "Identify refactoring opportunities...", "status": "pending"},
        # ... other todos
    ]
}
```

**Step 8: Agent Continues Execution**

The agent sees the ToolMessage in conversation history and proceeds:
1. **Sees first task is `in_progress`**: "Analyze current codebase structure"
2. **Executes analysis tools**: `ls`, `read_file`, `grep`, etc.
3. **Completes task**: Updates todo status

### Phase 6: Todo List Updates

**Step 9: Agent Updates Todo Status**

After completing first task, agent calls `write_todos` again:
```python
# New tool call from LLM
AIMessage(
    content="I've analyzed the codebase structure. Now updating plan...",
    tool_calls=[{
        "name": "write_todos",
        "args": {
            "todos": [
                {"content": "Analyze current codebase structure", "status": "completed"},
                {"content": "Identify refactoring opportunities...", "status": "in_progress"},  # Updated
                {"content": "Prioritize refactoring tasks...", "status": "pending"},
                # ... plus potentially new tasks discovered during analysis
                {"content": "Fix circular dependencies in utils module", "status": "pending"}
            ]
        },
        "id": "call_456"
    }]
)
```

**Step 10: State Update Loop Continues**

Each `write_todos` call:
1. Replaces entire `todos` list in state
2. Adds ToolMessage to conversation
3. Provides visibility into progress
4. Allows plan adaptation based on new information

### Phase 7: Completion and Final State

**Step 11: Final State After Completion**

```python
{
    "messages": [
        HumanMessage("Help me refactor this large codebase..."),
        AIMessage(..., tool_calls=[...]),  # First write_todos call
        ToolMessage("Updated todo list to [...]", tool_call_id="call_123"),
        # ... many more messages from execution
        AIMessage(..., tool_calls=[...]),  # Final write_todos call
        ToolMessage("Updated todo list to [...]", tool_call_id="call_final"),
        AIMessage("I've completed refactoring all modules. Here's a summary...")
    ],
    "todos": [
        {"content": "Analyze current codebase structure", "status": "completed"},
        {"content": "Identify refactoring opportunities...", "status": "completed"},
        # ... all tasks completed
        {"content": "Document changes and update documentation", "status": "completed"}
    ]
}
```

## Complete Execution Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            PHASE 1: INITIALIZATION                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  User Code: create_agent(middleware=[TodoListMiddleware()])                 │
│  ├─ Creates LangGraph StateGraph with middleware integration                │
│  ├─ Defines PlanningState schema with todos: list[Todo]                     │
│  └─ Registers write_todos tool in agent toolset                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PHASE 2: USER REQUEST & STATE SETUP                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  agent.invoke({"messages": [HumanMessage("Refactor codebase...")]})         │
│  ├─ Initial State: {messages: [HumanMessage], todos: []}                    │
│  └─ State Schema: PlanningState (todos marked OmitFromInput)                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                  PHASE 3: MIDDLEWARE PROMPT INJECTION                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  TodoListMiddleware.wrap_model_call()                                       │
│  ├─ Input: ModelRequest with base system prompt                             │
│  ├─ Process: Append WRITE_TODOS_SYSTEM_PROMPT to system message             │
│  └─ Output: Modified ModelRequest with todo instructions                    │
│                                                                             │
│  Data Flow:                                                                 │
│  ModelRequest → wrap_model_call() → ModelRequest'                           │
│  (system_message += todo_instructions)                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 4: LLM REASONING & PLANNING                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  LLM Receives:                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ System: [Base instructions] + [Todo usage guidelines]               │   │
│  │ Human: "Refactor codebase..."                                       │   │
│  │ Tools: write_todos: [100+ lines of detailed instructions]           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  LLM Outputs: AIMessage with tool_calls to write_todos                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ AIMessage:                                                          │   │
│  │   content: "I'll create a plan..."                                  │   │
│  │   tool_calls: [{                                                    │   │
│  │     name: "write_todos",                                            │   │
│  │     args: {todos: [{content: "Task 1", status: "in_progress"}, ...]},│   │
│  │     id: "call_123"                                                  │   │
│  │   }]                                                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PHASE 5: TOOL EXECUTION                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  write_todos(todos: list[Todo], tool_call_id: str) → Command               │
│  ├─ Input: List of Todo dictionaries                                       │
│  ├─ Process: Creates Command with state updates                            │
│  └─ Output: Command(update={todos: [...], messages: [ToolMessage]})        │
│                                                                             │
│  Command Structure:                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Command:                                                            │   │
│  │   update: {                                                         │   │
│  │     "todos": [Todo, Todo, ...],  # Complete replacement             │   │
│  │     "messages": [                                                   │   │
│  │       ToolMessage("Updated todo list to...", tool_call_id)          │   │
│  │     ]                                                               │   │
│  │   }                                                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PHASE 6: STATE UPDATE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  LangGraph applies Command to state:                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ State Before:                                                       │   │
│  │   messages: [HumanMessage, AIMessage]                               │   │
│  │   todos: []                                                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ State After:                                                        │   │
│  │   messages: [HumanMessage, AIMessage, ToolMessage]                  │   │
│  │   todos: [{content: "Task 1", status: "in_progress"}, ...]          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 PHASE 7: EXECUTION & ITERATIVE UPDATES                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  Loop:                                                                      │
│  1. Agent sees ToolMessage with current todos                              │
│  2. Executes in_progress task using other tools                            │
│  3. Completes task → calls write_todos with updated status                 │
│  4. State updates with new ToolMessage                                     │
│  5. Repeat until all tasks completed                                       │
│                                                                             │
│  Each iteration:                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ AIMessage → write_todos → Command → State Update → ToolMessage      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PHASE 8: COMPLETION                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  Final State:                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ messages: [HumanMessage, AIMessage₁, ToolMessage₁, ...,            │   │
│  │            AIMessageₙ, ToolMessageₙ, FinalAIMessage]               │   │
│  │ todos: [{content: "Task 1", status: "completed"}, ...,             │   │
│  │         {content: "Task N", status: "completed"}]                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  User receives: Complete conversation with visible planning progress        │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Data Flow Summary

1. **User Input** → `HumanMessage` in `state["messages"]`
2. **Middleware Injection** → `SystemMessage` extended with todo instructions
3. **LLM Decision** → `AIMessage` with `tool_calls` to `write_todos`
4. **Tool Execution** → `Command` with state updates
5. **State Update** → `state["todos"]` replaced, `ToolMessage` added
6. **Feedback Loop** → `ToolMessage` in conversation informs next steps
7. **Iterative Updates** → Multiple `write_todos` calls adapt plan
8. **Final State** → All todos `completed`, comprehensive message history

## Detailed Data Structures at Each Step

### Step 1: Initial Agent State
```python
# Type: PlanningState (extends AgentState)
{
    "messages": Annotated[list[AnyMessage], add_messages],  # Required
    "jump_to": Annotated[JumpTo | None, EphemeralValue, PrivateStateAttr],  # NotRequired
    "structured_response": Annotated[ResponseT, OmitFromInput],  # NotRequired
    "todos": Annotated[NotRequired[list[Todo]], OmitFromInput]  # Our addition
}
```

### Step 2: ModelRequest to LLM
```python
# Type: ModelRequest
ModelRequest(
    model=BaseChatModel("anthropic:claude-sonnet-4-5-20250929"),
    messages=[  # Conversation history
        HumanMessage(content="Help me refactor this large codebase...")
    ],
    system_message=SystemMessage(
        content_blocks=[
            {"type": "text", "text": "Base agent instructions..."},
            {"type": "text", "text": "\n\n## write_todos\n\nYou have access to..."}
        ]
    ),
    tools=[  # Available tools including write_todos
        BaseTool(
            name="write_todos",
            description=WRITE_TODOS_TOOL_DESCRIPTION,  # 100+ lines
            args_schema=type('WriteTodosArgs', (), {
                'todos': (list[Todo], ...),
                'tool_call_id': (Annotated[str, InjectedToolCallId], ...)
            })
        ),
        # ... other tools
    ],
    state={...},  # Current agent state
    runtime=Runtime[...],  # LangGraph runtime
    # ... other fields
)
```

### Step 3: LLM Response with Tool Call
```python
# Type: AIMessage
AIMessage(
    content="I'll help you refactor the codebase. Let me start by creating a plan.",
    tool_calls=[
        {
            "name": "write_todos",
            "args": {
                "todos": [
                    {
                        "content": "Analyze current codebase structure",
                        "status": "in_progress"  # Literal["pending","in_progress","completed"]
                    },
                    # ... more Todo dicts
                ]
            },
            "id": "call_123",  # Generated by LLM
            "type": "tool_call"
        }
    ],
    # ... other AIMessage fields
)
```

### Step 4: Tool Execution Command
```python
# Type: Command (LangGraph state update instruction)
Command(
    update={
        "todos": [  # Complete replacement of todos list
            {"content": "Analyze...", "status": "in_progress"},
            {"content": "Identify...", "status": "pending"},
            # ...
        ],
        "messages": [  # Append to messages list
            ToolMessage(
                content=f"Updated todo list to {todos}",
                tool_call_id="call_123",
                # Additional ToolMessage fields
            )
        ]
    },
    # Command can also have goto, interrupt, etc.
)
```

### Step 5: Updated State
```python
# After LangGraph applies Command
{
    "messages": [
        HumanMessage("Help me refactor..."),
        AIMessage(..., tool_calls=[...]),  # Original message
        ToolMessage("Updated todo list to [...]", tool_call_id="call_123")
    ],
    "todos": [
        {"content": "Analyze...", "status": "in_progress"},
        {"content": "Identify...", "status": "pending"},
        # ...
    ],
    # Other state fields maintained
}
```

### Step 6: Iterative Update Pattern
```python
# Subsequent write_todos calls follow same pattern:
# 1. LLM sees ToolMessage with current todos
# 2. LLM decides to update based on progress
# 3. New AIMessage with write_todos tool call
# 4. New Command updates todos and adds ToolMessage
# 5. State updated, loop continues

# Example update after completing first task:
Command(
    update={
        "todos": [
            {"content": "Analyze...", "status": "completed"},  # Updated
            {"content": "Identify...", "status": "in_progress"},  # Updated
            {"content": "Prioritize...", "status": "pending"},
            # Potentially new tasks added
            {"content": "Fix circular dependencies...", "status": "pending"}
        ],
        "messages": [
            ToolMessage("Updated todo list to [...]", tool_call_id="call_456")
        ]
    }
)
```

## Prompt Engineering Details

### System Prompt Injection
- **Base Prompt**: Original agent instructions
- **Todo Instructions**: `WRITE_TODOS_SYSTEM_PROMPT` appended
- **Result**: Combined prompt guiding agent on todo usage
- **Injection Method**: `wrap_model_call()` modifies `ModelRequest.system_message`

### Tool Description
- **Length**: 100+ lines of detailed instructions
- **Content**: When to use, how to use, task management guidelines
- **Purpose**: Ensure agent uses tool appropriately
- **Format**: Markdown-style with sections, examples, warnings

### Tool Message Content
- **Format**: `f"Updated todo list to {todos}"`
- **Visibility**: Human-readable representation of todo list
- **Context**: Provides agent with todo state in conversation
- **Persistence**: Remains in message history for agent reference

### State Schema Design
- **`OmitFromInput`**: `todos` not required in user input, but available in state
- **`NotRequired`**: Can be empty list initially
- **Type Safety**: `list[Todo]` with `Literal` status values
- **Integration**: Extends `AgentState` for LangGraph compatibility

## Key Design Decisions

### 1. **State-Based Approach**
- Todos are stored in agent state rather than external storage
- Enables seamless integration with LangGraph's checkpointing system
- Allows state to be shared across middleware components

### 2. **Complete Replacement Strategy**
- `write_todos` replaces the entire todos list each time
- Simpler than partial updates (add/remove/update individual items)
- Forces agent to think about the complete task list

### 3. **State-Based Access (No Separate Read Tool)**
- **Important Discovery**: The README mentions `read_todos` but no such tool exists in the implementation
- **State Access Pattern**: Agents access todos through the `request.state` object available in middleware
- **Schema Design**: The `todos` field uses `OmitFromInput` annotation:
  - `todos: Annotated[NotRequired[list[Todo]], OmitFromInput]`
  - This means `todos` is omitted from user input schema but available in agent state
- **Middleware Integration**: The `TodoListMiddleware` can access `request.state.todos` directly
- **Agent Context**: While the agent doesn't get `todos` in its prompt, it can infer them from tool call responses
- **Reduced Complexity**: Eliminates need for separate read tool while maintaining state persistence

### 4. **Comprehensive Guidance**
- Tool description includes extensive usage guidelines
- Covers when to use/not use the tool
- Provides task management best practices

## Integration in DeepAgents

### Default Configuration

In `deepagents/graph.py`, the middleware is included by default:

```python
deepagent_middleware = [
    TodoListMiddleware(),  # Planning tools
    FilesystemMiddleware(backend=backend),  # File operations
    SubAgentMiddleware(...),  # Subagent delegation
    # ... other middleware
]
```

### Subagent Support

Subagents also receive planning capabilities:
```python
default_middleware=[
    TodoListMiddleware(),  # Subagents can also plan!
    FilesystemMiddleware(backend=backend),
    # ...
]
```

## Usage Patterns

### When to Use `write_todos`

According to the tool description, agents should use `write_todos` for:
1. **Complex multi-step tasks** (3+ distinct steps)
2. **Non-trivial complex tasks** requiring careful planning
3. **User explicitly requests todo list**
4. **User provides multiple tasks**
5. **Plan may need revisions** based on early results

### When NOT to Use

Agents should NOT use `write_todos` for:
1. Single, straightforward tasks
2. Trivial tasks with no tracking benefit
3. Tasks completed in <3 trivial steps
4. Purely conversational/informational tasks

### Task Management Guidelines

The tool enforces these practices:
1. **Mark tasks `in_progress` BEFORE starting work**
2. **Mark tasks `completed` IMMEDIATELY after finishing**
3. **Always have at least one task `in_progress`** (unless all completed)
4. **Remove irrelevant tasks** from the list entirely
5. **Don't change completed tasks** - only update pending/in_progress

## Example Workflow

```python
# Agent creates initial plan
write_todos([
    {"content": "Explore repository structure", "status": "in_progress"},
    {"content": "Analyze core modules", "status": "pending"},
    {"content": "Document findings", "status": "pending"}
])

# After completing first task
write_todos([
    {"content": "Explore repository structure", "status": "completed"},
    {"content": "Analyze core modules", "status": "in_progress"},
    {"content": "Document findings", "status": "pending"}
])

# Discovering a new task during execution
write_todos([
    {"content": "Explore repository structure", "status": "completed"},
    {"content": "Analyze core modules", "status": "completed"},
    {"content": "Test key functionality", "status": "in_progress"},
    {"content": "Document findings", "status": "pending"}
])
```

## Advantages

1. **Visibility** - Users can see agent's plan and progress
2. **Structure** - Encourages systematic approach to complex tasks
3. **Persistence** - Task state survives interruptions/errors
4. **Simplicity** - Single tool with clear semantics
5. **Integration** - Works seamlessly with other middleware

## Limitations

1. **No Partial Updates** - Must replace entire list each time
2. **State Size** - Large todo lists increase state size
3. **No Explicit Read Tool** - README mentions `read_todos` but implementation only has `write_todos`
4. **Tool Call Overhead** - Each update requires a tool call

## Future Enhancements

Potential improvements could include:
1. **Partial update operations** (add/remove/update individual todos)
2. **Priority levels** for task prioritization
3. **Dependencies** between tasks
4. **Time estimates** for better planning
5. **Visualization tools** for user display

## Complete System Prompts for Planning

### 1. System Prompt Injected by TodoListMiddleware

This prompt is appended to the agent's system message via `wrap_model_call()`:

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

### 2. Tool Description Shown to LLM

This is the tool description that appears in the LLM's tool list (100+ lines):

```python
WRITE_TODOS_TOOL_DESCRIPTION = """Use this tool to create and manage a structured task list for your current work session. This helps you track progress, organize complex tasks, and demonstrate thoroughness to the user.

Only use this tool if you think it will be helpful in staying organized. If the user's request is trivial and takes less than 3 steps, it is better to NOT use this tool and just do the task directly.

## When to Use This Tool
Use this tool in these scenarios:

1. Complex multi-step tasks - When a task requires 3 or more distinct steps or actions
2. Non-trivial and complex tasks - Tasks that require careful planning or multiple operations
3. User explicitly requests todo list - When the user directly asks you to use the todo list
4. User provides multiple tasks - When users provide a list of things to be done (numbered or comma-separated)
5. The plan may need future revisions or updates based on results from the first few steps

## How to Use This Tool
1. When you start working on a task - Mark it as in_progress BEFORE beginning work.
2. After completing a task - Mark it as completed and add any new follow-up tasks discovered during implementation.
3. You can also update future tasks, such as deleting them if they are no longer necessary, or adding new tasks that are necessary. Don't change previously completed tasks.
4. You can make several updates to the todo list at once. For example, when you complete a task, you can mark the next task you need to start as in_progress.

## When NOT to Use This Tool
It is important to skip using this tool when:
1. There is only a single, straightforward task
2. The task is trivial and tracking it provides no benefit
3. The task can be completed in less than 3 trivial steps
4. The task is purely conversational or informational

## Task States and Management

1. **Task States**: Use these states to track progress:
   - pending: Task not yet started
   - in_progress: Currently working on (you can have multiple tasks in_progress at a time if they are not related to each other and can be run in parallel)
   - completed: Task finished successfully

2. **Task Management**:
   - Update task status in real-time as you work
   - Mark tasks complete IMMEDIATELY after finishing (don't batch completions)
   - Complete current tasks before starting new ones
   - Remove tasks that are no longer relevant from the list entirely
   - IMPORTANT: When you write this todo list, you should mark your first task (or tasks) as in_progress immediately!.
   - IMPORTANT: Unless all tasks are completed, you should always have at least one task in_progress to show the user that you are working on something.

3. **Task Completion Requirements**:
   - ONLY mark a task as completed when you have FULLY accomplished it
   - If you encounter errors, blockers, or cannot finish, keep the task as in_progress
   - When blocked, create a new task describing what needs to be resolved
   - Never mark a task as completed if:
     - There are unresolved issues or errors
     - Work is partial or incomplete
     - You encountered blockers that prevent completion
     - You couldn't find necessary resources or dependencies
     - Quality standards haven't been met

4. **Task Breakdown**:
   - Create specific, actionable items
   - Break complex tasks into smaller, manageable steps
   - Use clear, descriptive task names

Being proactive with task management demonstrates attentiveness and ensures you complete all requirements successfully
Remember: If you only need to make a few tool calls to complete a task, and it is clear what you need to do, it is better to just do the task directly and NOT call this tool at all."""
```

### 3. Combined Prompt Structure Seen by LLM

When the agent processes a request, the LLM sees this combined prompt structure:

```
[Base Agent System Instructions]
[Optional: User-provided custom system prompt]

## `write_todos`

You have access to the `write_todos` tool to help you manage and plan complex objectives.
Use this tool for complex objectives to ensure that you are tracking each necessary step and giving the user visibility into your progress.
This tool is very helpful for planning complex objectives, and for breaking down these larger complex objectives into smaller steps.

It is critical that you mark todos as completed as soon as you are done with a step. Do not batch up multiple steps before marking them as completed.
For simple objectives that only require a few steps, it is better to just complete the objective directly and NOT use this tool.
Writing todos takes time and tokens, use it when it is helpful for managing complex many-step problems! But not for simple few-step requests.

## Important To-Do List Usage Notes to Remember
- The `write_todos` tool should never be called multiple times in parallel.
- Don't be afraid to revise the To-Do list as you go. New information may reveal new tasks that need to be done, or old tasks that are irrelevant.

[User Message: "Help me refactor this large codebase with 10+ modules"]

Available Tools:
- write_todos: [The entire WRITE_TODOS_TOOL_DESCRIPTION shown here - 100+ lines]
- [Other available tools...]
```

### 4. Prompt Engineering Strategy

The prompts are designed with these principles:

1. **Progressive Disclosure**: System prompt gives high-level guidance, tool description provides detailed instructions
2. **Clear Decision Criteria**: Explicit "When to Use" and "When NOT to Use" sections
3. **Actionable Guidance**: Step-by-step instructions for proper usage
4. **Best Practices**: Task management patterns and quality standards
5. **Error Prevention**: Warnings about common mistakes (parallel calls, batching completions)
6. **User Visibility**: Emphasis on showing progress to user through todo tracking

### 5. Tool Input and Output Schema

The `write_todos` tool has a well-defined input and output schema that the LLM must adhere to:

#### Input Schema (`write_todos` function parameters):

```python
def write_todos(
    todos: list[Todo], 
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
```

**Parameter 1: `todos` (list[Todo])**
- **Type**: `list[Todo]` where `Todo` is a `TypedDict`
- **Structure**: List of todo items, each with:
  ```python
  Todo = TypedDict('Todo', {
      'content': str,  # The content/description of the todo item
      'status': Literal['pending', 'in_progress', 'completed']  # Current status
  })
  ```
- **Example**:
  ```python
  todos = [
      {
          "content": "Analyze current codebase structure",
          "status": "in_progress"  # Must be one of: pending, in_progress, completed
      },
      {
          "content": "Identify refactoring opportunities",
          "status": "pending"
      }
  ]
  ```

**Parameter 2: `tool_call_id` (Annotated[str, InjectedToolCallId])**
- **Type**: `str` annotated with `InjectedToolCallId`
- **Purpose**: Unique identifier for the tool call, automatically injected by LangGraph
- **Note**: The LLM doesn't need to provide this - it's injected by the system

#### Output Schema (Return type: `Command`):

```python
return Command(
    update={
        "todos": todos,  # The same todos list passed as input
        "messages": [
            ToolMessage(
                content=f"Updated todo list to {todos}",
                tool_call_id=tool_call_id
            )
        ]
    }
)
```

**Command Structure**:
- **Type**: `langgraph.types.Command`
- **Purpose**: Instructs LangGraph how to update the agent state
- **`update` field**: Dictionary specifying state updates:
  - `"todos"`: Replaces the entire `todos` list in agent state
  - `"messages"`: Appends a `ToolMessage` to the conversation history

**ToolMessage Structure**:
```python
ToolMessage(
    content=f"Updated todo list to {todos}",  # String representation of todos
    tool_call_id=tool_call_id  # Links this message to the tool call
)
```

#### Complete Schema Definition:

```python
# Pydantic-style schema (what the LLM sees):
class WriteTodosInputSchema:
    """Input schema for write_todos tool."""
    
    todos: list[Todo]
    """List of todo items to update in the agent state.
    
    Each todo item must have:
    - content: str - Description of the task
    - status: Literal['pending', 'in_progress', 'completed'] - Current status
    
    Important rules:
    1. When creating a new todo list, mark first task(s) as 'in_progress'
    2. Update status to 'completed' immediately after finishing a task
    3. Remove tasks that are no longer relevant
    4. Don't change completed tasks
    """
    
    # tool_call_id is automatically injected, not provided by LLM

# The tool returns:
Command(
    update={
        "todos": List[Todo],  # State update
        "messages": List[ToolMessage]  # Conversation update
    }
)
```

#### Schema Validation Rules:

1. **`todos` must be a list** (not a single item)
2. **Each item must have `content`** (string, non-empty)
3. **Each item must have `status`** (exactly one of: `'pending'`, `'in_progress'`, `'completed'`)
4. **First task(s) should be `'in_progress'`** when creating initial plan
5. **At least one task `'in_progress'`** unless all are completed
6. **Complete replacement**: Each call replaces entire `todos` list in state

#### How the Schema is Presented to the LLM:

When the LLM receives the tool list, `write_todos` is presented in a format the LLM can understand (e.g., OpenAI function calling format):

```json
{
  "name": "write_todos",
  "description": "Use this tool to create and manage a structured task list for your current work session. This helps you track progress, organize complex tasks, and demonstrate thoroughness to the user.\n\nOnly use this tool if you think it will be helpful in staying organized. If the user's request is trivial and takes less than 3 steps, it is better to NOT use this tool and just do the task directly.\n\n[Full 100+ line description continues...]",
  "parameters": {
    "type": "object",
    "properties": {
      "todos": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "content": {
              "type": "string",
              "description": "The content/description of the todo item."
            },
            "status": {
              "type": "string",
              "enum": ["pending", "in_progress", "completed"],
              "description": "The current status of the todo item."
            }
          },
          "required": ["content", "status"],
          "additionalProperties": false
        },
        "description": "List of todo items for tracking task progress."
      }
    },
    "required": ["todos"],
    "additionalProperties": false
  }
}
```

**Note**: The `tool_call_id` parameter is not shown to the LLM - it's automatically injected by LangGraph when the tool is called. The LLM only needs to provide the `todos` parameter.

#### Example LLM Tool Call:

When the LLM decides to use `write_todos`, it generates a tool call like:

```json
{
  "tool_calls": [
    {
      "id": "call_123",
      "type": "function",
      "function": {
        "name": "write_todos",
        "arguments": "{\"todos\": [{\"content\": \"Analyze codebase\", \"status\": \"in_progress\"}, {\"content\": \"Identify issues\", \"status\": \"pending\"}]}"
      }
    }
  ]
}
```

The system then:
1. Parses the JSON arguments
2. Injects the `tool_call_id` ("call_123")
3. Calls the `write_todos` function with both parameters
4. Executes the returned `Command` to update state

### 6. Prompt Injection Mechanism

The prompts are injected through the middleware's `wrap_model_call()` method:

```python
def wrap_model_call(self, request: ModelRequest, handler):
    # Append system prompt to existing system message
    if request.system_message is not None:
        new_system_content = [
            *request.system_message.content_blocks,
            {"type": "text", "text": f"\n\n{self.system_prompt}"},
        ]
    else:
        new_system_content = [{"type": "text", "text": self.system_prompt}]
    
    new_system_message = SystemMessage(content=new_system_content)
    return handler(request.override(system_message=new_system_message))
```

This ensures the planning instructions are always present when the agent needs to make decisions about task management.

## Conclusion

The planning tools in DeepAgents provide a simple yet effective task management system that integrates seamlessly with LangGraph's state management. By using a state-based approach with comprehensive guidance, it enables agents to tackle complex multi-step tasks while providing visibility into their planning process.

The prompt engineering is particularly sophisticated, with:
- A concise system prompt for high-level guidance
- A detailed tool description for specific instructions  
- Clear decision criteria for when to use the tool
- Best practices for task management
- Error prevention guidelines

This dual-prompt approach (system prompt + tool description) allows the agent to understand both the "why" and "how" of task planning, leading to more reliable and transparent execution of complex multi-step tasks.