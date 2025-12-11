# How Todos Are Passed to the LLM After Creation

## Key Finding

**Todos are NOT directly passed to the LLM in the prompt or system message. Instead, they are made visible through ToolMessages in the conversation history.**

## The Flow

### Step 1: Todo Creation
When the LLM calls `write_todos`, the tool executes and returns a `Command`:

```python
Command(
    update={
        "todos": todos,  # Updates state["todos"]
        "messages": [
            ToolMessage(
                content=f"Updated todo list to {todos}",
                tool_call_id=tool_call_id
            )
        ]
    }
)
```

### Step 2: State Update
LangGraph applies the Command:
- `state["todos"]` is updated with the new todos list
- A `ToolMessage` is **appended to `state["messages"]`**

The state now looks like:
```python
{
    "messages": [
        HumanMessage("Help me refactor..."),
        AIMessage(..., tool_calls=[...]),  # The write_todos call
        ToolMessage("Updated todo list to [...]", tool_call_id="call_123")  # ← Contains todos
    ],
    "todos": [
        {"content": "Analyze...", "status": "in_progress"},
        {"content": "Identify...", "status": "pending"},
        # ...
    ]
}
```

### Step 3: Next LLM Call
When the agent makes the next LLM call:

1. **LangGraph/LangChain creates a ModelRequest** that includes:
   - All messages from `state["messages"]` (including the ToolMessage)
   - System prompt (with todo instructions from middleware)
   - Available tools
   - Current state (but `todos` field is `OmitFromInput`, so not directly accessible)

2. **The LLM receives the conversation history**:
   ```
   System: [Base instructions + todo usage guidelines]
   
   Human: Help me refactor this large codebase with 10+ modules
   
   Assistant: I'll help you refactor the codebase. Let me create a plan first.
   [tool_calls: write_todos]
   
   Tool: Updated todo list to [{'content': 'Analyze current codebase structure', 'status': 'in_progress'}, {'content': 'Identify refactoring opportunities', 'status': 'pending'}, ...]
   ```

3. **The LLM sees the todos in the ToolMessage content** and can:
   - See what tasks are in progress
   - See what tasks are pending
   - Decide what to do next based on the current todo list

## Important Points

1. **No Direct State Access**: The `todos` field uses `OmitFromInput`, meaning it's not directly passed to the LLM in the prompt. The LLM doesn't see `state["todos"]` directly.

2. **Conversation History is the Bridge**: The ToolMessage containing the todos becomes part of the conversation history, which IS passed to the LLM.

3. **ToolMessage Format**: The ToolMessage content is: `f"Updated todo list to {todos}"` - a string representation of the todos list.

4. **Iterative Updates**: Each time `write_todos` is called:
   - A new ToolMessage is added to the conversation
   - The LLM sees the updated todo list in the latest ToolMessage
   - The LLM can track progress through the conversation history

## Where This Happens in the Code

### TodoListMiddleware (LangChain)
- **Location**: `langchain.agents.middleware.todo.TodoListMiddleware`
- **Tool Implementation**: The `write_todos` tool returns a Command with:
  - State update for `todos`
  - ToolMessage added to `messages`

### LangGraph Agent Execution
- **Location**: LangGraph's agent execution flow
- **Process**: 
  1. Tool execution returns Command
  2. LangGraph applies Command to state
  3. Next model call includes all messages from state
  4. LLM sees ToolMessage in conversation history

### DeepAgents Integration
- **Location**: `libs/deepagents/deepagents/graph.py`
- **Integration**: `TodoListMiddleware()` is included by default in `create_deep_agent()`
- **Flow**: Standard LangGraph agent flow with middleware

## Example: What the LLM Sees

After the first `write_todos` call, on the next LLM invocation, the conversation looks like:

```
System: [Instructions about using write_todos tool...]

Human: Help me refactor this large codebase with 10+ modules

Assistant: I'll help you refactor the codebase. Let me create a plan first.
[tool_calls: write_todos with todos list]

Tool: Updated todo list to [{'content': 'Analyze current codebase structure', 'status': 'in_progress'}, {'content': 'Identify refactoring opportunities', 'status': 'pending'}, {'content': 'Prioritize refactoring tasks by impact', 'status': 'pending'}, {'content': 'Execute refactoring with tests', 'status': 'pending'}]

[LLM now sees the todos and can proceed with the first task]
```

The LLM can then:
- See that "Analyze current codebase structure" is `in_progress`
- Decide to start working on that task
- Use other tools (like `ls`, `read_file`, etc.) to analyze the codebase
- Update todos when done

## Summary

**The first todo in the list is NOT explicitly passed to the LLM. Instead:**

1. The entire todo list is included in a ToolMessage
2. The ToolMessage becomes part of the conversation history
3. The conversation history (including the ToolMessage) is passed to the LLM
4. The LLM sees the todos in the ToolMessage content and can infer:
   - Which task is `in_progress` (should work on this)
   - Which tasks are `pending` (upcoming work)
   - Which tasks are `completed` (already done)

This design allows the LLM to naturally see the todo list as part of the conversation flow, rather than requiring special state injection mechanisms.

