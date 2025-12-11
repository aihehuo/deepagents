# Who Sets the Completed Status of Todo Tasks?

## Answer: **The LLM sets the completed status**

The completed status is **not automatically set** by the middleware or any other system component. It is **entirely controlled by the LLM** through explicit calls to the `write_todos` tool.

## How It Works

### 1. The Only Way to Update Todos: `write_todos` Tool

The `write_todos` tool is the **only interface** for updating todos. It accepts a complete list of todos where each todo has:
- `content`: The task description
- `status`: One of `"pending"`, `"in_progress"`, or `"completed"`

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

**Key Point**: The tool accepts a **complete list** of todos. The LLM must provide the full list with updated statuses each time it calls the tool.

### 2. The LLM Decides When to Mark Tasks as Completed

The LLM is instructed (via system prompts and tool descriptions) to:

1. **Mark tasks as completed after finishing them**:
   ```
   "After completing a task - Mark it as completed and add any new follow-up tasks discovered during implementation."
   ```

2. **Mark tasks complete immediately**:
   ```
   "Mark tasks complete IMMEDIATELY after finishing (don't batch completions)"
   ```

3. **Only mark as completed when fully done**:
   ```
   "ONLY mark a task as completed when you have FULLY accomplished it"
   ```

### 3. The Flow

Here's what happens when a task is completed:

```
1. LLM sees current todos in ToolMessage:
   [
     {"content": "Research market", "status": "in_progress"},
     {"content": "Create strategy", "status": "pending"}
   ]

2. LLM executes the task (e.g., calls execute_task tool or does work)

3. LLM decides: "I've completed the research task"

4. LLM calls write_todos with updated list:
   write_todos([
     {"content": "Research market", "status": "completed"},  ← LLM changed this
     {"content": "Create strategy", "status": "in_progress"}  ← LLM also updated this
   ])

5. Tool updates state with the new list
6. ToolMessage shows the updated todos
7. Next LLM call sees the updated status in conversation history
```

### 4. No Automatic Status Changes

**Important**: There is **no automatic mechanism** that:
- Detects when a task is done
- Automatically changes status from `in_progress` to `completed`
- Monitors tool execution to update todos

Everything is **explicit** and **LLM-driven**:
- The LLM must **explicitly call** `write_todos`
- The LLM must **explicitly set** the status to `"completed"`
- The LLM must **provide the complete list** with all statuses

## Why This Design?

### Complete Replacement Strategy

The `write_todos` tool uses a **complete replacement strategy**:
- It replaces the **entire** todos list each time
- This forces the LLM to think about the complete task list
- Simpler than partial updates (add/remove/update individual items)
- Ensures the LLM maintains awareness of all tasks

### LLM Control

This design gives the LLM **full control** over:
- When to mark tasks as completed
- What constitutes "completion"
- How to prioritize and organize tasks
- When to add new tasks or remove irrelevant ones

### State Persistence

The todos are stored in agent state (`state["todos"]`), but:
- The state is **not directly accessible** to the LLM (uses `OmitFromInput`)
- The LLM sees todos through **ToolMessage** in conversation history
- The LLM must **explicitly update** the list to change statuses

## Example from Test

In the test `test_agent_creates_and_executes_all_todos`, you can see this in action:

1. **Initial creation**: LLM calls `write_todos` with todos marked as `in_progress`
2. **After executing task 1**: LLM calls `write_todos` again with:
   - Task 1: `status: "completed"` ← LLM set this
   - Task 2: `status: "in_progress"` ← LLM also updated this
3. **After executing task 2**: LLM calls `write_todos` again with updated statuses
4. **And so on...**

Each status change is an **explicit LLM decision** communicated through `write_todos`.

## Summary

| Question | Answer |
|----------|--------|
| **Who sets completed status?** | The LLM |
| **How?** | By calling `write_todos` with updated todo list |
| **When?** | After the LLM determines a task is complete |
| **Automatic?** | No - entirely explicit and LLM-driven |
| **Can middleware auto-complete?** | No - middleware only provides the tool, doesn't change statuses |
| **Can tools auto-complete?** | No - tools don't have access to modify todos directly |

The completed status is a **conscious decision** made by the LLM based on:
- Instructions in system prompts
- Tool descriptions
- Its assessment of task completion
- The conversation history showing current todo states

