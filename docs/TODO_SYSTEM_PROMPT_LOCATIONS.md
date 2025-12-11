# System Prompt Locations for Todo List Instructions

## Overview

The instructions for working on todo lists come from **two main sources**:

1. **System Prompt** (injected by `TodoListMiddleware.wrap_model_call()`)
2. **Tool Description** (shown in the LLM's available tools list)

## Location 1: System Prompt from TodoListMiddleware

**Source**: `langchain.agents.middleware.todo.TodoListMiddleware`

**How it's injected**: The middleware's `wrap_model_call()` method appends this to the system message on every LLM call.

**Content**:
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

**Key Instructions**:
- Mark todos as completed as soon as done
- Don't batch completions
- Use for complex multi-step tasks only

## Location 2: Tool Description (Most Detailed Instructions)

**Source**: `langchain.agents.middleware.todo.TodoListMiddleware` - the `write_todos` tool's description

**How it's shown**: This appears in the LLM's tool list when tools are bound to the model.

**Key Section: "Task Management"** (Lines 794-800):

```python
2. **Task Management**:
   - Update task status in real-time as you work
   - Mark tasks complete IMMEDIATELY after finishing (don't batch completions)
   - Complete current tasks before starting new ones  # ← KEY INSTRUCTION
   - Remove tasks that are no longer relevant from the list entirely
   - IMPORTANT: When you write this todo list, you should mark your first task (or tasks) as in_progress immediately!.
   - IMPORTANT: Unless all tasks are completed, you should always have at least one task in_progress to show the user that you are working on something.
```

**Key Section: "How to Use This Tool"** (Lines 774-778):

```python
## How to Use This Tool
1. When you start working on a task - Mark it as in_progress BEFORE beginning work.
2. After completing a task - Mark it as completed and add any new follow-up tasks discovered during implementation.
3. You can also update future tasks, such as deleting them if they are no longer necessary, or adding new tasks that are necessary. Don't change previously completed tasks.
4. You can make several updates to the todo list at once. For example, when you complete a task, you can mark the next task you need to start as in_progress.
```

## Answer to Your Question: "Work on First In-Progress Task"

**Important Finding**: There is **NO explicit instruction** that says "work on the first in_progress task" or "pick the first in_progress task".

Instead, the LLM infers this from:

1. **"Complete current tasks before starting new ones"** (Task Management section)
   - This implies: if there's a task marked `in_progress`, complete it before starting pending tasks

2. **The ToolMessage in conversation history**
   - The LLM sees the todo list in the ToolMessage: `"Updated todo list to [{'content': 'Analyze...', 'status': 'in_progress'}, ...]"`
   - The LLM can see which tasks are `in_progress` and naturally works on them

3. **General task management guidance**
   - "Update task status in real-time as you work"
   - "Mark tasks complete IMMEDIATELY after finishing"
   - These encourage working on in_progress tasks

## How to Access These Prompts in Code

### System Prompt
```python
from langchain.agents.middleware.todo import TodoListMiddleware

middleware = TodoListMiddleware()
print(middleware.system_prompt)  # Shows WRITE_TODOS_SYSTEM_PROMPT
```

### Tool Description
```python
from langchain.agents.middleware.todo import TodoListMiddleware

middleware = TodoListMiddleware()
tool = next(tool for tool in middleware.tools if tool.name == "write_todos")
print(tool.description)  # Shows WRITE_TODOS_TOOL_DESCRIPTION (3654 chars)
```

## Additional DeepAgents-Specific Instructions

**Location**: `libs/deepagents-cli/deepagents_cli/agent.py` - `get_system_prompt()` function

**Content** (Lines 173-186):
```python
### Todo List Management

When using the write_todos tool:
1. Keep the todo list MINIMAL - aim for 3-6 items maximum
2. Only create todos for complex, multi-step tasks that truly need tracking
3. Break down work into clear, actionable items without over-fragmenting
4. For simple tasks (1-2 steps), just do them directly without creating todos
5. When first creating a todo list for a task, ALWAYS ask the user if the plan looks good before starting work
   - Create the todos, let them render, then ask: "Does this plan look good?" or similar
   - Wait for the user's response before marking the first todo as in_progress
   - If they want changes, adjust the plan accordingly
6. Update todo status promptly as you complete each item

The todo list is a planning tool - use it judiciously to avoid overwhelming the user with excessive task tracking.
```

**Note**: This is only for the CLI agent (`deepagents-cli`), not the core `deepagents` library.

## Summary

**The instruction to work on in_progress tasks is implicit, not explicit:**

1. ✅ **Explicit**: "Complete current tasks before starting new ones"
2. ✅ **Explicit**: "Mark it as in_progress BEFORE beginning work"
3. ✅ **Explicit**: "Mark tasks complete IMMEDIATELY after finishing"
4. ❌ **NOT Explicit**: "Work on the first in_progress task"

The LLM infers from the conversation history (ToolMessage showing todos with statuses) and the general guidance to work on tasks that are marked `in_progress` before starting new `pending` tasks.

## Where to Find the Implementation

1. **LangChain Source**: `langchain/agents/middleware/todo.py`
   - Contains `TodoListMiddleware` class
   - Contains `WRITE_TODOS_SYSTEM_PROMPT` constant
   - Contains `write_todos` tool with `WRITE_TODOS_TOOL_DESCRIPTION`

2. **DeepAgents Integration**: `libs/deepagents/deepagents/graph.py`
   - Line 114: `TodoListMiddleware()` is included by default

3. **DeepAgents CLI Custom Prompt**: `libs/deepagents-cli/deepagents_cli/agent.py`
   - Lines 173-186: Additional todo management instructions for CLI

