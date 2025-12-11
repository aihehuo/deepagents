# Tool Call Execution Flow

## Overview

**Important**: `SkillsMiddleware` does NOT execute tool calls. It only:
- Loads skills metadata from SKILL.md files
- Injects skills list into the system prompt
- Provides information about available skills

## Tool Call Execution Flow

### 1. LLM Returns Tool Calls

When the LLM decides to use a tool, it returns an `AIMessage` with `tool_calls`:

```python
AIMessage(
    content="...",
    tool_calls=[
        {
            "name": "execute",
            "args": {"command": "python3 /path/to/script.py 'query'"},
            "id": "call_123"
        }
    ]
)
```

### 2. LangChain Agent Framework Parses Tool Calls

**Location**: `langchain/agents/factory.py` (in LangChain library, not in this codebase)

The agent framework:
- Extracts `tool_calls` from the `AIMessage`
- Creates a `ToolCallRequest` for each tool call
- Routes to the appropriate tool node

### 3. Middleware Intercepts Tool Calls

**Location**: `libs/deepagents/deepagents/middleware/filesystem.py`

The `FilesystemMiddleware.wrap_tool_call()` method intercepts tool calls:

```python
def wrap_tool_call(
    self,
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    """Intercepts tool calls before execution."""
    # Can modify the request or result here
    tool_result = handler(request)  # This executes the tool
    return self._intercept_large_tool_result(tool_result, request.runtime)
```

### 4. Tool Execution

**For `execute` tool**: `libs/deepagents/deepagents/middleware/filesystem.py:683-714`

```python
def sync_execute(
    command: str,
    runtime: ToolRuntime[None, FilesystemState],
) -> str:
    """Synchronous wrapper for execute tool."""
    resolved_backend = _get_backend(backend, runtime)
    
    # Check if backend supports execution
    if not _supports_execution(resolved_backend):
        return "Error: Execution not available..."
    
    # Execute the command via backend
    result = resolved_backend.execute(command)  # ← ACTUAL EXECUTION HAPPENS HERE
    
    # Format output
    return format_output(result)
```

**For other tools** (read_file, write_file, etc.): Similar pattern in `_ls_tool_generator`, `_read_file_tool_generator`, etc.

### 5. Backend Execution

**Location**: Backend implementations (e.g., `libs/deepagents/deepagents/backends/`)

The backend's `execute()` method actually runs the command:

```python
# In a SandboxBackend implementation
def execute(self, command: str) -> ExecuteResponse:
    """Execute a shell command."""
    # Actually run the command (e.g., via subprocess, Docker, etc.)
    result = subprocess.run(command, shell=True, capture_output=True)
    return ExecuteResponse(
        output=result.stdout,
        exit_code=result.returncode,
        truncated=False
    )
```

## Key Files

### SkillsMiddleware (Does NOT Execute Tools)
- **File**: `libs/deepagents-cli/deepagents_cli/skills/middleware.py`
- **Purpose**: Load skills metadata, inject into system prompt
- **Methods**: `before_agent()`, `wrap_model_call()`
- **Does NOT have**: `wrap_tool_call()` - it doesn't intercept tool execution

### FilesystemMiddleware (Provides Execute Tool)
- **File**: `libs/deepagents/deepagents/middleware/filesystem.py`
- **Tool Definition**: `_execute_tool_generator()` (line 668)
- **Tool Execution**: `sync_execute()` (line 683) and `async_execute()` (line 716)
- **Interception**: `wrap_tool_call()` (line 1050) - can intercept and modify results

### Tool Call Request Structure
- **Type**: `ToolCallRequest` from `langchain.tools.tool_node`
- **Contains**: 
  - `tool_call`: The tool call dict with name, args, id
  - `runtime`: ToolRuntime with state access
  - `handler`: Function to call to execute the tool

## Execution Flow Diagram

```
LLM Response (AIMessage with tool_calls)
    ↓
LangChain Agent Framework (factory.py)
    ↓
Extract tool_calls → Create ToolCallRequest
    ↓
FilesystemMiddleware.wrap_tool_call()  ← Intercepts here
    ↓
Tool's invoke() method (e.g., sync_execute)
    ↓
Backend.execute(command)  ← ACTUAL EXECUTION
    ↓
Format result → ToolMessage
    ↓
Return to agent framework
    ↓
Add ToolMessage to state["messages"]
    ↓
Next model call sees the result
```

## Example: Executing aihehuo-member-search Script

1. **LLM decides to execute**: Returns `AIMessage` with tool_call:
   ```python
   tool_calls=[{
       "name": "execute",
       "args": {"command": "python3 /path/to/aihehuo_member_search.py 'search query'"},
       "id": "call_456"
   }]
   ```

2. **LangChain routes to execute tool**: Creates `ToolCallRequest`

3. **FilesystemMiddleware intercepts**: `wrap_tool_call()` is called

4. **Tool executes**: `sync_execute()` is called:
   ```python
   resolved_backend = _get_backend(backend, runtime)
   result = resolved_backend.execute("python3 /path/to/aihehuo_member_search.py 'search query'")
   ```

5. **Backend runs command**: Actually executes the Python script via subprocess/Docker/etc.

6. **Result returned**: Formatted as `ToolMessage` and added to conversation

## Summary

- **SkillsMiddleware**: Only provides skills metadata, does NOT execute tools
- **Tool execution**: Handled by LangChain agent framework + FilesystemMiddleware
- **Execute tool**: Defined in `FilesystemMiddleware._execute_tool_generator()`
- **Actual execution**: Happens in backend's `execute()` method
- **Interception point**: `FilesystemMiddleware.wrap_tool_call()` can modify tool calls/results

