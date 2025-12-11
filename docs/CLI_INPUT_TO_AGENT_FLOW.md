# CLI Input to Agent Flow

This document traces how user input from the command line is passed to the agent as a message.

---

## Complete Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. User Types Input in Terminal                            │
│    "Create a Python script to calculate fibonacci"        │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. CLI Loop Reads Input (main.py)                          │
│                                                              │
│    user_input = await session.prompt_async()                │
│                                                              │
│    Location: libs/deepagents-cli/deepagents_cli/main.py    │
│    Line: 231                                                 │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Input Processing (main.py)                               │
│                                                              │
│    - Strip whitespace                                       │
│    - Check for slash commands (/help, /clear, etc.)        │
│    - Check for bash commands (!command)                     │
│    - Check for quit keywords                                │
│                                                              │
│    Location: libs/deepagents-cli/deepagents_cli/main.py    │
│    Lines: 236-264                                            │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Call execute_task() (main.py)                           │
│                                                              │
│    await execute_task(                                       │
│        user_input, agent, assistant_id,                     │
│        session_state, token_tracker, backend=backend       │
│    )                                                         │
│                                                              │
│    Location: libs/deepagents-cli/deepagents_cli/main.py   │
│    Line: 266-268                                             │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Process Input in execute_task() (execution.py)          │
│                                                              │
│    - Parse file mentions (e.g., @file.py)                  │
│    - Inject file content if mentioned                       │
│    - Build final_input string                               │
│                                                              │
│    Location: libs/deepagents-cli/deepagents_cli/execution.py│
│    Lines: 180-208                                            │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. Create Message Dictionary (execution.py)               │
│                                                              │
│    stream_input = {                                         │
│        "messages": [                                         │
│            {"role": "user", "content": final_input}         │
│        ]                                                     │
│    }                                                         │
│                                                              │
│    Location: libs/deepagents-cli/deepagents_cli/execution.py│
│    Line: 264                                                 │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. Invoke Agent with Message (execution.py)                │
│                                                              │
│    async for chunk in agent.astream(                        │
│        stream_input,                                        │
│        stream_mode=["messages", "updates"],                 │
│        subgraphs=True,                                       │
│        config=config,                                       │
│        durability="exit",                                   │
│    ):                                                       │
│        # Process agent response...                         │
│                                                              │
│    Location: libs/deepagents-cli/deepagents_cli/execution.py│
│    Lines: 274-280                                            │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Code Locations

### 1. Input Reading Loop

**File**: `libs/deepagents-cli/deepagents_cli/main.py`

```229:268:libs/deepagents-cli/deepagents_cli/main.py
    while True:
        try:
            user_input = await session.prompt_async()
            if session_state.exit_hint_handle:
                session_state.exit_hint_handle.cancel()
                session_state.exit_hint_handle = None
            session_state.exit_hint_until = None
            user_input = user_input.strip()
        except EOFError:
            break
        except KeyboardInterrupt:
            console.print("\nGoodbye!", style=COLORS["primary"])
            break

        if not user_input:
            continue

        # Check for slash commands first
        if user_input.startswith("/"):
            result = handle_command(user_input, agent, token_tracker)
            if result == "exit":
                console.print("\nGoodbye!", style=COLORS["primary"])
                break
            if result:
                # Command was handled, continue to next input
                continue

        # Check for bash commands (!)
        if user_input.startswith("!"):
            execute_bash_command(user_input)
            continue

        # Handle regular quit keywords
        if user_input.lower() in ["quit", "exit", "q"]:
            console.print("\nGoodbye!", style=COLORS["primary"])
            break

        await execute_task(
            user_input, agent, assistant_id, session_state, token_tracker, backend=backend
        )
```

**Key Points**:
- Line 231: `user_input = await session.prompt_async()` - Reads user input
- Line 236: Strips whitespace
- Lines 247-259: Handles special commands (`/`, `!`)
- Line 266-268: Calls `execute_task()` with the user input

---

### 2. Input Processing and Message Creation

**File**: `libs/deepagents-cli/deepagents_cli/execution.py`

```180:264:libs/deepagents-cli/deepagents_cli/execution.py
async def execute_task(
    user_input: str,
    agent,
    assistant_id: str | None,
    session_state,
    token_tracker: TokenTracker | None = None,
    backend=None,
) -> None:
    """Execute any task by passing it directly to the AI agent."""
    # Parse file mentions and inject content if any
    prompt_text, mentioned_files = parse_file_mentions(user_input)

    if mentioned_files:
        context_parts = [prompt_text, "\n\n## Referenced Files\n"]
        for file_path in mentioned_files:
            try:
                content = file_path.read_text()
                # Limit file content to reasonable size
                if len(content) > 50000:
                    content = content[:50000] + "\n... (file truncated)"
                context_parts.append(
                    f"\n### {file_path.name}\nPath: `{file_path}`\n```\n{content}\n```"
                )
            except Exception as e:
                context_parts.append(f"\n### {file_path.name}\n[Error reading file: {e}]")

        final_input = "\n".join(context_parts)
    else:
        final_input = prompt_text

    config = {
        "configurable": {"thread_id": session_state.thread_id},
        "metadata": {"assistant_id": assistant_id} if assistant_id else {},
    }

    has_responded = False
    captured_input_tokens = 0
    captured_output_tokens = 0
    current_todos = None  # Track current todo list state

    status = console.status(f"[bold {COLORS['thinking']}]Agent is thinking...", spinner="dots")
    status.start()
    spinner_active = True

    tool_icons = {
        "read_file": "📖",
        "write_file": "✏️",
        "edit_file": "✂️",
        "ls": "📁",
        "glob": "🔍",
        "grep": "🔎",
        "shell": "⚡",
        "execute": "🔧",
        "web_search": "🌐",
        "http_request": "🌍",
        "task": "🤖",
        "write_todos": "📋",
    }

    file_op_tracker = FileOpTracker(assistant_id=assistant_id, backend=backend)

    # Track which tool calls we've displayed to avoid duplicates
    displayed_tool_ids = set()
    # Buffer partial tool-call chunks keyed by streaming index
    tool_call_buffers: dict[str | int, dict] = {}
    # Buffer assistant text so we can render complete markdown segments
    pending_text = ""

    def flush_text_buffer(*, final: bool = False) -> None:
        """Flush accumulated assistant text as rendered markdown when appropriate."""
        nonlocal pending_text, spinner_active, has_responded
        if not final or not pending_text.strip():
            return
        if spinner_active:
            status.stop()
            spinner_active = False
        if not has_responded:
            console.print("●", style=COLORS["agent"], markup=False, end=" ")
            has_responded = True
        markdown = Markdown(pending_text.rstrip())
        console.print(markdown, style=COLORS["agent"])
        pending_text = ""

    # Stream input - may need to loop if there are interrupts
    stream_input = {"messages": [{"role": "user", "content": final_input}]}
```

**Key Points**:
- Line 190: Parses file mentions (e.g., `@file.py`)
- Lines 192-208: Injects file content if files are mentioned
- Line 210-213: Creates config with thread_id and assistant_id
- **Line 264**: Creates the message dictionary: `{"messages": [{"role": "user", "content": final_input}]}`

---

### 3. Agent Invocation

**File**: `libs/deepagents-cli/deepagents_cli/execution.py`

```266:280:libs/deepagents-cli/deepagents_cli/execution.py
    try:
        while True:
            interrupt_occurred = False
            hitl_response: dict[str, HITLResponse] = {}
            suppress_resumed_output = False
            # Track all pending interrupts: {interrupt_id: request_data}
            pending_interrupts: dict[str, HITLRequest] = {}

            async for chunk in agent.astream(
                stream_input,
                stream_mode=["messages", "updates"],  # Dual-mode for HITL support
                subgraphs=True,
                config=config,
                durability="exit",
            ):
```

**Key Points**:
- **Line 274**: `agent.astream(stream_input, ...)` - This is where the message is passed to the agent
- `stream_input` contains: `{"messages": [{"role": "user", "content": final_input}]}`
- The agent processes this as a `HumanMessage` with the user's input

---

## Message Format

The message passed to the agent follows this structure:

```python
{
    "messages": [
        {
            "role": "user",
            "content": "Create a Python script to calculate fibonacci"
        }
    ]
}
```

If file mentions are included, the content is expanded:

```python
{
    "messages": [
        {
            "role": "user",
            "content": """Create a Python script to calculate fibonacci

## Referenced Files

### example.py
Path: `example.py`
```
def hello():
    print("world")
```
"""
        }
    ]
}
```

---

## Configuration

The `config` dictionary includes:

```python
config = {
    "configurable": {
        "thread_id": session_state.thread_id  # Conversation thread ID
    },
    "metadata": {
        "assistant_id": assistant_id  # Agent identifier for memory
    }
}
```

This ensures:
- Messages are associated with the correct conversation thread
- Agent memory is scoped to the correct assistant_id

---

## Summary

**Flow**:
1. **Input**: `session.prompt_async()` reads user input (main.py:231)
2. **Processing**: Input is stripped and checked for special commands (main.py:236-264)
3. **Execution**: `execute_task()` is called with user input (main.py:266-268)
4. **Enhancement**: File mentions are parsed and content injected (execution.py:190-208)
5. **Message Creation**: Message dictionary is created (execution.py:264)
6. **Agent Invocation**: `agent.astream()` is called with the message (execution.py:274)

**Key Line**: `execution.py:264` - Where the message dictionary is created
**Key Line**: `execution.py:274` - Where the agent is invoked with the message

The agent receives the input as a `HumanMessage` in the standard LangGraph/LangChain message format.

