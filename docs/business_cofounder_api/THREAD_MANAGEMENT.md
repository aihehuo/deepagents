# Thread Management and Separation in Multi-Agent Architecture

## Overview

When multiple agents are running (each potentially with different API endpoints), threads are managed and separated through a combination of:

1. **Thread ID Generation** - Unique identifiers for each conversation
2. **Checkpointer Isolation** - Separate checkpoint files per agent
3. **State Isolation** - Per-thread state management via LangGraph
4. **Concurrency Control** - Locks to prevent race conditions

---

## 1. Thread ID Generation

### Format
Thread IDs are generated using a consistent format:
```python
thread_id = f"bc::{user_id}::{conversation_id}"
```

**Example:**
- User `user123` with conversation `conv1` → `"bc::user123::conv1"`
- User `user456` with conversation `conv2` → `"bc::user456::conv2"`

### Uniqueness
- **Same user, different conversations** → Different thread IDs (isolated)
- **Different users, same conversation ID** → Different thread IDs (isolated)
- **Same user, same conversation** → Same thread ID (shared state)

### Location
- Defined in: `apps/business_cofounder_api/app/utils.py::thread_id()`
- Used in: All endpoints (`chat`, `chat_stream`, `deep_agent/call_async`, etc.)

---

## 2. Checkpointer Isolation

### Separate Checkpoint Files

Each agent type has its own checkpoint file:

#### Facilitator Agent (Frontend)
- **Checkpoint file**: `~/.deepagents/business_cofounder_api/facilitator_checkpoints.pkl`
- **Thread IDs stored**: All facilitator conversation threads
- **Storage structure**: `{thread_id: {namespace: {checkpoint_data}}}`

#### Expert Agent (Backend Analysis)
- **Checkpoint file**: `~/.deepagents/business_cofounder_api/expert_checkpoints.pkl`
- **Thread IDs stored**: Expert analysis threads (prefixed with `expert_analysis_`)
- **Storage structure**: `{thread_id: {namespace: {checkpoint_data}}}`

#### Legacy Single-Agent Mode
- **Checkpoint file**: `~/.deepagents/business_cofounder_api/business_cofounder_checkpoints.pkl`
- **Thread IDs stored**: All conversation threads

### Checkpoint Storage Structure

```python
# DiskBackedInMemorySaver storage structure
{
    "storage": {
        "bc::user123::conv1": {
            "checkpoint": {
                "v1": {...checkpoint_data...},
                "v2": {...checkpoint_data...}
            }
        },
        "bc::user456::conv2": {
            "checkpoint": {
                "v1": {...checkpoint_data...}
            }
        }
    },
    "writes": {...},
    "blobs": {...}
}
```

### Thread Isolation in Checkpoints

- **Each thread_id has its own namespace** in the checkpoint storage
- **No cross-thread contamination** - threads cannot access each other's state
- **Persistent across restarts** - checkpoints are saved to disk

### Implementation
- **Class**: `DiskBackedInMemorySaver` (extends `InMemorySaver`)
- **Location**: `apps/business_cofounder_api/checkpointer.py`
- **Persistence**: Atomic writes to pickle file

---

## 3. State Isolation via LangGraph

### Per-Thread State

LangGraph maintains separate state for each `thread_id`:

```python
# Each thread has its own state
config = {
    "configurable": {"thread_id": "bc::user123::conv1"}
}

# State includes:
{
    "messages": [...],           # Conversation history
    "canvas": {...},             # Business Model Canvas data
    "expert_guidance": "...",    # Expert guidance
    "detected_language": "en",   # Language detection
    "todos": [...],              # Task list
    "files": {...},              # File system state
    # ... other middleware state
}
```

### State Isolation Guarantees

1. **Separate state per thread_id**
   - Thread `bc::user123::conv1` has completely separate state from `bc::user456::conv2`
   - No shared variables or cross-contamination

2. **State persistence**
   - State is checkpointed after each agent step
   - Resumes from checkpoint when same `thread_id` is used again

3. **Middleware state isolation**
   - Each middleware (LanguageDetection, BusinessIdeaTracker, etc.) maintains per-thread state
   - State fields are merged into a single TypedDict per thread

### Expert Agent Thread Separation

The expert agent uses a **separate thread ID** to keep analysis isolated:

```python
# Expert agent uses prefixed thread ID
expert_thread_id = f"expert_analysis_{thread_id}"
# Example: "expert_analysis_bc::user123::conv1"
```

This ensures:
- Expert analysis doesn't interfere with facilitator conversations
- Expert can maintain its own analysis history
- Separate checkpoint namespace for expert

---

## 4. Concurrency Control

### Thread-Level Locks

To prevent race conditions when the same `thread_id` receives concurrent requests:

```python
# In AppState
thread_locks: dict[str, asyncio.Lock]  # One lock per thread_id

# Usage in endpoints
lock = state.thread_locks.get(tid)
if lock is None:
    lock = asyncio.Lock()
    state.thread_locks[tid] = lock

async with lock:
    # Process request for this thread_id
    # Only one request per thread_id at a time
```

### Lock Behavior

- **Same thread_id**: Requests are serialized (one at a time)
- **Different thread_ids**: Requests can run concurrently
- **Lock scope**: Per-thread, not global

### Example Scenario

```
Request 1: user123, conv1 → thread_id="bc::user123::conv1" → Lock A
Request 2: user123, conv1 → thread_id="bc::user123::conv1" → Lock A (waits)
Request 3: user456, conv2 → thread_id="bc::user456::conv2" → Lock B (runs concurrently)

Result:
- Request 1 and Request 3 can run at the same time (different locks)
- Request 2 waits for Request 1 to complete (same lock)
```

---

## 5. Multiple API Endpoints / Multiple Agents

### Scenario: Multiple Agents Running

If you have multiple agents running (e.g., different API instances):

#### Option A: Shared Checkpoint File (Same Process)
- **Same checkpoint file** → All agents share the same thread namespace
- **Thread IDs must be unique** across all agents
- **Risk**: If two agents use the same `user_id + conversation_id`, they'll share state

#### Option B: Separate Checkpoint Files (Different Processes)
- **Different checkpoint files** → Complete isolation
- **Same thread IDs are safe** → Each agent has its own checkpoint file
- **Recommended**: Use different checkpoint paths per agent instance

### Recommended Setup for Multiple Agents

```python
# Agent 1 (Instance A)
checkpoints_path = "~/.deepagents/business_cofounder_api/agent_a_checkpoints.pkl"

# Agent 2 (Instance B)
checkpoints_path = "~/.deepagents/business_cofounder_api/agent_b_checkpoints.pkl"

# Agent 3 (Instance C)
checkpoints_path = "~/.deepagents/business_cofounder_api/agent_c_checkpoints.pkl"
```

### Thread ID Collision Prevention

If you want to ensure no collisions across multiple agents:

```python
# Option 1: Prefix thread IDs with agent identifier
thread_id = f"agent_a::bc::{user_id}::{conversation_id}"

# Option 2: Include agent ID in checkpoint path
checkpoints_path = f"~/.deepagents/business_cofounder_api/agent_{agent_id}_checkpoints.pkl"
```

---

## 6. File System Isolation

### Virtual File System

Each agent uses a **virtual filesystem** with `virtual_mode=True`:

```python
backend = FilesystemBackend(
    root_dir=str(base_dir),
    virtual_mode=True  # All paths resolved relative to root_dir
)
```

### File Isolation

- **All file operations** are scoped to `backend_root` directory
- **Per-thread file state** stored in agent state (ephemeral)
- **Persistent files** stored in `backend_root/docs/` (shared across threads)

### Backend Root Structure

```
~/.deepagents/business_cofounder_api/
├── facilitator_checkpoints.pkl      # Facilitator agent checkpoints
├── expert_checkpoints.pkl           # Expert agent checkpoints
├── docs/                            # Shared document storage
│   ├── file1.md
│   └── file2.md
├── skills/                          # Shared skills
├── expertise/                       # Shared expertise templates
└── users/                           # Per-user memory
    └── {user_id}/
        └── conversations/
            └── {conversation_id}/
                └── agent.md
```

---

## 7. Memory Isolation

### User-Level Memory

Memory is organized by user and conversation:

```python
# User-level memory
~/.deepagents/business_cofounder_api/users/{user_id}/agent.md

# Conversation-level memory
~/.deepagents/business_cofounder_api/users/{user_id}/conversations/{conversation_id}/agent.md
```

### Isolation Guarantees

- **Different users**: Completely isolated memory files
- **Same user, different conversations**: Separate memory files
- **Same user, same conversation**: Shared memory file

---

## 8. Summary: Thread Separation Guarantees

### ✅ What IS Isolated

1. **State per thread_id** - Each conversation has separate state
2. **Checkpoints per thread_id** - Separate checkpoint data
3. **Memory per user/conversation** - Separate memory files
4. **File operations** - Scoped to backend_root (with virtual_mode)
5. **Concurrent requests** - Serialized per thread_id via locks

### ⚠️ What is NOT Isolated (Shared)

1. **Checkpoint file** - All threads in same agent share the same checkpoint file
2. **Backend root directory** - Files in `docs/` are shared across threads
3. **Skills directory** - Shared across all threads
4. **Expertise templates** - Shared across all threads
5. **Agent instance** - Same agent instance handles all threads

### 🔒 Isolation Strategy for Multiple Agents

To run multiple agents with complete isolation:

1. **Use separate checkpoint files** per agent instance
2. **Use separate backend_root** directories (optional, for file isolation)
3. **Prefix thread IDs** with agent identifier (optional, for extra safety)
4. **Run in separate processes** (recommended for production)

---

## 9. Example: Multiple Agents Running

### Setup

```python
# Agent Instance A (Port 8000)
agent_a = create_facilitator_agent(...)
checkpoints_a = "~/.deepagents/agent_a/facilitator_checkpoints.pkl"

# Agent Instance B (Port 8001)
agent_b = create_facilitator_agent(...)
checkpoints_b = "~/.deepagents/agent_b/facilitator_checkpoints.pkl"
```

### Request Flow

```
Request to Agent A:
  user_id="user1", conversation_id="conv1"
  → thread_id="bc::user1::conv1"
  → Uses checkpoints_a
  → State stored in agent_a's checkpoint file

Request to Agent B:
  user_id="user1", conversation_id="conv1"  # Same user/conv!
  → thread_id="bc::user1::conv1"  # Same thread_id!
  → Uses checkpoints_b
  → State stored in agent_b's checkpoint file (separate!)
```

**Result**: Even with the same `user_id` and `conversation_id`, the two agents maintain completely separate state because they use different checkpoint files.

---

## 10. Best Practices

1. **Use unique checkpoint files** per agent instance
2. **Use thread-level locks** to prevent race conditions
3. **Monitor thread_locks dictionary** size (clean up unused locks if needed)
4. **Use separate backend_root** if you want file system isolation
5. **Prefix thread IDs** if you want extra collision protection
6. **Run agents in separate processes** for production deployments

---

## References

- Thread ID generation: `apps/business_cofounder_api/app/utils.py::thread_id()`
- Checkpointer: `apps/business_cofounder_api/checkpointer.py`
- State management: `apps/business_cofounder_api/app/state.py`
- Lock management: `apps/business_cofounder_api/app/endpoints/chat.py`
- LangGraph state: `docs/AGENT_STATE_LIFECYCLE.md`
