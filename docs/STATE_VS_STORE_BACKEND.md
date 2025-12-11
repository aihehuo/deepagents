# StateBackend vs StoreBackend: Key Differences

## Overview

Both `StateBackend` and `StoreBackend` store files in memory-like structures, but they differ in **persistence**, **scope**, and **storage mechanism**.

---

## Quick Comparison Table

| Aspect | StateBackend | StoreBackend |
|--------|--------------|--------------|
| **Persistence** | Ephemeral (temporary) | Persistent (permanent) |
| **Scope** | Single conversation thread | All threads/conversations |
| **Storage** | LangGraph agent state | LangGraph BaseStore |
| **Lifespan** | Exists only during agent execution | Persists across agent restarts |
| **Isolation** | Per-thread (checkpoint-based) | Cross-thread (shared) |
| **State Updates** | Returns `files_update` in WriteResult/EditResult | Returns `files_update=None` |
| **Use Case** | Working files, temporary data | Long-term memory, shared knowledge |

---

## Detailed Differences

### 1. **Persistence Model**

#### StateBackend (Ephemeral)
```python
class StateBackend(BackendProtocol):
    """Backend that stores files in agent state (ephemeral).
    
    Uses LangGraph's state management and checkpointing. Files persist within
    a conversation thread but not across threads. State is automatically
    checkpointed after each agent step.
    """
```

- **Stored in**: `runtime.state["files"]` (in-memory dictionary)
- **Lifespan**: Only during the conversation thread
- **Checkpointing**: Automatically checkpointed after each agent step
- **Lost when**: Thread ends or agent restarts

#### StoreBackend (Persistent)
```python
class StoreBackend(BackendProtocol):
    """Backend that stores files in LangGraph's BaseStore (persistent).
    
    Uses LangGraph's Store for persistent, cross-conversation storage.
    Files are organized via namespaces and persist across all threads.
    """
```

- **Stored in**: LangGraph `BaseStore` (can be file-based, database, etc.)
- **Lifespan**: Permanent until explicitly deleted
- **Persistence**: Survives agent restarts, thread changes
- **Shared across**: All threads and conversations (within namespace)

---

### 2. **Storage Location**

#### StateBackend
```python
# Files stored in agent state
files = self.runtime.state.get("files", {})
# Structure: {"/path/to/file": {"content": [...], "created_at": "...", "modified_at": "..."}}
```

- Direct access to `runtime.state["files"]`
- Part of the agent's state graph
- Checkpointed with the rest of agent state

#### StoreBackend
```python
# Files stored in BaseStore
store = self.runtime.store
namespace = self._get_namespace()  # e.g., ("filesystem",) or ("assistant_id", "filesystem")
item = store.get(namespace, file_path)
```

- Uses LangGraph's `BaseStore` abstraction
- Supports different store implementations (InMemoryStore, PostgresStore, etc.)
- Organized by namespaces for multi-agent isolation

---

### 3. **State Update Mechanism**

#### StateBackend
```python
def write(self, file_path: str, content: str) -> WriteResult:
    new_file_data = create_file_data(content)
    return WriteResult(
        path=file_path, 
        files_update={file_path: new_file_data}  # ← Returns state update!
    )
```

**Key Point**: Returns `files_update` dictionary that LangGraph uses to update state.

```python
# LangGraph automatically applies the update:
state["files"].update(write_result.files_update)
```

#### StoreBackend
```python
def write(self, file_path: str, content: str) -> WriteResult:
    file_data = create_file_data(content)
    store_value = self._convert_file_data_to_store_value(file_data)
    store.put(namespace, file_path, store_value)  # ← Directly writes to store
    return WriteResult(
        path=file_path, 
        files_update=None  # ← No state update needed!
    )
```

**Key Point**: Returns `files_update=None` because it directly writes to the store.

---

### 4. **Thread/Conversation Scope**

#### StateBackend
- **Scope**: Single conversation thread
- **Isolation**: Each thread has its own state
- **Sharing**: Files in one thread are NOT visible to other threads
- **Example**:
  ```python
  # Thread 1
  agent.invoke({"messages": [HumanMessage("create /app/config.json")]})
  # File exists in Thread 1's state
  
  # Thread 2 (new conversation)
  agent.invoke({"messages": [HumanMessage("read /app/config.json")]})
  # File NOT found - different thread, different state!
  ```

#### StoreBackend
- **Scope**: All threads and conversations
- **Sharing**: Files are shared across all threads (within namespace)
- **Isolation**: Can use namespaces for multi-agent isolation
- **Example**:
  ```python
  # Thread 1
  agent.invoke({"messages": [HumanMessage("create /app/config.json")]})
  # File saved to store
  
  # Thread 2 (new conversation)
  agent.invoke({"messages": [HumanMessage("read /app/config.json")]})
  # File found! Shared across threads
  ```

---

### 5. **Namespace Support**

#### StateBackend
- **No namespace support**
- Files are stored directly in `state["files"]`
- No isolation mechanism beyond thread boundaries

#### StoreBackend
- **Full namespace support**
- Uses hierarchical namespaces: `(assistant_id, "filesystem")`
- Enables multi-agent isolation:
  ```python
  # Assistant A's files
  namespace = ("assistant_a", "filesystem")
  
  # Assistant B's files  
  namespace = ("assistant_b", "filesystem")
  
  # Shared files
  namespace = ("filesystem",)
  ```

---

### 6. **Implementation Details**

#### StateBackend Operations

```python
# Read: Direct state access
files = self.runtime.state.get("files", {})
file_data = files.get(file_path)

# Write: Returns state update
return WriteResult(files_update={file_path: new_file_data})

# Edit: Returns state update
return EditResult(files_update={file_path: updated_file_data})
```

**Pattern**: Read from state, return updates for LangGraph to apply.

#### StoreBackend Operations

```python
# Read: Store query
store = self._get_store()
namespace = self._get_namespace()
item = store.get(namespace, file_path)

# Write: Direct store write
store.put(namespace, file_path, store_value)
return WriteResult(files_update=None)

# Edit: Store update
store.put(namespace, file_path, updated_store_value)
return EditResult(files_update=None)
```

**Pattern**: Direct store operations, no state updates needed.

---

## When to Use Which?

### Use StateBackend When:
- ✅ Working on temporary files during a conversation
- ✅ Files are only needed for the current task
- ✅ You want automatic cleanup when conversation ends
- ✅ Files are conversation-specific (not shared)
- ✅ Default behavior (no configuration needed)

**Example Use Cases**:
- Generated code files during refactoring
- Temporary analysis results
- Intermediate computation files
- Files created during a single conversation

### Use StoreBackend When:
- ✅ Files need to persist across conversations
- ✅ Multiple threads/agents need to share files
- ✅ Building a knowledge base or memory system
- ✅ Long-term storage of important data
- ✅ Multi-agent systems with isolation needs

**Example Use Cases**:
- User preferences and settings
- Knowledge base articles
- Shared project documentation
- Cross-conversation memory
- Agent-specific data (with namespaces)

---

## Hybrid Approach: CompositeBackend

You can use both together with `CompositeBackend`:

```python
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.store import InMemoryStore

backend = CompositeBackend(
    default=StateBackend(runtime),  # Default: ephemeral
    routes={
        "/memories/": StoreBackend(runtime)  # Persistent storage
    }
)

# Files in /memories/ persist across conversations
# Other files are ephemeral
```

**Benefits**:
- Working files → StateBackend (auto-cleanup)
- Important data → StoreBackend (persistent)
- Best of both worlds!

---

## Code Examples

### StateBackend Example

```python
from deepagents import create_deep_agent
from deepagents.backends import StateBackend

# StateBackend is the default, but you can be explicit:
agent = create_deep_agent(
    backend=lambda runtime: StateBackend(runtime)
)

# In a conversation:
result = agent.invoke({
    "messages": [HumanMessage("Create /app/temp.py with print('hello')")]
})
# File exists in this thread's state

# New conversation:
result = agent.invoke({
    "messages": [HumanMessage("Read /app/temp.py")]
})
# File NOT found - different thread!
```

### StoreBackend Example

```python
from deepagents import create_deep_agent
from deepagents.backends import StoreBackend
from langgraph.store import InMemoryStore

# Create agent with persistent storage
agent = create_deep_agent(
    store=InMemoryStore(),  # Or PostgresStore, etc.
    backend=lambda runtime: StoreBackend(runtime)
)

# In conversation 1:
result = agent.invoke({
    "messages": [HumanMessage("Create /app/config.json")]
})
# File saved to store

# In conversation 2 (different thread):
result = agent.invoke({
    "messages": [HumanMessage("Read /app/config.json")]
})
# File found! Persisted across threads
```

---

## Summary

| Question | StateBackend | StoreBackend |
|----------|--------------|--------------|
| **Where stored?** | Agent state (`state["files"]`) | BaseStore |
| **How long?** | During conversation | Forever |
| **Shared?** | No (per-thread) | Yes (cross-thread) |
| **State updates?** | Yes (`files_update` dict) | No (`files_update=None`) |
| **Namespaces?** | No | Yes |
| **Use for?** | Temporary working files | Persistent knowledge/memory |

**Key Insight**: StateBackend is for **working memory**, StoreBackend is for **long-term memory**.

