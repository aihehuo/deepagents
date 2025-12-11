# How the LLM Decides Between Temporary and Persistent Storage

## The Key Insight

**The LLM doesn't directly choose between StateBackend and StoreBackend.** Instead:
1. The LLM chooses **file paths**
2. `CompositeBackend` routes paths to different backends based on **path prefixes**
3. The **system prompt** guides the LLM on which paths to use

---

## How It Works: Path-Based Routing

### The Routing Mechanism

```python
# CompositeBackend routes based on path prefixes
backend = CompositeBackend(
    default=StateBackend(runtime),      # Default: ephemeral
    routes={
        "/memories/": StoreBackend(runtime)  # Persistent storage
    }
)

# When LLM writes to /memories/notes.md:
# → CompositeBackend sees "/memories/" prefix
# → Routes to StoreBackend
# → File persists across conversations

# When LLM writes to /app/temp.py:
# → No matching prefix
# → Routes to default (StateBackend)
# → File is ephemeral
```

### The Routing Logic

```34:53:libs/deepagents/deepagents/backends/composite.py
    def _get_backend_and_key(self, key: str) -> tuple[BackendProtocol, str]:
        """Determine which backend handles this key and strip prefix.

        Args:
            key: Original file path

        Returns:
            Tuple of (backend, stripped_key) where stripped_key has the route
            prefix removed (but keeps leading slash).
        """
        # Check routes in order of length (longest first)
        for prefix, backend in self.sorted_routes:
            if key.startswith(prefix):
                # Strip full prefix and ensure a leading slash remains
                # e.g., "/memories/notes.txt" → "/notes.txt"; "/memories/" → "/"
                suffix = key[len(prefix) :]
                stripped_key = f"/{suffix}" if suffix else "/"
                return backend, stripped_key

        return self.default, key
```

**Key Point**: The LLM just uses file paths. The routing is automatic based on path prefixes.

---

## How the System Prompt Guides the LLM

### 1. Explicit Path Instructions

The system prompt explicitly tells the LLM which paths to use:

```6:23:libs/deepagents-cli/deepagents_cli/default_agent_prompt.md
## Memory-First Protocol
You have access to a persistent memory system. ALWAYS follow this protocol:

**At session start:**
- Check `ls /memories/` to see what knowledge you have stored
- If your role description references specific topics, check /memories/ for relevant guides

**Before answering questions:**
- If asked "what do you know about X?" or "how do I do Y?" → Check `ls /memories/` FIRST
- If relevant memory files exist → Read them and base your answer on saved knowledge
- Prefer saved knowledge over general knowledge when available

**When learning new information:**
- If user teaches you something or asks you to remember → Save to `/memories/[topic].md`
- Use descriptive filenames: `/memories/deep-agents-guide.md` not `/memories/notes.md`
- After saving, verify by reading back the key points

**Important:** Your memories persist across sessions. Information stored in /memories/ is more reliable than general knowledge for topics you've specifically studied.
```

### 2. Clear Semantic Guidance

The prompt provides semantic guidance on **what belongs where**:

**For Persistent Storage (`/memories/`)**:
- "Your memories persist across sessions"
- "Information stored in /memories/ is more reliable"
- "When user teaches you something or asks you to remember"

**For Temporary Storage (other paths)**:
- Working files go to default paths (e.g., `/app/`, current directory)
- No explicit mention of persistence
- Implicitly temporary

---

## Decision Framework for the LLM

Based on the system prompt, the LLM uses this decision framework:

### Use `/memories/` (Persistent) When:
- ✅ User asks to remember something
- ✅ Learning new information that should persist
- ✅ Building a knowledge base
- ✅ Storing preferences or instructions
- ✅ Information that will be useful in future sessions

**Examples**:
- `/memories/deep-agents-guide.md` - Guide learned from user
- `/memories/user-preferences.md` - User's coding style preferences
- `/memories/project-conventions.md` - Project-specific conventions

### Use Default Paths (Temporary) When:
- ✅ Creating working files for current task
- ✅ Generated code or temporary analysis
- ✅ Files only needed for this conversation
- ✅ Intermediate computation results

**Examples**:
- `/app/refactored_code.py` - Code being worked on
- `/app/analysis_results.txt` - Temporary analysis
- `/app/temp_data.json` - Intermediate data

---

## Complete Flow Example

```
┌─────────────────────────────────────────────────────────────┐
│ User: "Remember that I prefer functional programming"      │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ LLM Reasoning (based on system prompt):                      │
│                                                              │
│ "User wants me to remember something →                      │
│  System prompt says: 'Save to /memories/[topic].md'         │
│  This should persist across sessions"                        │
│                                                              │
│ Decision: Use /memories/ path                               │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ LLM Action:                                                  │
│                                                              │
│ write_file("/memories/coding-preferences.md",                │
│            "User prefers functional programming...")         │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ CompositeBackend Routing:                                    │
│                                                              │
│ 1. Receives path: "/memories/coding-preferences.md"          │
│ 2. Checks routes: Does it start with "/memories/"?          │
│ 3. YES → Routes to StoreBackend                              │
│ 4. StoreBackend writes to persistent store                   │
│ 5. File persists across conversations                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Configuration: Setting Up the Routes

### Default Setup (SDK)

By default, `create_deep_agent` doesn't set up `/memories/` routing. You need to configure it:

```python
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.store import InMemoryStore

agent = create_deep_agent(
    backend=lambda runtime: CompositeBackend(
        default=StateBackend(runtime),
        routes={
            "/memories/": StoreBackend(runtime)  # ← Configure the route
        }
    ),
    store=InMemoryStore()  # ← Required for StoreBackend
)
```

### CLI Setup

The CLI doesn't automatically set up `/memories/` routing either. The system prompt mentions `/memories/`, but if the route isn't configured, those files would go to the default backend (FilesystemBackend in local mode).

**Note**: This is a potential gap - the prompt tells the LLM to use `/memories/`, but the route might not be configured!

---

## Best Practices for Guiding the LLM

### 1. **Explicit Path Conventions**

Define clear path conventions in the system prompt:

```
## File Storage Guidelines

**Persistent Storage (`/memories/`):**
- Use for information that should persist across sessions
- Knowledge bases, preferences, learned patterns
- Example: `/memories/user-preferences.md`

**Temporary Storage (default paths):**
- Use for working files during current task
- Generated code, temporary analysis, intermediate results
- Example: `/app/refactored_code.py`
```

### 2. **Semantic Naming**

Guide the LLM with semantic path patterns:

```
**Path Patterns:**
- `/memories/*` → Persistent knowledge
- `/archive/*` → Historical records (if configured)
- `/cache/*` → Temporary cached data (if configured)
- `/app/*` → Working files (temporary)
- `/*` → Default temporary storage
```

### 3. **Examples in System Prompt**

Include concrete examples:

```
**Example: Storing User Preferences**
User: "I prefer TypeScript over JavaScript"
→ Save to: `/memories/coding-preferences.md`

**Example: Working File**
User: "Refactor this code"
→ Create: `/app/refactored_code.ts` (temporary)
```

### 4. **Route Configuration**

Always configure routes to match system prompt instructions:

```python
# If system prompt mentions /memories/, configure it:
backend = CompositeBackend(
    default=StateBackend(runtime),
    routes={
        "/memories/": StoreBackend(runtime),
        "/archive/": StoreBackend(runtime),  # Optional: additional routes
    }
)
```

---

## Current Implementation Gaps

### Issue 1: Prompt vs Configuration Mismatch

**Problem**: The system prompt in `default_agent_prompt.md` tells the LLM to use `/memories/`, but the CLI doesn't automatically configure this route.

**Current Behavior**:
- LLM tries to write to `/memories/notes.md`
- Route not configured → Goes to default backend (FilesystemBackend)
- File might persist (if FilesystemBackend) but not in the intended store

**Solution**: Either:
1. Auto-configure `/memories/` route when `enable_memory=True`
2. Update system prompt to match actual configuration
3. Make it explicit that routes must be configured

### Issue 2: No Explicit Guidance for Default Paths

**Problem**: The system prompt doesn't explicitly say "use default paths for temporary files."

**Current Behavior**:
- LLM might not realize other paths are temporary
- Might accidentally use `/memories/` for temporary files

**Solution**: Add explicit guidance:
```
**Temporary Files:**
- Use default paths (e.g., `/app/`, current directory) for working files
- These files are ephemeral and won't persist across sessions
- Example: `/app/temp_analysis.py`
```

---

## Recommended System Prompt Additions

Add these sections to guide the LLM better:

```markdown
## File Storage Decision Guide

When creating files, decide based on persistence needs:

**Use `/memories/` for:**
- Information that should persist across sessions
- User preferences and learned patterns
- Knowledge bases and reference materials
- Anything the user asks you to "remember"

**Use default paths (e.g., `/app/`) for:**
- Working files for current task
- Generated code being edited
- Temporary analysis or computation results
- Files only needed during this conversation

**Decision Flow:**
1. Will this be useful in future sessions? → `/memories/`
2. Is this just for current work? → Default path
3. User explicitly says "remember"? → `/memories/`
4. Unsure? → Default path (can move later if needed)
```

---

## Summary

### How the LLM Decides:

1. **System Prompt Guidance**: Explicit instructions on which paths to use
2. **Semantic Understanding**: LLM infers from context (remember vs. temporary)
3. **Path-Based Routing**: CompositeBackend automatically routes based on prefixes
4. **No Direct Backend Choice**: LLM doesn't know about StateBackend vs StoreBackend

### How to Assist the LLM:

1. ✅ **Configure routes** to match system prompt instructions
2. ✅ **Provide clear path conventions** in system prompt
3. ✅ **Include examples** of when to use each path
4. ✅ **Use semantic naming** (`/memories/`, `/archive/`, etc.)
5. ✅ **Make persistence explicit** ("persists across sessions" vs "temporary")

### Key Takeaway:

The LLM chooses **paths**, not backends. The system prompt guides path selection, and CompositeBackend handles the routing automatically. The challenge is ensuring the system prompt and route configuration are aligned!

