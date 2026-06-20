# Backend and Environment Comparison

## The Two Worlds

There are **two different ways** to use sandboxes in this codebase:

### 1. Direct CLI Usage (Standalone)
- Use `RunloopBackend`, `ModalBackend`, `DaytonaBackend` directly
- You manage the connection yourself
- Inherit from `BaseSandbox`
- Used in: `deepagents-cli` for interactive agent sessions

### 2. Harbor Framework Usage (Evaluation)
- Harbor provides `BaseEnvironment` instances
- `HarborSandbox` wraps them
- Used in: Harbor evaluation framework for benchmarking

---

## Side-by-Side: Docker vs RunLoop

### Docker Example Flow

```python
# Step 1: Harbor creates DockerEnvironment
docker_env = DockerEnvironment(
    image="ubuntu:22.04",
    session_id="harbor_session_abc123"
)

# Step 2: Wrap it
backend = HarborSandbox(environment=docker_env)

# Step 3: Agent uses it
result = await backend.aexecute("echo 'hello'")

# What happens:
# backend.aexecute()
#   → self.environment.exec("echo 'hello'")
#   → DockerEnvironment.exec()
#   → docker exec <container> bash -c "echo 'hello'"
#   → Returns: stdout="hello", exit_code=0
```

### RunLoop Example Flow

```python
# Step 1: Harbor creates RunLoopEnvironment
runloop_env = RunLoopEnvironment(
    api_key=os.getenv("RUNLOOP_API_KEY"),
    session_id="harbor_session_xyz789"
)

# Step 2: Wrap it
backend = HarborSandbox(environment=runloop_env)

# Step 3: Agent uses it
result = await backend.aexecute("echo 'hello'")

# What happens:
# backend.aexecute()
#   → self.environment.exec("echo 'hello'")
#   → RunLoopEnvironment.exec()
#   → HTTP POST to RunLoop API: /devboxes/{id}/execute
#   → Waits for completion
#   → Returns: stdout="hello", exit_code=0
```

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                    Harbor Framework                          │
│                                                              │
│  Creates environment based on --env flag:                   │
│                                                              │
│  --env docker  → DockerEnvironment                          │
│  --env runloop → RunLoopEnvironment                         │
│  --env modal   → ModalEnvironment                           │
│  --env daytona → DaytonaEnvironment                         │
│                                                              │
│  All implement BaseEnvironment interface:                    │
│    - exec(command) → runs command in sandbox                │
│    - session_id → unique identifier                         │
│    - trial_paths → access to task config                    │
│                                                              │
└─────────────────────┬────────────────────────────────────────┘
                      │
                      │ Provides environment to agent
                      ▼
┌──────────────────────────────────────────────────────────────┐
│              HarborSandbox (Wrapper)                         │
│                                                              │
│  class HarborSandbox(SandboxBackendProtocol):               │
│      def __init__(self, environment: BaseEnvironment):      │
│          self.environment = environment                     │
│                                                              │
│  Converts BaseEnvironment → SandboxBackendProtocol          │
│                                                              │
│  Implements all methods using environment.exec():           │
│    - aexecute() → environment.exec()                        │
│    - aread() → builds shell command, calls environment.exec()│
│    - awrite() → builds shell command, calls environment.exec()│
│    - aedit() → builds shell command, calls environment.exec()│
│                                                              │
└─────────────────────┬────────────────────────────────────────┘
                      │
                      │ Agent receives SandboxBackendProtocol
                      ▼
┌──────────────────────────────────────────────────────────────┐
│              DeepAgents Agent                                │
│                                                              │
│  Uses backend as SandboxBackendProtocol:                    │
│    - Doesn't know if it's Docker, RunLoop, etc.            │
│    - Just calls: backend.aexecute(), backend.aread(), etc. │
│                                                              │
│  All file operations work the same regardless of            │
│  underlying environment!                                     │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Key Insight: The Abstraction Layer

The **environment** is the actual sandbox (Docker container, RunLoop devbox, etc.)
The **backend** is the abstraction that provides a consistent interface.

**HarborSandbox** bridges the gap:
- Takes Harbor's `BaseEnvironment` (which could be any type)
- Provides DeepAgents' `SandboxBackendProtocol` interface
- Uses shell commands to implement file operations (no Python 3 required)

This allows:
- ✅ Same agent code works with Docker, RunLoop, Modal, Daytona
- ✅ Harbor manages environment lifecycle
- ✅ DeepAgents doesn't need to know about Harbor's internals

---

## Comparison Table

| Aspect | HarborSandbox | RunloopBackend (Direct) |
|--------|--------------|------------------------|
| **Created by** | Harbor framework | You create directly |
| **Inherits from** | `SandboxBackendProtocol` | `BaseSandbox` |
| **Takes** | `BaseEnvironment` instance | `devbox_id` + `Runloop` client |
| **Command execution** | `environment.exec()` | `client.devboxes.execute_and_await_completion()` |
| **File operations** | Shell commands via `environment.exec()` | Shell commands via RunLoop API |
| **Use case** | Harbor evaluation framework | Direct CLI usage |
| **Environment type** | Any (Docker, RunLoop, Modal, etc.) | RunLoop only |

---

## Real Code Examples

### Using HarborSandbox (Harbor framework)

```python
# This is what happens in deepagents_wrapper.py:
async def run(self, instruction: str, environment: BaseEnvironment, ...):
    # Harbor provides the environment - could be Docker, RunLoop, etc.
    backend = HarborSandbox(environment)

    # Create agent with the backend
    agent = create_cli_agent(..., sandbox=backend)

    # Agent uses it - doesn't care what environment type
    await agent.invoke({"messages": [HumanMessage(content=instruction)]})
```

### Using RunloopBackend (Direct CLI)

```python
# This is what happens in sandbox_factory.py:
from deepagents_cli.integrations.runloop import RunloopBackend
from runloop_api_client import Runloop

client = Runloop(bearer_token=os.getenv("RUNLOOP_API_KEY"))
devbox = client.devboxes.create()  # Create new devbox
backend = RunloopBackend(devbox_id=devbox.id, client=client)

# Use it directly
agent = create_cli_agent(..., sandbox=backend)
```

---

## Summary

**Environment** = The actual sandbox (Docker container, RunLoop devbox, etc.)
- Provided by Harbor framework
- Implements `BaseEnvironment` interface
- Knows how to execute commands in that specific sandbox type

**Backend** = The abstraction layer
- `HarborSandbox` wraps Harbor environments
- `RunloopBackend` directly uses RunLoop API
- Both implement `SandboxBackendProtocol` interface
- Agent code doesn't need to know the difference!
