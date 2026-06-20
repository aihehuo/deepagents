# RunLoop Environment Example

## The Flow

### Step 1: Harbor Framework Creates RunLoop Environment

When you run:
```bash
harbor run --env runloop --agent-import-path deepagents_harbor:DeepAgentsWrapper
```

**What Harbor does internally (pseudo-code):**
```python
# Harbor framework creates a RunLoopEnvironment
from harbor.environments.runloop import RunLoopEnvironment

# Harbor internally does something like:
runloop_env = RunLoopEnvironment(
    api_key=os.getenv("RUNLOOP_API_KEY"),
    session_id="harbor_session_xyz789"
)

# This RunLoopEnvironment provides:
# - environment.exec(command) -> runs command via RunLoop API
# - environment.session_id -> "harbor_session_xyz789"
# - environment.trial_paths.config_path -> path to task config
```

### Step 2: HarborSandbox Wraps the Environment

```python
# In deepagents_wrapper.py, line 173:
from deepagents_harbor.backend import HarborSandbox

# Harbor provides the environment to our agent:
backend = HarborSandbox(environment=runloop_env)

# Now backend.environment is the RunLoopEnvironment instance
```

### Step 3: Agent Executes a Command

When the agent wants to run `ls -la`:

```python
# Agent calls:
result = await backend.aexecute("ls -la")

# Inside HarborSandbox.aexecute() (line 24-29):
async def aexecute(self, command: str) -> ExecuteResponse:
    # This calls the RunLoop environment's exec method
    result = await self.environment.exec(command)

    # Inside RunLoopEnvironment.exec(), Harbor probably does:
    # POST https://api.runloop.com/devboxes/{id}/execute
    # {
    #   "command": "ls -la"
    # }
    # and waits for completion, capturing stdout/stderr/exit_code

    # Returns ExecuteResponse with the output
    return ExecuteResponse(
        output=result.stdout + "\n stderr: " + result.stderr,
        exit_code=result.return_code
    )
```

### Comparison: Direct RunLoopBackend vs HarborSandbox

**Option A: Direct RunLoopBackend (CLI usage)**
```python
from deepagents_cli.integrations.runloop import RunloopBackend
from runloop_api_client import Runloop

# You create the backend directly:
client = Runloop(bearer_token="your_api_key")
backend = RunloopBackend(devbox_id="devbox_123", client=client)

# When you call execute:
result = backend.execute("ls -la")
# Internally calls: client.devboxes.execute_and_await_completion(...)
```

**Option B: HarborSandbox with RunLoopEnvironment (Harbor usage)**
```python
from deepagents_harbor.backend import HarborSandbox

# Harbor creates the environment, you wrap it:
backend = HarborSandbox(environment=runloop_env)

# When you call execute:
result = await backend.aexecute("ls -la")
# Internally calls: runloop_env.exec(...)
# Which Harbor's RunLoopEnvironment implements using RunLoop API
```

## Visual Flow

```
┌─────────────────────────────────────────────────────────────┐
│ Harbor Framework (runs command: --env runloop)              │
│                                                              │
│  Creates: RunLoopEnvironment                                │
│    - Connects to RunLoop cloud devbox via API               │
│    - Uses RunLoop API: POST /devboxes/{id}/execute          │
│    - session_id: "harbor_session_xyz789"                    │
│                                                              │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      │ Provides environment to agent
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ DeepAgentsWrapper.run()                                     │
│                                                              │
│  backend = HarborSandbox(environment=runloop_env)           │
│                                                              │
│  HarborSandbox wraps RunLoopEnvironment                     │
│    - backend.environment = runloop_env                      │
│    - backend.environment.exec() → RunLoop API call          │
│                                                              │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      │ Agent calls backend methods
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ Agent executes: backend.aexecute("ls -la")                  │
│                                                              │
│  HarborSandbox.aexecute()                                   │
│    ↓                                                         │
│  self.environment.exec("ls -la")                            │
│    ↓                                                         │
│  RunLoopEnvironment.exec()                                  │
│    ↓                                                         │
│  HTTP POST to RunLoop API:                                  │
│    https://api.runloop.com/devboxes/{id}/execute            │
│    { "command": "ls -la" }                                  │
│    ↓                                                         │
│  Waits for completion, returns:                             │
│    stdout="file1.txt\nfile2.py", exit_code=0               │
│    ↓                                                         │
│  HarborSandbox formats as ExecuteResponse                   │
│    ↓                                                         │
│  Agent receives result                                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Key Differences: Harbor vs Direct CLI Usage

### Harbor Usage (with HarborSandbox)
```python
# Harbor manages the environment lifecycle
# You just wrap whatever Harbor gives you:
backend = HarborSandbox(environment=runloop_env)  # or docker_env, or modal_env
```

### Direct CLI Usage (with RunloopBackend)
```python
# You manage the RunLoop connection yourself:
from deepagents_cli.integrations.runloop import RunloopBackend
from runloop_api_client import Runloop

client = Runloop(bearer_token=api_key)
backend = RunloopBackend(devbox_id="devbox_123", client=client)
# This inherits from BaseSandbox, so it has full file operations
```

## Why HarborSandbox Exists

The HarborSandbox is needed because:
1. Harbor framework provides different environment types (Docker, RunLoop, Modal, etc.)
2. All provide the same `BaseEnvironment` interface
3. HarborSandbox converts `BaseEnvironment` → `SandboxBackendProtocol`
4. This allows DeepAgents to work with any Harbor environment transparently

The same `HarborSandbox` code works for Docker, RunLoop, Modal, Daytona - you just pass different `BaseEnvironment` instances!
