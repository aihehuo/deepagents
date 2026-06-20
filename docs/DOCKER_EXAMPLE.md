# Docker Environment Example

## The Flow

### Step 1: Harbor Framework Creates Docker Environment

When you run:
```bash
harbor run --env docker --agent-import-path deepagents_harbor:DeepAgentsWrapper
```

**What Harbor does internally (pseudo-code):**
```python
# Harbor framework creates a DockerEnvironment
from harbor.environments.docker import DockerEnvironment

# Harbor internally does something like:
docker_env = DockerEnvironment(
    image="ubuntu:22.04",  # or whatever Harbor uses
    session_id="harbor_session_abc123"
)

# This DockerEnvironment provides:
# - environment.exec(command) -> runs command in Docker container
# - environment.session_id -> "harbor_session_abc123"
# - environment.trial_paths.config_path -> path to task config
```

### Step 2: HarborSandbox Wraps the Environment

```python
# In deepagents_wrapper.py, line 173:
from deepagents_harbor.backend import HarborSandbox

# Harbor provides the environment to our agent:
backend = HarborSandbox(environment=docker_env)

# Now backend.environment is the DockerEnvironment instance
```

### Step 3: Agent Executes a Command

When the agent wants to run `ls -la`:

```python
# Agent calls:
result = await backend.aexecute("ls -la")

# Inside HarborSandbox.aexecute() (line 24-29):
async def aexecute(self, command: str) -> ExecuteResponse:
    # This calls the Docker environment's exec method
    result = await self.environment.exec(command)

    # Inside DockerEnvironment.exec(), Harbor probably does:
    # docker exec <container_id> bash -c "ls -la"
    # and captures stdout/stderr/exit_code

    # Returns ExecuteResponse with the output
    return ExecuteResponse(
        output=result.stdout + "\n stderr: " + result.stderr,
        exit_code=result.return_code
    )
```

### Step 4: File Operations Use Shell Commands

When the agent reads a file:

```python
# Agent calls:
content = await backend.aread("/app/script.py", offset=0, limit=100)

# Inside HarborSandbox.aread() (line 82-115):
async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
    # Builds a shell command using awk
    cmd = f"""
    if [ ! -f /app/script.py ]; then
        echo "Error: File not found"
        exit 1
    fi
    awk -v offset=0 -v limit=100 '...' /app/script.py
    """

    # Executes via environment (which runs in Docker container)
    result = await self.aexecute(cmd)

    return result.output  # Formatted with line numbers
```

## Visual Flow

```
┌─────────────────────────────────────────────────────────────┐
│ Harbor Framework (runs command: --env docker)               │
│                                                              │
│  Creates: DockerEnvironment                                 │
│    - Uses docker exec to run commands in container          │
│    - session_id: "harbor_session_abc123"                    │
│                                                              │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      │ Provides environment to agent
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ DeepAgentsWrapper.run()                                     │
│                                                              │
│  backend = HarborSandbox(environment=docker_env)            │
│                                                              │
│  HarborSandbox wraps DockerEnvironment                      │
│    - backend.environment = docker_env                       │
│    - backend.environment.exec() → docker exec               │
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
│  DockerEnvironment.exec()                                   │
│    ↓                                                         │
│  docker exec <container> bash -c "ls -la"                   │
│    ↓                                                         │
│  Returns: stdout="file1.txt\nfile2.py", exit_code=0        │
│    ↓                                                         │
│  HarborSandbox formats as ExecuteResponse                   │
│    ↓                                                         │
│  Agent receives result                                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Key Points

1. **DockerEnvironment** is created by Harbor framework
2. **HarborSandbox** wraps it to provide `SandboxBackendProtocol` interface
3. All commands run inside the Docker container via `docker exec`
4. No Python 3 assumption - uses shell commands (bash, awk, grep, etc.)
