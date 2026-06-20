# How Harbor Initializes Docker Environments

## Overview

Yes, you're absolutely right! When you use `--env docker`, Harbor **does initialize a Docker container locally** on your machine. Here's exactly what happens:

---

## The Docker Initialization Flow

### Step 1: Harbor Creates the Docker Container

When you run:
```bash
harbor run --env docker --agent-import-path deepagents_harbor:DeepAgentsWrapper
```

**Harbor internally does something like this:**

```python
# Harbor's DockerEnvironment (in the Harbor framework codebase)
from harbor.environments.docker import DockerEnvironment
import docker  # docker-py library

class DockerEnvironment(BaseEnvironment):
    def __init__(self, session_id: str, image: str = "ubuntu:22.04"):
        self.session_id = session_id
        self.docker_client = docker.from_env()  # Connect to local Docker daemon

        # Create and start a new Docker container
        self.container = self.docker_client.containers.run(
            image=image,  # e.g., "ubuntu:22.04" or Harbor's custom image
            detach=True,  # Run in background
            tty=False,    # No TTY
            stdin_open=False,
            remove=False, # Don't auto-remove (Harbor will clean up)
            name=f"harbor_{session_id}",  # Unique container name
            working_dir="/app",  # Working directory
            # Harbor might also mount volumes for task files
        )

        # Wait for container to be ready
        self.container.wait(condition="running")
```

### Step 2: Container is Running

After initialization:
- ✅ Docker container is running locally
- ✅ Container ID stored in `DockerEnvironment`
- ✅ Container persists for the duration of the evaluation task
- ✅ Working directory typically set to `/app` or similar

### Step 3: Commands Execute in the Container

When the agent executes a command:

```python
# Agent calls:
result = await backend.aexecute("ls -la")

# HarborSandbox.aexecute() calls:
result = await self.environment.exec("ls -la")

# DockerEnvironment.exec() does:
async def exec(self, command: str):
    # Execute command in the running container
    exec_result = self.container.exec_run(
        cmd=f"bash -c {shlex.quote(command)}",
        stdout=True,
        stderr=True,
        demux=True  # Separate stdout/stderr
    )

    # Returns:
    # - exit_code: 0 for success, non-zero for errors
    # - output: (stdout_bytes, stderr_bytes) tuple

    return ExecutionResult(
        stdout=exec_result.output[0].decode('utf-8') if exec_result.output[0] else "",
        stderr=exec_result.output[1].decode('utf-8') if exec_result.output[1] else "",
        return_code=exec_result.exit_code
    )
```

### Step 4: Container Cleanup

After the task completes, Harbor cleans up:

```python
# Harbor framework cleans up
def cleanup(self):
    if self.container:
        self.container.stop()      # Stop the container
        self.container.remove()    # Remove the container
```

---

## What Docker Image Does Harbor Use?

Harbor likely uses one of these approaches:

### Option 1: Standard Base Image
```python
# Harbor might use a standard image:
image = "ubuntu:22.04"  # or "debian:bookworm", "alpine:latest", etc.
```

### Option 2: Harbor's Custom Image
```python
# Harbor might maintain a custom image with common tools:
image = "harborai/terminal-bench:latest"
# This image might include:
# - Common utilities (git, curl, wget, etc.)
# - Programming language runtimes (python, node, etc.)
# - Development tools (gcc, make, etc.)
```

### Option 3: Configurable Image
```python
# Harbor might allow configuration:
image = os.getenv("HARBOR_DOCKER_IMAGE", "ubuntu:22.04")
```

**To find out what Harbor actually uses**, you can:

1. **Check Harbor's documentation** (https://github.com/HarborAI/harbor)
2. **Inspect running containers**:
   ```bash
   docker ps  # While a Harbor task is running
   # Look for containers named "harbor_*"
   ```
3. **Check Harbor's source code** in the Harbor repository

---

## Complete Docker Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ You run: harbor run --env docker ...                        │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ Harbor Framework Initialization                             │
│                                                              │
│  1. Creates DockerEnvironment                               │
│  2. Connects to local Docker daemon                         │
│  3. Pulls image if not already present                      │
│  4. Creates new container:                                  │
│     docker run -d --name harbor_abc123 \                    │
│                -w /app \                                    │
│                ubuntu:22.04                                 │
│                                                              │
│  Container is now running locally on your machine!          │
│                                                              │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      │ DockerEnvironment created with
                      │ reference to running container
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ Harbor provides environment to agent                        │
│                                                              │
│  backend = HarborSandbox(environment=docker_env)            │
│                                                              │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      │ Agent executes commands
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ Command Execution                                            │
│                                                              │
│  Agent: backend.aexecute("echo 'hello'")                    │
│    ↓                                                         │
│  HarborSandbox: self.environment.exec("echo 'hello'")       │
│    ↓                                                         │
│  DockerEnvironment: container.exec_run("bash -c 'echo hello'")│
│    ↓                                                         │
│  Docker API: Executes command in running container          │
│    ↓                                                         │
│  Returns: stdout="hello", exit_code=0                       │
│                                                              │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      │ Task completes
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ Harbor Cleanup                                               │
│                                                              │
│  1. Stops container: docker stop harbor_abc123              │
│  2. Removes container: docker rm harbor_abc123              │
│  3. Container is gone from your system                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Points

1. **Local Docker Required**: Harbor requires Docker to be installed and running locally when using `--env docker`

2. **Container Lifecycle**:
   - Created when task starts
   - Persists for entire task duration
   - Removed when task completes

3. **Isolation**: Each Harbor task gets its own isolated Docker container

4. **File Access**:
   - Container has its own filesystem
   - Harbor may mount volumes for task files (test cases, expected outputs, etc.)
   - Files created by the agent exist only in the container (unless volumes are used)

5. **No Python Required in Container**:
   - HarborSandbox uses shell commands (bash, awk, grep, etc.)
   - Container doesn't need Python installed
   - Works with minimal base images

---

## Example: Inspecting Harbor Containers

While a Harbor task is running, you can inspect the container:

```bash
# List running containers
docker ps

# You might see something like:
# CONTAINER ID   IMAGE          COMMAND     CREATED         STATUS
# abc123def456   ubuntu:22.04   "/bin/bash" 2 minutes ago   Up 2 minutes
# NAMES
# harbor_session_xyz789

# Execute a command in the container (for debugging)
docker exec -it harbor_session_xyz789 bash

# Check what's in the container
docker exec harbor_session_xyz789 ls -la /app

# View container logs
docker logs harbor_session_xyz789
```

---

## Differences: Docker vs Other Environments

| Aspect | Docker | RunLoop | Modal |
|--------|--------|---------|-------|
| **Location** | Local machine | Cloud (RunLoop API) | Cloud (Modal API) |
| **Setup** | `docker run` locally | HTTP API call | HTTP API call |
| **Isolation** | Local container | Remote devbox | Remote sandbox |
| **Prerequisites** | Docker installed locally | RunLoop API key | Modal account |
| **Speed** | Very fast (local) | Network latency | Network latency |
| **Cost** | Free (local resources) | Cloud pricing | Cloud pricing |

---

## Summary

Yes! When you use `--env docker`:

1. ✅ Harbor **creates a Docker container locally** on your machine
2. ✅ It uses a base image (likely `ubuntu:22.04` or Harbor's custom image)
3. ✅ The container runs for the duration of the evaluation task
4. ✅ All commands execute inside that container via `docker exec`
5. ✅ Harbor cleans up the container when the task completes

The container is a **real Docker container** running on your local Docker daemon, providing full isolation and security for the evaluation task.
