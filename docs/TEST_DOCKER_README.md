# Docker Environment Test Script

This test script demonstrates how Harbor initializes Docker containers and executes commands in them, similar to what happens with `HarborSandbox` when using `--env docker`.

## Prerequisites

1. **Docker installed and running**
   ```bash
   # Check if Docker is running
   docker ps

   # If not installed, install Docker:
   # macOS: brew install --cask docker
   # Linux: Follow Docker installation guide
   # Windows: Install Docker Desktop
   ```

2. **Python dependencies**
   ```bash
   pip install docker
   ```

## Running the Test

```bash
# Make sure Docker is running
docker ps

# Run the test script
python test_docker_environment.py

# Or make it executable and run directly:
chmod +x test_docker_environment.py
./test_docker_environment.py
```

## What the Test Does

The script simulates Harbor's Docker environment initialization:

1. **Container Creation**: Creates a Docker container from `ubuntu:22.04` image
2. **Command Execution**: Runs various bash commands in the container
3. **File Operations**: Creates, reads, and lists files in the container
4. **Error Handling**: Tests commands that fail and verifies error capture
5. **Isolation**: Tests multiple containers to verify isolation

## Test Output

The script will output:
- ✓ Container initialization status
- ✓ Command execution results
- ✓ File operation results
- ✓ Error handling verification
- ✓ Cleanup confirmation

## Expected Output

```
======================================================================
Docker Environment Test - Simulating Harbor's Docker Environment
======================================================================

======================================================================
TEST 1: Container Initialization
======================================================================

✓ Connected to Docker daemon

📦 Creating Docker container...
   Image: ubuntu:22.04
   Name: harbor_test_001
   ✓ Image ubuntu:22.04 found locally
   ✓ Container created and running: abc123def456

✓ Container initialized successfully
  Session ID: test_001
  Container ID: abc123def456

======================================================================
TEST 2: Simple Command Execution
======================================================================

📝 Executing: echo 'Hello from Docker!'
   Exit code: 0
   Stdout: 'Hello from Docker!'
   ✓ Output matches expected result

... (more tests)

✅ All tests completed successfully!
```

## How It Relates to Harbor

This script demonstrates the same flow that Harbor uses:

```
┌─────────────────────────────────────────────────────────┐
│ Harbor Framework                                        │
│   Creates: DockerEnvironment                           │
│   - docker run ubuntu:22.04                            │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ HarborSandbox                                           │
│   - Wraps DockerEnvironment                            │
│   - Provides SandboxBackendProtocol interface          │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ Your Agent                                              │
│   - Calls: backend.aexecute("ls -la")                  │
│   - Gets results from container                         │
└─────────────────────────────────────────────────────────┘
```

## Troubleshooting

### Docker not running
```
ERROR: Failed to connect to Docker: ...
```
**Solution**: Start Docker daemon
```bash
# macOS/Linux
sudo systemctl start docker  # Linux
open -a Docker  # macOS
```

### Image not found
```
ImageNotFound: ...
```
**Solution**: The script will automatically pull the image, but you can pre-pull it:
```bash
docker pull ubuntu:22.04
```

### Permission denied
```
PermissionError: ...
```
**Solution**:
- Linux: Add your user to docker group: `sudo usermod -aG docker $USER`
- macOS/Windows: Usually not an issue with Docker Desktop

## Understanding the Code

Key components:

1. **`DockerEnvironment`**: Mimics Harbor's Docker environment
   - Creates containers
   - Executes commands via `docker exec`
   - Manages container lifecycle

2. **`exec()` method**: What Harbor calls internally
   - Runs `container.exec_run()`
   - Returns stdout, stderr, and exit code
   - Similar to Harbor's `environment.exec()`

3. **Context manager**: Automatic cleanup
   - Container created in `__enter__`
   - Container removed in `__exit__`
   - Ensures cleanup even on errors

## Next Steps

After running this test, you can:
- Modify commands to test different scenarios
- Try different base images
- Test file upload/download operations
- Simulate more complex Harbor workflows
