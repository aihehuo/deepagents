# Inspecting Running Tasks in the Business Co-Founder API Server

When the API server is busy and you can't terminate it, use these commands to see what's happening.

## 1. Find the Python Process

First, identify the running server process:

```bash
# Find uvicorn/python processes
ps aux | grep -E "(uvicorn|python.*app)" | grep -v grep

# Or more specifically for the BC API
ps aux | grep "business_cofounder_api" | grep -v grep

# Get process ID (PID)
pgrep -f "business_cofounder_api"
```

## 2. Check Process Details

Once you have the PID (let's call it `$PID`), inspect what it's doing:

```bash
# Replace $PID with actual process ID
PID=$(pgrep -f "business_cofounder_api")

# See detailed process information
ps -p $PID -o pid,ppid,cmd,%cpu,%mem,etime,state

# See threads within the process
ps -p $PID -T -o pid,tid,comm,state,%cpu,%mem

# See what files/sockets the process has open
lsof -p $PID

# See network connections
lsof -p $PID -i

# See current working directory and open files
lsof -p $PID -a -d cwd
```

## 3. Monitor CPU and Memory Usage

```bash
# Real-time monitoring
top -p $PID

# Or with htop (if installed)
htop -p $PID

# Continuous monitoring with 1-second intervals
watch -n 1 "ps -p $PID -o pid,%cpu,%mem,vsz,rss,etime,state,cmd"
```

## 4. See Current Activity (System Calls)

```bash
# Trace system calls (shows what the process is doing)
strace -p $PID -c  # Summary mode (press Ctrl+C after a few seconds)

# See detailed system calls
strace -p $PID -f -e trace=file,network  # Only file and network operations

# See all system calls (verbose)
strace -p $PID -f -t -s 200
```

## 5. Inspect Python Threads and Stack Traces

```bash
# Install py-spy first: pip install py-spy

# Show current Python call stack
py-spy top --pid $PID

# Dump all thread stack traces
py-spy dump --pid $PID

# Generate a flamegraph
py-spy record -o profile.svg --pid $PID --duration 10

# Show real-time Python stack traces
py-spy top --pid $PID --threads
```

## 6. Check Logs

The server logs to stdout/stderr (uvicorn logger). If you're running it directly:

```bash
# If running in a terminal, you should see logs there
# If running in background, check:
tail -f /proc/$PID/fd/1  # stdout
tail -f /proc/$PID/fd/2  # stderr

# Or if logs are redirected to a file
tail -f /path/to/logfile
```

## 7. See Active Network Connections

```bash
# See what ports the process is listening on and connections
netstat -tulpn | grep $PID
# Or with ss (modern alternative)
ss -tulpn | grep $PID

# See active connections
lsof -p $PID -i -a
```

## 8. Quick One-Liner to Get Full Status

```bash
PID=$(pgrep -f "business_cofounder_api")
if [ -n "$PID" ]; then
  echo "=== Process Info ==="
  ps -p $PID -o pid,ppid,user,%cpu,%mem,etime,state,cmd
  echo ""
  echo "=== Threads ==="
  ps -p $PID -T -o tid,comm,state,%cpu
  echo ""
  echo "=== Open Files/Sockets ==="
  lsof -p $PID | head -20
  echo ""
  echo "=== Network Connections ==="
  lsof -p $PID -i
else
  echo "Process not found"
fi
```

## 9. Check for Locked Resources

If the process seems stuck, it might be waiting on a lock:

```bash
# Check for file locks
lsof -p $PID | grep -E "(LOCK|\.lock)"

# Check for Python locks (in process memory)
# This requires gdb or py-spy to inspect Python objects
```

## 10. Use Python's Built-in Debugging (if process is responsive)

If the process is still accepting signals, you can try to get a Python traceback:

```bash
# Send SIGUSR1 to print stack trace (if the app handles it)
kill -USR1 $PID

# Or use faulthandler (if enabled) to dump traceback on SIGUSR1
# The app would need to have: import faulthandler; faulthandler.enable()
```

## 11. Check if it's an LLM API Call

Since the agent makes LLM API calls, check if it's waiting on network:

```bash
# See if process is in "D" (uninterruptible sleep) state - usually I/O wait
ps -p $PID -o state

# If state is "D", it's likely waiting on I/O (network/disk)
# Check network activity
iftop -i lo  # Monitor localhost traffic (if installed)

# Or use tcpdump to see network packets
sudo tcpdump -i lo -n host localhost and port 8001
```

## 12. Monitor Agent State (via API)

If the server is still responding, you can check the agent's current state:

```bash
# Check agent state for a specific conversation
curl "http://localhost:8001/state?user_id=u1&conversation_id=default" | jq

# Check health endpoint
curl http://localhost:8001/health
```

## Common Scenarios

### Scenario 1: Stuck on LLM API call
- **Symptoms**: Process in "S" (sleeping) or "D" (waiting) state, high wait time
- **Check**: Network connections with `lsof -p $PID -i`
- **Solution**: The LLM API might be slow or unresponsive

### Scenario 2: Lock contention
- **Symptoms**: Process active but not progressing, multiple threads
- **Check**: `py-spy dump --pid $PID` to see if threads are waiting on locks
- **Solution**: Check the lock implementation in `app.py` (thread_locks)

### Scenario 3: CPU-bound processing
- **Symptoms**: High CPU usage, process in "R" (running) state
- **Check**: `py-spy top --pid $PID` to see what function is consuming CPU
- **Solution**: Agent is actively processing (token generation, file operations)

### Scenario 4: Disk I/O wait
- **Symptoms**: Process in "D" state, high I/O wait in top
- **Check**: `iostat -x 1` or `iotop` to see disk activity
- **Solution**: Agent might be reading/writing large files or checkpoints

## Example: Full Diagnostic Script

Save this as `inspect_api.sh`:

```bash
#!/bin/bash
PID=$(pgrep -f "business_cofounder_api")

if [ -z "$PID" ]; then
  echo "❌ Business Co-Founder API process not found"
  exit 1
fi

echo "🔍 Inspecting process $PID"
echo "=================================="
echo ""
echo "📊 Process Status:"
ps -p $PID -o pid,ppid,user,%cpu,%mem,vsz,rss,etime,state,cmd
echo ""
echo "🧵 Threads:"
ps -p $PID -T -o tid,comm,state,%cpu,%mem | head -10
echo ""
echo "📁 Open Files/Sockets (top 20):"
lsof -p $PID | head -20
echo ""
echo "🌐 Network Connections:"
lsof -p $PID -i
echo ""
echo "💾 Memory Map:"
pmap -x $PID | tail -5
echo ""
echo "⏱️  Runtime:"
ps -p $PID -o etime=
echo ""
echo "🔗 To get Python stack traces, install py-spy and run:"
echo "   py-spy dump --pid $PID"
```

Make it executable: `chmod +x inspect_api.sh` and run: `./inspect_api.sh`
