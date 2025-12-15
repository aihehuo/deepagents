# Event Loop and Blocking Explained

## What is the Event Loop?

The **event loop** is like a single-threaded "task manager" that runs your async code. Think of it as a waiter in a restaurant:

- The waiter (event loop) can handle multiple tables (requests) by going back and forth between them
- When one table is waiting for food (I/O operation like network request), the waiter goes to another table
- This allows one waiter to serve many tables efficiently

In FastAPI/async Python:
- The event loop runs in a **single thread**
- It can handle many requests concurrently by switching between them
- When one request is waiting (e.g., for a database query), the event loop handles other requests

## What Does "Blocking" Mean?

**Blocking** means the event loop is stuck doing one thing and can't handle anything else.

### Example: Non-Blocking (Good) ✅

```python
async def handle_request():
    # This is async - while waiting for the API call, event loop can handle other requests
    result = await some_async_api_call()  # Event loop can switch to other requests here
    return result
```

### Example: Blocking (Bad) ❌

```python
async def handle_request():
    # This is sync - event loop is STUCK here, can't handle other requests
    result = some_sync_function_that_takes_10_seconds()  # Event loop blocked for 10 seconds!
    return result
```

## What Happens in Our Code?

### Current Implementation (with fallback):

```python
# First try: Non-blocking (uses asyncio.to_thread)
try:
    result = await asyncio.to_thread(sync_function)  # ✅ Event loop free to handle other requests
except RuntimeError:
    # Fallback: Blocking (runs directly)
    result = sync_function()  # ❌ Event loop stuck here, can't handle other requests
```

### Why We Need the Fallback?

In your production environment, thread creation is blocked. So:
- `asyncio.to_thread()` tries to run code in a background thread
- But it can't create a thread → fails with "can't start new thread"
- So we fall back to running the sync code directly
- This **blocks the event loop** because sync code runs in the main thread

## Real-World Impact

### Scenario 1: Non-Blocking (with threads working)

```
Request 1: [waiting for LLM] → Event loop switches to Request 2
Request 2: [waiting for LLM] → Event loop switches to Request 3
Request 3: [waiting for LLM] → Event loop can handle all 3 concurrently
```

**Result**: All 3 requests can be processed at the same time (concurrent)

### Scenario 2: Blocking (our fallback)

```
Request 1: [running sync LLM call - 10 seconds] → Event loop STUCK
Request 2: [waiting...] → Can't be handled, event loop is stuck
Request 3: [waiting...] → Can't be handled, event loop is stuck
```

**Result**: Only 1 request at a time, others wait in line (serialized)

## In Our Specific Case

With our current implementation:

1. **If `asyncio.to_thread()` works**: 
   - Multiple requests can be processed concurrently
   - Event loop stays free

2. **If `asyncio.to_thread()` fails** (thread creation blocked):
   - Falls back to direct execution
   - Event loop is blocked during graph execution
   - But we use `sync_execution_lock` to serialize requests anyway
   - So even if blocking, only one request runs at a time (which is what we want)

## The Good News

Even though the fallback "blocks" the event loop, it's actually **okay** in our case because:

1. We serialize all requests with `sync_execution_lock` anyway
2. So only one request runs at a time regardless
3. The blocking just means other requests wait in the FastAPI queue (which they would anyway)
4. The server still works, just processes requests one-by-one

## Summary

- **Event loop** = Single-threaded task manager that handles async code
- **Blocking** = Event loop stuck doing one thing, can't handle other requests
- **Our fallback blocks** = But it's okay because we serialize requests anyway
- **Result** = Server works, processes requests one at a time (which is fine for our use case)

