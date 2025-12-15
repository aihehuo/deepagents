# API Refactoring: Background Thread Pattern

## Overview

This refactoring changes the Business Co-Founder API to match BPGenerationAgent's threading pattern, avoiding the "can't start new thread" errors that occur in production environments with strict thread limits.

## Key Changes

### 1. Removed Async Executor Workarounds
- **Removed**: `_configure_asyncio_default_executor()` - no longer needed
- **Removed**: `_patch_openai_no_thread()` - no longer needed
- These were workarounds for async LLM calls that tried to spawn threads. We now avoid async LLM calls entirely.

### 2. Sync Graph Execution in Background Threads
- **Changed**: `/chat` endpoint now uses `agent.invoke()` (sync) in a background thread instead of `agent.ainvoke()` (async)
- **Changed**: `/chat/stream` endpoint now uses `agent.stream()` (sync) in a background thread instead of `agent.astream()` (async)
- **Pattern**: Matches BPGenerationAgent's `_run_async_generation` function that runs `graph.invoke()` in `threading.Thread`

### 3. Lock Coordination
- **Changed**: Using `threading.Lock` instead of `asyncio.Lock` for coordinating per-thread_id execution
- **Reason**: Background threads need threading primitives, not asyncio primitives

### 4. Queue-Based SSE Bridge
- **New**: For `/chat/stream`, we bridge sync background thread → queue → async SSE generator
- **Implementation**: Background thread puts chunks in `queue.Queue`, async generator polls queue with timeout
- **Benefit**: Keeps sync graph execution while maintaining async SSE response

## Files

- `app_refactored.py`: New refactored implementation
- `app.py`: Original implementation (keep for comparison)

## How to Test

1. **Backup current app.py**:
   ```bash
   cp apps/business_cofounder_api/app.py apps/business_cofounder_api/app_original.py
   ```

2. **Replace with refactored version**:
   ```bash
   cp apps/business_cofounder_api/app_refactored.py apps/business_cofounder_api/app.py
   ```

3. **Test locally**:
   ```bash
   cd apps/business_cofounder_api
   python -m uvicorn app:app --port 8001
   ```

4. **Test endpoints**:
   - `POST /chat` - synchronous chat (should work without thread errors)
   - `POST /chat/stream` - streaming chat (should stream properly)

5. **Test in Docker** (mimics production):
   ```bash
   docker-compose -f docker-compose.yml up --build
   ```

## Benefits

1. **Avoids thread creation during LLM calls**: Sync graph execution doesn't trigger async networking's thread pool requirements
2. **Matches proven pattern**: BPGenerationAgent uses this pattern successfully in production
3. **Simpler code**: Removed complex workarounds (`BC_API_OPENAI_NO_THREAD`, `BC_API_ASYNCIO_EXECUTOR_WORKERS`)
4. **Better production stability**: No risk of "can't start new thread" errors under load

## Migration Notes

- Environment variables `BC_API_OPENAI_NO_THREAD` and `BC_API_ASYNCIO_EXECUTOR_WORKERS` are no longer used (but won't cause errors if set)
- All other functionality remains the same (checkpoints, state endpoints, etc.)
- Response format is identical to original implementation

