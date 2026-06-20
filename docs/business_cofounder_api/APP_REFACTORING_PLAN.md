# App.py Refactoring Plan

## Current Status
- **File size**: 3521 lines
- **Structure**: Monolithic file with all functionality

## Refactoring Strategy

### Phase 1: Core Infrastructure (✅ Completed)
- ✅ `app/models.py` - Pydantic request/response models
- ✅ `app/state.py` - Application state management
- ✅ `app/utils.py` - Utility functions (logging, path resolution, etc.)

### Phase 2: Callbacks & Streaming (In Progress)
- ⏳ `app/callbacks.py` - Callback functions (~900 lines)
  - `_invoke_callback`
  - `_send_artifacts_callback`
  - `_send_heartbeat`
  - `_run_async_stream_with_callback` (very large, ~850 lines)
  - `_compose_concise_callback_message`
  - `_serialize_for_json`

### Phase 3: Startup & Configuration
- ⏳ `app/startup.py` - Startup/shutdown handlers
  - `_configure_asyncio_default_executor`
  - `_patch_openai_no_thread`
  - `_startup` event handler

### Phase 4: Endpoints
- ⏳ `app/endpoints/health.py` - Health check endpoint
- ⏳ `app/endpoints/canvas.py` - Canvas/state endpoint
- ⏳ `app/endpoints/chat.py` - Chat endpoints (sync & stream)
- ⏳ `app/endpoints/reset.py` - Reset endpoint
- ⏳ `app/endpoints/deep_agent.py` - Deep agent async endpoint

### Phase 5: Main App Assembly
- ⏳ `app/__init__.py` - FastAPI app instance and endpoint registration
- ⏳ `app.py` - Backward compatibility wrapper (re-exports from app package)

## File Size Breakdown (Current)
- Models: ~90 lines
- State: ~25 lines
- Utils: ~380 lines
- Callbacks: ~900 lines (estimated)
- Startup: ~100 lines (estimated)
- Endpoints: ~2000 lines (estimated)
- Exception handlers: ~20 lines

## Benefits
1. **Better organization**: Each module has a clear responsibility
2. **Easier navigation**: Smaller, focused files
3. **Maintainability**: Changes are localized to specific modules
4. **Testability**: Easier to test individual components
5. **Backward compatibility**: Original `app.py` still works via re-exports

## Next Steps
Continue extracting callbacks module, then endpoints, then assemble in `__init__.py`.
