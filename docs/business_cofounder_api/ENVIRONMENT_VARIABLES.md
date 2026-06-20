# Business Co-Founder API - Environment Variables

Complete list of all environment variables used in the Business Co-Founder API application.

## Model Configuration

### Main Model Settings
- **`MODEL_API_PROVIDER`** (default: `deepseek`)
  - Provider selection: `deepseek` | `qwen`
  - `deepseek`: Uses Anthropic-compatible `ChatAnthropic`
  - `qwen`: Uses OpenAI-compatible `ChatOpenAI` (DashScope compatible-mode)

- **`MODEL_NAME`** (optional)
  - Model name to use
  - Defaults: `deepseek-chat` (for deepseek) or `qwen-plus` (for qwen)
  - Location: `agent_factory.py`

- **`MODEL_BASE_URL`** (optional)
  - Base URL for the model API endpoint
  - Required for most proxy/compatible endpoints
  - Location: `agent_factory.py`

- **`MODEL_API_KEY`** (optional)
  - API key for the model provider
  - Location: `agent_factory.py`

- **`MODEL_API_MAX_TOKENS`** (default: `20000`)
  - Maximum tokens for model responses
  - Location: `agent_factory.py`

- **`MODEL_API_TEMPERATURE`** (optional)
  - Temperature setting for the model (0.0-2.0)
  - Default: not set (uses model default)
  - Location: `agent_factory.py`

- **`MODEL_API_TIMEOUT_S`** (default: `180.0`)
  - Timeout in seconds for model API calls
  - Location: `agent_factory.py`

### Coder Subagent Model Settings (Optional)
- **`CODER_MODEL_API_PROVIDER`** (optional)
  - Provider for the coder subagent (defaults to `MODEL_API_PROVIDER` if unset)
  - Location: Used by `build_coder_subagent_from_env()` in `deepagents`

- **`CODER_MODEL_BASE_URL`** (optional)
  - Base URL for coder subagent model
  - Location: Used by `build_coder_subagent_from_env()` in `deepagents`

- **`CODER_MODEL_API_KEY`** (optional)
  - API key for coder subagent model
  - Location: Used by `build_coder_subagent_from_env()` in `deepagents`

- **`CODER_MODEL_NAME`** (optional)
  - Model name for coder subagent (defaults to `MODEL_NAME` if unset)
  - Location: Used by `build_coder_subagent_from_env()` in `deepagents`

## API Configuration

### Logging
- **`BC_API_LOG_CHAT_IO`** (default: `false`)
  - Enable logging of chat input/output
  - Set to `1`, `true`, `TRUE`, `yes`, `YES`, `on`, or `ON` to enable
  - Location: `app.py:52`, `app_sync_calls.py:108`

- **`BC_API_LOG_TRUNCATE_CHARS`** (default: `2000`)
  - Maximum characters to log per message (truncates longer messages)
  - Location: `app.py:54`, `app_sync_calls.py:110`

- **`BC_API_LOG_STATE`** (default: `false`)
  - Enable debug logging of milestones/todos/tool calls
  - Set to `1`, `true`, `TRUE`, `yes`, `YES`, `on`, or `ON` to enable
  - Location: `app.py:67`, `app_sync_calls.py:123`

- **`BC_API_STREAM_DEBUG`** (default: `false`)
  - Enable debug logging for streaming endpoint
  - Logs delta count, message types, HTML file paths, etc.
  - Set to `1`, `true`, `TRUE`, `yes`, `YES`, `on`, or `ON` to enable
  - Location: `app.py:581`, `app_sync_calls.py:918`

### Endpoints
- **`BC_API_ENABLE_STATE_ENDPOINT`** (default: `false`)
  - Enable the `/state` debug endpoint
  - Set to `1`, `true`, `TRUE`, `yes`, `YES`, `on`, or `ON` to enable
  - Location: `app.py:366`, `app_sync_calls.py:536`

### Error Handling
- **`BC_API_RETURN_TRACEBACK`** (default: `false`)
  - Include full traceback in HTTP error responses
  - Useful for local dev; avoid enabling in production
  - Set to `1`, `true`, `TRUE`, `yes`, `YES`, `on`, or `ON` to enable
  - Location: `app.py:456`, `app_sync_calls.py:664,748,849,950`

### Timeouts
- **`BC_API_INVOKE_TIMEOUT_S`** (default: `300.0`)
  - Timeout in seconds for `/chat` endpoint (non-streaming)
  - Location: `app_sync_calls.py:612`

### Threading & Execution
- **`BC_API_ASYNCIO_EXECUTOR_WORKERS`** (default: `1`)
  - Number of workers for asyncio default executor
  - Used for DNS resolution and async networking
  - Location: `app.py:320`
  - Note: Only used in `app.py` (async version), not in `app_sync_calls.py`

- **`BC_API_OPENAI_NO_THREAD`** (default: `false`)
  - Patch OpenAI SDK to avoid `asyncio.to_thread()` calls
  - Prevents "can't start new thread" errors in restricted environments
  - Set to `1`, `true`, `TRUE`, `yes`, `YES`, `on`, or `ON` to enable
  - Location: `app.py:299`
  - Note: Only used in `app.py` (async version), not in `app_sync_calls.py`

- **`BC_API_DISABLE_CHECKPOINT_EXECUTOR`** (default: `true`)
  - Disable LangGraph's checkpoint ThreadPoolExecutor
  - Patches ThreadPoolExecutor to run synchronously
  - Prevents thread creation errors in restricted environments
  - Set to `1`, `true`, `TRUE`, `yes`, `YES`, `on`, or `ON` to enable
  - Location: `app_sync_calls.py:421`
  - Note: Only used in `app_sync_calls.py` (sync version)

### Version
- **`BC_API_VERSION`** (optional)
  - Version string for deployment verification
  - Falls back to git commit hash if available, or "dev" if not
  - Location: `app_sync_calls.py:52`

## Client/CLI Configuration (bc_api.sh)

These are used by the `bc_api.sh` CLI script, not the API server itself:

- **`API_TYPE`** (default: `bc`)
  - API type selection: `bc` (Business Co-Founder) | `bp` (BP Generation Agent)

- **`BC_API_BASE_URL`** (default: `http://127.0.0.1:8001`)
  - Base URL for Business Co-Founder API
  - Location: `bc_api.sh`

- **`BC_API_PORT`** (default: `8001`)
  - Port for Business Co-Founder API
  - Location: `bc_api.sh`

- **`BC_API_USER_ID`** (default: `u1`)
  - Default user ID for API calls
  - Location: `bc_api.sh`

- **`BC_API_CONV_ID`** (default: `default`)
  - Default conversation ID for API calls
  - Location: `bc_api.sh`

- **`BP_API_BASE_URL`** (default: `http://127.0.0.1:8000`)
  - Base URL for BP Generation Agent API
  - Location: `bc_api.sh`

- **`BP_API_PORT`** (default: `8000`)
  - Port for BP Generation Agent API
  - Location: `bc_api.sh`

## Summary by Category

### Required (for basic operation)
- None - all have defaults or are optional

### Recommended (for production)
- `MODEL_API_PROVIDER`
- `MODEL_API_KEY`
- `MODEL_BASE_URL` (if using proxy/compatible endpoint)
- `MODEL_NAME` (if not using default)

### Optional (for debugging/development)
- `BC_API_LOG_CHAT_IO=1`
- `BC_API_LOG_STATE=1`
- `BC_API_ENABLE_STATE_ENDPOINT=1`
- `BC_API_RETURN_TRACEBACK=1` (development only)
- `BC_API_STREAM_DEBUG=1`

### Optional (for restricted environments)
- `BC_API_OPENAI_NO_THREAD=1` (async version only)
- `BC_API_ASYNCIO_EXECUTOR_WORKERS=1` (async version only)
- `BC_API_DISABLE_CHECKPOINT_EXECUTOR=1` (sync version only)

## Example Configuration

See `docker.env.example` for a complete example configuration file.
