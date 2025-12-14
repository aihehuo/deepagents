# Business Co-Founder Agent API (Internal)

Internal HTTP API that exposes DeepAgents as a **Business Co-Founder Agent**.

## Quick start

Install dependencies (in your runtime environment):

```bash
pip install fastapi uvicorn
```

Run (from repo root):

```bash
PYTHONPATH="libs/deepagents:libs/deepagents-cli" \
  uvicorn apps.business_cofounder_api.app:app --host 0.0.0.0 --port 8000
```

## Model configuration

The server supports switching between DeepSeek and Qwen via a single generic env var set:

- `MODEL_API_PROVIDER`: `deepseek` (default) or `qwen`
- `MODEL_NAME` (optional; provider default will be used if unset)
- `MODEL_API_KEY`
- `MODEL_BASE_URL` (optional; required for most proxy/compatible endpoints)
- `MODEL_API_TEMPERATURE` (optional)
- `MODEL_API_MAX_TOKENS` (optional)
- `MODEL_API_TIMEOUT_S` (optional)

### Provider notes
- `MODEL_API_PROVIDER=deepseek`: uses Anthropic-compatible `ChatAnthropic`
- `MODEL_API_PROVIDER=qwen`: uses OpenAI-compatible `ChatOpenAI` (DashScope compatible-mode works here)

## Request/response logging (optional)

To print every `/chat` request’s user message + the assistant’s final reply to server logs:

- `BC_API_LOG_CHAT_IO=1`
- `BC_API_LOG_TRUNCATE_CHARS=2000` (optional)

## Endpoints

- `GET /health`
- `POST /chat`
- `POST /reset`

## Persistence model

- Conversations are keyed by:
  - `thread_id = "bc::{user_id}::{conversation_id}"`
- Checkpoints are persisted to:
  - `~/.deepagents/business_cofounder_api/checkpoints.pkl`

This persists the full LangGraph state, including any **SummarizationMiddleware**-compressed
`state["messages"]`.

## agent.md (API runtime)

The CLI has an `agent.md` memory file under `~/.deepagents/<assistant_id>/agent.md`.
This API server is not the CLI, so it uses its own `agent.md` location:

- `~/.deepagents/business_cofounder_api/agent.md`

On first start, the server will auto-create a default `agent.md` if missing, and inject it
into the system prompt for every call.


