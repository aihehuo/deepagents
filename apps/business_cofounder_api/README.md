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

The server uses `ChatAnthropic` and is **env-configurable** (so you can point it at Anthropic, or an Anthropic-compatible proxy such as DeepSeek):

- `BC_API_MODEL` (preferred) or `ANTHROPIC_MODEL`
- `ANTHROPIC_API_KEY`
- `ANTHROPIC_BASE_URL` (optional; use this for proxies)
- `BC_API_TEMPERATURE` (optional)
- `BC_API_MAX_TOKENS` (optional)

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


