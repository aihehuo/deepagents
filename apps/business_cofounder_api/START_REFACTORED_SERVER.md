# How to Start the Refactored Server

## Option 1: Run Directly from Project Root

From the `deepagents` project root directory:

```bash
cd /Users/yc/workspace/deepagents
python -m uvicorn apps.business_cofounder_api.app_refactored:app --host 0.0.0.0 --port 8001
```

## Option 2: Run from API Directory (Temporary Replacement)

If you want to test it as `app.py` (replacing the original temporarily):

```bash
cd /Users/yc/workspace/deepagents/apps/business_cofounder_api

# Backup original
cp app.py app_original.py

# Use refactored version
cp app_refactored.py app.py

# Run normally
python -m uvicorn app:app --host 0.0.0.0 --port 8001
```

## Option 3: Run with Environment Variables

Set up your environment variables first (e.g., from `.docker.env` or `.env`):

```bash
cd /Users/yc/workspace/deepagents

# Load environment variables (adjust path as needed)
export $(cat apps/business_cofounder_api/.docker.env | xargs)

# Run the refactored server
python -m uvicorn apps.business_cofounder_api.app_refactored:app --host 0.0.0.0 --port 8001
```

## Option 4: Run with Debug/State Endpoints Enabled

Enable debug endpoints for testing:

```bash
cd /Users/yc/workspace/deepagents

export BC_API_ENABLE_STATE_ENDPOINT=1
export BC_API_LOG_STATE=1
export BC_API_LOG_CHAT_IO=1

python -m uvicorn apps.business_cofounder_api.app_refactored:app --host 0.0.0.0 --port 8001 --reload
```

Note: `--reload` enables auto-reload during development.

## Quick Test

Once the server is running, test it:

```bash
# Health check
curl http://localhost:8001/health

# Chat endpoint (if BC_API_ENABLE_STATE_ENDPOINT=1)
curl "http://localhost:8001/state?user_id=test&conversation_id=default"
```

## Differences from Original

The refactored server:
- Uses the **same endpoints** (`/chat`, `/chat/stream`, `/health`, `/state`, `/reset`)
- Uses the **same port** (8001 by default)
- Has the **same API interface** - clients don't need to change
- Uses background threads instead of async LLM calls internally

## Troubleshooting

1. **Import errors**: Make sure you're running from the project root or have `PYTHONPATH` set:
   ```bash
   export PYTHONPATH=/Users/yc/workspace/deepagents:$PYTHONPATH
   ```

2. **Missing dependencies**: Install requirements:
   ```bash
   pip install -r apps/business_cofounder_api/requirements.txt
   ```

3. **Port already in use**: Use a different port:
   ```bash
   python -m uvicorn apps.business_cofounder_api.app_refactored:app --port 8002
   ```

