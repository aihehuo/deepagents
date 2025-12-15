# Business Co-Founder Worker (Celery)

Minimal Celery worker implementation to test threading in production.

## Architecture

- **Celery Worker**: Processes tasks in background worker processes
- **Redis**: Message broker and result backend
- **Tasks**: Business Co-Founder agent execution tasks

## Why Celery?

Celery workers run in separate processes (not threads), which avoids the "can't start new thread" errors we encountered with the API server approach.

## Setup

### 1. Install Dependencies

```bash
cd apps/business_cofounder_worker
pip install -r requirements.txt
```

### 2. Configure Redis (Broker)

- Use existing host Redis at `redis://host.docker.internal:6379/0` (default)
- Or set a custom `CELERY_BROKER_URL`

### 3. Start Celery Worker

```bash
# From project root
export PYTHONPATH="libs/deepagents:libs/deepagents-cli:$PYTHONPATH"
# Uses host.docker.internal:6379 by default
celery -A apps.business_cofounder_worker.celery_app worker --loglevel=info --pool=solo
```

Options:
- `--pool=solo`: Single-threaded (avoids thread creation, good for testing)
- `--pool=prefork`: Multi-process (better performance, if threading works)

### 4. Test Worker

```bash
# From project root
python apps/business_cofounder_worker/test_client.py
```

## Docker

### Build

```bash
docker build -t business-cofounder-worker -f apps/business_cofounder_worker/Dockerfile .
```

### Run

```bash
docker run -d \
  --name business-cofounder-worker \
  -e CELERY_BROKER_URL=redis://host.docker.internal:6379/0 \
  -e MODEL_API_KEY=... \
  -e MODEL_BASE_URL=... \
  -v ~/.deepagents:/root/.deepagents \
  business-cofounder-worker
```

## Environment Variables

- `CELERY_BROKER_URL`: Redis broker URL (default: `redis://host.docker.internal:6379/0`)
- `MODEL_API_KEY`, `MODEL_BASE_URL`, etc.: Same as Business Co-Founder API

## Testing Threading

The worker uses `--pool=solo` by default, which runs tasks in the same process/thread. This avoids thread creation issues.

To test if threading works:
1. Change to `--pool=prefork` (multi-process)
2. Submit a task
3. Check logs for thread creation errors

## Next Steps

Once threading is confirmed to work:
1. Add proper task routing
2. Add result persistence
3. Add monitoring/health checks
4. Add API endpoint to submit tasks
5. Add task status tracking

