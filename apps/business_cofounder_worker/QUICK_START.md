# Quick Start - Business Co-Founder Worker

Minimal Celery worker implementation to test threading in production.

## Why Celery?

Celery workers run in separate processes (not threads), which should avoid the "can't start new thread" errors we encountered with the API server approach.

## Quick Test (Local)

### 1. Redis

- Use existing host Redis at `redis://host.docker.internal:6379/0` (default)
- Or set `CELERY_BROKER_URL` to a custom Redis endpoint

### 2. Start Worker

```bash
# From project root
cd /Users/yc/workspace/deepagents
# Uses host.docker.internal:6379 by default; override CELERY_BROKER_URL if needed
./apps/business_cofounder_worker/start_worker.sh
```

Or manually:
```bash
export PYTHONPATH="libs/deepagents:libs/deepagents-cli:$PYTHONPATH"
export CELERY_BROKER_URL="redis://host.docker.internal:6379/0"  # or your redis
celery -A apps.business_cofounder_worker.celery_app worker --loglevel=info --pool=solo
```

### 3. Test Worker

In another terminal:
```bash
cd /Users/yc/workspace/deepagents
python apps/business_cofounder_worker/test_client.py
```

## Docker Test

### Build

```bash
docker build -t business-cofounder-worker -f apps/business_cofounder_worker/Dockerfile .
```

### Run with Docker Compose

```bash
cd apps/business_cofounder_worker
docker-compose up
```

### Run Standalone

```bash
docker run -d \
  --name business-cofounder-worker \
  -e CELERY_BROKER_URL=redis://host.docker.internal:6379/0 \
  -e MODEL_API_KEY=... \
  -e MODEL_BASE_URL=... \
  -v ~/.deepagents:/root/.deepagents \
  business-cofounder-worker
```

## Production Test

1. Build and push image (similar to API)
2. Deploy worker container
3. Ensure Redis is accessible
4. Submit test task
5. Check logs for threading errors

## What to Test

1. **Worker starts successfully** - No import errors
2. **Task submission works** - Task is accepted by worker
3. **Agent execution works** - No "can't start new thread" errors
4. **Result is returned** - Task completes and returns reply

## Pool Options

- `--pool=solo`: Single-threaded (default, safest for testing)
- `--pool=prefork`: Multi-process (better performance, if threading works)

## Next Steps

Once threading is confirmed to work:
- Add proper task routing
- Add result persistence
- Add monitoring/health checks
- Add API endpoint to submit tasks
- Add task status tracking

