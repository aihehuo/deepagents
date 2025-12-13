# Business Co-Founder API — Docker Deployment

This folder provides a Docker setup for running `apps.business_cofounder_api.app:app`.

**Important:** This setup intentionally mimics the conventions used in `BPGenerationAgent`
(base image, compose layout, Aliyun registry push script style).

## Base image note (Python 3.11)

`deepagents` requires **Python >= 3.11**. The HuaweiCloud SWR mirror used by `BPGenerationAgent`
for Python 3.10 may not have a Python 3.11 tag.  
So this service defaults to `python:3.11-slim-bullseye`, but you can override via compose build args:

- `BASE_IMAGE: <your-mirror-python-3.11-image>`

## Quick start (docker compose)

From repo root:

```bash
cd apps/business_cofounder_api
cp .docker.env.example .docker.env
docker compose up -d --build
docker compose logs -f
```

Then:
- Health: `GET http://localhost:8001/health`
- Chat: `POST http://localhost:8001/chat`

## Persistence

The API persists LangGraph checkpoints to:
- `/root/.deepagents/business_cofounder_api/checkpoints.pkl`

### Local dev (bind mount)

`docker-compose.yml` mounts:
- `./data` → `/root/.deepagents/business_cofounder_api`

So checkpoints + copied skills survive restarts and are visible on your host.

### Production (named volume)

`docker-compose.prod.yml` uses a named volume:
- `business_cofounder_api_data` → `/root/.deepagents/business_cofounder_api`

This avoids relying on a relative `./data` directory existing on the production host
(and is typically the right default for server deployments).

## Production compose

```bash
cd apps/business_cofounder_api
docker compose -f docker-compose.prod.yml up -d --build
```

## Logging

Enable request/response logging:
- `BC_API_LOG_CHAT_IO=1`

Optional truncation:
- `BC_API_LOG_TRUNCATE_CHARS=2000`

## Aliyun registry build/push

See `build_and_push.sh` (same registry + username pattern as BPGenerationAgent).

## Deployment config (.deploy.env)

To avoid committing internal hostnames / registry usernames to a public repo:

1. Copy `apps/business_cofounder_api/deploy.env.example` to `apps/business_cofounder_api/.deploy.env`
2. Fill in:
   - `ALIYUN_DOCKER_REGISTRY`
   - `ALIYUN_DOCKER_USERNAME`
   - `REMOTE_HOST` (for deploy script)

`.deploy.env` is ignored by git.

## Note on production stability (exit 139 / segfault)

If you observe containers restarting with `exit=139` and kernel logs like `traps: python ... in libc.so.6`,
this is usually a native dependency crash.

This API intentionally runs uvicorn in a **pure-python** configuration to reduce that risk:
- `--loop asyncio` (avoid `uvloop`)
- `--http h11` (avoid `httptools`)


