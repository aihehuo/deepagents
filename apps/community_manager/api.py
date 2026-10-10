"""Dedicated manager HTTP admission with its own broker, replay cache and queue bound."""
import os
import uuid
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, ConfigDict
from redis import Redis
from apps.community_manager.protocol import verify, rpc, ProtocolError
from apps.community_manager.celery_app import app as celery, QUEUE

app = FastAPI(title="Community Manager", docs_url=None, redoc_url=None)


class Admission(BaseModel):
    """Only an opaque run reference; scope is fetched from Micro by the worker."""
    model_config = ConfigDict(extra="forbid")
    run_id: uuid.UUID


def redis_client() -> Redis:
    """Redis client.
    """
    return Redis.from_url(os.environ.get("COMMUNITY_MANAGER_REDIS_URL", "redis://localhost:6379/6"), socket_connect_timeout=2, socket_timeout=2)


@app.get("/health")
def health() -> dict[str, str]:
    """Health.
    """
    return {"status": "ok", "service": "community_manager"}


@app.post("/manager/call_async", status_code=202)
async def admit(request: Request) -> dict:
    """Authenticate before enqueue; do not import any customer graph or customer queue.

    Args:
        request: Incoming FastAPI request.
    """
    if request.headers.get("content-type", "").split(";")[0] != "application/json":
        raise HTTPException(415, "unsupported_media_type")
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > 1024:
            raise HTTPException(413, "payload_too_large")
    try:
        redis = redis_client()
        verify(bytes(body), dict(request.headers), redis)
        data = Admission.model_validate_json(bytes(body))
        # Kombu's global_keyprefix also applies to the list used by this admission bound.
        if redis.llen("community_manager:v1:" + QUEUE) >= 100:
            raise HTTPException(429, "manager_queue_full")
        celery.send_task("community_manager.run", args=[str(data.run_id)], queue=QUEUE, expires=600, retry=False)
        return {"run_id": str(data.run_id), "status": "queued"}
    except ProtocolError as error:
        raise HTTPException(401, str(error)) from None
    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(422, "invalid_manager_request") from None
    except Exception:
        raise HTTPException(503, "manager_queue_unavailable") from None


@app.get("/manager/runs/{run_id}")
def run_status(run_id: uuid.UUID, request: Request) -> dict:
    """Read status for a signed service request; no user/session transcript is returned.

    Args:
        run_id: Opaque Micro run identifier.
        request: Incoming FastAPI request.
    """
    try:
        # Include the exact path in the signed bytes to prevent cross-run replay.
        verify(request.url.path.encode(), dict(request.headers), redis_client())
        return rpc("status", run_id=str(run_id))
    except ProtocolError as error:
        raise HTTPException(401, str(error)) from None
    except Exception:
        raise HTTPException(503, "manager_status_unavailable") from None


@app.get("/ready")
def ready() -> dict[str, str]:
    """Check required admission configuration and the isolated broker connection."""
    try:
        from apps.community_manager.protocol import secret
        secret()
        if not os.environ.get("COMMUNITY_MANAGER_AGENT_KEY") or not os.environ.get("COMMUNITY_MANAGER_OPERATIONS_URL"):
            raise ValueError("unconfigured")
        redis_client().ping()
        return {"status": "ready", "service": "community_manager"}
    except Exception:
        raise HTTPException(503, "manager_not_ready") from None
