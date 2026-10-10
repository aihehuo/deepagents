"""Dedicated manager HTTP admission with its own broker, replay cache and queue bound."""
import os
import uuid
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, ConfigDict
from redis import Redis
from apps.community_manager.protocol import verify, rpc, ProtocolError
from apps.community_manager.celery_app import app as celery, QUEUE, CUSTOMER_QUEUE

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
    return await admit_run(request, queue=QUEUE, task="community_manager.run")


@app.post("/customer/assess_async", status_code=202)
async def admit_customer(request: Request) -> dict:
    """Admit a signed opaque assessment run on the independent customer queue.

    Args:
        request: Incoming service request, without employee or user scope claims.
    """
    return await admit_run(request, queue=CUSTOMER_QUEUE, task="community_customer.run")


async def admit_run(request: Request, *, queue: str, task: str) -> dict:
    """Authenticate exact bytes and bound the selected internal queue.

    Args:
        request: Incoming service request.
        queue: Server-selected queue.
        task: Server-selected task name.
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
        if redis.llen("community_manager:v1:" + queue) >= 100:
            raise HTTPException(429, "manager_queue_full")
        celery.send_task(task, args=[str(data.run_id)], queue=queue, expires=600, retry=False)
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
        if not all(os.environ.get(name) for name in ("COMMUNITY_MANAGER_AGENT_KEY", "COMMUNITY_MANAGER_OPERATIONS_URL", "COMMUNITY_MANAGER_MODEL", "COMMUNITY_MANAGER_MODEL_API_KEY", "COMMUNITY_MANAGER_MODEL_BASE_URL")):
            raise ValueError("unconfigured")
        redis_client().ping()
        return {"status": "ready", "service": "community_manager"}
    except Exception:
        raise HTTPException(503, "manager_not_ready") from None


@app.get("/customer/ready")
def customer_ready() -> dict[str, str]:
    """Check background model and service configuration without calling a provider."""
    try:
        from apps.community_manager.protocol import secret
        secret()
        required = ("COMMUNITY_MANAGER_AGENT_KEY", "COMMUNITY_MANAGER_OPERATIONS_URL", "COMMUNITY_CUSTOMER_MODEL", "COMMUNITY_CUSTOMER_MODEL_API_KEY", "COMMUNITY_CUSTOMER_MODEL_BASE_URL")
        if not all(os.environ.get(name) for name in required):
            raise ValueError("unconfigured")
        redis_client().ping()
        return {"status": "ready", "service": "community_customer_assessment"}
    except Exception:
        raise HTTPException(503, "customer_not_ready") from None
