"""Retired synthesis route; assessments run on the independent customer queue."""
import hmac
import os

from fastapi import HTTPException, Request


def authenticate(secret: str) -> None:
    """Authenticate the legacy service route before returning its retirement notice.

    Args:
        secret: Service secret supplied by the caller.
    """
    expected = os.environ.get("GROUP_AGENT_SERVICE_SECRET") or os.environ.get("MICRO_SERVICE_SECRET")
    if not expected:
        raise HTTPException(503, detail="service_auth_unconfigured")
    if not secret or not hmac.compare_digest(secret.encode(), expected.encode()):
        raise HTTPException(403, detail="service_forbidden")


async def refresh(request: Request) -> dict[str, object]:
    """Reject the former synchronous synthesis path without loading a model.

    Args:
        request: Legacy authenticated service request.
    """
    authenticate(request.headers.get("X-GA-Service-Secret", ""))
    raise HTTPException(410, detail="service_journey_moved_to_customer_assessment")
