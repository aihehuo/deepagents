"""The retired route is authenticated and never invokes a model."""
from unittest.mock import patch

import httpx
import pytest
from fastapi import HTTPException

from apps.group_agent_api.app.endpoints.service_journey import authenticate


def test_service_secret_is_fail_closed(monkeypatch):
    monkeypatch.delenv("GROUP_AGENT_SERVICE_SECRET", raising=False)
    monkeypatch.delenv("MICRO_SERVICE_SECRET", raising=False)
    with pytest.raises(HTTPException) as error:
        authenticate("any")
    assert error.value.status_code == 503
    monkeypatch.setenv("GROUP_AGENT_SERVICE_SECRET", "service-secret")
    for key in ("", "agent-key", "jwt"):
        with pytest.raises(HTTPException):
            authenticate(key)
    authenticate("service-secret")


async def test_retired_endpoint_never_loads_model_or_accepts_agent_key(monkeypatch):
    from apps.group_agent_api.app import create_app
    monkeypatch.setenv("GROUP_AGENT_SERVICE_SECRET", "service-secret")
    transport = httpx.ASGITransport(app=create_app())
    with patch("apps.group_agent_api.agent_factory.model_builder.create_model") as model:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/wecom/service_journey", json={}, headers={"X-Agent-Key": "service-secret"})
            assert response.status_code == 403
            response = await client.post("/wecom/service_journey", json={}, headers={"X-GA-Service-Secret": "service-secret"})
            assert response.status_code == 410
            assert response.json()["detail"] == "service_journey_moved_to_customer_assessment"
    model.assert_not_called()
