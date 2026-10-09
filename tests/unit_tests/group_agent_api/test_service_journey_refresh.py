"""Independent internal analysis fails explicitly, never generating customer text."""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from apps.group_agent_api.app.endpoints.service_journey import authenticate, synthesize
from tests.unit_tests.group_agent_api.test_wecom_journey import inputs, summary


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


async def test_internal_synthesis_is_cited_and_tool_free():
    context, _ = inputs()
    model = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content=json.dumps({"reply": "", "service_journey": summary()}))))
    result = await synthesize(context, model)
    assert result["founder_profile"]["clarity"] == "clear"
    assert len(result["needs"]) == 2
    model.ainvoke.assert_awaited_once()
    assert "不生成客户回复" in model.ainvoke.call_args.args[0][0].content


@pytest.mark.parametrize("output", ["普通客户回复", '{"reply":"给客户发这句","service_journey":{}}', '{"reply":"","service_journey":{}}'])
async def test_plain_or_invalid_analysis_is_failure_not_silent_success(output):
    context, _ = inputs()
    model = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content=output)))
    with pytest.raises(ValueError):
        await synthesize(context, model)


async def test_foreign_evidence_cannot_enter_summary():
    context, _ = inputs()
    value = summary()
    value["founder_profile"]["evidence_keys"] = ["wecom_turn:999:input:0"]
    model = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content=json.dumps({"reply": "", "service_journey": value}))))
    with pytest.raises(ValueError):
        await synthesize(context, model)


async def test_http_endpoint_auth_size_and_valid_result(monkeypatch):
    import httpx
    from unittest.mock import patch
    from apps.group_agent_api.app import create_app
    monkeypatch.setenv("GROUP_AGENT_SERVICE_SECRET", "service-secret")
    transport = httpx.ASGITransport(app=create_app())
    context, _ = inputs()
    model = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content=json.dumps({"reply": "", "service_journey": summary()}))))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/wecom/service_journey", json=context, headers={"X-Agent-Key": "service-secret"})
        assert response.status_code == 403
        headers = {"X-GA-Service-Secret": "service-secret"}
        response = await client.post("/wecom/service_journey", content=b"x" * 24001, headers=headers)
        assert response.status_code == 413
        response = await client.post("/wecom/service_journey", json={"user_number": "wrong"}, headers=headers)
        assert response.status_code == 422
        with patch("apps.group_agent_api.agent_factory.model_builder.create_model", return_value=model):
            response = await client.post("/wecom/service_journey", json=context, headers=headers)
        assert response.status_code == 200
        assert set(response.json()) == {"service_journey"}
        assert response.json()["service_journey"]["founder_profile"]["clarity"] == "clear"
        assert "reply" not in response.json()
    model.ainvoke.assert_awaited_once()
