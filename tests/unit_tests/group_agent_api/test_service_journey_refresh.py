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


@pytest.mark.parametrize("field,bad,code", [
    ("has_need", True, "service_journey_has_need"),
    ("has_need", "有需求", "service_journey_has_need"),
    ("needs_truncated", "false", "service_journey_needs_truncated"),
])
async def test_invalid_intent_is_corrected_once_without_coercion(field, bad, code):
    context, _ = inputs()
    value = summary()
    value[field] = bad
    model = SimpleNamespace(ainvoke=AsyncMock(side_effect=[
        SimpleNamespace(content=json.dumps({"reply": "", "service_journey": value})),
        SimpleNamespace(content=json.dumps({"reply": "", "service_journey": summary()})),
    ]))
    result = await synthesize(context, model)
    assert result == summary()
    assert model.ainvoke.await_count == 2
    assert code in model.ainvoke.call_args.args[0][0].content
    assert "最多10条" in model.ainvoke.call_args.args[0][0].content


async def test_excess_evidence_stays_rejected_after_one_correction():
    from apps.group_agent_api.app.endpoints.service_journey import OutputError
    context, _ = inputs()
    value = summary()
    value["founder_profile"]["evidence_keys"] = list(context["sources"]) * 11
    model = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content=json.dumps({"reply": "", "service_journey": value}))))
    with pytest.raises(OutputError) as error:
        await synthesize(context, model)
    assert error.value.code == "service_journey_evidence_count"
    assert model.ainvoke.await_count == 2


async def test_model_runtime_error_is_not_retried():
    context, _ = inputs()
    model = SimpleNamespace(ainvoke=AsyncMock(side_effect=RuntimeError("provider unavailable")))
    with pytest.raises(RuntimeError):
        await synthesize(context, model)
    model.ainvoke.assert_awaited_once()


async def test_http_failure_exposes_only_fixed_error_code(monkeypatch, caplog):
    import httpx
    from unittest.mock import patch
    from apps.group_agent_api.app import create_app
    monkeypatch.setenv("GROUP_AGENT_SERVICE_SECRET", "service-secret")
    context, _ = inputs()
    model = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content="PRIVATE_INVALID_MODEL_TEXT")))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()), base_url="http://test") as client:
        with patch("apps.group_agent_api.agent_factory.model_builder.create_model", return_value=model):
            response = await client.post("/wecom/service_journey", json=context, headers={"X-GA-Service-Secret": "service-secret"})
    assert response.status_code == 422
    assert response.json() == {"detail": "service_journey_output_json"}
    assert "PRIVATE_INVALID_MODEL_TEXT" not in caplog.text
    assert "PRIVATE_INVALID_MODEL_TEXT" not in model.ainvoke.call_args.args[0][0].content


async def test_correction_shares_original_deadline(monkeypatch):
    import asyncio
    from apps.group_agent_api.app.endpoints import service_journey
    context, _ = inputs()
    value = summary()
    value["has_need"] = True
    model = SimpleNamespace(ainvoke=AsyncMock(side_effect=[
        SimpleNamespace(content=json.dumps({"reply": "", "service_journey": value})),
        SimpleNamespace(content=json.dumps({"reply": "", "service_journey": summary()})),
    ]))
    wait_for = asyncio.wait_for
    budgets = []
    async def track(awaitable, timeout):
        budgets.append(timeout)
        return await wait_for(awaitable, timeout)
    monkeypatch.setattr(service_journey.asyncio, "wait_for", track)
    await synthesize(context, model)
    assert len(budgets) == 2
    assert 0 < budgets[1] <= budgets[0] <= 75


@pytest.mark.parametrize("count", [5, 6, 10, 11])
def test_evidence_limit_matches_decoder_and_journey_validator(count):
    from apps.group_agent_api.app.endpoints.service_journey import validate_output, OutputError
    from apps.group_agent_api.app.wecom_journey import validate_journey
    context, _ = inputs()
    source = next(iter(context["sources"].values()))
    keys = ["wecom_turn:%s:input:0" % i for i in range(1, count + 1)]
    sources = {key: source for key in keys}
    value = summary()
    value["intent_evidence_keys"] = keys
    value["founder_profile"]["evidence_keys"] = keys
    for need in value["needs"]:
        need["evidence_keys"] = keys
    text = json.dumps({"reply": "", "service_journey": value})
    if count <= 10:
        assert validate_output(text, sources) == value
        assert validate_journey(value, sources) == value
    else:
        with pytest.raises(OutputError) as error:
            validate_output(text, sources)
        assert error.value.code == "service_journey_evidence_count"
        with pytest.raises(ValueError):
            validate_journey(value, sources)
