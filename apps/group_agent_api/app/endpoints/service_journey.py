"""Experimental service-only synthesis; no tools, conversation or outgoing reply."""
from __future__ import annotations

import asyncio
import hmac
import json
import os
from typing import Any

from fastapi import HTTPException, Request
from langchain_core.messages import HumanMessage, SystemMessage

from apps.group_agent_api.app.wecom_journey import CONVERGENCE_PROMPT, structured_prompt, validate_context, validate_journey


def authenticate(secret: str) -> None:
    """Require a configured shared host secret without JWT or cookie fallback.

    Args:
        secret: Secret supplied in the service header.
    """
    expected = os.environ.get("GROUP_AGENT_SERVICE_SECRET") or os.environ.get("MICRO_SERVICE_SECRET")
    if not expected:
        raise HTTPException(503, detail="service_auth_unconfigured")
    if not secret or not hmac.compare_digest(secret.encode(), expected.encode()):
        raise HTTPException(403, detail="service_forbidden")


async def synthesize(context: dict[str, Any], model: Any) -> dict[str, Any]:
    """Generate only bounded, cited internal progress.

    Args:
        context: Host-authored source catalogue, verified independently on saving.
        model: A tool-free language model.
    """
    metadata = {"source": "wecom_service", "wecom_service_action": "conversation",
                "wecom_service_customer_number": context.get("user_number")}
    context = validate_context(context, metadata)
    if context is None:
        raise ValueError("missing service journey context")
    prompt = structured_prompt(context) + "\n这是独立的内部整理任务：reply必须为空，不生成客户回复、不提问、不执行工具。"
    output = await asyncio.wait_for(model.ainvoke([SystemMessage(content=CONVERGENCE_PROMPT + prompt),
                                                 HumanMessage(content="根据提供的证据更新客服简报。")]), timeout=75)
    text = output.content
    if not isinstance(text, str) or len(text.encode()) > 20000:
        raise ValueError("invalid service journey output")
    if text.strip().startswith("```"):
        text = text.strip().split("\n", 1)[1].rsplit("```", 1)[0]
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {"reply", "service_journey"} or value["reply"] != "":
        raise ValueError("invalid service journey envelope")
    return validate_journey(value["service_journey"], context["sources"])


async def refresh(request: Request) -> dict[str, Any]:
    """Service endpoint. Authenticate before reading sources or loading a model.

    Args:
        request: Request with shared secret and a bounded JSON source catalogue.
    """
    authenticate(request.headers.get("X-GA-Service-Secret", ""))
    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw) > 24000:
            raise HTTPException(413, detail="service_journey_too_large")
    try:
        context = json.loads(raw)
        if not isinstance(context, dict):
            raise ValueError("invalid context")
        validate_context(context, {"source": "wecom_service", "wecom_service_customer_number": context.get("user_number")})
    except (ValueError, TypeError):
        raise HTTPException(422, detail="invalid_service_journey_context") from None
    from apps.group_agent_api.agent_factory.model_builder import create_model
    try:
        result = await synthesize(context, create_model(timeout_s=70, max_retries=0))
    except (ValueError, TypeError, KeyError):
        raise HTTPException(422, detail="invalid_service_journey_output") from None
    except Exception as error:
        import logging
        logging.getLogger("uvicorn.error").error("service_journey_failed type=%s", type(error).__name__)
        raise HTTPException(503, detail="service_journey_model_failed") from None
    return {"service_journey": result}
