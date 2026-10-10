"""Experimental service-only synthesis; no tools, conversation or outgoing reply."""
from __future__ import annotations

import asyncio
import hmac
import json
import os
from typing import Any

from fastapi import HTTPException, Request
from langchain_core.messages import HumanMessage, SystemMessage

from apps.group_agent_api.app.wecom_journey import MAX_EVIDENCE_KEYS, CONVERGENCE_PROMPT, structured_prompt, validate_context, validate_journey


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
    prompt = structured_prompt(context) + (
        "\n这是独立内部整理：reply必须为零长度字符串，不生成客户回复，不执行工具。"
        "has_need必须是字符串yes/no/unknown，不能使用true/false或中文；needs_truncated必须是JSON布尔值。"
        "所有evidence_keys和intent_evidence_keys各最多10条，只选最相关且存在于sources的证据，不能堆砌引用。"
        "doing/background/offer/title/target/purpose/conditions每项最多160个字符；next_action最多240字符。"
        "missing最多5项，每项最多100字符；needs最多8项；clarity仅clear/unclear。"
        "schema_version必须为整数1；简报JSON最多12000字节；引用至少支持非空画像和需求内容。"
        "需求id必须唯一，格式need_后接1至40位小写字母/数字/下划线/连字符。"
        "严格遵守模板字段，未知用空字符串/空数组，不用null，不增加字段。"
    )
    deadline = asyncio.get_running_loop().time() + 75
    correction = ""
    for attempt in range(2):
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise asyncio.TimeoutError
        output = await asyncio.wait_for(model.ainvoke([
            SystemMessage(content=CONVERGENCE_PROMPT + prompt + correction),
            HumanMessage(content="根据提供的证据更新客服简报。"),
        ]), timeout=remaining)
        try:
            return validate_output(output.content, context["sources"])
        except OutputError as error:
            if attempt:
                raise
            # Only a fixed error code is echoed; never replay invalid model text.
            correction = "\n上次输出校验失败：" + error.code + "。请按上述约束重新生成完整JSON，不改变证据事实。"
    raise AssertionError("unreachable")


class OutputError(ValueError):
    """A fixed, privacy-safe output validation code."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def validate_output(text: Any, sources: dict[str, Any]) -> dict[str, Any]:
    """Decode bounded synthesis without coercing intent or dropping evidence.

    Args:
        text: Model output, never logged or included in correction feedback.
        sources: Trusted finite source catalogue.
    """
    if not isinstance(text, str) or len(text.encode()) > 20000:
        raise OutputError("service_journey_output_size")
    try:
        if text.strip().startswith("```"):
            text = text.strip().split("\n", 1)[1].rsplit("```", 1)[0]
        value = json.loads(text)
    except (ValueError, IndexError):
        raise OutputError("service_journey_output_json") from None
    if not isinstance(value, dict) or set(value) != {"reply", "service_journey"} or value["reply"] != "":
        raise OutputError("service_journey_output_envelope")
    data = value["service_journey"]
    if isinstance(data, dict):
        if not isinstance(data.get("has_need"), str) or data["has_need"] not in {"yes", "no", "unknown"}:
            raise OutputError("service_journey_has_need")
        if type(data.get("needs_truncated")) is not bool:
            raise OutputError("service_journey_needs_truncated")
        entries = [data, data.get("founder_profile")]
        if isinstance(data.get("needs"), list):
            entries.extend(data["needs"])
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            for field in ("intent_evidence_keys", "evidence_keys"):
                keys = entry.get(field)
                if isinstance(keys, list) and len(keys) > MAX_EVIDENCE_KEYS:
                    raise OutputError("service_journey_evidence_count")
                if isinstance(keys, list) and any(not isinstance(key, str) or key not in sources for key in keys):
                    raise OutputError("service_journey_evidence_reference")
    try:
        return validate_journey(data, sources)
    except (ValueError, TypeError, KeyError):
        raise OutputError("service_journey_output_schema") from None



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
    except OutputError as error:
        import logging
        logging.getLogger("uvicorn.error").warning("service_journey_rejected code=%s", error.code)
        raise HTTPException(422, detail=error.code) from None
    except (ValueError, TypeError, KeyError):
        raise HTTPException(422, detail="service_journey_output_schema") from None
    except Exception as error:
        import logging
        logging.getLogger("uvicorn.error").error("service_journey_failed type=%s", type(error).__name__)
        raise HTTPException(503, detail="service_journey_model_failed") from None
    return {"service_journey": result}
