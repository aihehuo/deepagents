"""Authenticated prompt sandbox: a model call without conversation or delivery writes."""
from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID

from fastapi import HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, StrictStr, ValidationError

from apps.group_agent_api.app.endpoints.ai_assistant import Input, PromptRevision, execute, prompt, service_prompt, default_prompt, SERVICE_DEFAULT
from apps.group_agent_api.app.endpoints.service_journey import authenticate
from apps.group_agent_api.app.assistant_usage import UsageReceipt, sink, endpoint_role

_slots = asyncio.Semaphore(1)


class Preview(BaseModel):
    """Explicit test input with no database fetch or queue admission."""
    model_config = ConfigDict(extra="forbid")
    revision: PromptRevision
    fixture: dict[str, Any]
    request_id: StrictStr = Field(max_length=36)


async def run(value: Preview) -> dict[str, object]:
    """Run the normal executor with a test receipt and no delivery callback.

    Args:
        value: Host-selected immutable revision and explicitly supplied test data.
    """
    UUID(value.request_id, version=4)
    if "prompt_revision" in value.fixture:
        raise ValueError("fixture_cannot_override_revision")
    payload = Input.model_validate({**value.fixture, "prompt_revision": value.revision.model_dump()})
    if payload.kind != "conversation":
        raise ValueError("preview_requires_personal_conversation")
    plain = value.revision.prompt_key == "ai_assistant.service_reply"
    if value.revision.version == 0:
        baseline = SERVICE_DEFAULT if plain else default_prompt(payload)
        if value.revision.content != baseline:
            raise ValueError("baseline_revision_mismatch")
        payload.prompt_revision = None
    effective = service_prompt(payload) if plain else prompt(payload)
    if _slots.locked():
        raise HTTPException(429, detail="prompt_preview_busy")

    async def discard(_event: str, _data: dict[str, Any]) -> bool:
        return True

    receipt = UsageReceipt(discard)
    requested: dict[str, str] = {}

    async def capture(event: str, data: dict[str, Any]) -> bool | None:
        call = dict(data.get("supplier_usage") or {})
        ident = call.get("call_id", "")
        role = endpoint_role.get()
        if call.get("status") == "started":
            requested[ident] = call.get("model") or "unknown"
        call["requested_model"] = requested.get(ident, "unknown")
        call["endpoint_role"] = role
        call["provider"] = "token_plan" if role == "primary" else (call.get("provider") or "unknown")
        call["model"] = call.get("model") or "unknown"
        return await receipt.emit(event, {"supplier_usage": call})

    token = sink.set(capture)
    try:
        async with _slots:
            result = await execute(payload, reply_only=plain)
        return {"success": True, "content_sha256": value.revision.content_sha256,
                "effective_prompt": effective, "result": result, "usage": receipt.manifest(), "side_effects": False}
    except Exception as error:
        return {"success": False, "error": type(error).__name__, "content_sha256": value.revision.content_sha256,
                "usage": receipt.manifest(), "side_effects": False}
    finally:
        sink.reset(token)


async def preview(request: Request) -> dict[str, object]:
    """Authenticate before reading bounded JSON; never forwards to a business queue.

    Args:
        request: Internal Micro request using the existing service secret.
    """
    authenticate(request.headers.get("X-GA-Service-Secret", ""))
    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw) > 60000:
            raise HTTPException(413, detail="preview_too_large")
    try:
        value = Preview.model_validate_json(bytes(raw))
        return await run(value)
    except (ValueError, ValidationError) as error:
        raise HTTPException(422, detail="invalid_prompt_preview") from error

