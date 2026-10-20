"""Metadata-only supplier metering for authenticated APP assistant runs."""
from __future__ import annotations

import logging
import os
import time
import uuid
from contextvars import ContextVar
from typing import Any, Awaitable, Callable

Sink = Callable[[str, dict[str, Any]], Awaitable[bool | None]]
sink: ContextVar[Sink | None] = ContextVar('assistant_usage_sink', default=None)


def counters(output: Any) -> dict[str, Any]:
    """Normalize LangChain metadata without retaining prompts or model output."""
    meta = getattr(output, 'response_metadata', None) or {}
    usage = getattr(output, 'usage_metadata', None) or meta.get('token_usage') or {}
    def count(key: str, fallback: str) -> int | None:
        value = usage.get(key, usage.get(fallback))
        return value if type(value) is int and value >= 0 else None
    incoming, outgoing = count('input_tokens', 'prompt_tokens'), count('output_tokens', 'completion_tokens')
    details = usage.get('input_token_details') or usage.get('prompt_tokens_details') or {}
    completion = usage.get('output_token_details') or usage.get('completion_tokens_details') or {}
    def optional(value: Any) -> int | None:
        return value if type(value) is int and value >= 0 else None
    return {'input_tokens': incoming, 'output_tokens': outgoing,
            'total_tokens': incoming + outgoing if incoming is not None and outgoing is not None else None,
            'cached_tokens': optional(details.get('cache_read', details.get('cached_tokens'))),
            'reasoning_tokens': optional(completion.get('reasoning', completion.get('reasoning_tokens'))),
            'provider_request_id': str(meta.get('request_id') or '')[:200],
            'model': str(meta.get('model_name') or meta.get('model') or os.getenv('GROUP_AGENT_MODEL', ''))[:120],
            'usage_status': 'reported' if incoming is not None and outgoing is not None else 'unknown'}


async def report(emit: Sink, data: dict[str, Any]) -> None:
    """Telemetry failure cannot replace a completed model result or exception."""
    try:
        delivered = await emit('progress', {'supplier_usage': data})
        if delivered is False:
            logging.getLogger(__name__).warning('assistant_usage_not_delivered')
    except Exception as error:
        logging.getLogger(__name__).warning('assistant_usage_failed class=%s', type(error).__name__)


async def invoke(model: Any, messages: list[Any], *, operation: str, timeout: float) -> Any:
    """Try the optional primary once, then use standby within the same deadline."""
    from apps.group_agent_api.app.assistant_model_fallback import primary_client, may_fallback
    primary = primary_client(model)
    if primary is None:
        return await _invoke_once(model, messages, operation=operation, timeout=timeout)
    started = time.monotonic()
    try:
        return await _invoke_once(primary, messages, operation=operation, timeout=min(20, timeout / 2))
    except Exception as error:
        if not may_fallback(error):
            raise
        logging.getLogger(__name__).warning('assistant_primary_fallback class=%s', type(error).__name__)
    remaining = timeout - (time.monotonic() - started)
    if remaining <= 0:
        raise TimeoutError('assistant_model_deadline')
    return await _invoke_once(model, messages, operation=operation, timeout=remaining)


async def _invoke_once(model: Any, messages: list[Any], *, operation: str, timeout: float) -> Any:
    """Journal each actual SDK invocation; failed calls retain unknown usage."""
    import asyncio
    emit = sink.get()
    if emit is None:
        return await asyncio.wait_for(model.ainvoke(messages), timeout)
    call_id = str(uuid.uuid4())
    base = {'call_id': call_id, 'operation': operation, 'provider': os.getenv('GROUP_AGENT_PROVIDER', '')[:40],
            'model': str(getattr(model, 'model_name', None) or os.getenv('GROUP_AGENT_MODEL', ''))[:120]}
    await report(emit, {**base, 'status': 'started', 'usage_status': 'unknown'})
    started = time.monotonic()
    try:
        output = await asyncio.wait_for(model.ainvoke(messages), timeout)
    except BaseException as error:
        from openai import APIStatusError
        rejected = isinstance(error, APIStatusError) and error.status_code in {401, 403, 404, 429}
        await report(emit, {**base, 'status': 'failed', 'usage_status': 'not_applicable' if rejected else 'unknown',
                    'duration_ms': int((time.monotonic() - started) * 1000)})
        raise
    try:
        usage = counters(output)
    except Exception as error:
        logging.getLogger(__name__).warning('assistant_usage_parse_failed class=%s', type(error).__name__)
        usage = {'usage_status': 'unknown'}
    await report(emit, {**base, **usage, 'status': 'completed',
                'duration_ms': int((time.monotonic() - started) * 1000)})
    return output


async def tool(name: str, *, status: str = "completed") -> None:
    """Record tool execution separately from supplier token usage."""
    emit = sink.get()
    if emit:
        await report(emit, {'call_id': str(uuid.uuid4()), 'operation': name,
                   'provider': 'internal', 'model': '', 'status': status, 'usage_status': 'not_applicable'})


class UsageReceipt:
    """Bounded metadata receipt carried by the durable final callback.

    Independent progress delivery stays best effort. An incomplete receipt must
    never be mistaken for a complete zero-cost invocation set.
    """
    def __init__(self, forward: Sink) -> None:
        self.forward = forward
        self.calls: dict[str, dict[str, Any]] = {}
        self.complete = True

    async def emit(self, event: str, payload: dict[str, Any]) -> bool | None:
        data = payload.get('supplier_usage')
        if event == 'progress' and isinstance(data, dict):
            call_id = data.get('call_id')
            if isinstance(call_id, str) and (call_id in self.calls or len(self.calls) < 16):
                self.calls[call_id] = dict(data)
            else:
                self.complete = False
        return await self.forward(event, payload)

    def manifest(self) -> list[dict[str, Any]]:
        return list(self.calls.values()) if self.complete else []
