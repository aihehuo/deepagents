"""Unauthenticated public conversation, separated from member pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from apps.group_agent_api.app.state import AppState

import asyncio
import logging
import uuid

from fastapi import HTTPException
from langchain_core.messages import AIMessage, HumanMessage

from apps.group_agent_api.agent_factory.anonymous import anonymous_read_scope
from apps.group_agent_api.app.anonymous_config import anonymous_enabled, anonymous_limits
from apps.group_agent_api.app.anonymous_models import AnonymousChatRequest, AnonymousChatResponse

_logger = logging.getLogger("uvicorn.error")


async def anonymous_chat(req: AnonymousChatRequest, state: AppState) -> AnonymousChatResponse:
    """Run one anonymous request using public tools and no private persistence.

    Args:
        req: Validated text-only public request.
        state: Application state with a separately configured public graph.

    Returns:
        Stateless response with write capability fixed to false.

    Raises:
        HTTPException: Entry disabled, unavailable, busy, timed out, or failed.
    """
    if not anonymous_enabled():
        raise HTTPException(status_code=404, detail="anonymous_entry_disabled")
    if state.anonymous_agent is None:
        raise HTTPException(status_code=503, detail="anonymous_agent_unavailable")
    limits = anonymous_limits()
    request_id = uuid.uuid4().hex
    if not state.try_start_anonymous_run(request_id, max_active=limits.max_active):
        raise HTTPException(status_code=429, detail="anonymous_entry_busy", headers={"Retry-After": "5"})
    try:
        with anonymous_read_scope(max_queries=limits.max_queries):
            result = await asyncio.wait_for(
                state.anonymous_agent.ainvoke(
                    {"messages": [HumanMessage(content=req.message)]},
                    {"recursion_limit": 24, "metadata": {"source": "anonymous_public", "can_write_profile": False}},
                ),
                timeout=limits.timeout_s,
            )
        messages = result.get("messages", [])
        replies = [message for message in messages if isinstance(message, AIMessage) and not message.tool_calls]
        reply = _require_reply(replies)
        return AnonymousChatResponse(request_id=request_id, reply=reply)
    except TimeoutError as error:
        raise HTTPException(status_code=504, detail="anonymous_request_timeout") from error
    except HTTPException:
        raise
    except Exception as error:  # do not expose provider errors or credentials
        _logger.warning("Anonymous request failed request_id=%s error_type=%s", request_id, type(error).__name__)
        raise HTTPException(status_code=502, detail="anonymous_request_failed") from error
    finally:
        state.finish_agent_run(request_id, "anonymous")


def _require_reply(replies: list[AIMessage]) -> str:
    if not replies or not replies[-1].text.strip():
        raise HTTPException(status_code=502, detail="anonymous_reply_missing")
    return replies[-1].text
