"""Opt-in primary endpoint for APP assistant calls; existing client is standby."""
from __future__ import annotations

import logging
import os
from typing import Any
from urllib.parse import urlparse


def primary_client(standby: Any) -> Any | None:
    """Build an isolated client without changing the process-wide provider key."""
    if os.getenv('AI_ASSISTANT_PRIMARY_ENABLED', '0') != '1' or os.getenv('GROUP_AGENT_PROVIDER', '').lower() not in {'qwen', 'dashscope'}:
        return None
    from langchain_openai import ChatOpenAI
    if not isinstance(standby, ChatOpenAI):
        return None
    key = os.getenv('AI_ASSISTANT_PRIMARY_API_KEY', '').strip()
    url = os.getenv('AI_ASSISTANT_PRIMARY_BASE_URL', '').strip()
    model = os.getenv('AI_ASSISTANT_PRIMARY_MODEL', '').strip()
    parsed = urlparse(url)
    if not key or not model or parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password:
        logging.getLogger(__name__).warning('assistant_primary_configuration_unavailable')
        return None
    return ChatOpenAI(model=model, api_key=key, base_url=url,
                      max_tokens=standby.max_tokens, temperature=None,
                      extra_body={**(standby.extra_body or {}), 'enable_thinking': False}, timeout=20, max_retries=0)


def may_fallback(error: Exception) -> bool:
    """Retry endpoint failures only, never cancellations or content parsing."""
    import asyncio
    from openai import APIConnectionError, APIStatusError
    error = provider_error(error)
    if isinstance(error, (asyncio.TimeoutError, APIConnectionError)):
        return True
    return isinstance(error, APIStatusError) and error.status_code in {400, 401, 403, 404, 408, 422, 429, 500, 502, 503, 504}


def provider_error(error: BaseException) -> BaseException:
    """Unwrap SDK request wrappers without depending on a LangChain version."""
    for _ in range(5):
        if error.__cause__ is None:
            break
        error = error.__cause__
    return error
