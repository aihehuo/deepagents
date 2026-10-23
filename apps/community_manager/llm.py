"""Sync primary/standby routing and metadata-only metering for background agents."""
from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Iterator, Protocol
from urllib.parse import urlparse

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APIStatusError

class _Model(Protocol):
    def invoke(self, messages: list, **kwargs: object) -> BaseMessage: ...
    def bind_tools(self, tools: list, **kwargs: object) -> _Model: ...


LOGGER = logging.getLogger(__name__)
Reporter = Callable[[dict], None]
_LAST: ContextVar[str | None] = ContextVar("community_llm_last_model", default=None)
_REPORTER: ContextVar[Reporter | None] = ContextVar("community_llm_reporter", default=None)


@contextmanager
def usage_scope(run_id: str, token: str, *, rpc: Callable[..., dict]) -> Iterator[None]:
    """Bind metering to an authenticated execution without passing secrets to models.

    Args:
        run_id: Backend-issued run identity.
        token: Backend-issued execution credential.
        rpc: Existing signed operation transport.
    """
    def report(event: dict) -> None:
        result = rpc("usage", run_id=run_id, token=token, usage=event)
        if isinstance(result, dict) and result.get("recorded") is False:
            raise RuntimeError("usage_not_recorded")

    context = _REPORTER.set(report)
    previous = _LAST.set(None)
    try:
        yield
    finally:
        _REPORTER.reset(context)
        _LAST.reset(previous)


def reported_model() -> str:
    """Return the current scoped model label without mixing worker executions."""
    return _LAST.get() or os.environ.get("COMMUNITY_MANAGER_MODEL", "unknown")


def emit(event: dict) -> None:
    """Send best-effort accounting without replacing a business result.

    Args:
        event: Whitelisted supplier metadata, never messages or provider errors.
    """
    reporter = _REPORTER.get()
    if reporter is None:
        return
    try:
        reporter(event)
    except Exception as error:
        LOGGER.warning("community_llm_usage_failed class=%s", type(error).__name__)


def counters(output: BaseMessage) -> dict:
    """Extract reported counters without estimating unknown tokens.

    Args:
        output: Supplier message; its content is never included in the result.
    """
    meta = getattr(output, "response_metadata", None)
    meta = meta if isinstance(meta, dict) else {}
    usage = getattr(output, "usage_metadata", None)
    usage = usage if isinstance(usage, dict) else meta.get("token_usage", {})
    usage = usage if isinstance(usage, dict) else {}

    def number(value: object) -> int | None:
        return value if type(value) is int and 0 <= value <= 9_000_000_000_000_000 else None

    incoming = number(usage.get("input_tokens", usage.get("prompt_tokens")))
    outgoing = number(usage.get("output_tokens", usage.get("completion_tokens")))
    details = usage.get("input_token_details", usage.get("prompt_tokens_details", {}))
    completion = usage.get("output_token_details", usage.get("completion_tokens_details", {}))
    details = details if isinstance(details, dict) else {}
    completion = completion if isinstance(completion, dict) else {}
    model = meta.get("model_name", meta.get("model"))
    request = meta.get("request_id")
    return {"input_tokens": incoming, "output_tokens": outgoing,
            "total_tokens": incoming + outgoing if incoming is not None and outgoing is not None else None,
            "cached_tokens": number(details.get("cache_read", details.get("cached_tokens"))),
            "reasoning_tokens": number(completion.get("reasoning", completion.get("reasoning_tokens"))),
            "response_model": model[:120] if isinstance(model, str) and model else None,
            "provider_request_id": request[:200] if isinstance(request, str) and request else None,
            "usage_status": "reported" if incoming is not None and outgoing is not None else "unknown"}


def retryable(error: Exception) -> bool:
    """Permit only connection, timeout and fixed supplier HTTP failures.

    Args:
        error: Original supplier exception, never logged as text.
    """
    return isinstance(error, (APIConnectionError, TimeoutError, MissingPrimary)) or (
        isinstance(error, APIStatusError) and error.status_code in
        {400, 401, 403, 404, 408, 422, 429, 500, 502, 503, 504})


class MissingPrimary(RuntimeError):
    """Primary credentials have not been loaded; no supplier call was made."""


class RoutedModel:
    """Preserve sync message/tool contracts while metering every actual attempt."""

    def __init__(self, standby: _Model, primary: _Model | None, *, node: str,
                 standby_model: str, primary_model: str = "auto", timeout: float = 30,
                 state: dict | None = None) -> None:
        self.standby, self.primary = standby, primary
        self.node, self.standby_model, self.primary_model = node, standby_model, primary_model
        self.timeout = timeout
        self.state = state if state is not None else {"last_model": standby_model}

    @property
    def last_model(self) -> str:
        """Return the last successful response model or requested model."""
        return self.state["last_model"]

    def bind_tools(self, tools: list, **kwargs: object) -> RoutedModel:
        """Apply identical tool definitions and tool choice to both endpoints.

        Args:
            tools: Existing business tool definitions.
            kwargs: Existing LangChain tool binding options.
        """
        return RoutedModel(self.standby.bind_tools(tools, **kwargs),
            self.primary.bind_tools(tools, **kwargs) if self.primary is not None else None,
            node=self.node, standby_model=self.standby_model, primary_model=self.primary_model,
            timeout=self.timeout, state=self.state)

    def invoke(self, messages: list, *, usage_node: str | None = None, **kwargs: object) -> BaseMessage:
        """Attempt primary once then standby within the original per-call deadline.

        Args:
            messages: Unchanged business messages, never added to accounting.
            usage_node: Fixed internal suboperation, for example output repair.
            kwargs: Existing model invocation options.
        """
        invocation = str(uuid.uuid4())
        deadline = time.monotonic() + self.timeout
        for attempt, model in enumerate((self.primary, self.standby), start=1):
            role = "primary" if attempt == 1 else "standby"
            name = self.primary_model if attempt == 1 else self.standby_model
            event = {"invocation_id": invocation, "attempt": attempt, "node": usage_node or self.node,
                     "provider": "token_plan" if attempt == 1 else "dashscope",
                     "endpoint_role": role, "requested_model": name}
            emit({**event, "status": "started", "usage_status": "unknown"})
            started = time.monotonic()
            try:
                if model is None:
                    raise MissingPrimary("primary_not_configured")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("model_deadline")
                timeout = min(20, remaining / 2) if attempt == 1 else remaining
                output = model.invoke(messages, **{**kwargs, "timeout": timeout})
            except Exception as error:
                emit({**event, "status": "failed", "usage_status": "unknown",
                      "duration_ms": int((time.monotonic() - started) * 1000),
                      "error_class": type(error).__name__[:160],
                      "http_status": error.status_code if isinstance(error, APIStatusError) else None})
                if attempt == 2 or not retryable(error):
                    raise
                LOGGER.warning("community_llm_primary_fallback class=%s", type(error).__name__)
                continue
            try:
                usage = counters(output)
            except Exception as error:
                LOGGER.warning("community_llm_usage_parse_failed class=%s", type(error).__name__)
                usage = {"usage_status": "unknown"}
            self.state["last_model"] = (usage.get("response_model") or name)[:96]
            if _REPORTER.get() is not None:
                _LAST.set(self.state["last_model"])
            emit({**event, **usage, "status": "completed", "duration_ms": int((time.monotonic() - started) * 1000)})
            return output
        raise RuntimeError("model_routes_exhausted")


def validate_primary() -> None:
    """Fail readiness/deployment before workers start with missing primary setup."""
    key = os.getenv("COMMUNITY_LLM_PRIMARY_API_KEY", "").strip()
    base = os.getenv("COMMUNITY_LLM_PRIMARY_BASE_URL", "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1")
    url = urlparse(base)
    if not key or not os.getenv("COMMUNITY_LLM_PRIMARY_MODEL", "auto").strip():
        raise ValueError("community_primary_unconfigured")
    if url.scheme != "https" or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError("invalid_primary_endpoint")


def routed(standby: _Model, *, node: str, model: str, timeout: float,
           max_tokens: int, response_format: dict | None = None) -> RoutedModel:
    """Configure the isolated Token Plan primary without modifying existing keys.

    Args:
        standby: Existing scoped qwen-plus client.
        node: Fixed accounting node name.
        model: Existing standby model.
        timeout: Existing total per-invocation timeout.
        max_tokens: Existing output limit.
        response_format: Optional structured-output format, preserved on primary.
    """
    key = os.getenv("COMMUNITY_LLM_PRIMARY_API_KEY", "").strip()
    base = os.getenv("COMMUNITY_LLM_PRIMARY_BASE_URL", "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1")
    name = os.getenv("COMMUNITY_LLM_PRIMARY_MODEL", "auto")
    url = urlparse(base)
    primary = None
    if key:
        if url.scheme != "https" or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError("invalid_primary_endpoint")
        primary = ChatOpenAI(model=name, api_key=key, base_url=base, temperature=None,
            timeout=min(20, timeout / 2), max_retries=0, max_tokens=max_tokens,
            extra_body={"enable_thinking": False},
            model_kwargs={"response_format": response_format} if response_format else {})
    return RoutedModel(standby, primary, node=node, standby_model=model, primary_model=name, timeout=timeout)
