"""Model creation utilities for Wu Tanchang API."""

from __future__ import annotations

import logging

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from apps.wu_tanchang_api.config import resolve_model_config
from apps.wu_tanchang_api.agent_factory.utils import mask_sensitive_value

_logger = logging.getLogger("uvicorn.error")


def create_model(
    *,
    provider: str = "qwen",
    model_name_suffix: str = "MAIN_AGENT_MODEL",
    log_prefix: str = "[ModelConfig]",
    max_tokens: int | None = None,
    model_name_override: str | None = None,
) -> ChatOpenAI | ChatAnthropic:
    """Create a configured chat model instance.

    Args:
        provider: Model provider key from `config.json`.
        model_name_suffix: Env var suffix for model name.
        log_prefix: Log message prefix.
        max_tokens: Optional output token cap override.
        model_name_override: Explicit model name; bypasses env-var resolution.

    Returns:
        Configured LangChain chat model.
    """
    model_config = resolve_model_config(
        provider=provider,
        model_name_suffix=model_name_suffix,
        model_name_override=model_name_override,
    )

    _logger.info(
        "%s provider=%s api_type=%s model=%s",
        log_prefix,
        model_config.provider,
        model_config.api_type,
        model_config.model,
    )

    model_kwargs: dict[str, object] = {
        "model": model_config.model,
        "max_tokens": max_tokens or model_config.max_tokens,
        "timeout": model_config.timeout_s,
    }
    if model_config.temperature is not None:
        model_kwargs["temperature"] = model_config.temperature
    if model_config.base_url:
        model_kwargs["base_url"] = model_config.base_url
    if model_config.api_key:
        model_kwargs["api_key"] = model_config.api_key

    if model_config.api_type == "openai-compatible":
        model = ChatOpenAI(**model_kwargs)
    elif model_config.api_type == "anthropic-compatible":
        model = ChatAnthropic(**model_kwargs)
    else:
        msg = f"Unsupported API type for provider `{provider}`: {model_config.api_type}"
        raise ValueError(msg)

    if not hasattr(model, "profile") or model.profile is None:
        model.profile = {}
    if not isinstance(model.profile, dict):
        model.profile = {}
    model.profile["max_input_tokens"] = model_config.max_input_tokens

    _logger.info(
        "%s api_key=%s max_input_tokens=%s",
        log_prefix,
        mask_sensitive_value(model_config.api_key),
        model_config.max_input_tokens,
    )
    return model
