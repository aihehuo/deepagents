"""Centralized model configuration parsing from environment variables.

This module provides a single source of truth for parsing model configuration
to ensure consistency across the codebase.

New design:
- Provider-specific env vars: [PROVIDER]_BASE_URL, [PROVIDER]_API_KEY (e.g., QWEN_BASE_URL, DEEPSEEK_API_KEY)
- Model name vars: [PROVIDER]_MAIN_AGENT_MODEL, [PROVIDER]_CODER_SUBAGENT_MODEL, [PROVIDER]_AIHEHUO_SUBAGENT_MODEL
  (e.g., QWEN_MAIN_AGENT_MODEL, DEEPSEEK_MAIN_AGENT_MODEL, QWEN_CODER_SUBAGENT_MODEL)
- Shared config: MODEL_API_MAX_TOKENS, MODEL_API_TIMEOUT_S, MODEL_API_TEMPERATURE
- supported_model_providers: comma-separated list of available providers
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Model configuration parsed from environment variables.

    Attributes:
        provider: Model provider ("deepseek" | "qwen")
        model: Model name (e.g., "deepseek-chat", "qwen-plus")
        base_url: Base URL for the model API endpoint (optional)
        api_key: API key for the model provider (optional)
        max_tokens: Maximum tokens for model responses
        timeout_s: Timeout in seconds for model API calls
        temperature: Temperature setting (optional)
    """

    provider: str
    model: str
    base_url: str | None
    api_key: str | None
    max_tokens: int
    timeout_s: float
    temperature: float | None


def _get_env_str(name: str, default: str | None = None) -> str | None:
    """Get string environment variable, returning None if not set or empty."""
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip()


def _get_env_int(name: str, default: int) -> int:
    """Get integer environment variable with fallback to default."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_env_float(name: str, default: float | None = None) -> float | None:
    """Get float environment variable with fallback to default."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_supported_providers() -> list[str]:
    """Get list of supported model providers from environment variable."""
    providers_str = os.environ.get("SUPPORTED_MODEL_PROVIDERS", "")
    if not providers_str:
        return []
    # Parse comma-separated list, strip whitespace, convert to lowercase
    providers = [p.strip().lower() for p in providers_str.split(",") if p.strip()]
    return providers


def _normalize_provider_name(provider: str) -> str:
    """Normalize provider name to uppercase for env var lookup."""
    return provider.strip().upper()


def parse_model_config(
    *,
    provider: str | None = None,
    model_name_suffix: str = "MAIN_AGENT_MODEL",
    default_provider: str = "deepseek",
) -> ModelConfig:
    """Parse model configuration from environment variables.

    New design: Uses provider-specific env vars and provider-specific model name variables.
    - Provider-specific: [PROVIDER]_BASE_URL, [PROVIDER]_API_KEY (e.g., QWEN_BASE_URL, DEEPSEEK_API_KEY)
    - Model name: [PROVIDER]_MAIN_AGENT_MODEL, [PROVIDER]_CODER_SUBAGENT_MODEL, [PROVIDER]_AIHEHUO_SUBAGENT_MODEL
      (e.g., QWEN_MAIN_AGENT_MODEL, DEEPSEEK_MAIN_AGENT_MODEL, QWEN_CODER_SUBAGENT_MODEL)
    - Shared config: MODEL_API_MAX_TOKENS, MODEL_API_TIMEOUT_S, MODEL_API_TEMPERATURE

    Args:
        provider: Explicit provider to use (e.g., "qwen", "deepseek"). If None, tries to
                  infer from supported_model_providers or falls back to default_provider.
        model_name_suffix: Suffix for the model name environment variable (default: "MAIN_AGENT_MODEL").
                          The full variable name will be [PROVIDER]_[model_name_suffix].
                          For subagents, use "CODER_SUBAGENT_MODEL", "AIHEHUO_SUBAGENT_MODEL", etc.
        default_provider: Default provider if provider is not specified and cannot be inferred.

    Returns:
        ModelConfig instance with parsed values.

    Examples:
        # Main agent with qwen provider (looks for QWEN_MAIN_AGENT_MODEL)
        config = parse_model_config(provider="qwen", model_name_suffix="MAIN_AGENT_MODEL")

        # Coder subagent (looks for QWEN_CODER_SUBAGENT_MODEL)
        config = parse_model_config(provider="qwen", model_name_suffix="CODER_SUBAGENT_MODEL")

        # Auto-detect provider from supported_model_providers
        config = parse_model_config(model_name_suffix="MAIN_AGENT_MODEL")
    """
    # Determine provider
    if provider is None:
        # Try to infer from supported_model_providers
        supported = _get_supported_providers()
        if len(supported) == 1:
            provider = supported[0]
        elif len(supported) > 1:
            # Multiple providers supported, use default
            provider = default_provider
        else:
            # No supported providers listed, use default
            provider = default_provider
    provider = provider.strip().lower()
    provider_upper = _normalize_provider_name(provider)

    # Parse provider-specific config: [PROVIDER]_BASE_URL, [PROVIDER]_API_KEY
    base_url = _get_env_str(f"{provider_upper}_BASE_URL")
    api_key = _get_env_str(f"{provider_upper}_API_KEY")

    # Parse model name from provider-specific variable: [PROVIDER]_[model_name_suffix]
    model_name_var = f"{provider_upper}_{model_name_suffix}"
    model = _get_env_str(model_name_var)

    # For subagents, fall back to main agent model name if not specified
    if not model and model_name_suffix != "MAIN_AGENT_MODEL":
        model = _get_env_str(f"{provider_upper}_MAIN_AGENT_MODEL")

    # Set default model name based on provider if still not specified
    if not model:
        if provider == "qwen":
            model = "qwen-plus"
        elif provider == "deepseek":
            model = "deepseek-chat"
        else:
            # Generic fallback
            model = f"{provider}-chat"

    # Parse shared config variables (can be overridden with provider-specific if needed)
    max_tokens = _get_env_int(f"{provider_upper}_MAX_TOKENS", 20000)
    timeout_s = _get_env_float(f"{provider_upper}_TIMEOUT_S", 180.0)
    temperature = _get_env_float(f"{provider_upper}_TEMPERATURE", 0.2)

    return ModelConfig(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        max_tokens=max_tokens,
        timeout_s=timeout_s,
        temperature=temperature,
    )


def get_model_base_url_and_key(
    *,
    provider: str | None = None,
    default_provider: str = "deepseek",
) -> tuple[str | None, str | None]:
    """Get model base URL and API key from environment variables.

    Convenience function for cases where only base_url and api_key are needed.

    Args:
        provider: Explicit provider to use (e.g., "qwen", "deepseek").
                  If None, uses default_provider.
        default_provider: Default provider if provider is not specified.

    Returns:
        Tuple of (base_url, api_key), both may be None.

    Examples:
        # Get QWEN_BASE_URL and QWEN_API_KEY
        base_url, api_key = get_model_base_url_and_key(provider="qwen")

        # Get DEEPSEEK_BASE_URL and DEEPSEEK_API_KEY
        base_url, api_key = get_model_base_url_and_key(provider="deepseek")
    """
    config = parse_model_config(provider=provider, default_provider=default_provider)
    return config.base_url, config.api_key

