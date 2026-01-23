"""Model creation utilities for agent factory."""

from __future__ import annotations

import logging
import os

from deepagents.model_config import parse_model_config
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from apps.business_cofounder_api.agent_factory.utils import mask_sensitive_value, mask_url

# Use uvicorn's configured logger so output reliably shows up in the terminal.
_logger = logging.getLogger("uvicorn.error")


def create_model(
    *,
    provider: str = "qwen",
    model_name_suffix: str = "MAIN_AGENT_MODEL",
    log_prefix: str = "[ModelConfig]",
    set_max_input_tokens: bool = True,
) -> ChatOpenAI | ChatAnthropic:
    """Create a model instance based on provider configuration.
    
    Args:
        provider: Model provider to use ("qwen" or "deepseek", default: "qwen")
        model_name_suffix: Suffix for model name env var (e.g., "MAIN_AGENT_MODEL" or "EXPERT_AGENT_MODEL")
        log_prefix: Prefix for log messages
        set_max_input_tokens: Whether to set max_input_tokens in model profile (for summarization)
    
    Returns:
        Configured model instance (ChatOpenAI or ChatAnthropic)
    """
    # Model configuration using new provider-specific design:
    # - supported_model_providers: comma-separated list (e.g., "deepseek,qwen")
    # - Provider-specific: [PROVIDER]_BASE_URL, [PROVIDER]_API_KEY (e.g., QWEN_BASE_URL, DEEPSEEK_API_KEY)
    # - Model name: [PROVIDER]_[SUFFIX] (e.g., QWEN_MAIN_AGENT_MODEL="qwen-plus")
    # - Shared config: MODEL_API_MAX_TOKENS, MODEL_API_TEMPERATURE, MODEL_API_TIMEOUT_S
    
    model_config = parse_model_config(
        provider=provider,
        model_name_suffix=model_name_suffix,
        default_provider=provider,
    )

    # Log model configuration during initialization
    _logger.info("%s Model provider configuration:", log_prefix)
    _logger.info("  Provider: %s", model_config.provider)
    _logger.info("  Model: %s", model_config.model)
    _logger.info("  Base URL: %s", mask_url(model_config.base_url))
    _logger.info("  API Key: %s", mask_sensitive_value(model_config.api_key))
    _logger.info("  Max Tokens: %s", model_config.max_tokens)
    _logger.info("  Timeout: %ss", model_config.timeout_s)
    if model_config.temperature is not None:
        _logger.info("  Temperature: %s", model_config.temperature)

    # Create model based on provider
    if model_config.provider == "qwen":
        from langchain_openai import ChatOpenAI  # lazy import (avoid import-time side effects in tests)
        
        model_kwargs: dict[str, object] = {
            "model": model_config.model,
            "max_tokens": model_config.max_tokens,
            "timeout": model_config.timeout_s,
        }
        if model_config.temperature is not None:
            model_kwargs["temperature"] = model_config.temperature
        if model_config.base_url:
            model_kwargs["base_url"] = model_config.base_url
        if model_config.api_key:
            model_kwargs["api_key"] = model_config.api_key
        
        model = ChatOpenAI(**model_kwargs)
    else:
        # DeepSeek / Anthropic-compatible proxy
        model_kwargs: dict[str, object] = {
            "model": model_config.model,
            "max_tokens": model_config.max_tokens,
            "timeout": model_config.timeout_s,
        }
        if model_config.temperature is not None:
            model_kwargs["temperature"] = model_config.temperature
        if model_config.base_url:
            model_kwargs["base_url"] = model_config.base_url
        if model_config.api_key:
            model_kwargs["api_key"] = model_config.api_key
        
        model = ChatAnthropic(**model_kwargs)

    # Set model profile with max_input_tokens to enable fraction-based summarization trigger
    # This allows SummarizationMiddleware to use fraction-based triggers (85% of max) instead of
    # the hardcoded 170000 token fallback.
    # Use provider-specific env vars: DEEPSEEK_MAX_TOKENS or QWEN_MAX_TOKENS
    # Note: This should be the actual model context limit (e.g., 131072), not the output max_tokens
    if set_max_input_tokens:
        provider_upper = model_config.provider.upper()
        max_input_tokens_env = os.environ.get(f"{provider_upper}_MAX_TOKENS")
        if max_input_tokens_env:
            max_input_tokens = int(max_input_tokens_env)
        else:
            # Fallback to 131072 if not set (common limit for many models like Qwen/DeepSeek)
            # This is the context window size, not the output max_tokens
            # Note: The error message shows the actual limit is 131072, so use that if env var not set
            max_input_tokens = 131072
            _logger.warning(
                "  Max Input Tokens not set via %s_MAX_TOKENS, defaulting to %d (set this to your model's actual context limit)",
                provider_upper,
                max_input_tokens,
            )
        
        if not hasattr(model, "profile") or model.profile is None:
            model.profile = {}
        if not isinstance(model.profile, dict):
            model.profile = {}
        model.profile["max_input_tokens"] = max_input_tokens
        trigger_threshold = int(max_input_tokens * 0.85)
        _logger.info("  Max Input Tokens (for summarization): %s (trigger at 85%% = %d tokens)", max_input_tokens, trigger_threshold)
        
        # Verify the profile was set correctly (for debugging)
        if model.profile.get("max_input_tokens") != max_input_tokens:
            _logger.error("  ERROR: Failed to set max_input_tokens in model profile!")
        else:
            _logger.info("  ✓ Model profile max_input_tokens verified: %s", model.profile.get("max_input_tokens"))
    else:
        # Still set a basic max_input_tokens for facilitator (simpler, no summarization)
        provider_upper = model_config.provider.upper()
        max_input_tokens_env = os.environ.get(f"{provider_upper}_MAX_TOKENS")
        if max_input_tokens_env:
            max_input_tokens = int(max_input_tokens_env)
        else:
            max_input_tokens = 131072
            _logger.warning(
                "  Max Input Tokens not set via %s_MAX_TOKENS, defaulting to %d",
                provider_upper,
                max_input_tokens,
            )
        
        if not hasattr(model, "profile") or model.profile is None:
            model.profile = {}
        if not isinstance(model.profile, dict):
            model.profile = {}
        model.profile["max_input_tokens"] = max_input_tokens
        _logger.info("  Max Input Tokens: %s", max_input_tokens)

    return model
