from __future__ import annotations

import os
from pathlib import Path

import pytest
from deepagents.model_config import ModelConfig, parse_model_config
from langchain_core.language_models import BaseChatModel


_DID_PRINT_MODEL_SELECTION = False


def _maybe_print_model_selection(cfg: ModelConfig) -> None:
    """Print the selected model config once per test process for quick visibility."""
    global _DID_PRINT_MODEL_SELECTION
    if _DID_PRINT_MODEL_SELECTION:
        return
    _DID_PRINT_MODEL_SELECTION = True

    base_url = cfg.base_url or "(default)"
    temperature = cfg.temperature if cfg.temperature is not None else "(default)"
    print(
        "\n[tests] Model selected: "
        f"provider={cfg.provider} model={cfg.model} base_url={base_url} "
        f"max_tokens={cfg.max_tokens} timeout_s={cfg.timeout_s} temperature={temperature}",
        flush=True,
    )


def _read_dotenv_exports(path: Path) -> dict[str, str]:
    """Parse a very small subset of dotenv: lines like `export KEY="VALUE"`."""
    raw = path.read_text(encoding="utf-8")
    env_vars: dict[str, str] = {}
    for line in raw.splitlines():
        line = line.strip()
        if line and not line.startswith("#") and line.startswith("export "):
            key_value = line.replace("export ", "", 1).split("=", 1)
            if len(key_value) == 2:
                key, value = key_value
                env_vars[key] = value.strip().strip('"').strip("'")
    return env_vars


def load_test_model_config(*, repo_root: Path) -> ModelConfig:
    """Load model config for tests.

    Uses the centralized parse_model_config() and adds test-specific features:
    - Fallback to .env.qwen or .env.deepseek files if env vars are missing
    - Fallback to BC_API_TEMPERATURE for temperature
    - pytest.skip() if API key is missing

    Supports both new and old env var patterns:
    - New: [PROVIDER]_BASE_URL, [PROVIDER]_API_KEY, [PROVIDER]_MAIN_AGENT_MODEL
    - Old (for backward compatibility): MODEL_BASE_URL, MODEL_API_KEY, MODEL_NAME
    """
    # Start with centralized config parsing (auto-detect provider)
    config = parse_model_config(
        provider=None,  # Auto-detect from supported_model_providers
        model_name_suffix="MAIN_AGENT_MODEL",
        default_provider="deepseek",
    )

    # Handle BC_API_TEMPERATURE fallback (test-specific)
    if config.temperature is None:
        temperature_env = os.environ.get("BC_API_TEMPERATURE")
        if temperature_env:
            try:
                temperature = float(temperature_env)
            except ValueError:
                temperature = None
        else:
            temperature = None
    else:
        temperature = config.temperature

    # Handle .env file fallback (test-specific developer convenience)
    # Support both new provider-specific vars and old MODEL_* vars in .env files
    base_url = config.base_url
    api_key = config.api_key
    model = config.model
    provider_upper = config.provider.upper()

    if config.provider == "qwen":
        # Optional local fallback file (developer convenience)
        if (not base_url or not api_key) and (repo_root / ".env.qwen").exists():
            try:
                env = _read_dotenv_exports(repo_root / ".env.qwen")
                # Try new provider-specific vars first, then old MODEL_* vars
                base_url = base_url or env.get("QWEN_BASE_URL") or env.get("MODEL_BASE_URL")
                api_key = api_key or env.get("QWEN_API_KEY") or env.get("MODEL_API_KEY")
                model = env.get("QWEN_MAIN_AGENT_MODEL") or env.get("MAIN_AGENT_MODEL") or env.get("MODEL_NAME") or model
            except PermissionError:
                # Some sandboxes block dotfiles; fall through to skip below.
                pass

        if not api_key:
            pytest.skip(
                "Qwen requested but QWEN_API_KEY (or MODEL_API_KEY) is not set. "
                "Set supported_model_providers=qwen and QWEN_API_KEY (and optionally QWEN_BASE_URL, QWEN_MAIN_AGENT_MODEL) to run."
            )
    else:
        # Default: DeepSeek/Anthropic-compatible
        env_file = repo_root / ".env.deepseek"
        if (not base_url or not api_key) and env_file.exists():
            try:
                env = _read_dotenv_exports(env_file)
                # Try new provider-specific vars first, then old MODEL_* vars
                base_url = base_url or env.get("DEEPSEEK_BASE_URL") or env.get("MODEL_BASE_URL")
                api_key = api_key or env.get("DEEPSEEK_API_KEY") or env.get("MODEL_API_KEY")
                model = env.get("DEEPSEEK_MAIN_AGENT_MODEL") or env.get("MAIN_AGENT_MODEL") or env.get("MODEL_NAME") or model
            except PermissionError:
                # Some sandboxes block dotfiles; fall through to skip below.
                pass

        if not api_key:
            pytest.skip(
                "DeepSeek (Anthropic-compatible) requested but DEEPSEEK_API_KEY (or MODEL_API_KEY) is not set. "
                "Set supported_model_providers=deepseek and DEEPSEEK_API_KEY (and optionally DEEPSEEK_BASE_URL, DEEPSEEK_MAIN_AGENT_MODEL) to run."
            )

    # Return updated config with test-specific overrides
    return ModelConfig(
        provider=config.provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        max_tokens=config.max_tokens,
        timeout_s=config.timeout_s,
        temperature=temperature,
    )


def create_test_model(*, cfg: ModelConfig) -> BaseChatModel:
    """Create the chat model based on ModelConfig.

    Qwen uses OpenAI-compatible ChatOpenAI (lazy import to avoid import-time side effects).
    DeepSeek uses ChatAnthropic.
    """
    _maybe_print_model_selection(cfg)
    if cfg.provider == "qwen":
        try:
            from langchain_openai import ChatOpenAI  # lazy import
        except Exception as e:  # pragma: no cover
            pytest.skip(
                f"MODEL_API_PROVIDER=qwen but langchain_openai is not usable in this environment: {type(e).__name__}: {e}"
            )

        kwargs: dict[str, object] = {
            "model": cfg.model,
            "max_tokens": cfg.max_tokens,
            "timeout": cfg.timeout_s,
        }
        if cfg.base_url:
            kwargs["base_url"] = cfg.base_url
        if cfg.api_key:
            kwargs["api_key"] = cfg.api_key
        if cfg.temperature is not None:
            kwargs["temperature"] = cfg.temperature
        return ChatOpenAI(**kwargs)

    from langchain_anthropic import ChatAnthropic

    kwargs2: dict[str, object] = {
        "model": cfg.model,
        "max_tokens": cfg.max_tokens,
        "timeout": cfg.timeout_s,
    }
    if cfg.base_url:
        kwargs2["base_url"] = cfg.base_url
    if cfg.api_key:
        kwargs2["api_key"] = cfg.api_key
    if cfg.temperature is not None:
        kwargs2["temperature"] = cfg.temperature
    return ChatAnthropic(**kwargs2)


