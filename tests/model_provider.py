from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pytest
from langchain_core.language_models import BaseChatModel


@dataclass(frozen=True)
class ModelConfig:
    provider: str  # "deepseek" (anthropic-compatible) | "qwen" (openai-compatible)
    model: str
    base_url: str | None
    api_key: str | None
    max_tokens: int
    timeout_s: float
    temperature: float | None


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

    Preferred (generic) env vars:
    - MODEL_API_PROVIDER: "qwen" | "deepseek"
    - MODEL_API_KEY
    - MODEL_BASE_URL (optional)
    - MODEL_NAME
    - MODEL_API_MAX_TOKENS (optional)
    """
    provider = (
        os.environ.get("MODEL_API_PROVIDER")
        or "deepseek"
    ).strip().lower()

    max_tokens = int(
        os.environ.get("MODEL_API_MAX_TOKENS")
        or "20000"
    )
    timeout_s = float(
        os.environ.get("MODEL_API_TIMEOUT_S")
        or "180.0"
    )
    temperature_env = os.environ.get("MODEL_API_TEMPERATURE") or os.environ.get(
        "BC_API_TEMPERATURE"
    )
    temperature = float(temperature_env) if temperature_env is not None else None

    if provider == "qwen":
        # Prefer generic env vars, then fall back to historical ones.
        base_url = os.environ.get("MODEL_BASE_URL")
        api_key = os.environ.get("MODEL_API_KEY")
        model = (
            os.environ.get("MODEL_NAME")
            or "qwen-plus"
        )

        # Optional local fallback file (developer convenience)
        if (not base_url or not api_key) and (repo_root / ".env.qwen").exists():
            try:
                env = _read_dotenv_exports(repo_root / ".env.qwen")
                base_url = base_url or env.get("MODEL_BASE_URL")
                api_key = api_key or env.get("MODEL_API_KEY")
                model = env.get("MODEL_NAME") or model
            except PermissionError:
                # Some sandboxes block dotfiles; fall through to skip below.
                pass

        if not api_key:
            pytest.skip(
                "Qwen requested but MODEL_API_KEY is not set. "
                "Set MODEL_API_PROVIDER=qwen and MODEL_API_KEY (and optionally MODEL_BASE_URL, MODEL_NAME) to run."
            )

        return ModelConfig(
            provider="qwen",
            model=model,
            base_url=base_url,
            api_key=api_key,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
            temperature=temperature,
        )

    # Default: DeepSeek/Anthropic-compatible
    base_url = os.environ.get("MODEL_BASE_URL")
    api_key = os.environ.get("MODEL_API_KEY")
    model = (
        os.environ.get("MODEL_NAME")
        or "deepseek-chat"
    )

    env_file = repo_root / ".env.deepseek"
    if (not base_url or not api_key) and env_file.exists():
        try:
            env = _read_dotenv_exports(env_file)
            base_url = base_url or env.get("MODEL_BASE_URL")
            api_key = api_key or env.get("MODEL_API_KEY") 
            model = env.get("MODEL_NAME")
        except PermissionError:
            # Some sandboxes block dotfiles; fall through to skip below.
            pass

    if not api_key:
        pytest.skip(
            "DeepSeek (Anthropic-compatible) requested but MODEL_API_KEY is not set. "
            "Set MODEL_API_PROVIDER=deepseek and MODEL_API_KEY (and optionally MODEL_BASE_URL, MODEL_NAME) to run."
        )

    return ModelConfig(
        provider="deepseek",
        model=model,
        base_url=base_url,
        api_key=api_key,
        max_tokens=max_tokens,
        timeout_s=timeout_s,
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


