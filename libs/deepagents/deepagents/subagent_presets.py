"""Helpers for defining subagent specs in a reusable, extensible way.

We keep these as plain dict specs (matching `SubAgent` TypedDict) so callers can pass
them directly into `create_deep_agent(subagents=[...])`.
"""

from __future__ import annotations

import os
from typing import Any, Sequence

from langchain_core.tools import BaseTool


def _get_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def build_coder_subagent_from_env(
    *,
    tools: Sequence[BaseTool | Any] | None,
    name: str = "coder",
) -> dict[str, Any] | None:
    """Create a single coder subagent spec from environment variables.

    Design goals:
    - Default to "off" unless API key + base URL are available.
    - Keep it provider-agnostic (qwen=openai-compatible, deepseek=anthropic-compatible).
    - Easy to extend (add more builders for more subagents later).

    Env vars (recommended, coder-specific):
    - CODER_MODEL_API_PROVIDER: qwen | deepseek
    - CODER_MODEL_API_KEY
    - CODER_MODEL_BASE_URL (optional depending on provider; recommended for qwen-compatible endpoints)
    - CODER_MODEL_NAME
    - CODER_MODEL_API_MAX_TOKENS (optional)
    - CODER_MODEL_API_TEMPERATURE (optional)
    - CODER_MODEL_API_TIMEOUT_S (optional)

    Defaults/fallbacks:
    - If a CODER_* variable is missing, we fall back to the main MODEL_* variable (if set).
    """

    provider = (
        os.environ.get("CODER_MODEL_API_PROVIDER")
        or os.environ.get("MODEL_API_PROVIDER")
        or "qwen"
    ).strip().lower()

    api_key = os.environ.get("CODER_MODEL_API_KEY") or os.environ.get("MODEL_API_KEY")
    base_url = os.environ.get("CODER_MODEL_BASE_URL") or os.environ.get("MODEL_BASE_URL")
    model_name = os.environ.get("CODER_MODEL_NAME") or os.environ.get("MODEL_NAME")

    # If not configured, don't enable the coder subagent.
    if not api_key or not model_name:
        return None

    temperature = _get_env_float("CODER_MODEL_API_TEMPERATURE", _get_env_float("MODEL_API_TEMPERATURE", 0.2))
    timeout_s = _get_env_float("CODER_MODEL_API_TIMEOUT_S", _get_env_float("MODEL_API_TIMEOUT_S", 180.0))
    max_tokens = _get_env_int("CODER_MODEL_API_MAX_TOKENS", _get_env_int("MODEL_API_MAX_TOKENS", 20000))

    if provider == "deepseek":
        from langchain_anthropic import ChatAnthropic  # noqa: WPS433

        coder_model = ChatAnthropic(
            model=model_name,
            max_tokens=max_tokens,
            timeout=timeout_s,
            temperature=temperature,
            base_url=base_url,
            api_key=api_key,
        )
    else:
        # Default: qwen / OpenAI-compatible mode.
        # Lazy import to avoid sandbox/CI import-time side effects when coder is disabled.
        from langchain_openai import ChatOpenAI  # noqa: WPS433

        kwargs: dict[str, object] = {
            "model": model_name,
            "max_tokens": max_tokens,
            "timeout": timeout_s,
            "temperature": temperature,
        }
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key

        coder_model = ChatOpenAI(**kwargs)

    system_prompt = """You are a coding-focused subagent.

You specialize in:
- Writing and editing code in existing repositories
- Creating clean, correct HTML/CSS/JS when asked (HTML is treated as code)
- Making minimal, well-scoped changes and explaining them clearly

Operating rules:
- Prefer making concrete file edits using available file tools.
- If multiple files are involved, be explicit about which ones you changed and why.
- If you are unsure, ask for the missing detail *once* with the smallest set of clarifying questions.
"""

    return {
        "name": name,
        "description": "Use for coding tasks (including HTML/CSS/JS), repo edits, debugging, and implementation work.",
        "system_prompt": system_prompt,
        # Important: pass through the main agent's tools so the coder subagent can edit files.
        "tools": list(tools or []),
        "model": coder_model,
    }


