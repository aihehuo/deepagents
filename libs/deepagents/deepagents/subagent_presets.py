"""Helpers for defining subagent specs in a reusable, extensible way.

We keep these as plain dict specs (matching `SubAgent` TypedDict) so callers can pass
them directly into `create_deep_agent(subagents=[...])`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

from langchain_core.tools import BaseTool

from deepagents.model_config import parse_model_config


def _load_subagent_prompt(subagent_name: str, fallback: str) -> str:
    """Load subagent system prompt from markdown file, with fallback to default.
    
    Args:
        subagent_name: Name of the subagent (e.g., "coder", "aihehuo")
        fallback: Default prompt to use if file cannot be loaded
        
    Returns:
        System prompt string, either from file or fallback
    """
    try:
        # Get the directory where this file is located
        current_dir = Path(__file__).parent
        prompt_file = current_dir / "subagent_prompts" / f"{subagent_name}.md"
        
        if prompt_file.exists():
            content = prompt_file.read_text(encoding="utf-8")
            # Remove markdown header if present (lines starting with #)
            lines = content.split("\n")
            # Skip lines that are just markdown headers
            prompt_lines = []
            skip_header = True
            for line in lines:
                if skip_header and line.strip().startswith("#"):
                    continue
                if skip_header and line.strip() == "":
                    continue
                skip_header = False
                prompt_lines.append(line)
            
            prompt = "\n".join(prompt_lines).strip()
            if prompt:
                return prompt
    except Exception:  # noqa: BLE001
        # If anything goes wrong, fall back to default
        pass
    
    return fallback


def build_coder_subagent_from_env(
    *,
    tools: Sequence[BaseTool | Any] | None,
    name: str = "coder",
    provider: str | None = None,
) -> dict[str, Any] | None:
    """Create a single coder subagent spec from environment variables.

    Uses the same provider config as the main agent, but with CODER_SUBAGENT_MODEL for model name.

    Design goals:
    - Default to "off" unless API key + model are available.
    - Keep it provider-agnostic (qwen=openai-compatible, deepseek=anthropic-compatible).
    - Uses same provider config as main agent (same base_url, api_key, etc.), different model name.

    Env vars:
    - Uses provider-specific vars: [PROVIDER]_BASE_URL, [PROVIDER]_API_KEY (e.g., QWEN_BASE_URL, QWEN_API_KEY)
    - Model name: CODER_SUBAGENT_MODEL (e.g., "qwen-coder-plus")
    - Shared config: MODEL_API_MAX_TOKENS, MODEL_API_TEMPERATURE, MODEL_API_TIMEOUT_S

    Args:
        tools: Tools to pass to the subagent.
        name: Name of the subagent.
        provider: Provider to use (e.g., "qwen", "deepseek"). If None, auto-detects from main agent.
    """
    model_config = parse_model_config(
        provider=provider,
        model_name_suffix="CODER_SUBAGENT_MODEL",
        default_provider="qwen",
    )

    # If not configured, don't enable the coder subagent.
    if not model_config.api_key or not model_config.model:
        return None

    # Use config values, with special default temperature for coder (0.2)
    temperature = model_config.temperature if model_config.temperature is not None else 0.2
    timeout_s = model_config.timeout_s
    max_tokens = model_config.max_tokens
    provider = model_config.provider

    if provider == "deepseek":
        from langchain_anthropic import ChatAnthropic  # noqa: WPS433

        coder_model = ChatAnthropic(
            model=model_config.model,
            max_tokens=max_tokens,
            timeout=timeout_s,
            temperature=temperature,
            base_url=model_config.base_url,
            api_key=model_config.api_key,
        )
    else:
        # Default: qwen / OpenAI-compatible mode.
        # Lazy import to avoid sandbox/CI import-time side effects when coder is disabled.
        from langchain_openai import ChatOpenAI  # noqa: WPS433

        kwargs: dict[str, object] = {
            "model": model_config.model,
            "max_tokens": max_tokens,
            "timeout": timeout_s,
            "temperature": temperature,
        }
        if model_config.base_url:
            kwargs["base_url"] = model_config.base_url
        if model_config.api_key:
            kwargs["api_key"] = model_config.api_key

        coder_model = ChatOpenAI(**kwargs)

    # Load system prompt from file, with fallback to default
    default_coder_prompt = """You are a coding-focused subagent.

You specialize in:
- Writing and editing code in existing repositories
- Creating clean, correct HTML/CSS/JS when asked (HTML is treated as code)
- Making minimal, well-scoped changes and explaining them clearly

Operating rules:
- Prefer making concrete file edits using available file tools.
- If multiple files are involved, be explicit about which ones you changed and why.
- If you are unsure, ask for the missing detail *once* with the smallest set of clarifying questions.
"""
    system_prompt = _load_subagent_prompt("coder", default_coder_prompt)

    return {
        "name": name,
        "description": "Use for coding tasks (including HTML/CSS/JS), repo edits, debugging, and implementation work.",
        "system_prompt": system_prompt,
        # Important: pass through the main agent's tools so the coder subagent can edit files.
        "tools": list(tools or []),
        "model": coder_model,
    }


def build_aihehuo_subagent_from_env(
    *,
    tools: Sequence[BaseTool | Any] | None,
    name: str = "aihehuo",
    provider: str | None = None,
) -> dict[str, Any] | None:
    """Create an AI He Huo (爱合伙) search subagent spec from environment variables.
    
    This subagent is specialized for searching the AI He Huo platform to find:
    - Co-founders and business partners
    - Investors
    - Domain experts
    - Related business ideas and projects
    
    Uses the same provider config as the main agent, but with AIHEHUO_SUBAGENT_MODEL for model name.
    
    Design goals:
    - Default to "off" unless API key + model are available.
    - Keep it provider-agnostic (qwen=openai-compatible, deepseek=anthropic-compatible).
    - Equipped with AihehuoMiddleware for search capabilities.
    - Uses same provider config as main agent (same base_url, api_key, etc.), different model name.
    
    Env vars:
    - Uses provider-specific vars: [PROVIDER]_BASE_URL, [PROVIDER]_API_KEY (e.g., QWEN_BASE_URL, QWEN_API_KEY)
    - Model name: AIHEHUO_SUBAGENT_MODEL (e.g., "qwen-aihehuo-plus")
    - Shared config: MODEL_API_MAX_TOKENS, MODEL_API_TEMPERATURE, MODEL_API_TIMEOUT_S
    - AIHEHUO_API_KEY (required for search functionality)
    - AIHEHUO_API_BASE (optional, defaults to https://new-api.aihehuo.com)
    
    Args:
        tools: Tools to pass to the subagent.
        name: Name of the subagent.
        provider: Provider to use (e.g., "qwen", "deepseek"). If None, auto-detects from main agent.
    """
    model_config = parse_model_config(
        provider=provider,
        model_name_suffix="AIHEHUO_SUBAGENT_MODEL",
        default_provider="qwen",
    )
    
    # If not configured, don't enable the AI He Huo subagent.
    if not model_config.api_key or not model_config.model:
        return None
    
    # Check for AI He Huo API key (required for search functionality)
    aihehuo_api_key = os.environ.get("AIHEHUO_API_KEY")
    if not aihehuo_api_key:
        # Still create the subagent, but it won't be able to search without the API key
        # The middleware will handle the error gracefully
        pass
    
    # Use config values, with special default temperature for aihehuo (0.7)
    temperature = model_config.temperature if model_config.temperature is not None else 0.7
    timeout_s = model_config.timeout_s
    max_tokens = model_config.max_tokens
    provider = model_config.provider
    
    if provider == "deepseek":
        from langchain_anthropic import ChatAnthropic  # noqa: WPS433
        
        aihehuo_model = ChatAnthropic(
            model=model_config.model,
            max_tokens=max_tokens,
            timeout=timeout_s,
            temperature=temperature,
            base_url=model_config.base_url,
            api_key=model_config.api_key,
        )
    else:
        # Default: qwen / OpenAI-compatible mode.
        # Lazy import to avoid sandbox/CI import-time side effects when aihehuo is disabled.
        from langchain_openai import ChatOpenAI  # noqa: WPS433
        
        kwargs: dict[str, object] = {
            "model": model_config.model,
            "max_tokens": max_tokens,
            "timeout": timeout_s,
            "temperature": temperature,
        }
        if model_config.base_url:
            kwargs["base_url"] = model_config.base_url
        if model_config.api_key:
            kwargs["api_key"] = model_config.api_key
        
        aihehuo_model = ChatOpenAI(**kwargs)
    
    # Load system prompt from file, with fallback to default
    default_aihehuo_prompt = """You are an AI He Huo (爱合伙) search specialist subagent.

You specialize in:
- Searching for co-founders, business partners, and team members on the AI He Huo platform
- Finding investors interested in specific industries, technologies, or business models
- Discovering domain experts with relevant experience
- Exploring related business ideas and projects
- Matching entrepreneurs with potential partners based on business needs

Operating rules:
- Use aihehuo_search_members to find people (co-founders, investors, experts)
- Use aihehuo_search_ideas to find related business ideas and projects
- Create multiple targeted searches for different roles/needs
- Use natural language queries (full sentences, not just keywords)
- Query must be longer than 5 characters for member searches
- Use the investor parameter when searching specifically for investors
- Synthesize results and provide clear recommendations
- Be specific and targeted in your search queries

**Report Writing Requirements:**
- When writing reports or summaries, use Chinese (中文) as the language
- For each candidate you recommend, you MUST include their profile page link/URL if it's available in the search results
- Profile links are essential - always extract and include them from the search response data
"""
    system_prompt = _load_subagent_prompt("aihehuo", default_aihehuo_prompt)
    
    # Import middleware
    from deepagents.middleware.aihehuo import AihehuoMiddleware
    from deepagents.middleware.datetime import DateTimeMiddleware
    
    # Build middleware list - include AihehuoMiddleware for search capabilities
    # and DateTimeMiddleware for accurate timestamps in reports
    subagent_middleware = [
        DateTimeMiddleware(),  # Provides get_current_datetime tool for accurate timestamps
        AihehuoMiddleware(),
    ]
    
    return {
        "name": name,
        "description": "Use for searching the AI He Huo (爱合伙) platform to find co-founders, investors, partners, and related business ideas. Specialized in matching entrepreneurs with potential collaborators.",
        "system_prompt": system_prompt,
        # Pass through tools from main agent (filesystem tools, etc.)
        "tools": list(tools or []),
        "model": aihehuo_model,
        "middleware": subagent_middleware,
    }


