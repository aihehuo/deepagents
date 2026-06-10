"""Configuration loading for Wu Tanchang API."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WuModelConfig:
    """Resolved model configuration for one Wu Tanchang model client."""

    provider: str
    api_type: str
    model: str
    base_url: str | None
    api_key: str | None
    max_tokens: int
    timeout_s: float
    temperature: float | None
    max_input_tokens: int


@dataclass(frozen=True)
class WuAgentConfig:
    """Resolved configuration for one named agent profile."""

    name: str
    provider: str
    model: str
    max_tokens: int
    workspace: str


@dataclass(frozen=True)
class WuAgentRegistry:
    """Registry of all configured agent profiles."""

    defaults: WuAgentConfig
    agents: dict[str, WuAgentConfig]
    default_name: str


def default_config_path() -> Path:
    """Return the default JSON config path."""
    return Path(__file__).resolve().parent / "config.json"


def default_env_path() -> Path:
    """Return the default dotenv path."""
    return Path(__file__).resolve().parent / ".env"


def _strip_env_value(value: str) -> str:
    """Strip whitespace and one layer of shell-style quotes."""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_env_file(path: Path | None = None) -> None:
    """Load dotenv values without overriding shell-provided environment.

    Args:
        path: Optional dotenv path. Defaults to `WU_API_ENV_FILE`, then app `.env`.
    """
    env_path = path or Path(os.environ.get("WU_API_ENV_FILE", str(default_env_path())))
    if not env_path.exists():
        return

    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = _strip_env_value(value)


def _load_config(path: Path | None = None) -> dict[str, Any]:
    """Load the Wu Tanchang JSON config."""
    config_path = path or Path(os.environ.get("WU_API_CONFIG", str(default_config_path())))
    with config_path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        msg = f"Config root must be an object: {config_path}"
        raise ValueError(msg)
    return data


def _resolve(value: Any) -> Any:
    """Resolve `env:VAR` string references from the process environment."""
    if isinstance(value, str) and value.startswith("env:"):
        return os.environ.get(value[len("env:") :])
    if isinstance(value, dict):
        return {key: _resolve(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve(item) for item in value]
    return value


def _get_str(data: dict[str, Any], key: str, default: str | None = None) -> str | None:
    """Return a string value, treating empty strings as missing."""
    value = data.get(key, default)
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _get_int(data: dict[str, Any], key: str, default: int) -> int:
    """Return an integer value with a default fallback."""
    value = data.get(key)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _get_float(data: dict[str, Any], key: str, default: float | None = None) -> float | None:
    """Return a float value with a default fallback."""
    value = data.get(key)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _role_key(model_name_suffix: str) -> str:
    """Convert a model env suffix to a config role key."""
    role = model_name_suffix.lower()
    if role.endswith("_model"):
        role = role[: -len("_model")] + "_model"
    return role


def get_selected_provider() -> str:
    """Return the configured model provider key."""
    load_env_file()
    data = _resolve(_load_config())
    provider = _get_str(data, "model_provider") or _get_str(data, "default_model_provider")
    if provider:
        return provider
    return "qwen"


def resolve_model_config(
    *,
    provider: str,
    model_name_suffix: str,
    model_name_override: str | None = None,
) -> WuModelConfig:
    """Resolve one model config from `config.json` and environment references.

    Args:
        provider: Provider key under `providers`.
        model_name_suffix: Model role suffix, e.g. `MAIN_AGENT_MODEL`.
        model_name_override: Explicit model name; bypasses env-var resolution when set.

    Returns:
        Resolved model configuration.
    """
    load_env_file()
    data = _resolve(_load_config())
    providers = data.get("providers")
    if not isinstance(providers, dict):
        msg = "Config must define a `providers` object."
        raise ValueError(msg)

    provider_data = providers.get(provider)
    if not isinstance(provider_data, dict):
        msg = f"Unknown Wu Tanchang model provider: {provider}"
        raise ValueError(msg)

    if model_name_override:
        model = model_name_override
    else:
        role_key = _role_key(model_name_suffix)
        model = (
            _get_str(provider_data, role_key)
            or _get_str(provider_data, f"default_{role_key}")
            or _get_str(provider_data, "model")
            or _get_str(provider_data, "default_model")
        )
        if not model:
            msg = f"Provider `{provider}` must define `{role_key}`, `default_{role_key}`, or `model`."
            raise ValueError(msg)

    models = provider_data.get("models")
    model_data: dict[str, Any] = {}
    if isinstance(models, dict):
        raw_model_data = models.get(model)
        if isinstance(raw_model_data, dict):
            model_data = raw_model_data

    api_type = _get_str(provider_data, "api_type", "openai-compatible")
    max_tokens_default = _get_int(model_data, "max_output_tokens", 20000)
    max_input_tokens_default = _get_int(model_data, "context_length", 131072)
    return WuModelConfig(
        provider=provider,
        api_type=api_type or "openai-compatible",
        model=model,
        base_url=_get_str(provider_data, "base_url"),
        api_key=_get_str(provider_data, "api_key"),
        max_tokens=_get_int(provider_data, "max_tokens", max_tokens_default),
        timeout_s=_get_float(provider_data, "timeout_s", 180.0) or 180.0,
        temperature=_get_float(provider_data, "temperature", 0.2),
        max_input_tokens=_get_int(provider_data, "max_input_tokens", max_input_tokens_default),
    )


def _get_agent_defaults(data: dict[str, Any]) -> dict[str, Any]:
    """Extract agent defaults from config data, with sensible fallbacks."""
    agents_section = data.get("agents")
    if not isinstance(agents_section, dict):
        return {}
    defaults = agents_section.get("defaults")
    if isinstance(defaults, dict):
        return defaults
    return {}


def load_agent_registry(data: dict[str, Any] | None = None) -> WuAgentRegistry | None:
    """Build the full agent registry from config.json.

    Args:
        data: Pre-loaded resolved config dict. Loads fresh if None.

    Returns:
        WuAgentRegistry if agents section exists, or None for legacy mode.
    """
    if data is None:
        load_env_file()
        data = _resolve(_load_config())

    agents_section = data.get("agents")
    if not isinstance(agents_section, dict):
        return None

    agent_list = agents_section.get("list")
    if not isinstance(agent_list, list) or not agent_list:
        return None

    defaults = _get_agent_defaults(data)
    default_provider = _get_str(defaults, "provider") or get_selected_provider()

    # Helper: get fallback model for a provider from providers section
    def _provider_default_model(provider_name: str) -> str:
        providers = data.get("providers")
        if isinstance(providers, dict):
            p = providers.get(provider_name)
            if isinstance(p, dict):
                return _get_str(p, "default_model", "qwen-flash") or "qwen-flash"
        return "qwen-flash"

    agents: dict[str, WuAgentConfig] = {}
    default_name: str = ""
    for entry in agent_list:
        if not isinstance(entry, dict):
            continue
        name = _get_str(entry, "name") or "unnamed"
        merged = {**defaults, **entry}
        provider = _get_str(merged, "provider") or default_provider or "qwen"
        fallback = _provider_default_model(provider)
        model = _get_str(merged, "model") or _get_str(defaults, "model") or fallback
        max_tokens = _get_int(merged, "max_tokens", 800)
        workspace = _get_str(merged, "workspace", "") or ""

        agent_config = WuAgentConfig(
            name=name,
            provider=provider,
            model=model,
            max_tokens=max_tokens,
            workspace=workspace,
        )
        agents[name] = agent_config

        if entry.get("default") or not default_name:
            default_name = name

    if not agents:
        return None

    defaults_config = WuAgentConfig(
        name="__defaults__",
        provider=default_provider,
        model=_get_str(defaults, "model") or _provider_default_model(default_provider),
        max_tokens=_get_int(defaults, "max_tokens", 800),
        workspace=_get_str(defaults, "workspace", "") or "",
    )
    return WuAgentRegistry(defaults=defaults_config, agents=agents, default_name=default_name)


def resolve_agent_config(
    *,
    name: str | None = None,
    data: dict[str, Any] | None = None,
) -> WuAgentConfig | None:
    """Resolve a named agent config from the registry.

    Args:
        name: Agent name. None or empty returns the default agent.
        data: Pre-loaded resolved config dict. Loads fresh if None.

    Returns:
        WuAgentConfig for the named/default agent, or None if no agents section.
    """
    registry = load_agent_registry(data)
    if registry is None:
        return None
    agent_name = name or registry.default_name
    return registry.agents.get(agent_name, registry.agents.get(registry.default_name))
