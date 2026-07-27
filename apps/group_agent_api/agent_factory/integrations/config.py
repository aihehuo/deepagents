"""REQ-007 / REQ-009 integration config (local HTTP & prod security locking)."""

from __future__ import annotations

import os
from typing import Literal

IntegrationMode = Literal["stub", "http"]


def integration_mode() -> IntegrationMode:
    raw = (os.environ.get("GROUP_AGENT_INTEGRATION") or "stub").strip().lower()
    return "http" if raw in {"http", "live", "real"} else "stub"


def is_production_locked() -> bool:
    """True when misconfig must abort startup (prod / explicit require)."""
    env = (
        os.environ.get("GROUP_AGENT_ENV")
        or os.environ.get("APP_ENV")
        or ""
    ).strip().lower()
    if env in {"production", "prod"}:
        return True
    flag = (
        os.environ.get("GROUP_AGENT_REQUIRE_TRUSTED_PRINCIPAL") or ""
    ).strip().lower()
    return flag in {"1", "true", "yes", "on"}


def principal_hmac_secret() -> str:
    return (os.environ.get("GROUP_AGENT_PRINCIPAL_HMAC_SECRET") or "").strip()


def principal_max_skew_s() -> int:
    return int(os.environ.get("GROUP_AGENT_PRINCIPAL_MAX_SKEW_S", "300"))


def callback_hmac_secret() -> str:
    return (os.environ.get("GROUP_AGENT_CALLBACK_HMAC_SECRET") or "").strip()


def callback_allowed_base_urls() -> list[str]:
    raw = os.environ.get("GROUP_AGENT_CALLBACK_ALLOWED_BASE_URLS") or ""
    urls = [u.strip().rstrip("/") for u in raw.split(",") if u.strip()]
    if not urls:
        if is_production_locked():
            return []
        # Local development / stub fallback
        urls = [f"{micro_base()}/group_agent_callbacks"]
    return urls


def callback_timeout_s() -> float:
    return float(os.environ.get("GROUP_AGENT_CALLBACK_TIMEOUT_S", "10"))


def callback_max_retries() -> int:
    return int(os.environ.get("GROUP_AGENT_CALLBACK_MAX_RETRIES", "3"))


def async_max_active() -> int:
    return int(os.environ.get("GROUP_AGENT_ASYNC_MAX_ACTIVE", "10"))


def async_run_timeout_s() -> float:
    return float(os.environ.get("GROUP_AGENT_ASYNC_RUN_TIMEOUT_S", "120"))


def assert_startup_security() -> None:
    """Fail closed: prod/http must have principal secret, callback secret, allowlist, and directional isolation."""
    mode = integration_mode()
    p_secret = principal_hmac_secret()
    c_secret = callback_hmac_secret()
    locked = is_production_locked()

    if mode == "http" or locked:
        if mode != "http" and locked:
            raise RuntimeError(
                "production/locked mode forbids GROUP_AGENT_INTEGRATION=stub; "
                "set GROUP_AGENT_INTEGRATION=http"
            )

        if not p_secret:
            raise RuntimeError(
                "GROUP_AGENT_PRINCIPAL_HMAC_SECRET is required when "
                "GROUP_AGENT_INTEGRATION=http or in production"
            )

        if not c_secret:
            raise RuntimeError(
                "GROUP_AGENT_CALLBACK_HMAC_SECRET is required when "
                "GROUP_AGENT_INTEGRATION=http or in production"
            )

        if c_secret == p_secret:
            raise RuntimeError(
                "GROUP_AGENT_CALLBACK_HMAC_SECRET must not be identical to "
                "GROUP_AGENT_PRINCIPAL_HMAC_SECRET (directional isolation required)"
            )

        allowlist = callback_allowed_base_urls()
        if not allowlist or not os.environ.get("GROUP_AGENT_CALLBACK_ALLOWED_BASE_URLS", "").strip():
            raise RuntimeError(
                "GROUP_AGENT_CALLBACK_ALLOWED_BASE_URLS must be explicitly configured when "
                "GROUP_AGENT_INTEGRATION=http or in production"
            )

    # REQ-010 Fail closed: http/prod mode forbids mock/fixture configurations and stub model mode
    if mode == "http" or locked:
        if (os.environ.get("GROUP_AGENT_MODEL_MODE") or "").strip().lower() == "stub":
            raise RuntimeError(
                "http/production mode forbids GROUP_AGENT_MODEL_MODE=stub"
            )
        for mock_k in ("GROUP_AGENT_TEST_LEVEL", "GROUP_AGENT_MOCK_SEED", "GROUP_AGENT_MOCK_FIXTURE_FILE", "GROUP_AGENT_MOCK_FIXTURE_DIR"):
            if os.environ.get(mock_k):
                raise RuntimeError(
                    f"http/production mode forbids mock fixture configuration '{mock_k}'"
                )

    # Validate numeric configs
    try:
        t_out = callback_timeout_s()
        if not (0 < t_out <= 300):
            raise ValueError(f"callback_timeout_s out of bounds: {t_out}")

        retries = callback_max_retries()
        if not (0 <= retries <= 10):
            raise ValueError(f"callback_max_retries out of bounds: {retries}")

        max_act = async_max_active()
        if not (0 < max_act <= 1000):
            raise ValueError(f"async_max_active out of bounds: {max_act}")

        run_t_out = async_run_timeout_s()
        if not (0 < run_t_out <= 3600):
            raise ValueError(f"async_run_timeout_s out of bounds: {run_t_out}")

    except ValueError as exc:
        raise RuntimeError(f"Invalid async/callback numeric configuration: {exc}") from exc


def new_api_base() -> str:
    """New API origin.

    Production / http mode: must be supplied explicitly via GROUP_AGENT_NEW_API_BASE
    (or AIHEHUO_API_BASE) — no real production URL is baked in as a default.
    Local dev / stub only falls back to a loopback placeholder.
    """
    explicit = (
        os.environ.get("GROUP_AGENT_NEW_API_BASE")
        or os.environ.get("AIHEHUO_API_BASE")
        or ""
    ).strip()
    if explicit:
        return explicit.rstrip("/")
    if is_production_locked() or integration_mode() == "http":
        raise RuntimeError(
            "GROUP_AGENT_NEW_API_BASE must be explicitly configured when "
            "GROUP_AGENT_INTEGRATION=http or in production"
        )
    # Local development / stub fallback only (never a real production host).
    return "http://localhost:3000"


def micro_base() -> str:
    """aihehuomicro origin. Prod proxy: {new_api}/micro ; local often :3001."""
    explicit = os.environ.get("GROUP_AGENT_MICRO_BASE") or os.environ.get(
        "AIHEHUO_MICRO_BASE"
    )
    if explicit:
        return explicit.rstrip("/")
    return f"{new_api_base()}/micro"


def new_api_bearer() -> str:
    return (
        os.environ.get("GROUP_AGENT_NEW_API_TOKEN")
        or os.environ.get("AIHEHUO_API_KEY")
        or ""
    )


def http_timeout_s() -> float:
    return float(os.environ.get("GROUP_AGENT_HTTP_TIMEOUT_S", "15"))


def llm_polish_enabled() -> bool:
    raw = (os.environ.get("GROUP_AGENT_LLM_POLISH") or "1").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    if (os.environ.get("GROUP_AGENT_MODEL_MODE") or "").lower() == "stub":
        return False

    provider = os.environ.get("GROUP_AGENT_PROVIDER", "deepseek").strip().lower()
    prefix = provider.upper()
    provider_fallback_key = None
    if provider in {"qwen", "dashscope"}:
        provider_fallback_key = os.environ.get("DASHSCOPE_API_KEY")
    elif provider == "deepseek":
        provider_fallback_key = os.environ.get("DEEPSEEK_API_KEY")

    key = (
        os.environ.get(f"{prefix}_API_KEY")
        or provider_fallback_key
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("AIHEHUO_API_KEY")
    )
    return bool(key and key.strip() and key.strip() != "EMPTY")
