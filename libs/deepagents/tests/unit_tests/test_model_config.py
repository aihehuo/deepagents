"""Tests for deepagents.model_config module."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from deepagents.model_config import (
    ModelConfig,
    get_model_base_url_and_key,
    parse_model_config,
)


class TestParseModelConfig:
    """Tests for parse_model_config function."""

    def test_parse_model_config_with_explicit_provider_qwen(self, monkeypatch):
        """Test parsing config with explicit qwen provider."""
        monkeypatch.setenv("QWEN_BASE_URL", "https://qwen.example.com")
        monkeypatch.setenv("QWEN_API_KEY", "qwen-key-123")
        monkeypatch.setenv("QWEN_MAIN_AGENT_MODEL", "qwen-plus")

        config = parse_model_config(provider="qwen", model_name_suffix="MAIN_AGENT_MODEL")

        assert config.provider == "qwen"
        assert config.model == "qwen-plus"
        assert config.base_url == "https://qwen.example.com"
        assert config.api_key == "qwen-key-123"
        assert config.max_tokens == 20000  # default
        assert config.timeout_s == 180.0  # default
        assert config.temperature == 0.2  # default

    def test_parse_model_config_with_explicit_provider_deepseek(self, monkeypatch):
        """Test parsing config with explicit deepseek provider."""
        monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://deepseek.example.com")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key-456")
        monkeypatch.setenv("DEEPSEEK_MAIN_AGENT_MODEL", "deepseek-chat")

        config = parse_model_config(provider="deepseek", model_name_suffix="MAIN_AGENT_MODEL")

        assert config.provider == "deepseek"
        assert config.model == "deepseek-chat"
        assert config.base_url == "https://deepseek.example.com"
        assert config.api_key == "deepseek-key-456"

    def test_parse_model_config_auto_detect_single_provider(self, monkeypatch):
        """Test auto-detection when only one provider is supported."""
        monkeypatch.setenv("SUPPORTED_MODEL_PROVIDERS", "qwen")
        monkeypatch.setenv("QWEN_BASE_URL", "https://qwen.example.com")
        monkeypatch.setenv("QWEN_API_KEY", "qwen-key")
        monkeypatch.setenv("QWEN_MAIN_AGENT_MODEL", "qwen-plus")

        config = parse_model_config(model_name_suffix="MAIN_AGENT_MODEL")

        assert config.provider == "qwen"
        assert config.base_url == "https://qwen.example.com"
        assert config.api_key == "qwen-key"

    def test_parse_model_config_auto_detect_multiple_providers_uses_default(self, monkeypatch):
        """Test auto-detection when multiple providers are supported uses default."""
        monkeypatch.setenv("SUPPORTED_MODEL_PROVIDERS", "qwen,deepseek")
        monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://deepseek.example.com")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
        monkeypatch.setenv("DEEPSEEK_MAIN_AGENT_MODEL", "deepseek-chat")

        config = parse_model_config(model_name_suffix="MAIN_AGENT_MODEL", default_provider="deepseek")

        assert config.provider == "deepseek"
        assert config.base_url == "https://deepseek.example.com"
        assert config.api_key == "deepseek-key"

    def test_parse_model_config_no_supported_providers_uses_default(self, monkeypatch):
        """Test that missing SUPPORTED_MODEL_PROVIDERS uses default provider."""
        # Ensure SUPPORTED_MODEL_PROVIDERS is not set
        monkeypatch.delenv("SUPPORTED_MODEL_PROVIDERS", raising=False)
        monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://deepseek.example.com")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
        monkeypatch.setenv("DEEPSEEK_MAIN_AGENT_MODEL", "deepseek-chat")

        config = parse_model_config(model_name_suffix="MAIN_AGENT_MODEL", default_provider="deepseek")

        assert config.provider == "deepseek"
        assert config.base_url == "https://deepseek.example.com"

    def test_parse_model_config_default_model_name_qwen(self, monkeypatch):
        """Test default model name for qwen when not specified."""
        monkeypatch.setenv("QWEN_BASE_URL", "https://qwen.example.com")
        monkeypatch.setenv("QWEN_API_KEY", "qwen-key")
        # Don't set QWEN_MAIN_AGENT_MODEL

        config = parse_model_config(provider="qwen", model_name_suffix="MAIN_AGENT_MODEL")

        assert config.model == "qwen-plus"  # default for qwen

    def test_parse_model_config_default_model_name_deepseek(self, monkeypatch):
        """Test default model name for deepseek when not specified."""
        monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://deepseek.example.com")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
        # Don't set DEEPSEEK_MAIN_AGENT_MODEL

        config = parse_model_config(provider="deepseek", model_name_suffix="MAIN_AGENT_MODEL")

        assert config.model == "deepseek-chat"  # default for deepseek

    def test_parse_model_config_default_model_name_unknown_provider(self, monkeypatch):
        """Test default model name for unknown provider."""
        monkeypatch.setenv("CUSTOM_BASE_URL", "https://custom.example.com")
        monkeypatch.setenv("CUSTOM_API_KEY", "custom-key")
        # Don't set CUSTOM_MAIN_AGENT_MODEL

        config = parse_model_config(provider="custom", model_name_suffix="MAIN_AGENT_MODEL")

        assert config.model == "custom-chat"  # generic fallback

    def test_parse_model_config_coder_subagent_model(self, monkeypatch):
        """Test parsing config for coder subagent."""
        monkeypatch.setenv("QWEN_BASE_URL", "https://qwen.example.com")
        monkeypatch.setenv("QWEN_API_KEY", "qwen-key")
        monkeypatch.setenv("QWEN_CODER_SUBAGENT_MODEL", "qwen-coder-plus")

        config = parse_model_config(provider="qwen", model_name_suffix="CODER_SUBAGENT_MODEL")

        assert config.provider == "qwen"
        assert config.model == "qwen-coder-plus"
        assert config.base_url == "https://qwen.example.com"  # Same provider config

    def test_parse_model_config_aihehuo_subagent_model(self, monkeypatch):
        """Test parsing config for aihehuo subagent."""
        monkeypatch.setenv("QWEN_BASE_URL", "https://qwen.example.com")
        monkeypatch.setenv("QWEN_API_KEY", "qwen-key")
        monkeypatch.setenv("QWEN_AIHEHUO_SUBAGENT_MODEL", "qwen-max")

        config = parse_model_config(provider="qwen", model_name_suffix="AIHEHUO_SUBAGENT_MODEL")

        assert config.provider == "qwen"
        assert config.model == "qwen-max"
        assert config.base_url == "https://qwen.example.com"

    def test_parse_model_config_provider_specific_max_tokens(self, monkeypatch):
        """Test provider-specific max_tokens override."""
        monkeypatch.setenv("QWEN_BASE_URL", "https://qwen.example.com")
        monkeypatch.setenv("QWEN_API_KEY", "qwen-key")
        monkeypatch.setenv("QWEN_MAX_TOKENS", "40000")
        monkeypatch.setenv("QWEN_MAIN_AGENT_MODEL", "qwen-plus")

        config = parse_model_config(provider="qwen", model_name_suffix="MAIN_AGENT_MODEL")

        assert config.max_tokens == 40000

    def test_parse_model_config_provider_specific_timeout(self, monkeypatch):
        """Test provider-specific timeout override."""
        monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://deepseek.example.com")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
        monkeypatch.setenv("DEEPSEEK_TIMEOUT_S", "300.0")
        monkeypatch.setenv("DEEPSEEK_MAIN_AGENT_MODEL", "deepseek-chat")

        config = parse_model_config(provider="deepseek", model_name_suffix="MAIN_AGENT_MODEL")

        assert config.timeout_s == 300.0

    def test_parse_model_config_provider_specific_temperature(self, monkeypatch):
        """Test provider-specific temperature override."""
        monkeypatch.setenv("QWEN_BASE_URL", "https://qwen.example.com")
        monkeypatch.setenv("QWEN_API_KEY", "qwen-key")
        monkeypatch.setenv("QWEN_TEMPERATURE", "0.7")
        monkeypatch.setenv("QWEN_MAIN_AGENT_MODEL", "qwen-plus")

        config = parse_model_config(provider="qwen", model_name_suffix="MAIN_AGENT_MODEL")

        assert config.temperature == 0.7

    def test_parse_model_config_missing_base_url_and_key(self, monkeypatch):
        """Test parsing config when base_url and api_key are missing."""
        # Don't set QWEN_BASE_URL or QWEN_API_KEY
        monkeypatch.setenv("QWEN_MAIN_AGENT_MODEL", "qwen-plus")

        config = parse_model_config(provider="qwen", model_name_suffix="MAIN_AGENT_MODEL")

        assert config.provider == "qwen"
        assert config.model == "qwen-plus"
        assert config.base_url is None
        assert config.api_key is None

    def test_parse_model_config_provider_name_normalization(self, monkeypatch):
        """Test that provider names are normalized (case-insensitive)."""
        monkeypatch.setenv("QWEN_BASE_URL", "https://qwen.example.com")
        monkeypatch.setenv("QWEN_API_KEY", "qwen-key")
        monkeypatch.setenv("QWEN_MAIN_AGENT_MODEL", "qwen-plus")

        # Test with different case variations
        config1 = parse_model_config(provider="QWEN", model_name_suffix="MAIN_AGENT_MODEL")
        config2 = parse_model_config(provider="qwen", model_name_suffix="MAIN_AGENT_MODEL")
        config3 = parse_model_config(provider="Qwen", model_name_suffix="MAIN_AGENT_MODEL")

        assert config1.provider == "qwen"
        assert config2.provider == "qwen"
        assert config3.provider == "qwen"
        assert config1.base_url == config2.base_url == config3.base_url

    def test_parse_model_config_provider_with_whitespace(self, monkeypatch):
        """Test that provider names with whitespace are trimmed."""
        monkeypatch.setenv("QWEN_BASE_URL", "https://qwen.example.com")
        monkeypatch.setenv("QWEN_API_KEY", "qwen-key")
        monkeypatch.setenv("QWEN_MAIN_AGENT_MODEL", "qwen-plus")

        config = parse_model_config(provider="  qwen  ", model_name_suffix="MAIN_AGENT_MODEL")

        assert config.provider == "qwen"
        assert config.base_url == "https://qwen.example.com"

    def test_parse_model_config_supported_providers_with_whitespace(self, monkeypatch):
        """Test that SUPPORTED_MODEL_PROVIDERS handles whitespace correctly."""
        monkeypatch.setenv("SUPPORTED_MODEL_PROVIDERS", " qwen , deepseek ")
        monkeypatch.setenv("QWEN_BASE_URL", "https://qwen.example.com")
        monkeypatch.setenv("QWEN_API_KEY", "qwen-key")
        monkeypatch.setenv("QWEN_MAIN_AGENT_MODEL", "qwen-plus")

        # Should use default when multiple providers (with whitespace trimmed)
        config = parse_model_config(model_name_suffix="MAIN_AGENT_MODEL", default_provider="qwen")

        assert config.provider == "qwen"


class TestGetModelBaseUrlAndKey:
    """Tests for get_model_base_url_and_key convenience function."""

    def test_get_model_base_url_and_key_qwen(self, monkeypatch):
        """Test getting base URL and key for qwen provider."""
        monkeypatch.setenv("QWEN_BASE_URL", "https://qwen.example.com")
        monkeypatch.setenv("QWEN_API_KEY", "qwen-key-123")

        base_url, api_key = get_model_base_url_and_key(provider="qwen")

        assert base_url == "https://qwen.example.com"
        assert api_key == "qwen-key-123"

    def test_get_model_base_url_and_key_deepseek(self, monkeypatch):
        """Test getting base URL and key for deepseek provider."""
        monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://deepseek.example.com")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key-456")

        base_url, api_key = get_model_base_url_and_key(provider="deepseek")

        assert base_url == "https://deepseek.example.com"
        assert api_key == "deepseek-key-456"

    def test_get_model_base_url_and_key_missing_values(self, monkeypatch):
        """Test getting base URL and key when values are missing."""
        # Don't set any env vars

        base_url, api_key = get_model_base_url_and_key(provider="qwen")

        assert base_url is None
        assert api_key is None

    def test_get_model_base_url_and_key_uses_default_provider(self, monkeypatch):
        """Test that missing provider uses default."""
        monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://deepseek.example.com")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")

        base_url, api_key = get_model_base_url_and_key(default_provider="deepseek")

        assert base_url == "https://deepseek.example.com"
        assert api_key == "deepseek-key"


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_model_config_creation(self):
        """Test creating a ModelConfig instance."""
        config = ModelConfig(
            provider="qwen",
            model="qwen-plus",
            base_url="https://qwen.example.com",
            api_key="qwen-key",
            max_tokens=20000,
            timeout_s=180.0,
            temperature=0.2,
        )

        assert config.provider == "qwen"
        assert config.model == "qwen-plus"
        assert config.base_url == "https://qwen.example.com"
        assert config.api_key == "qwen-key"
        assert config.max_tokens == 20000
        assert config.timeout_s == 180.0
        assert config.temperature == 0.2

    def test_model_config_with_none_values(self):
        """Test ModelConfig with None values for optional fields."""
        config = ModelConfig(
            provider="qwen",
            model="qwen-plus",
            base_url=None,
            api_key=None,
            max_tokens=20000,
            timeout_s=180.0,
            temperature=None,
        )

        assert config.base_url is None
        assert config.api_key is None
        assert config.temperature is None

    def test_model_config_is_frozen(self):
        """Test that ModelConfig is immutable (frozen dataclass)."""
        config = ModelConfig(
            provider="qwen",
            model="qwen-plus",
            base_url="https://qwen.example.com",
            api_key="qwen-key",
            max_tokens=20000,
            timeout_s=180.0,
            temperature=0.2,
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            config.provider = "deepseek"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_parse_model_config_invalid_int_fallback(self, monkeypatch):
        """Test that invalid integer values fall back to default."""
        monkeypatch.setenv("QWEN_BASE_URL", "https://qwen.example.com")
        monkeypatch.setenv("QWEN_API_KEY", "qwen-key")
        monkeypatch.setenv("QWEN_MAX_TOKENS", "not-a-number")
        monkeypatch.setenv("QWEN_MAIN_AGENT_MODEL", "qwen-plus")

        config = parse_model_config(provider="qwen", model_name_suffix="MAIN_AGENT_MODEL")

        assert config.max_tokens == 20000  # Should fall back to default

    def test_parse_model_config_invalid_float_fallback(self, monkeypatch):
        """Test that invalid float values fall back to default."""
        monkeypatch.setenv("QWEN_BASE_URL", "https://qwen.example.com")
        monkeypatch.setenv("QWEN_API_KEY", "qwen-key")
        monkeypatch.setenv("QWEN_TIMEOUT_S", "not-a-number")
        monkeypatch.setenv("QWEN_MAIN_AGENT_MODEL", "qwen-plus")

        config = parse_model_config(provider="qwen", model_name_suffix="MAIN_AGENT_MODEL")

        assert config.timeout_s == 180.0  # Should fall back to default

    def test_parse_model_config_empty_string_env_vars(self, monkeypatch):
        """Test that empty string env vars are treated as None."""
        monkeypatch.setenv("QWEN_BASE_URL", "")
        monkeypatch.setenv("QWEN_API_KEY", "")
        monkeypatch.setenv("QWEN_MAIN_AGENT_MODEL", "")

        config = parse_model_config(provider="qwen", model_name_suffix="MAIN_AGENT_MODEL")

        assert config.base_url is None
        assert config.api_key is None
        assert config.model == "qwen-plus"  # Should use default model name

    def test_parse_model_config_whitespace_in_env_vars(self, monkeypatch):
        """Test that whitespace in env vars is trimmed."""
        monkeypatch.setenv("QWEN_BASE_URL", "  https://qwen.example.com  ")
        monkeypatch.setenv("QWEN_API_KEY", "  qwen-key  ")
        monkeypatch.setenv("QWEN_MAIN_AGENT_MODEL", "  qwen-plus  ")

        config = parse_model_config(provider="qwen", model_name_suffix="MAIN_AGENT_MODEL")

        assert config.base_url == "https://qwen.example.com"
        assert config.api_key == "qwen-key"
        assert config.model == "qwen-plus"

    def test_parse_model_config_supported_providers_empty_string(self, monkeypatch):
        """Test that empty SUPPORTED_MODEL_PROVIDERS is handled."""
        monkeypatch.setenv("SUPPORTED_MODEL_PROVIDERS", "")
        monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://deepseek.example.com")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
        monkeypatch.setenv("DEEPSEEK_MAIN_AGENT_MODEL", "deepseek-chat")

        config = parse_model_config(model_name_suffix="MAIN_AGENT_MODEL", default_provider="deepseek")

        assert config.provider == "deepseek"  # Should use default

    def test_parse_model_config_supported_providers_with_empty_items(self, monkeypatch):
        """Test that SUPPORTED_MODEL_PROVIDERS with empty items is handled."""
        monkeypatch.setenv("SUPPORTED_MODEL_PROVIDERS", "qwen,,deepseek,")
        monkeypatch.setenv("QWEN_BASE_URL", "https://qwen.example.com")
        monkeypatch.setenv("QWEN_API_KEY", "qwen-key")
        monkeypatch.setenv("QWEN_MAIN_AGENT_MODEL", "qwen-plus")

        # Should detect multiple providers and use default
        config = parse_model_config(model_name_suffix="MAIN_AGENT_MODEL", default_provider="qwen")

        assert config.provider == "qwen"  # Should use default when multiple detected

