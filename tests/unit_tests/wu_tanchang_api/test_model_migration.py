"""Regression checks for Alibaba model retirement migration."""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("wu_migration_config", ROOT / "apps/wu_tanchang_api/config.py")
assert SPEC is not None and SPEC.loader is not None
CONFIG = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CONFIG
SPEC.loader.exec_module(CONFIG)


def test_legacy_alibaba_override_routes_to_supported_model(monkeypatch) -> None:
    """Keep old caller overrides usable after the retirement."""
    monkeypatch.setattr(CONFIG, "load_env_file", lambda: None)
    model = CONFIG.resolve_model_config(provider="qwen", model_name_suffix="MAIN_AGENT_MODEL", model_name_override="deepseek-v4-flash")
    assert model.model == "deepseek-v4.1-flash"
    assert model.max_input_tokens == 1000000


def test_official_deepseek_provider_is_not_remapped(monkeypatch) -> None:
    """Apply the Alibaba migration only to the Alibaba provider."""
    monkeypatch.setattr(CONFIG, "load_env_file", lambda: None)
    model = CONFIG.resolve_model_config(provider="deepseek", model_name_suffix="MAIN_AGENT_MODEL", model_name_override="deepseek-v4-flash")
    assert model.model == "deepseek-v4-flash"
