"""Load group-agent Module / Check switches from one YAML file (TSD-14 §4.4).

Secrets / URLs / queues remain in ENV. Business toggles live here so operators
can oversee them in a single file.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_logger = logging.getLogger("uvicorn.error")

_DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "modules.yaml"
)

_lock = threading.Lock()
_cached: ModulesConfig | None = None
_cached_path: Path | None = None


@dataclass(frozen=True)
class ModulesConfig:
    version: int
    preset: str
    context: dict[str, bool] = field(default_factory=dict)
    checks: dict[str, bool] = field(default_factory=dict)
    modules: dict[str, bool] = field(default_factory=dict)
    reply_grounding: dict[str, Any] = field(default_factory=dict)
    debug: dict[str, bool] = field(default_factory=dict)
    ingress: dict[str, Any] = field(default_factory=dict)
    source_path: str = ""
    fingerprint: str = ""

    def is_module_enabled(self, module_id: str) -> bool:
        return bool(self.modules.get(module_id, False))

    def is_check_enabled(self, check_id: str) -> bool:
        return bool(self.checks.get(check_id, False))

    def reply_grounding_enabled(self) -> bool:
        """Module switch is authoritative; check id mirrors it when both set."""
        module_on = self.is_module_enabled("mod.brain.reply_grounding")
        check_on = self.is_check_enabled("chk.reply_fact_grounding_llm")
        if module_on and not check_on:
            _logger.warning(
                "action=modules_config_mismatch module=mod.brain.reply_grounding "
                "on but chk.reply_fact_grounding_llm off; treating as enabled"
            )
        return module_on

    def reply_grounding_max_attempts(self) -> int:
        raw = self.reply_grounding.get("max_attempts", 2)
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = 2
        return max(1, min(5, value))


def default_modules_config_path() -> Path:
    override = (os.environ.get("GROUP_AGENT_MODULES_CONFIG") or "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return _DEFAULT_CONFIG_PATH


def load_modules_config(path: Path | str | None = None) -> ModulesConfig:
    """Load (or return cached) modules YAML. Missing file → empty fail-safe defaults.

    After ``reload_modules_config(custom_path)``, pathless loads keep using that
    cached file until reset — so tests can point at a temporary YAML.
    """
    global _cached, _cached_path
    with _lock:
        if path is None and _cached is not None:
            return _cached
        target = (
            Path(path).expanduser().resolve()
            if path
            else default_modules_config_path()
        )
        if _cached is not None and _cached_path == target:
            return _cached
        cfg = _parse_modules_yaml(target)
        _cached = cfg
        _cached_path = target
        return cfg


def reload_modules_config(path: Path | str | None = None) -> ModulesConfig:
    """Drop cache and reload — used by tests and hot-reload operators."""
    reset_modules_config_cache()
    return load_modules_config(path)


def reset_modules_config_cache() -> None:
    global _cached, _cached_path
    with _lock:
        _cached = None
        _cached_path = None


def reply_grounding_enabled() -> bool:
    return load_modules_config().reply_grounding_enabled()


def reply_grounding_max_attempts() -> int:
    return load_modules_config().reply_grounding_max_attempts()


def config_fingerprint() -> str:
    return load_modules_config().fingerprint


def _parse_modules_yaml(path: Path) -> ModulesConfig:
    if not path.is_file():
        _logger.warning(
            "action=modules_config_missing path=%s; modules default off",
            path,
        )
        return ModulesConfig(
            version=1,
            preset="current",
            source_path=str(path),
            fingerprint=_fingerprint_text(""),
        )

    raw_text = path.read_text(encoding="utf-8")
    try:
        import yaml

        data = yaml.safe_load(raw_text) or {}
    except Exception as exc:  # noqa: BLE001
        _logger.error(
            "action=modules_config_parse_failed path=%s error=%s; modules default off",
            path,
            exc,
        )
        return ModulesConfig(
            version=1,
            preset="current",
            source_path=str(path),
            fingerprint=_fingerprint_text(raw_text),
        )

    if not isinstance(data, dict):
        data = {}

    return ModulesConfig(
        version=int(data.get("version") or 1),
        preset=str(data.get("preset") or "current"),
        context=_bool_map(data.get("context")),
        checks=_bool_map(data.get("checks")),
        modules=_bool_map(data.get("modules")),
        reply_grounding=_dict_section(data.get("reply_grounding")),
        debug=_bool_map(data.get("debug")),
        ingress=_dict_section(data.get("ingress")),
        source_path=str(path),
        fingerprint=_fingerprint_text(raw_text),
    )


def _bool_map(raw: Any) -> dict[str, bool]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, bool] = {}
    for key, value in raw.items():
        kid = str(key).strip()
        if not kid:
            continue
        if isinstance(value, bool):
            out[kid] = value
        elif isinstance(value, (int, float)):
            out[kid] = bool(value)
        else:
            text = str(value or "").strip().lower()
            out[kid] = text in {"1", "true", "yes", "on"}
    return out


def _dict_section(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    return {str(k): v for k, v in raw.items()}


def _fingerprint_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
