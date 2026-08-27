"""YAML switch for finalize confirmation / next_step templates."""

from __future__ import annotations

from apps.group_agent_api.agent_factory.checks.finalize_templates.ids import CHECK_ID

__all__ = ["finalize_templates_enabled"]


def finalize_templates_enabled(*, enabled: bool | None = None) -> bool:
    """True when confirmation + next_step template assembly may run."""
    if enabled is not None:
        return bool(enabled)
    from apps.group_agent_api.agent_factory.module_config import load_modules_config

    return load_modules_config().is_check_enabled(CHECK_ID)
