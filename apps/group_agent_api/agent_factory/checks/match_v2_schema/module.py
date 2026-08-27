"""YAML-gated CandidateV2 contract validation (chk.match_v2_schema).

Hard grounding boundary: missing YAML key → **on** (fail-closed).
Not soft under ``mod.brain.check``. Off = skip ``CandidateV2.model_validate``
(illegal candidates may reach orchestrator — debug only).
"""

from __future__ import annotations

import logging

from apps.group_agent_api.agent_factory.checks.match_v2_schema.ids import CHECK_ID

_logger = logging.getLogger("uvicorn.error")

__all__ = ["match_v2_schema_enabled"]


def match_v2_schema_enabled(*, enabled: bool | None = None) -> bool:
    """Resolve YAML; explicit ``enabled`` wins. Missing key → True."""
    if enabled is not None:
        return bool(enabled)
    from apps.group_agent_api.agent_factory.module_config import load_modules_config

    cfg = load_modules_config()
    if CHECK_ID not in cfg.checks:
        return True
    return cfg.is_check_enabled(CHECK_ID)
