"""chk.invented_candidate — scrub model-invented candidate bios before finalize.

Thin YAML-gated facade over ``content_quality.scrub_invented_candidate_narrative``.
Detector helpers stay in ``content_quality`` for reply_grounding L0
(``invented_candidate``) reuse.

Transitional vs RG L0 (TSD): scrub empties unsafe paragraphs so finalize can
fall back to templates; L0 fails the post-finalize reply. Prefer RG as the
production primary semantic gate; this scrub is YAML-offable.

Off = return text unchanged (no paragraph drop).
"""

from __future__ import annotations

from apps.group_agent_api.agent_factory.checks.invented_candidate.ids import (
    CHECK_ID,
    PROTOCOL_NAME,
)
from apps.group_agent_api.agent_factory.checks.invented_candidate.module import (
    invented_candidate_enabled,
    scrub_invented_candidate_if_enabled,
)

__all__ = [
    "CHECK_ID",
    "PROTOCOL_NAME",
    "invented_candidate_enabled",
    "scrub_invented_candidate_if_enabled",
]
