"""chk.profile_quality_llm — second-path LLM semantic richness gate.

Wraps ``profile_quality.assess_profile_match_ready`` Layer 2.

When **on** (default / ``current`` preset): Layer1 length/role then Layer2 LLM
(or fingerprint cache).

When **off** (or ``mod.brain.check`` soft-master off): Layer1 rules only —
length-ok profiles are treated as ready without a second LLM call
(TSD-14 §4.3 degrade: 只走规则闸).

Soft under ``mod.brain.check``; not a Backbone hard gate.
"""

from __future__ import annotations

from apps.group_agent_api.agent_factory.checks.profile_quality.ids import (
    CHECK_ID,
    PROTOCOL_NAME,
)
from apps.group_agent_api.agent_factory.checks.profile_quality.module import (
    profile_quality_llm_enabled,
)

__all__ = [
    "CHECK_ID",
    "PROTOCOL_NAME",
    "profile_quality_llm_enabled",
]
