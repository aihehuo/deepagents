"""chk.match_v2_schema — CandidateV2.model_validate trust boundary in match_client.

Default **on**. Explicit YAML ``false`` skips schema reject (debug only).
Not soft under ``mod.brain.check``.
"""

from __future__ import annotations

from apps.group_agent_api.agent_factory.checks.match_v2_schema.ids import (
    CHECK_ID,
    PROTOCOL_NAME,
)
from apps.group_agent_api.agent_factory.checks.match_v2_schema.module import (
    match_v2_schema_enabled,
)

__all__ = [
    "CHECK_ID",
    "PROTOCOL_NAME",
    "match_v2_schema_enabled",
]
