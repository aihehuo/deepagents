"""bb.brain.repair — mouth 422 → ingress_repair → same-seq re-final.

Backbone seam extracted from orchestrator/callback (BSD-01 §4.3 / §10 P2).
Reject maps and peel helpers remain in ``ingress_repair`` (re-exported here).
"""

from apps.group_agent_api.agent_factory.brain_repair.reject import (
    MouthIngressRejected,
    parse_mouth_reject_body,
)
from apps.group_agent_api.agent_factory.brain_repair.seam import (
    abandon_error,
    decide_mouth_repair_action,
    emit_final_with_mouth_repair,
    prepare_repaired_final,
)
from apps.group_agent_api.agent_factory.brain_repair.turn_guards import noop_turn_guards
from apps.group_agent_api.agent_factory.ingress_repair import (
    MOUTH_INGRESS_MAX_ATTEMPTS,
    apply_mouth_repair,
    build_abandon_final_payload,
    format_ingress_deny,
    peel_final_payload,
    reject_meta,
)

__all__ = [
    "MOUTH_INGRESS_MAX_ATTEMPTS",
    "MouthIngressRejected",
    "abandon_error",
    "apply_mouth_repair",
    "build_abandon_final_payload",
    "decide_mouth_repair_action",
    "emit_final_with_mouth_repair",
    "format_ingress_deny",
    "noop_turn_guards",
    "parse_mouth_reject_body",
    "peel_final_payload",
    "prepare_repaired_final",
    "reject_meta",
]
