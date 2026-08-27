"""bb.brain.turn_guards stub (BSD-01 P2 · thin structural asserts before mouth).

Hang points today (do **not** relocate reply_grounding gates here):

- ``agent_factory.guard.post_process_assert`` — candidate/in-group sanitization
- ``checks.action_claim.apply_action_claim_guard`` — YAML ``chk.action_claim``
- ``checks.reply_grounding.apply_reply_grounding_gate`` — semantic reply-vs-ground
  (Module switch; orchestrator hang point must stay in chat/async_manager)

This module is a path placeholder for future Allow|Deny → ``check_deny``
extraction. Callers may import ``noop_turn_guards`` as a documented no-op;
semantics must remain identical to pre-extract behavior.
"""

from __future__ import annotations

from typing import Any


def noop_turn_guards(payload: dict[str, Any]) -> dict[str, Any]:
    """Identity stub — no structural assert policy yet (P2 thin extract)."""
    return payload
