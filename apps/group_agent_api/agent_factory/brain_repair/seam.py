"""bb.brain.repair seam: mouth reject → ingress_repair → same-seq re-final.

Behavior (BSD-01 P1 / P2 extract): N≈1 (first emit + one repair), no blind
replay when ``repairable_by=none`` or attempts exhausted →
``mouth_ingress_rejected:<reason>`` (not user-visible ``safe_failed``).
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Literal

from apps.group_agent_api.agent_factory.brain_repair.reject import MouthIngressRejected
from apps.group_agent_api.agent_factory.debug_trace import record_decision_point
from apps.group_agent_api.agent_factory.ingress_repair import (
    MOUTH_INGRESS_MAX_ATTEMPTS,
    apply_mouth_repair,
    build_abandon_final_payload,
)

_logger = logging.getLogger("uvicorn.error")

MouthRepairAction = Literal["repair", "abandon"]

EmitFinalFn = Callable[[str, dict[str, Any]], Awaitable[bool]]


def decide_mouth_repair_action(
    *,
    reject: MouthIngressRejected,
    attempt: int,
    max_attempts: int = MOUTH_INGRESS_MAX_ATTEMPTS,
) -> MouthRepairAction:
    """Decide re-emit after repair vs abandon (``mouth_ingress_rejected``)."""
    if attempt >= max_attempts or reject.repairable_by == "none":
        return "abandon"
    return "repair"


def prepare_repaired_final(
    payload: dict[str, Any],
    *,
    reject: MouthIngressRejected,
    model: Any | None = None,
    attempt: int,
) -> dict[str, Any]:
    """Peel/rewrite via ``ingress_repair``; abandon shape if still risky."""
    repaired = apply_mouth_repair(
        payload,
        reject=reject,
        model=model,
        attempt=attempt,
    )
    # If peel left a still-risky recommendation shape, fall back to abandon.
    if repaired.get("reply_mode") == "recommendation" and not (
        repaired.get("candidates") or []
    ):
        return build_abandon_final_payload(payload)
    return repaired


def abandon_error(reject: MouthIngressRejected) -> RuntimeError:
    """Terminal brain error after mouth reject — not user ``safe_failed``."""
    return RuntimeError(f"mouth_ingress_rejected:{reject.reason_code}")


async def emit_final_with_mouth_repair(
    *,
    emit_callback: EmitFinalFn,
    final_payload: dict[str, Any],
    model: Any | None = None,
    max_attempts: int = MOUTH_INGRESS_MAX_ATTEMPTS,
    run_id: str | None = None,
    logger: logging.Logger | None = None,
) -> bool:
    """Emit ``final`` with at most one same-seq repair re-emit.

    Raises:
        RuntimeError: ``mouth_ingress_rejected:<code>`` or ``final_callback_failed``.
    """
    log = logger or _logger
    mouth_attempt = 1
    current_final = final_payload
    while True:
        try:
            final_ok = await emit_callback("final", current_final)
        except MouthIngressRejected as exc:
            log.warning(
                "Mouth rejected final run_id=%s attempt=%s/%s reason_code=%s "
                "repairable_by=%s",
                run_id,
                mouth_attempt,
                max_attempts,
                exc.reason_code,
                exc.repairable_by,
            )
            record_decision_point(
                phase="ingress_mouth",
                detail={
                    "status": "rejected",
                    "attempts": mouth_attempt,
                    "max_attempts": max_attempts,
                    "reason_code": exc.reason_code,
                    "repairable_by": exc.repairable_by,
                    "action": decide_mouth_repair_action(
                        reject=exc,
                        attempt=mouth_attempt,
                        max_attempts=max_attempts,
                    ),
                },
                run_id=run_id,
            )
            if decide_mouth_repair_action(
                reject=exc,
                attempt=mouth_attempt,
                max_attempts=max_attempts,
            ) == "abandon":
                raise abandon_error(exc) from exc
            mouth_attempt += 1
            current_final = prepare_repaired_final(
                current_final,
                reject=exc,
                model=model,
                attempt=mouth_attempt,
            )
            continue
        if final_ok is False:
            record_decision_point(
                phase="ingress_mouth",
                detail={
                    "status": "failed",
                    "attempts": mouth_attempt,
                    "reason": "final_callback_failed",
                },
                run_id=run_id,
            )
            raise RuntimeError("final_callback_failed")

        record_decision_point(
            phase="ingress_mouth",
            detail={
                "status": "delivered",
                "attempts": mouth_attempt,
                "repaired": mouth_attempt > 1,
                "delivery_kind": current_final.get("delivery_kind"),
                "reply_mode": current_final.get("reply_mode"),
                "candidates_count": len(current_final.get("candidates") or []),
                "profile_status": current_final.get("profile_status"),
                "match_status": current_final.get("match_status"),
            },
            run_id=run_id,
        )
        return True
