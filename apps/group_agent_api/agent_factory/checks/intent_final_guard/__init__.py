"""Deterministic current-run action and evidence guard."""

from .ids import CHECK_ID
from .module import (
    IntentFinalGuardResult,
    apply_intent_final_guard,
    intent_final_guard_enabled,
)

__all__ = [
    "CHECK_ID",
    "IntentFinalGuardResult",
    "apply_intent_final_guard",
    "intent_final_guard_enabled",
]
