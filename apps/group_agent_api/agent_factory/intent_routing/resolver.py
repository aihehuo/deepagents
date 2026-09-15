"""Stable resolver API and observe-only instrumentation for INTENT-DA-001."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from apps.group_agent_api.agent_factory.debug_trace import record_decision_point

from .classifier import ClassifierResult, aclassify_intent, classify_intent
from .context import (
    IntentContextScope,
    TrustedIntentContext,
    extract_completed_context,
    is_elliptical_followup,
    merge_followup_constraints,
)
from .policy import build_decision
from .schema import (
    Confidence,
    DecisionSource,
    IntentConstraintsV1,
    IntentDecisionV1,
    IntentKind,
)
from .signals import SignalResult, detect_signals

_CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}
_READ_INTENTS = {
    "people_search",
    "opportunity_search",
    "project_partnership_search",
    "community_search",
    "startup_digest",
    "platform_help",
}


@dataclass(frozen=True)
class IntentResolution:
    """Decision plus bounded diagnostics safe for observe logs."""

    decision: IntentDecisionV1
    latency_ms: int
    signal_ids: tuple[str, ...]
    classifier_status: str
    classifier_attempts: int
    context_inherited: bool


def intent_routing_mode() -> str:
    """Return the explicit off, observe, or enforce routing mode."""
    mode = os.environ.get("GROUP_AGENT_INTENT_ROUTING_MODE", "off").strip().lower()
    if mode not in {"off", "observe", "enforce"}:
        msg = "GROUP_AGENT_INTENT_ROUTING_MODE must be off, observe, or enforce"
        raise RuntimeError(msg)
    return mode


def intent_classifier_timeout_s() -> float:
    """Return the bounded total L2 classification timeout."""
    try:
        value = float(os.environ.get("GROUP_AGENT_INTENT_CLASSIFIER_TIMEOUT_S", "1.5"))
    except ValueError as exc:
        msg = "GROUP_AGENT_INTENT_CLASSIFIER_TIMEOUT_S must be numeric"
        raise RuntimeError(msg) from exc
    return min(5.0, max(0.05, value))


def _classifier_choice(
    result: ClassifierResult,
) -> tuple[IntentKind, Confidence, tuple[str, ...]]:
    classification = result.classification
    if classification is None:
        return "clarification", "low", ()
    if classification.missing_object:
        return "clarification", "low", tuple(classification.evidence_spans[:16])
    best_value = max(
        _CONFIDENCE_ORDER[item.confidence] for item in classification.candidates
    )
    best = [
        item
        for item in classification.candidates
        if _CONFIDENCE_ORDER[item.confidence] == best_value
    ]
    if len(best) != 1:
        return "clarification", "low", tuple(classification.evidence_spans[:16])
    selected = best[0]
    return (
        selected.intent,
        selected.confidence,
        tuple(classification.evidence_spans[:16]),
    )


def _context_decision(
    signal: SignalResult,
    message: str,
    context: TrustedIntentContext | None,
) -> (
    tuple[
        IntentKind,
        Confidence,
        tuple[str, ...],
        tuple[str, ...],
        IntentConstraintsV1,
    ]
    | None
):
    if (
        context is None
        or not is_elliptical_followup(message)
        or signal.intent not in {None, "conversation"}
        or "explicit_negation" in signal.signal_ids
    ):
        return None
    constraints, context_negations = merge_followup_constraints(
        context.constraints, signal.constraints, message
    )
    return (
        context.intent,
        "high",
        tuple(signal.evidence_spans),
        context_negations,
        constraints,
    )


def _finalize(
    *,
    started: float,
    signal: SignalResult,
    intent: IntentKind,
    confidence: Confidence,
    source: DecisionSource,
    evidence: tuple[str, ...],
    negations: tuple[str, ...],
    constraints: IntentConstraintsV1,
    execution_mode: str,
    can_search: bool,
    classifier: ClassifierResult,
    inherited: bool,
) -> IntentResolution:
    decision = build_decision(
        intent=intent,
        confidence=confidence,
        source=source,
        evidence_spans=evidence,
        constraints=constraints,
        negations=negations,
        execution_mode=execution_mode,
        can_search=can_search,
    )
    return IntentResolution(
        decision=decision,
        latency_ms=max(0, int((time.perf_counter() - started) * 1000)),
        signal_ids=signal.signal_ids,
        classifier_status=classifier.status,
        classifier_attempts=classifier.attempts,
        context_inherited=inherited,
    )


def resolve_intent(
    message: str,
    *,
    execution_mode: str,
    can_search: bool,
    prior_messages: list[Any] | None = None,
    scope: IntentContextScope | None = None,
    classifier_model: Any = None,
) -> IntentResolution:
    """Resolve one turn without changing the execution path.

    Args:
        message: Current user message.
        execution_mode: Trusted mode (`group_agent` or `cafe_host`).
        can_search: Trusted public-read capability.
        prior_messages: Messages from the exact scoped checkpoint.
        scope: Provenance for those messages; required for inheritance.
        classifier_model: Optional bare model used only for L2 structured output.

    Returns:
        A contract-valid decision and bounded observe diagnostics.
    """
    started = time.perf_counter()
    signal = detect_signals(message)
    context = (
        extract_completed_context(prior_messages or [], scope=scope)
        if scope is not None
        else None
    )
    inherited = _context_decision(signal, message, context)
    unavailable = ClassifierResult(None, "not_needed", 0)
    if inherited is not None:
        intent, confidence, evidence, negations, constraints = inherited
        return _finalize(
            started=started,
            signal=signal,
            intent=intent,
            confidence=confidence,
            source="context",
            evidence=evidence,
            negations=negations,
            constraints=constraints,
            execution_mode=execution_mode,
            can_search=can_search,
            classifier=unavailable,
            inherited=True,
        )
    if signal.intent is not None and not signal.needs_classifier:
        return _finalize(
            started=started,
            signal=signal,
            intent=signal.intent,
            confidence=signal.confidence,
            source="rules",
            evidence=signal.evidence_spans,
            negations=signal.negations,
            constraints=signal.constraints,
            execution_mode=execution_mode,
            can_search=can_search,
            classifier=unavailable,
            inherited=False,
        )
    classified = classify_intent(
        message,
        model=classifier_model,
        prior_intent=context.intent if context else None,
    )
    intent, confidence, evidence = _classifier_choice(classified)
    safe_evidence = tuple(span for span in evidence if span and span in message)[:16]
    return _finalize(
        started=started,
        signal=signal,
        intent=intent,
        confidence=confidence,
        source="classifier",
        evidence=safe_evidence,
        negations=signal.negations,
        constraints=signal.constraints,
        execution_mode=execution_mode,
        can_search=can_search,
        classifier=classified,
        inherited=False,
    )


async def aresolve_intent(
    message: str,
    *,
    execution_mode: str,
    can_search: bool,
    prior_messages: list[Any] | None = None,
    scope: IntentContextScope | None = None,
    classifier_model: Any = None,
) -> IntentResolution:
    """Async equivalent of `resolve_intent`, including a bounded L2 timeout."""
    started = time.perf_counter()
    signal = detect_signals(message)
    context = (
        extract_completed_context(prior_messages or [], scope=scope)
        if scope is not None
        else None
    )
    inherited = _context_decision(signal, message, context)
    unavailable = ClassifierResult(None, "not_needed", 0)
    if inherited is not None:
        intent, confidence, evidence, negations, constraints = inherited
        return _finalize(
            started=started,
            signal=signal,
            intent=intent,
            confidence=confidence,
            source="context",
            evidence=evidence,
            negations=negations,
            constraints=constraints,
            execution_mode=execution_mode,
            can_search=can_search,
            classifier=unavailable,
            inherited=True,
        )
    if signal.intent is not None and not signal.needs_classifier:
        return _finalize(
            started=started,
            signal=signal,
            intent=signal.intent,
            confidence=signal.confidence,
            source="rules",
            evidence=signal.evidence_spans,
            negations=signal.negations,
            constraints=signal.constraints,
            execution_mode=execution_mode,
            can_search=can_search,
            classifier=unavailable,
            inherited=False,
        )
    classified = await aclassify_intent(
        message,
        model=classifier_model,
        prior_intent=context.intent if context else None,
        timeout_s=intent_classifier_timeout_s(),
    )
    intent, confidence, evidence = _classifier_choice(classified)
    safe_evidence = tuple(span for span in evidence if span and span in message)[:16]
    return _finalize(
        started=started,
        signal=signal,
        intent=intent,
        confidence=confidence,
        source="classifier",
        evidence=safe_evidence,
        negations=signal.negations,
        constraints=signal.constraints,
        execution_mode=execution_mode,
        can_search=can_search,
        classifier=classified,
        inherited=False,
    )


def record_intent_prediction(
    resolution: IntentResolution,
    *,
    run_id: str | None,
    thread_id: str,
) -> None:
    """Record only bounded policy metadata, never message or evidence text."""
    decision = resolution.decision
    record_decision_point(
        phase="intent_prediction_v1",
        detail={
            "primary_intent": decision.primary_intent,
            "action": decision.action,
            "confidence": decision.confidence,
            "decision_source": decision.decision_source,
            "policy_version": decision.policy_version,
            "signal_ids": list(resolution.signal_ids),
            "classifier_status": resolution.classifier_status,
            "classifier_attempts": resolution.classifier_attempts,
            "context_inherited": resolution.context_inherited,
            "latency_ms": resolution.latency_ms,
        },
        run_id=run_id,
        thread_id=thread_id,
    )


def record_intent_outcome(
    resolution: IntentResolution,
    *,
    observed_route: str,
    run_id: str | None,
    thread_id: str,
) -> None:
    """Record a confusion-matrix row without enforcing the prediction."""
    predicted = (
        "read_search"
        if resolution.decision.primary_intent in _READ_INTENTS
        else resolution.decision.primary_intent
    )
    observed = {
        "search": "read_search",
        "profile_save": "profile_update",
        "clarification": "clarification",
        "chit_chat": "conversation",
    }.get(observed_route, "unknown")
    record_decision_point(
        phase="intent_observed_outcome_v1",
        detail={
            "predicted": predicted,
            "predicted_intent": resolution.decision.primary_intent,
            "predicted_action": resolution.decision.action,
            "observed": observed,
            "observed_route": observed_route,
            "matched": predicted == observed,
            "policy_version": resolution.decision.policy_version,
        },
        run_id=run_id,
        thread_id=thread_id,
    )
