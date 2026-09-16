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
    bound_intent_constraints,
    build_classifier_context,
    extract_completed_context,
    has_structured_search_object,
    is_elliptical_followup,
    merge_classifier_constraints,
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
_OBJECT_REQUIRED_INTENTS = {
    "people_search",
    "opportunity_search",
    "project_partnership_search",
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

    @property
    def routing_class(self) -> str:
        """Thin-harness observability label; does not affect routing decisions."""
        return _routing_class(
            source=self.decision.decision_source,
            action=self.decision.action,
            classifier_status=self.classifier_status,
        )


def _routing_class(
    *,
    source: DecisionSource,
    action: str,
    classifier_status: str,
) -> str:
    """Map a finished decision onto the thin-harness observability classes.

    This does not change routing. It only labels the path already taken.
    Classifier success (including intentional clarify) is ``semantic_planner``;
    classifier failure fallbacks are ``clarification_fallback``.
    """
    if action == "reject":
        return "policy_rejection"
    if source == "context":
        return "context_inheritance"
    if source == "classifier":
        if classifier_status == "ok":
            return "semantic_planner"
        return "clarification_fallback"
    if action == "execute":
        return "l0_positive"
    if action == "delegate_existing_flow":
        return "delegate_existing_flow"
    if action == "clarify":
        return "clarification_fallback"
    return "l0_conversation"


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


def _classifier_constraints(
    *,
    intent: IntentKind,
    signal: SignalResult,
    trusted: TrustedIntentContext | None,
    profile: IntentConstraintsV1 | None,
) -> IntentConstraintsV1:
    """Merge profile → trusted → current L0 constraints into the frozen decision."""
    if intent not in _READ_INTENTS:
        return bound_intent_constraints(signal.constraints)
    merged = IntentConstraintsV1()
    if profile is not None:
        merged = merge_classifier_constraints(merged, profile)
    if trusted is not None:
        merged = merge_classifier_constraints(merged, trusted.constraints)
    return merge_classifier_constraints(merged, signal.constraints)


def _apply_object_requirement(
    *,
    intent: IntentKind,
    confidence: Confidence,
    evidence: tuple[str, ...],
    constraints: IntentConstraintsV1,
) -> tuple[IntentKind, Confidence, tuple[str, ...], IntentConstraintsV1]:
    """Fail closed when a search intent has no structured object to execute."""
    if intent in _OBJECT_REQUIRED_INTENTS and not has_structured_search_object(
        constraints
    ):
        return "clarification", "low", evidence, constraints
    return intent, confidence, evidence, constraints


def _merge_negations(*groups: tuple[str, ...]) -> tuple[str, ...]:
    merged: list[str] = []
    for group in groups:
        for item in group:
            text = str(item or "").strip()
            if text and text not in merged:
                merged.append(text)
            if len(merged) >= 16:
                return tuple(merged)
    return tuple(merged)


def _classify_and_finalize(
    *,
    started: float,
    message: str,
    signal: SignalResult,
    trusted: TrustedIntentContext | None,
    profile_constraints: IntentConstraintsV1 | None,
    profile_negations: tuple[str, ...],
    execution_mode: str,
    can_search: bool,
    classified: ClassifierResult,
) -> IntentResolution:
    intent, confidence, evidence = _classifier_choice(classified)
    safe_evidence = tuple(span for span in evidence if span and span in message)[:16]
    constraints = _classifier_constraints(
        intent=intent,
        signal=signal,
        trusted=trusted,
        profile=profile_constraints,
    )
    intent, confidence, safe_evidence, constraints = _apply_object_requirement(
        intent=intent,
        confidence=confidence,
        evidence=safe_evidence,
        constraints=constraints,
    )
    return _finalize(
        started=started,
        signal=signal,
        intent=intent,
        confidence=confidence,
        source="classifier",
        evidence=safe_evidence,
        negations=_merge_negations(signal.negations, profile_negations),
        constraints=constraints,
        execution_mode=execution_mode,
        can_search=can_search,
        classifier=classified,
        inherited=False,
    )


def resolve_intent(
    message: str,
    *,
    execution_mode: str,
    can_search: bool,
    prior_messages: list[Any] | None = None,
    scope: IntentContextScope | None = None,
    classifier_model: Any = None,
    profile_constraints: IntentConstraintsV1 | None = None,
    profile_negations: tuple[str, ...] = (),
) -> IntentResolution:
    """Resolve one turn without changing the execution path.

    Args:
        message: Current user message.
        execution_mode: Trusted mode (`group_agent` or `cafe_host`).
        can_search: Trusted public-read capability.
        prior_messages: Messages from the exact scoped checkpoint.
        scope: Provenance for those messages; required for inheritance.
        classifier_model: Optional bare model used only for L2 structured output.
        profile_constraints: Optional structured profile fields for L1 context.
        profile_negations: Optional profile exclusions as ``kind:value`` tokens.

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
            negations=_merge_negations(negations, profile_negations),
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
            negations=_merge_negations(signal.negations, profile_negations),
            constraints=signal.constraints,
            execution_mode=execution_mode,
            can_search=can_search,
            classifier=unavailable,
            inherited=False,
        )
    classifier_context = build_classifier_context(
        trusted=context,
        prior_messages=prior_messages,
        profile=profile_constraints,
        current_negations=_merge_negations(signal.negations, profile_negations),
    )
    classified = classify_intent(
        message,
        model=classifier_model,
        context=classifier_context,
    )
    return _classify_and_finalize(
        started=started,
        message=message,
        signal=signal,
        trusted=context,
        profile_constraints=profile_constraints,
        profile_negations=profile_negations,
        execution_mode=execution_mode,
        can_search=can_search,
        classified=classified,
    )


async def aresolve_intent(
    message: str,
    *,
    execution_mode: str,
    can_search: bool,
    prior_messages: list[Any] | None = None,
    scope: IntentContextScope | None = None,
    classifier_model: Any = None,
    profile_constraints: IntentConstraintsV1 | None = None,
    profile_negations: tuple[str, ...] = (),
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
            negations=_merge_negations(negations, profile_negations),
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
            negations=_merge_negations(signal.negations, profile_negations),
            constraints=signal.constraints,
            execution_mode=execution_mode,
            can_search=can_search,
            classifier=unavailable,
            inherited=False,
        )
    classifier_context = build_classifier_context(
        trusted=context,
        prior_messages=prior_messages,
        profile=profile_constraints,
        current_negations=_merge_negations(signal.negations, profile_negations),
    )
    classified = await aclassify_intent(
        message,
        model=classifier_model,
        context=classifier_context,
        timeout_s=intent_classifier_timeout_s(),
    )
    return _classify_and_finalize(
        started=started,
        message=message,
        signal=signal,
        trusted=context,
        profile_constraints=profile_constraints,
        profile_negations=profile_negations,
        execution_mode=execution_mode,
        can_search=can_search,
        classified=classified,
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
            "routing_class": resolution.routing_class,
            "query_plan_would_run": bool(
                decision.action == "execute" and decision.query_plan_ref is not None
            ),
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
