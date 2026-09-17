"""Authoritative planner-backed resolver for group-agent intent routing."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from apps.group_agent_api.agent_factory.debug_trace import record_decision_point

from .context import (
    IntentContextScope,
    build_planner_context,
    extract_completed_context,
)
from .planner import PlannerResult, aplan_intent, plan_intent
from .policy import build_decision_from_planner, clarification_decision
from .schema import IntentDecisionV1

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
    context_compressed: bool = False

    @property
    def planner_status(self) -> str:
        """Alias for thin-harness observability; same value as classifier_status."""
        return self.classifier_status

    @property
    def planner_attempts(self) -> int:
        return self.classifier_attempts

    @property
    def routing_class(self) -> str:
        """Thin-harness observability label; does not affect routing decisions."""
        return _routing_class(
            source=self.decision.decision_source,
            action=self.decision.action,
            planner_status=self.classifier_status,
        )


def _routing_class(
    *,
    source: str,
    action: str,
    planner_status: str,
) -> str:
    """Map a finished decision onto the thin-harness observability classes."""
    if action == "reject":
        return "policy_rejection"
    if source == "planner":
        if planner_status == "ok":
            return "authoritative_planner"
        return "clarification_fallback"
    if action == "clarify":
        return "clarification_fallback"
    if action == "delegate_existing_flow":
        return "delegate_existing_flow"
    return "authoritative_planner"


def intent_routing_mode() -> str:
    """Return the explicit off, observe, or enforce routing mode."""
    mode = os.environ.get("GROUP_AGENT_INTENT_ROUTING_MODE", "off").strip().lower()
    if mode not in {"off", "observe", "enforce"}:
        msg = "GROUP_AGENT_INTENT_ROUTING_MODE must be off, observe, or enforce"
        raise RuntimeError(msg)
    return mode


def intent_classifier_timeout_s() -> float:
    """Return the bounded total planner timeout (legacy env name retained)."""
    try:
        value = float(os.environ.get("GROUP_AGENT_INTENT_CLASSIFIER_TIMEOUT_S", "1.5"))
    except ValueError as exc:
        msg = "GROUP_AGENT_INTENT_CLASSIFIER_TIMEOUT_S must be numeric"
        raise RuntimeError(msg) from exc
    return min(30.0, max(0.05, value))


def intent_planner_timeout_s() -> float:
    """Return the bounded total planner timeout."""
    return intent_classifier_timeout_s()


def _finalize(
    *,
    started: float,
    decision: IntentDecisionV1,
    planner: PlannerResult,
    context_inherited: bool,
    context_compressed: bool,
    signal_ids: tuple[str, ...] = (),
) -> IntentResolution:
    return IntentResolution(
        decision=decision,
        latency_ms=max(0, int((time.perf_counter() - started) * 1000)),
        signal_ids=signal_ids,
        classifier_status=planner.status,
        classifier_attempts=planner.attempts,
        context_inherited=context_inherited,
        context_compressed=context_compressed,
    )


def _decision_from_planner(
    planner: PlannerResult,
    *,
    execution_mode: str,
    can_search: bool,
) -> IntentDecisionV1:
    if planner.proposal is None or planner.status != "ok":
        return clarification_decision(source="planner")
    return build_decision_from_planner(
        planner.proposal,
        execution_mode=execution_mode,
        can_search=can_search,
    )


def resolve_intent(
    message: str,
    *,
    execution_mode: str,
    can_search: bool,
    prior_messages: list[Any] | None = None,
    scope: IntentContextScope | None = None,
    classifier_model: Any = None,
    planner_model: Any = None,
    profile: Any | None = None,
    profile_constraints: Any | None = None,
    profile_negations: tuple[str, ...] = (),
) -> IntentResolution:
    """Resolve one turn via the authoritative planner on every turn.

    Args:
        message: Current user message.
        execution_mode: Trusted mode (`group_agent` or `cafe_host`).
        can_search: Trusted public-read capability.
        prior_messages: Messages from the exact scoped checkpoint.
        scope: Provenance for those messages; required for completed-context hints.
        classifier_model: Deprecated alias for ``planner_model``.
        planner_model: Bare model used for structured planner output.
        profile: Persistent profile object including free-text doing/need/offer.
        profile_constraints: Ignored legacy Phase 1 argument.
        profile_negations: Ignored legacy Phase 1 argument.

    Returns:
        A contract-valid decision and bounded observe diagnostics.
    """
    del profile_constraints, profile_negations
    started = time.perf_counter()
    model = planner_model if planner_model is not None else classifier_model
    trusted = (
        extract_completed_context(prior_messages or [], scope=scope)
        if scope is not None
        else None
    )
    context = build_planner_context(
        message=message,
        prior_messages=prior_messages,
        profile=profile,
        trusted=trusted,
    )
    planned = plan_intent(model=model, context=context)
    decision = _decision_from_planner(
        planned, execution_mode=execution_mode, can_search=can_search
    )
    return _finalize(
        started=started,
        decision=decision,
        planner=planned,
        context_inherited=trusted is not None,
        context_compressed=context.context_compressed,
        signal_ids=("context_compressed",) if context.context_compressed else (),
    )


async def aresolve_intent(
    message: str,
    *,
    execution_mode: str,
    can_search: bool,
    prior_messages: list[Any] | None = None,
    scope: IntentContextScope | None = None,
    classifier_model: Any = None,
    planner_model: Any = None,
    profile: Any | None = None,
    profile_constraints: Any | None = None,
    profile_negations: tuple[str, ...] = (),
) -> IntentResolution:
    """Async equivalent of `resolve_intent`, including a bounded planner timeout."""
    del profile_constraints, profile_negations
    started = time.perf_counter()
    model = planner_model if planner_model is not None else classifier_model
    trusted = (
        extract_completed_context(prior_messages or [], scope=scope)
        if scope is not None
        else None
    )
    context = build_planner_context(
        message=message,
        prior_messages=prior_messages,
        profile=profile,
        trusted=trusted,
    )
    planned = await aplan_intent(
        model=model,
        context=context,
        timeout_s=intent_planner_timeout_s(),
    )
    decision = _decision_from_planner(
        planned, execution_mode=execution_mode, can_search=can_search
    )
    return _finalize(
        started=started,
        decision=decision,
        planner=planned,
        context_inherited=trusted is not None,
        context_compressed=context.context_compressed,
        signal_ids=("context_compressed",) if context.context_compressed else (),
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
            "planner_version": decision.planner_version,
            "signal_ids": list(resolution.signal_ids),
            "classifier_status": resolution.classifier_status,
            "classifier_attempts": resolution.classifier_attempts,
            "planner_status": resolution.planner_status,
            "planner_attempts": resolution.planner_attempts,
            "context_inherited": resolution.context_inherited,
            "context_compressed": resolution.context_compressed,
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
