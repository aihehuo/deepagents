"""Pure decision policy and frozen read-plan references for INTENT-DA-001."""

from __future__ import annotations

from .schema import (
    Confidence,
    DecisionSource,
    IntentConstraintsV1,
    IntentDecisionV1,
    IntentKind,
    QueryPlanRefV1,
)

_READ_ROUTES: dict[str, tuple[str, str]] = {
    "people_search": ("get-ai-cofounders", "people-search-v1"),
    "opportunity_search": ("get-ai-opportunities", "opportunity-search-v1"),
    "project_partnership_search": (
        "get-ai-project-partnerships",
        "project-partnership-search-v1",
    ),
    "community_search": ("get-ai-communities", "community-search-v1"),
    "startup_digest": ("get-ai-startup-digest", "startup-digest-v1"),
    "platform_help": ("get-ai-platform-help", "platform-help-v1"),
}


def build_decision(
    *,
    intent: IntentKind,
    confidence: Confidence,
    source: DecisionSource,
    evidence_spans: tuple[str, ...],
    constraints: IntentConstraintsV1,
    negations: tuple[str, ...],
    execution_mode: str,
    can_search: bool,
) -> IntentDecisionV1:
    """Build a contract-valid decision from trusted inputs.

    Args:
        intent: Resolved primary intent.
        confidence: Bounded resolver confidence.
        source: Highest routing layer used.
        evidence_spans: Short matched phrases only.
        constraints: Normalized query constraints.
        negations: Explicit negative phrases.
        execution_mode: Trusted execution mode.
        can_search: Trusted public-read capability.

    Returns:
        A closed `IntentDecisionV1`. This function never executes a tool.
    """
    route = _READ_ROUTES.get(intent)
    if route is not None:
        if confidence == "low":
            intent = "clarification"
            action = "clarify"
            route = None
        elif execution_mode != "group_agent" or not can_search:
            action = "reject"
        else:
            action = "execute"
    elif intent == "profile_update":
        action = "delegate_existing_flow"
    elif intent == "clarification":
        action = "clarify"
    else:
        action = "conversation"

    skill, plan_id = (
        route if action == "execute" and route is not None else (None, None)
    )
    return IntentDecisionV1(
        primary_intent=intent,
        confidence=confidence,
        decision_source=source,
        evidence_spans=evidence_spans[:16],
        constraints=constraints,
        negations=negations[:16],
        action=action,
        required_skill=skill,
        query_plan_ref=QueryPlanRefV1(plan_id=plan_id) if plan_id else None,
        requires_plan_receipts=bool(plan_id),
    )
