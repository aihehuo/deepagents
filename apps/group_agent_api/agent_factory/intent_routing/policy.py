"""Pure decision policy mapping planner proposals to frozen IntentDecisionV1."""

from __future__ import annotations

from .schema import (
    IntentConstraintsV1,
    IntentDecisionV1,
    IntentKind,
    PlannerProposalV2,
    QueryPlanRefV1,
    ResourceQueriesV2,
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


def _clip_queries(queries: ResourceQueriesV2) -> ResourceQueriesV2:
    def clip(value: str | None) -> str | None:
        if value is None:
            return None
        text = value.strip()[:160]
        return text or None

    return ResourceQueriesV2(
        users=clip(queries.users),
        ideas=clip(queries.ideas),
        projects=clip(queries.projects),
        groups=clip(queries.groups),
        faqs=clip(queries.faqs),
    )


def build_decision_from_planner(
    proposal: PlannerProposalV2,
    *,
    execution_mode: str,
    can_search: bool,
    paid_execution_authorized: bool = False,
) -> IntentDecisionV1:
    """Map a validated planner proposal onto the server-owned decision contract.

    The planner owns semantic intent and query wording. This function only
    applies authorization and registered plan binding. It never invents query
    terms from lexicons.
    """
    queries = _clip_queries(proposal.queries)
    constraints = IntentConstraintsV1(time_range=proposal.time_range)

    if proposal.action == "query":
        intent: IntentKind = proposal.intent or "clarification"
        route = _READ_ROUTES.get(intent)
        if route is None:
            return IntentDecisionV1(
                primary_intent="clarification",
                confidence="low",
                decision_source="planner",
                evidence_spans=(),
                constraints=constraints,
                negations=(),
                action="clarify",
                planner_version=proposal.planner_version,
            )
        cafe_paid = execution_mode == "cafe_host" and paid_execution_authorized and intent in {
            "people_search",
            "opportunity_search",
            "project_partnership_search",
        }
        if not ((execution_mode == "group_agent" and can_search) or cafe_paid):
            return IntentDecisionV1(
                primary_intent=intent,
                confidence="high",
                decision_source="planner",
                evidence_spans=(),
                constraints=constraints,
                negations=(),
                action="reject",
                planner_version=proposal.planner_version,
            )
        skill, plan_id = route
        return IntentDecisionV1(
            primary_intent=intent,
            confidence="high",
            decision_source="planner",
            evidence_spans=(),
            constraints=constraints,
            negations=(),
            action="execute",
            required_skill=skill,
            query_plan_ref=QueryPlanRefV1(plan_id=plan_id),
            requires_plan_receipts=True,
            resource_queries=queries,
            group_member_scope=proposal.group_member_scope,
            group_event_id=proposal.group_event_id,
            planner_version=proposal.planner_version,
        )

    if proposal.action == "clarify":
        return IntentDecisionV1(
            primary_intent="clarification",
            confidence="low",
            decision_source="planner",
            evidence_spans=(),
            constraints=constraints,
            negations=(),
            action="clarify",
            planner_version=proposal.planner_version,
        )

    intent = proposal.intent or "conversation"
    if intent == "profile_update":
        return IntentDecisionV1(
            primary_intent="profile_update",
            confidence="high",
            decision_source="planner",
            evidence_spans=(),
            constraints=constraints,
            negations=(),
            action="delegate_existing_flow",
            planner_version=proposal.planner_version,
        )
    return IntentDecisionV1(
        primary_intent="conversation",
        confidence="high",
        decision_source="planner",
        evidence_spans=(),
        constraints=constraints,
        negations=(),
        action="conversation",
        planner_version=proposal.planner_version,
    )


def clarification_decision(*, source: str = "planner") -> IntentDecisionV1:
    """Fail-closed clarification used when the planner is unavailable or invalid."""
    decision_source = "planner" if source == "planner" else "classifier"
    return IntentDecisionV1(
        primary_intent="clarification",
        confidence="low",
        decision_source=decision_source,  # type: ignore[arg-type]
        evidence_spans=(),
        constraints=IntentConstraintsV1(),
        negations=(),
        action="clarify",
        planner_version="intent-planner-v2" if source == "planner" else None,
    )
