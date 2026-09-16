"""Frozen six-intent public read registry."""

from __future__ import annotations

from .contracts import QueryBudgetV1, QueryPlanStepV1, QueryPlanV1, RegistryEntryV1
from .schema import ReadIntent

REGISTRY_VERSION = "aihehuo-read-registry-v1"
_PARAMETER_FOR_RESOURCE = {
    "users": "people-query-v1",
    "ideas": "ideas-query-v1",
    "projects": "projects-query-v1",
    "groups": "groups-query-v1",
    "group-members": "group-members-query-v1",
    "weekly": "weekly-query-v1",
    "blog": "blog-query-v1",
    "faqs": "faqs-query-v1",
}


def _step(
    step_id: str,
    resource: str,
    *,
    kind: str = "required",
    depends_on: tuple[str, ...] = (),
    trigger: str = "always",
) -> QueryPlanStepV1:
    return QueryPlanStepV1(
        step_id=step_id,
        kind=kind,
        resource=resource,
        depends_on=depends_on,
        trigger=trigger,
        parameter_policy=_PARAMETER_FOR_RESOURCE[resource],
    )


def _plan(plan_id: str, *steps: QueryPlanStepV1) -> QueryPlanV1:
    people = plan_id == "people-search-v1"
    max_requests = 2 if people else 4
    return QueryPlanV1(
        plan_id=plan_id,
        steps=steps,
        budget=QueryBudgetV1(
            max_requests=max_requests,
            max_pages=2 if people else 4,
            max_items=40 if people else 100,
            max_elapsed_ms=15_000 * max_requests,
        ),
        scope_disclosure=("pages_queried", "items_examined"),
    )


READ_REGISTRY: dict[ReadIntent, RegistryEntryV1] = {
    "people_search": RegistryEntryV1(
        intent="people_search",
        skill="get-ai-cofounders",
        plan=_plan("people-search-v1", _step("users-primary", "users")),
    ),
    "opportunity_search": RegistryEntryV1(
        intent="opportunity_search",
        skill="get-ai-opportunities",
        plan=_plan(
            "opportunity-search-v1",
            _step("ideas-primary", "ideas"),
            _step(
                "projects-if-requested",
                "projects",
                kind="conditional",
                depends_on=("ideas-primary",),
                trigger="user_object_includes_projects",
            ),
        ),
    ),
    "project_partnership_search": RegistryEntryV1(
        intent="project_partnership_search",
        skill="get-ai-project-partnerships",
        plan=_plan(
            "project-partnership-search-v1",
            _step("projects-primary", "projects"),
            _step(
                "ideas-if-needed",
                "ideas",
                kind="conditional",
                depends_on=("projects-primary",),
                trigger="primary_needs_idea_evidence",
            ),
        ),
    ),
    "community_search": RegistryEntryV1(
        intent="community_search",
        skill="get-ai-communities",
        plan=_plan(
            "community-search-v1",
            _step("groups-primary", "groups"),
            _step(
                "members-if-selected",
                "group-members",
                kind="conditional",
                depends_on=("groups-primary",),
                trigger="primary_result_selected",
            ),
        ),
    ),
    "startup_digest": RegistryEntryV1(
        intent="startup_digest",
        skill="get-ai-startup-digest",
        plan=_plan(
            "startup-digest-v1",
            _step("weekly-primary", "weekly"),
            _step(
                "blog-if-complete",
                "blog",
                kind="conditional",
                depends_on=("weekly-primary",),
                trigger="user_requested_complete_scope",
            ),
        ),
    ),
    "platform_help": RegistryEntryV1(
        intent="platform_help",
        skill="get-ai-platform-help",
        plan=_plan("platform-help-v1", _step("faqs-primary", "faqs")),
    ),
}


def registry_entry(intent: str) -> RegistryEntryV1 | None:
    """Return the immutable registry entry for one read intent."""
    return READ_REGISTRY.get(intent)  # type: ignore[arg-type]


def validate_registry(*, installed_skills: set[str] | None = None) -> None:
    """Validate cardinality, unique plans, decision bindings, and installation."""
    if len(READ_REGISTRY) != 6:
        raise RuntimeError("query_plan_invalid")
    plans = {entry.plan.plan_id for entry in READ_REGISTRY.values()}
    if len(plans) != 6:
        raise RuntimeError("query_plan_invalid")
    for intent, entry in READ_REGISTRY.items():
        if entry.intent != intent or not any(step.kind == "required" for step in entry.plan.steps):
            raise RuntimeError("query_plan_invalid")
        for step in entry.plan.steps:
            if step.parameter_policy != _PARAMETER_FOR_RESOURCE[step.resource]:
                raise RuntimeError("query_plan_invalid")
        if installed_skills is not None and entry.skill not in installed_skills:
            raise RuntimeError("query_plan_invalid")
