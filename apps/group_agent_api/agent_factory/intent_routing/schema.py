"""Closed Pydantic schemas for the INTENT-XCUT-000 decision contract."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

IntentKind = Literal[
    "people_search",
    "opportunity_search",
    "project_partnership_search",
    "community_search",
    "startup_digest",
    "platform_help",
    "profile_update",
    "conversation",
    "clarification",
]
ReadIntent = Literal[
    "people_search",
    "opportunity_search",
    "project_partnership_search",
    "community_search",
    "startup_digest",
    "platform_help",
]
Confidence = Literal["low", "medium", "high"]
DecisionSource = Literal["rules", "context", "classifier", "combined", "planner"]
DecisionAction = Literal[
    "execute", "clarify", "conversation", "reject", "delegate_existing_flow"
]
PlannerAction = Literal["query", "clarify", "reply"]
PlannerTimeRange = Literal["本周", "本月", "最近", "近期"]
EpisodeTurnRole = Literal["human", "assistant"]
GroupMemberScope = Literal["none", "first_verified"]

_INTENT_PRIMARY_QUERY_FIELD: dict[str, str | None] = {
    "people_search": "users",
    "opportunity_search": "ideas",
    "project_partnership_search": "projects",
    "community_search": "groups",
    "startup_digest": None,
    "platform_help": "faqs",
}


class ClosedModel(BaseModel):
    """Reject forward fields until a new contract version supports them."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class IntentConstraintsV1(ClosedModel):
    """Normalized constraints retained for markers and calendar parameters."""

    location: tuple[str, ...] = Field(default=(), max_length=10)
    industry: tuple[str, ...] = Field(default=(), max_length=10)
    role: tuple[str, ...] = Field(default=(), max_length=10)
    capability: tuple[str, ...] = Field(default=(), max_length=10)
    time_range: str | None = Field(default=None, max_length=64)


class ResourceQueriesV2(ClosedModel):
    """Closed resource-specific search text frozen before any public request."""

    users: str | None = Field(default=None, max_length=160)
    ideas: str | None = Field(default=None, max_length=160)
    projects: str | None = Field(default=None, max_length=160)
    groups: str | None = Field(default=None, max_length=160)
    faqs: str | None = Field(default=None, max_length=160)

    def for_resource(self, resource: str) -> str | None:
        """Return the frozen query string for a registered resource, if any."""
        mapping = {
            "users": self.users,
            "ideas": self.ideas,
            "projects": self.projects,
            "groups": self.groups,
            "faqs": self.faqs,
        }
        value = mapping.get(resource)
        if value is None:
            return None
        text = value.strip()
        return text or None

    def non_empty(self) -> bool:
        return any(
            value and value.strip()
            for value in (
                self.users,
                self.ideas,
                self.projects,
                self.groups,
                self.faqs,
            )
        )


class ClassifierContextV1(ClosedModel):
    """Historical L1 context shape retained only for snapshot/test compatibility."""

    prior_intent: IntentKind | None = None
    prior_constraints: IntentConstraintsV1 = Field(default_factory=IntentConstraintsV1)
    profile_location: tuple[str, ...] = Field(default=(), max_length=10)
    profile_industry: tuple[str, ...] = Field(default=(), max_length=10)
    profile_role: tuple[str, ...] = Field(default=(), max_length=10)
    profile_capability: tuple[str, ...] = Field(default=(), max_length=10)
    recent_user_turns: tuple[str, ...] = Field(default=(), max_length=5)
    current_negations: tuple[str, ...] = Field(default=(), max_length=16)


class PlannerTurnV2(ClosedModel):
    """One bounded current-episode turn supplied to the planner as data."""

    role: EpisodeTurnRole
    content: str = Field(min_length=1, max_length=1000)


class PlannerProfileV2(ClosedModel):
    """Persistent profile fields the planner may read; never credentials."""

    doing: str | None = Field(default=None, max_length=500)
    need: str | None = Field(default=None, max_length=500)
    offer: str | None = Field(default=None, max_length=500)
    match_constraints: tuple[str, ...] = Field(default=(), max_length=32)


class PlannerContextV2(ClosedModel):
    """Closed planner input: full retained episode, profile, and provenance flags."""

    current_message: str = Field(min_length=1, max_length=2000)
    episode_turns: tuple[PlannerTurnV2, ...] = Field(default=(), max_length=40)
    profile: PlannerProfileV2 = Field(default_factory=PlannerProfileV2)
    prior_completed_intent: IntentKind | None = None
    prior_completed_queries: ResourceQueriesV2 = Field(default_factory=ResourceQueriesV2)
    context_compressed: bool = False
    compression_reason: str | None = Field(default=None, max_length=64)


class PlannerProposalV2(ClosedModel):
    """Authoritative semantic and query proposal for one turn."""

    version: Literal[2] = 2
    action: PlannerAction
    intent: IntentKind | None = None
    queries: ResourceQueriesV2 = Field(default_factory=ResourceQueriesV2)
    clarification: str | None = Field(default=None, max_length=240)
    time_range: PlannerTimeRange | None = None
    group_member_scope: GroupMemberScope = "none"
    group_event_id: str | None = Field(
        default=None, max_length=18, pattern=r"^[0-9]{1,18}$"
    )
    planner_version: Literal["intent-planner-v2"] = "intent-planner-v2"

    @model_validator(mode="after")
    def validate_action_bindings(self) -> PlannerProposalV2:
        """Enforce closed action/intent/query bindings before policy mapping."""
        if self.action == "query":
            if self.intent is None or self.intent not in _INTENT_PRIMARY_QUERY_FIELD:
                msg = "query action requires a registered read intent"
                raise ValueError(msg)
            if self.intent in {"profile_update", "conversation", "clarification"}:
                msg = "query action cannot use non-read intents"
                raise ValueError(msg)
            primary = _INTENT_PRIMARY_QUERY_FIELD[self.intent]
            if primary is not None and self.queries.for_resource(primary) is None:
                msg = f"query action for {self.intent} requires a {primary} query"
                raise ValueError(msg)
            if self.clarification is not None:
                msg = "query action cannot include clarification text"
                raise ValueError(msg)
            if self.intent != "community_search" and (
                self.group_member_scope != "none" or self.group_event_id is not None
            ):
                msg = "group member scope is only valid for community_search"
                raise ValueError(msg)
            return self
        if self.group_member_scope != "none" or self.group_event_id is not None:
            msg = "non-query actions cannot request group members"
            raise ValueError(msg)
        if self.action == "clarify":
            if self.intent not in {None, "clarification"}:
                msg = "clarify action requires clarification intent"
                raise ValueError(msg)
            if self.queries.non_empty():
                msg = "clarify action cannot include resource queries"
                raise ValueError(msg)
            return self
        if self.intent not in {None, "conversation", "profile_update"}:
            msg = "reply action must be conversation or profile_update"
            raise ValueError(msg)
        if self.queries.non_empty():
            msg = "reply action cannot include resource queries"
            raise ValueError(msg)
        return self


class QueryPlanRefV1(ClosedModel):
    """Reference to a plan frozen by INTENT-XCUT-000."""

    plan_version: Literal[1] = 1
    plan_id: str = Field(min_length=3, max_length=64, pattern=r"^[a-z][a-z0-9-]+-v1$")


class IntentCandidateV1(ClosedModel):
    """One bounded classifier candidate (historical Phase 1 shape)."""

    intent: IntentKind
    confidence: Confidence


class IntentClassificationV1(ClosedModel):
    """Historical L2 classifier response retained for test fixtures only."""

    version: Literal[1] = 1
    candidates: tuple[IntentCandidateV1, ...] = Field(min_length=1, max_length=3)
    evidence_spans: tuple[str, ...] = Field(default=(), max_length=16)
    missing_object: str | None = Field(default=None, max_length=128)
    classifier_version: Literal["intent-classifier-v1"] = "intent-classifier-v1"


class IntentDecisionV1(ClosedModel):
    """Authoritative pre-generation decision shape used by later INTENT REQs."""

    version: Literal[1] = 1
    primary_intent: IntentKind
    confidence: Confidence
    decision_source: DecisionSource
    evidence_spans: tuple[str, ...] = Field(default=(), max_length=16)
    constraints: IntentConstraintsV1 = Field(default_factory=IntentConstraintsV1)
    negations: tuple[str, ...] = Field(default=(), max_length=16)
    action: DecisionAction
    required_skill: str | None = Field(default=None, max_length=128)
    query_plan_ref: QueryPlanRefV1 | None = None
    requires_plan_receipts: bool = False
    resource_queries: ResourceQueriesV2 = Field(default_factory=ResourceQueriesV2)
    group_member_scope: GroupMemberScope = "none"
    group_event_id: str | None = Field(
        default=None, max_length=18, pattern=r"^[0-9]{1,18}$"
    )
    planner_version: Literal["intent-planner-v2"] | None = None
    policy_version: Literal["intent-policy-v1"] = "intent-policy-v1"

    @model_validator(mode="after")
    def validate_action_binding(self) -> IntentDecisionV1:
        """Enforce execute/profile/clarification bindings from the frozen contract."""
        execute = self.action == "execute"
        if execute != bool(
            self.required_skill
            and self.query_plan_ref is not None
            and self.requires_plan_receipts
        ):
            msg = "execute decisions must exclusively own skill, plan, and receipt requirements"
            raise ValueError(msg)
        if execute:
            primary = _INTENT_PRIMARY_QUERY_FIELD.get(self.primary_intent)
            missing_primary = (
                primary is not None
                and self.resource_queries.for_resource(primary) is None
            )
            # Phase-2 planner decisions must freeze queries. Legacy Phase-1
            # snapshots (planner_version is None) may resume via constraint
            # synthesis in the snapshot reader / plan_executor.
            if missing_primary and self.planner_version is not None:
                msg = "execute decisions must freeze the primary resource query"
                raise ValueError(msg)
            if self.primary_intent != "community_search" and (
                self.group_member_scope != "none" or self.group_event_id is not None
            ):
                msg = "group member scope is only valid for community_search"
                raise ValueError(msg)
        elif self.resource_queries.non_empty():
            msg = "non-execute decisions cannot carry resource queries"
            raise ValueError(msg)
        elif self.group_member_scope != "none" or self.group_event_id is not None:
            msg = "non-execute decisions cannot request group members"
            raise ValueError(msg)
        if (
            self.primary_intent == "profile_update"
            and self.action != "delegate_existing_flow"
        ):
            msg = "profile_update must delegate to save_group_profile"
            raise ValueError(msg)
        if (
            self.action == "delegate_existing_flow"
            and self.primary_intent != "profile_update"
        ):
            msg = "only profile_update may delegate"
            raise ValueError(msg)
        if self.action == "clarify" and self.primary_intent != "clarification":
            msg = "clarify action requires clarification intent"
            raise ValueError(msg)
        return self
