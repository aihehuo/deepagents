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
DecisionSource = Literal["rules", "context", "classifier", "combined"]
DecisionAction = Literal[
    "execute", "clarify", "conversation", "reject", "delegate_existing_flow"
]


class ClosedModel(BaseModel):
    """Reject forward fields until a new contract version supports them."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class IntentConstraintsV1(ClosedModel):
    """Normalized constraints extracted without retaining the full message."""

    location: tuple[str, ...] = Field(default=(), max_length=10)
    industry: tuple[str, ...] = Field(default=(), max_length=10)
    role: tuple[str, ...] = Field(default=(), max_length=10)
    capability: tuple[str, ...] = Field(default=(), max_length=10)
    time_range: str | None = Field(default=None, max_length=64)


class QueryPlanRefV1(ClosedModel):
    """Reference to a plan frozen by INTENT-XCUT-000."""

    plan_version: Literal[1] = 1
    plan_id: str = Field(min_length=3, max_length=64, pattern=r"^[a-z][a-z0-9-]+-v1$")


class IntentCandidateV1(ClosedModel):
    """One bounded classifier candidate."""

    intent: IntentKind
    confidence: Confidence


class IntentClassificationV1(ClosedModel):
    """Tool-free L2 classifier response."""

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
