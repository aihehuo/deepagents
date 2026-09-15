"""Closed INTENT-DA-002 QueryPlan, routing snapshot, and receipt models."""

from __future__ import annotations

import datetime
import re
from typing import Literal

from pydantic import Field, model_validator

from .canonical import canonical_hash
from .schema import ClosedModel, IntentDecisionV1, ReadIntent

_HASH = r"^sha256:[0-9a-f]{64}$"
_ID = r"^.{1,128}$"
_TIMESTAMP = r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{3})?Z$"
ResourceV1 = Literal[
    "users", "ideas", "projects", "groups", "group-members", "weekly", "blog", "faqs"
]
TriggerV1 = Literal[
    "always",
    "user_object_includes_projects",
    "primary_needs_idea_evidence",
    "primary_result_selected",
    "user_requested_complete_scope",
]
ParameterPolicyV1 = Literal[
    "people-query-v1",
    "ideas-query-v1",
    "projects-query-v1",
    "groups-query-v1",
    "group-members-query-v1",
    "weekly-query-v1",
    "blog-query-v1",
    "faqs-query-v1",
]


class QueryPlanStepV1(ClosedModel):
    step_id: str = Field(pattern=r"^[a-z][a-z0-9-]{2,63}$")
    kind: Literal["required", "conditional"]
    tool: Literal["query_aihehuo_public_data"] = "query_aihehuo_public_data"
    resource: ResourceV1
    depends_on: tuple[str, ...] = Field(default=(), max_length=8)
    trigger: TriggerV1
    parameter_policy: ParameterPolicyV1


class QueryBudgetV1(ClosedModel):
    max_requests: int = Field(ge=1, le=20)
    max_pages: int = Field(ge=1, le=20)
    max_items: int = Field(ge=1, le=500)
    max_elapsed_ms: int = Field(ge=100, le=60_000)


class MinSuccessConditionV1(ClosedModel):
    all_required_steps: Literal["succeeded_or_empty"] = "succeeded_or_empty"


class QueryPlanV1(ClosedModel):
    plan_version: Literal[1] = 1
    plan_id: str = Field(pattern=r"^[a-z][a-z0-9-]{2,63}-v1$")
    steps: tuple[QueryPlanStepV1, ...] = Field(min_length=1, max_length=8)
    min_success_condition: MinSuccessConditionV1 = Field(default_factory=MinSuccessConditionV1)
    budget: QueryBudgetV1
    scope_disclosure: tuple[
        Literal["pages_queried", "items_examined", "time_range", "resources_queried"], ...
    ] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def validate_graph(self) -> QueryPlanV1:
        ids = [step.step_id for step in self.steps]
        if len(ids) != len(set(ids)) or len(self.scope_disclosure) != len(set(self.scope_disclosure)):
            raise ValueError("query_plan_invalid")
        by_id = {step.step_id: step for step in self.steps}
        for step in self.steps:
            if len(step.depends_on) != len(set(step.depends_on)) or any(
                re.fullmatch(r"[a-z][a-z0-9-]{2,63}", item) is None
                for item in step.depends_on
            ):
                raise ValueError("query_plan_invalid")
            if (step.kind == "required") != (step.trigger == "always"):
                raise ValueError("query_plan_invalid")
            if step.step_id in step.depends_on or any(item not in by_id for item in step.depends_on):
                raise ValueError("query_plan_invalid")
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(step_id: str) -> None:
            if step_id in visiting:
                raise ValueError("query_plan_invalid")
            if step_id in visited:
                return
            visiting.add(step_id)
            for dependency in by_id[step_id].depends_on:
                visit(dependency)
            visiting.remove(step_id)
            visited.add(step_id)

        for step_id in ids:
            visit(step_id)
        return self


class RoutingSnapshotV1(ClosedModel):
    routing_snapshot_version: Literal[1] = 1
    run_id: str = Field(pattern=_ID)
    intent_decision: IntentDecisionV1
    query_plan: QueryPlanV1 | None
    policy_version: Literal["intent-policy-v1"] = "intent-policy-v1"
    registry_version: Literal["aihehuo-read-registry-v1"] = "aihehuo-read-registry-v1"
    canonical_content_hash: str = Field(pattern=_HASH)
    created_at: str = Field(pattern=_TIMESTAMP)

    @model_validator(mode="after")
    def validate_bindings(self) -> RoutingSnapshotV1:
        decision = self.intent_decision
        if decision.policy_version != self.policy_version:
            raise ValueError("routing_snapshot_invalid")
        if decision.action == "execute":
            if self.query_plan is None or decision.query_plan_ref is None:
                raise ValueError("query_plan_invalid")
            if (
                decision.query_plan_ref.plan_id != self.query_plan.plan_id
                or decision.query_plan_ref.plan_version != self.query_plan.plan_version
            ):
                raise ValueError("query_plan_invalid")
        elif self.query_plan is not None:
            raise ValueError("query_plan_invalid")
        without_hash = self.model_dump(mode="json")
        without_hash.pop("canonical_content_hash")
        if canonical_hash(without_hash) != self.canonical_content_hash:
            raise ValueError("routing_snapshot_invalid")
        return self


class QueryBudgetUsageV1(ClosedModel):
    usage_version: Literal[1] = 1
    run_id: str = Field(pattern=_ID)
    routing_snapshot_hash: str = Field(pattern=_HASH)
    requests_reserved: int = Field(ge=0)
    pages_reserved: int = Field(ge=0)
    items_committed: int = Field(ge=0)
    tool_elapsed_committed_ms: int = Field(ge=0)
    tool_elapsed_reserved_ms: int = Field(ge=0)
    next_request_index_by_step: dict[str, int]

    @model_validator(mode="after")
    def validate_indices(self) -> QueryBudgetUsageV1:
        if any(
            value < 1 or re.fullmatch(r"[a-z][a-z0-9-]{2,63}", key) is None
            for key, value in self.next_request_index_by_step.items()
        ):
            raise ValueError("receipt_invalid")
        return self


class ToolReceiptV1(ClosedModel):
    receipt_version: Literal[1] = 1
    receipt_id: str = Field(pattern=_HASH)
    run_id: str = Field(pattern=_ID)
    tool_call_id: str = Field(pattern=_ID)
    routing_snapshot_hash: str = Field(pattern=_HASH)
    plan_id: str = Field(pattern=r"^[a-z][a-z0-9-]{2,63}-v1$")
    plan_step_id: str = Field(pattern=r"^[a-z][a-z0-9-]{2,63}$")
    request_index: int = Field(ge=1)
    skill: str = Field(pattern=r"^get-ai-[a-z-]{2,80}$")
    tool: Literal["query_aihehuo_public_data"] = "query_aihehuo_public_data"
    resource: ResourceV1
    status: Literal["succeeded", "empty"]
    result_count: int = Field(ge=0, le=500)
    evidence_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    source_urls: tuple[str, ...] = Field(min_length=1, max_length=20)
    produced_by_attempt_id: str
    produced_by_attempt_number: int = Field(ge=1)
    started_at: str = Field(pattern=_TIMESTAMP)
    finished_at: str = Field(pattern=_TIMESTAMP)
    committed_at: str = Field(pattern=_TIMESTAMP)

    @model_validator(mode="after")
    def validate_status_count(self) -> ToolReceiptV1:
        if (self.status == "empty") != (self.result_count == 0):
            raise ValueError("receipt_invalid")
        if any(not url.startswith("https://www.aihehuo.com/") for url in self.source_urls):
            raise ValueError("receipt_invalid")
        timestamps = [
            datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
            for value in (self.started_at, self.finished_at, self.committed_at)
        ]
        if timestamps != sorted(timestamps):
            raise ValueError("receipt_invalid")
        return self


class ToolAttemptAuditV1(ClosedModel):
    audit_version: Literal[1] = 1
    run_id: str = Field(pattern=_ID)
    routing_snapshot_hash: str = Field(pattern=_HASH)
    plan_id: str = Field(pattern=r"^[a-z][a-z0-9-]{2,63}-v1$")
    plan_step_id: str = Field(pattern=r"^[a-z][a-z0-9-]{2,63}$")
    request_index: int = Field(ge=1)
    status: Literal["failed", "timed_out"]
    error_code: Literal["tool_failed", "tool_timed_out"]
    attempt_id: str = Field(pattern=_ID)
    attempt_number: int = Field(ge=1)
    started_at: str = Field(pattern=_TIMESTAMP)
    finished_at: str = Field(pattern=_TIMESTAMP)


class ReceiptAdoptionV1(ClosedModel):
    adoption_version: Literal[1] = 1
    receipt_id: str = Field(pattern=_HASH)
    produced_by_attempt_id: str = Field(pattern=_ID)
    produced_by_attempt_number: int = Field(ge=1)
    adopted_by_attempt_id: str = Field(pattern=_ID)
    adopted_by_attempt_number: int = Field(ge=1)
    adopted_at: str = Field(pattern=_TIMESTAMP)

    @model_validator(mode="after")
    def validate_attempt_advance(self) -> ReceiptAdoptionV1:
        if self.adopted_by_attempt_number <= self.produced_by_attempt_number:
            raise ValueError("receipt_invalid")
        return self


class ReceiptRefV1(ClosedModel):
    plan_step_id: str = Field(pattern=r"^[a-z][a-z0-9-]{2,63}$")
    request_index: int = Field(ge=1)
    receipt_id: str = Field(pattern=_HASH)


class AuditRefV1(ClosedModel):
    plan_step_id: str = Field(pattern=r"^[a-z][a-z0-9-]{2,63}$")
    request_index: int = Field(ge=1)
    status: Literal["failed", "timed_out"]


class AdoptionRefV1(ClosedModel):
    receipt_id: str = Field(pattern=_HASH)
    adopted_by_attempt_number: int = Field(ge=1)


class ReceiptLedgerV1(ClosedModel):
    ledger_version: Literal[1] = 1
    run_id: str = Field(pattern=_ID)
    routing_snapshot_hash: str = Field(pattern=_HASH)
    receipt_refs: tuple[ReceiptRefV1, ...] = Field(default=(), max_length=20)
    audit_refs: tuple[AuditRefV1, ...] = Field(default=(), max_length=20)
    adoption_refs: tuple[AdoptionRefV1, ...] = Field(default=(), max_length=20)
    budget: QueryBudgetUsageV1

    @model_validator(mode="after")
    def validate_unique_refs(self) -> ReceiptLedgerV1:
        keys = [(item.plan_step_id, item.request_index) for item in self.receipt_refs]
        if len(keys) != len(set(keys)):
            raise ValueError("receipt_invalid")
        return self


class RegistryEntryV1(ClosedModel):
    intent: ReadIntent
    skill: str
    plan: QueryPlanV1
