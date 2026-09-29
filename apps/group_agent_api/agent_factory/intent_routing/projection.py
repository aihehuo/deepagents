"""Project a completed server QueryPlan into reply and client-facing state."""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from apps.group_agent_api.agent_factory.checks.reply_grounding.protocol import (
    PublicReadGround,
)
from apps.group_agent_api.agent_factory.contact_scrub import (
    scrub_contact_text,
    scrub_display_name,
)
from apps.group_agent_api.agent_factory.match_stub import MAX_CANDIDATES

from .canonical import canonical_hash
from .constraint_lexicon import expand_constraint_token, phrase_in_text
from .execution import IntentExecution

_PUBLIC_NUMBER = re.compile(r"^[1-9][0-9]*$")


@dataclass(frozen=True)
class IntentSearchProjection:
    """Trusted current-turn search facts derived only from valid receipts."""

    completed: bool = False
    result_count: int = 0
    candidates: list[dict[str, Any]] = field(default_factory=list)
    # Only internal reads with a real App user_id become actionable cards.
    public_result_names: list[str] = field(default_factory=list)
    # Full receipt evidence (debug / audit). May include constraint-mismatched people.
    public_reads: list[PublicReadGround] = field(default_factory=list)
    # Public reads are retained for compatibility with existing callers.
    grounding_public_reads: list[PublicReadGround] = field(default_factory=list)
    pages: int = 0


def empty_search_reply(projection: IntentSearchProjection) -> str | None:
    """Return a truthful final reply when completed public reads returned nothing."""
    if not projection.completed or projection.result_count != 0:
        return None
    return (
        "本次公开检索已完成，但没有返回符合条件的公开记录。"
        "你可以调整检索条件后再试。"
    )


def project_intent_search(
    execution: IntentExecution | None,
) -> IntentSearchProjection:
    """Return the server-executed search state without inspecting model tool calls."""
    if (
        execution is None
        or execution.decision.action != "execute"
        or not execution.receipts
        or not execution.required_steps_satisfied()
    ):
        return IntentSearchProjection()

    reads: list[PublicReadGround] = []
    grounding_reads: list[PublicReadGround] = []
    candidates: list[dict[str, Any]] = []
    public_result_names: list[str] = []
    seen_ids: set[str] = set()
    hard_locations = execution.decision.constraints.location
    hard_location_result_count = 0
    steps = {step.step_id: step for step in execution.plan.steps}
    for receipt in execution.receipts:
        evidence = execution.evidence_by_receipt.get(receipt.receipt_id)
        if not _valid_receipt(execution, receipt, evidence, steps):
            return IntentSearchProjection()
        items = evidence if isinstance(evidence, list) else [evidence]
        document_data: Any = items
        candidate_items = items
        group_wrapper: dict[str, Any] | None = None
        if (
            receipt.resource == "group-members"
            and len(items) == 1
            and isinstance(items[0], dict)
        ):
            group_wrapper = items[0]
            document_data = items[0]
            members = items[0].get("members")
            candidate_items = members if isinstance(members, list) else []
        reads.append(
            PublicReadGround(
                resource=receipt.resource,
                source_url=receipt.source_urls[0],
                document={"data": document_data},
            )
        )
        if receipt.resource not in {"users", "group-members"}:
            grounding_reads.append(reads[-1])
            continue
        filtered_items: list[Any] = []
        for item in candidate_items:
            if (
                receipt.resource in {"users", "group-members"}
                and hard_locations
                and not _matches_public_location(item, hard_locations)
            ):
                continue
            filtered_items.append(item)
            if receipt.resource in {"users", "group-members"}:
                hard_location_result_count += 1
            if isinstance(item, dict) and len(public_result_names) < MAX_CANDIDATES:
                raw_name = str(item.get("name") or item.get("display_name") or item.get("nickname") or "").strip()
                if raw_name:
                    name = scrub_display_name(
                        raw_name, fallback_user_id=str(item.get("id") or "")
                    )[:64]
                    if name not in public_result_names:
                        public_result_names.append(name)
            candidate = _public_candidate(item)
            if candidate is None or candidate["user_id"] in seen_ids:
                continue
            if len(candidates) >= MAX_CANDIDATES:
                continue
            seen_ids.add(candidate["user_id"])
            candidates.append(candidate)
        if receipt.resource == "group-members" and group_wrapper is not None:
            filtered_document: Any = {**group_wrapper, "members": filtered_items}
        else:
            filtered_document = filtered_items
        grounding_reads.append(
            PublicReadGround(
                resource=receipt.resource,
                source_url=receipt.source_urls[0],
                document={"data": filtered_document},
            )
        )

    frozen = execution.recommendation_candidates
    if frozen is not None:
        candidates = deepcopy(list(frozen))
        public_result_names = [
            str(candidate.get("display_name") or "").strip()
            for candidate in candidates
            if str(candidate.get("display_name") or "").strip()
        ]

    return IntentSearchProjection(
        completed=True,
        result_count=(
            hard_location_result_count
            if hard_locations
            and execution.decision.primary_intent == "people_search"
            else execution.unique_result_count()
        ),
        candidates=candidates,
        public_result_names=public_result_names,
        public_reads=reads,
        grounding_public_reads=grounding_reads,
        pages=len(execution.receipts),
    )


def _matches_public_location(raw: Any, allowed: tuple[str, ...]) -> bool:
    """Require a public city/location field to satisfy an explicit hard boundary."""
    if not isinstance(raw, dict):
        return False
    location = _first_public_text(raw, ("city", "location"))
    if not location:
        return False
    return any(
        phrase_in_text(location, alias)
        for value in allowed
        for alias in expand_constraint_token("location", value)
    )


def _public_candidate(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    # The public API's `id` is User.number. Never use it as an App route ID.
    # The authenticated internal response adds the actual database user_id.
    raw_id = raw.get("user_id")
    user_id = str(raw_id).strip() if raw_id is not None else ""
    if not _PUBLIC_NUMBER.fullmatch(user_id):
        return None

    display = str(
        raw.get("display_name") or raw.get("name") or raw.get("nickname") or ""
    ).strip()[:64]
    display = scrub_display_name(display, fallback_user_id=user_id)
    candidate: dict[str, Any] = {
        "kind": "public",
        "identity_kind": "app_user_id",
        "user_id": user_id,
        "display_name": display or f"用户{user_id}",
        "bound": bool(raw.get("bound", True)),
    }
    profile_url = raw.get("profile_url") or raw.get("url")
    if isinstance(profile_url, str) and profile_url.strip():
        candidate["profile_url"] = profile_url.strip()
    facts: list[dict[str, str]] = []
    for dimension in ("doing", "need", "offer"):
        value = _public_text(raw.get(dimension))
        if value:
            fact = {
                "value": value,
                "disclosure": "confirmed_public",
            }
            candidate[dimension] = fact
            facts.append({"field": dimension, **fact})
    if "doing" not in candidate:
        value = _first_public_text(
            raw,
            ("bio", "summary", "description", "introduction", "role", "title"),
        )
        if value:
            fact = {
                "value": value,
                "disclosure": "confirmed_public",
            }
            candidate["doing"] = fact
            facts.insert(0, {"field": "doing", **fact})
    if facts:
        candidate["facts"] = facts
        candidate["match_evidence"] = [{"summary": facts[0]["value"][:1000]}]
        return candidate
    return None


def freeze_intent_search_projection(
    execution: IntentExecution | None,
    *,
    allow_candidates: bool = True,
) -> IntentSearchProjection:
    """Freeze the exact candidate cards used by composition and delivery.

    Args:
        execution: Current receipt-backed execution, if any.
        allow_candidates: Whether the trusted session may receive people cards.

    Returns:
        Projection whose candidates are frozen on `execution`.
    """
    projection = project_intent_search(execution)
    if execution is None or not projection.completed:
        return projection
    execution.freeze_recommendation_candidates(
        projection.candidates if allow_candidates else []
    )
    return project_intent_search(execution)


def _first_public_text(raw: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = _public_text(raw.get(key))
        if value:
            return value
    return ""


def _public_text(raw: Any) -> str:
    if isinstance(raw, dict):
        raw = raw.get("value")
    if isinstance(raw, list):
        raw = "、".join(str(item) for item in raw if item is not None)
    if not isinstance(raw, (str, int, float)):
        return ""
    return scrub_contact_text(str(raw).strip())[:200]


def _valid_receipt(
    execution: IntentExecution,
    receipt: Any,
    evidence: Any,
    steps: dict[str, Any],
) -> bool:
    if evidence is None or not receipt.source_urls:
        return False
    step = steps.get(receipt.plan_step_id)
    body = receipt.model_dump(mode="json")
    body.pop("receipt_id", None)
    items = evidence if isinstance(evidence, list) else [evidence]
    return bool(
        step is not None
        and receipt.run_id == execution.snapshot.run_id
        and receipt.routing_snapshot_hash == execution.snapshot.canonical_content_hash
        and receipt.plan_id == execution.plan.plan_id
        and receipt.tool == step.tool
        and receipt.resource == step.resource
        and receipt.status in {"succeeded", "empty"}
        and receipt.result_count == len(items)
        and canonical_hash(body) == receipt.receipt_id
        and canonical_hash(evidence) == receipt.evidence_digest
        and (
            receipt.produced_by_attempt_id == execution.attempt_id
            or receipt.receipt_id in execution.adopted_receipt_ids
        )
        and all(_official_source(url) for url in receipt.source_urls)
    )


def _official_source(url: str) -> bool:
    try:
        parsed = urlsplit(url)
    except ValueError:
        return False
    return (
        parsed.scheme == "https"
        and parsed.netloc == "www.aihehuo.com"
        and parsed.path.startswith("/ai/")
    )
