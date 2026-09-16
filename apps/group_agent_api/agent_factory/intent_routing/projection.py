"""Project a completed server QueryPlan into reply and client-facing state."""

from __future__ import annotations

import re
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

from .execution import IntentExecution
from .canonical import canonical_hash
from .constraint_lexicon import expand_constraint_token, phrase_in_text

_STABLE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


@dataclass(frozen=True)
class IntentSearchProjection:
    """Trusted current-turn search facts derived only from valid receipts."""

    completed: bool = False
    result_count: int = 0
    candidates: list[dict[str, Any]] = field(default_factory=list)
    # Full receipt evidence (debug / audit). May include constraint-mismatched people.
    public_reads: list[PublicReadGround] = field(default_factory=list)
    # Users/group-members filtered by frozen constraints — for reply grounding only.
    grounding_public_reads: list[PublicReadGround] = field(default_factory=list)
    pages: int = 0


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
    seen_ids: set[str] = set()
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
            if not _matches_frozen_constraints(execution, item):
                continue
            if _violates_frozen_negations(execution, item):
                continue
            filtered_items.append(item)
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

    return IntentSearchProjection(
        completed=True,
        result_count=execution.unique_result_count(),
        candidates=candidates,
        public_reads=reads,
        grounding_public_reads=grounding_reads,
        pages=len(execution.receipts),
    )


def _public_candidate(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    raw_id = raw.get("user_id", raw.get("id"))
    user_id = str(raw_id).strip() if raw_id is not None else ""
    if not _STABLE_ID.fullmatch(user_id):
        return None

    display = str(
        raw.get("display_name") or raw.get("name") or raw.get("nickname") or ""
    ).strip()[:64]
    display = scrub_display_name(display, fallback_user_id=user_id)
    candidate: dict[str, Any] = {
        "user_id": user_id,
        "display_name": display or f"用户{user_id}",
        "bound": bool(raw.get("bound", True)),
    }
    profile_url = raw.get("profile_url") or raw.get("url")
    if isinstance(profile_url, str) and profile_url.strip():
        candidate["profile_url"] = profile_url.strip()
    for key in ("group_id", "source_group_id", "is_reachable"):
        if key in raw and raw[key] is not None:
            candidate[key] = raw[key]

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


def _dimension_haystack(raw: dict[str, Any], keys: tuple[str, ...]) -> str:
    return " ".join(_search_text(raw.get(key)) for key in keys).casefold()


def _location_haystack(raw: dict[str, Any]) -> str:
    """Trust an explicit city over broader location text such as street addresses."""
    city = _search_text(raw.get("city")).strip()
    if city:
        return city.casefold()
    return _dimension_haystack(raw, ("location", "province", "address"))


def _matches_frozen_constraints(execution: IntentExecution, raw: Any) -> bool:
    if not isinstance(raw, dict):
        return False
    constraints = execution.decision.constraints
    dimensions = (
        (constraints.location, "location", ("city", "province", "location", "address")),
        (
            constraints.industry,
            "industry",
            (
                "industry",
                "tags",
                "bio",
                "summary",
                "description",
                "job_descriptions",
                "title",
                "status_role",
            ),
        ),
        (
            constraints.role,
            "role",
            ("role", "status_role", "title", "bio", "summary", "description"),
        ),
        (
            constraints.capability,
            "capability",
            (
                "capability",
                "skills",
                "tags",
                "bio",
                "summary",
                "description",
                "job_descriptions",
            ),
        ),
    )
    for expected, kind, keys in dimensions:
        if not expected:
            continue
        haystack = (
            _location_haystack(raw)
            if kind == "location"
            else _dimension_haystack(raw, keys)
        )
        aliases = [
            alias
            for value in expected
            if value.strip()
            for alias in expand_constraint_token(kind, value)
            if alias.strip()
        ]
        if not aliases or not any(phrase_in_text(haystack, alias) for alias in aliases):
            return False
    return True


def _violates_frozen_negations(execution: IntentExecution, raw: Any) -> bool:
    """Exclude candidates that match explicit field exclusions."""
    if not isinstance(raw, dict):
        return True
    key_map = {
        "location": ("city", "province", "location", "address"),
        "industry": (
            "industry",
            "tags",
            "bio",
            "summary",
            "description",
            "job_descriptions",
            "title",
            "status_role",
        ),
        "role": ("role", "status_role", "title", "bio", "summary", "description"),
        "capability": (
            "capability",
            "skills",
            "tags",
            "bio",
            "summary",
            "description",
            "job_descriptions",
        ),
    }
    for negation in execution.decision.negations:
        kind, separator, value = negation.partition(":")
        value = value.strip()
        if separator != ":" or kind not in key_map or not value:
            continue
        haystack = (
            _location_haystack(raw)
            if kind == "location"
            else _dimension_haystack(raw, key_map[kind])
        )
        aliases = [
            alias for alias in expand_constraint_token(kind, value) if alias.strip()
        ]
        if any(phrase_in_text(haystack, alias) for alias in aliases):
            return True
    return False


def _search_text(raw: Any) -> str:
    if isinstance(raw, dict):
        return " ".join(_search_text(value) for value in raw.values())
    if isinstance(raw, list):
        return " ".join(_search_text(value) for value in raw)
    return str(raw or "")


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
