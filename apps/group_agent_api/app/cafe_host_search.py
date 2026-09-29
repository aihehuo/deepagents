"""Cafe Host v4 paid-search execution, projection, and pending-search offers.

The module reuses the existing read registry and QueryPlan. It does not copy
search execution. Flags default closed, so an active drink still returns
``search_capability_unavailable`` until an operator opens the allowlist.
"""

from __future__ import annotations

import hashlib
import os
import re
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

from apps.group_agent_api.agent_factory.contact_scrub import scrub_contact_text
from apps.group_agent_api.agent_factory.intent_routing.canonical import canonical_hash
from apps.group_agent_api.agent_factory.intent_routing.execution import IntentExecution
from apps.group_agent_api.agent_factory.intent_routing.projection import (
    IntentSearchProjection,
)
from apps.group_agent_api.agent_factory.intent_routing.registry import (
    REGISTRY_VERSION,
    registry_entry,
)
from apps.group_agent_api.agent_factory.intent_routing.schema import (
    IntentDecisionV1,
    QueryPlanRefV1,
    ResourceQueriesV2,
)
from apps.group_agent_api.app.execution_modes import PAID_SEARCH_INTENTS

_PRIMARY_QUERY = {
    "people_search": "users",
    "opportunity_search": "ideas",
    "project_partnership_search": "projects",
}
_PRIMARY_RESOURCE = dict(_PRIMARY_QUERY)
_QUERY_FIELDS = ("users", "ideas", "projects")
_BAR_LIMIT = 3
_BOUND_DIGEST: ContextVar[str | None] = ContextVar(
    "cafe_host_search_credential_digest",
    default=None,
)
_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")


def _enabled(name: str) -> bool:
    return os.environ.get(name, "0").strip() == "1"


def search_execution_enabled() -> bool:
    """Return whether Cafe Host may execute the three paid QueryPlans."""
    return _enabled("CAFE_HOST_V4_SEARCH_EXECUTION_ENABLED")


def search_user_allowed(user_id: str) -> bool:
    """Return whether one user is on the internal search allowlist.

    An empty allowlist allows nobody. ``*`` is an explicit full-rollout
    operator choice, matching v4 admission and Micro's ChatAccess gates.
    The execution flag alone is not enough.
    """
    raw = os.environ.get("CAFE_HOST_V4_SEARCH_USER_ALLOWLIST", "")
    allowed = {item.strip() for item in raw.split(",") if item.strip()}
    return bool(user_id) and ("*" in allowed or user_id in allowed)


def pending_offer_enabled() -> bool:
    """Return whether ordinary runs may emit a pending-search offer."""
    return _enabled("CAFE_HOST_PENDING_SEARCH_OFFER_ENABLED")


def v4_search_log_enabled() -> bool:
    """v4 search logs stay off until their own schema flag is confirmed."""
    return os.environ.get("CAFE_HOST_V4_SEARCH_LOG_SCHEMA_VERSION", "0") == "1"


def v4_search_result_enabled() -> bool:
    """v4 search results stay off until their own schema flag is confirmed."""
    return os.environ.get("CAFE_HOST_V4_SEARCH_RESULT_SCHEMA_VERSION", "0") == "1"


def message_sha256(message: str) -> str:
    """Hash the exact UTF-8 user message bytes Micro already persisted."""
    digest = hashlib.sha256(message.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def current_bound_credential_digest() -> str | None:
    """Return the digest bound for the current plan step, if any."""
    return _BOUND_DIGEST.get()


@contextmanager
def bound_search_credential(user_token: str | None) -> Iterator[None]:
    """Expose only a digest to the scoped wrapper for the duration of one plan.

    The raw token is not stored on the context. Callers must pass the worker
    memory token that matches the run's frozen credential digest.
    """
    digest = hashlib.sha256((user_token or "").encode("utf-8")).hexdigest()
    token = _BOUND_DIGEST.set(digest)
    try:
        yield
    finally:
        _BOUND_DIGEST.reset(token)


def _clean_queries(intent: str, raw: dict[str, Any]) -> dict[str, str]:
    entry = registry_entry(intent)
    if entry is None:
        raise ValueError("search_capability_unavailable")
    allowed = {step.resource for step in entry.plan.steps}
    queries: dict[str, str] = {}
    for field in _QUERY_FIELDS:
        value = raw.get(field)
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip() or field not in allowed:
            raise ValueError("query_plan_invalid")
        queries[field] = value.strip()[:160]
    primary = _PRIMARY_QUERY[intent]
    if primary not in queries:
        raise ValueError("query_plan_invalid")
    return queries


def canonical_query_plan(intent: str, queries: dict[str, Any]) -> dict[str, Any]:
    """Build the registry plan after this turn's parameter policy.

    Args:
        intent: One of the three paid search intents.
        queries: Planner or restored resource queries.

    Returns:
        A JSON object whose canonical hash is stable across Deep and Micro.
    """
    if intent not in PAID_SEARCH_INTENTS:
        raise ValueError("query_plan_invalid")
    entry = registry_entry(intent)
    if entry is None:
        raise ValueError("search_capability_unavailable")
    cleaned = _clean_queries(intent, queries)
    steps = [
        {
            "step_id": step.step_id,
            "kind": step.kind,
            "tool": step.tool,
            "resource": step.resource,
            "depends_on": list(step.depends_on),
            "trigger": step.trigger,
            "parameter_policy": step.parameter_policy,
        }
        for step in entry.plan.steps
    ]
    return {
        "registry_version": REGISTRY_VERSION,
        "intent": intent,
        "plan_id": entry.plan.plan_id,
        "skill": entry.skill,
        "primary_resource": _PRIMARY_RESOURCE[intent],
        "steps": steps,
        "queries": cleaned,
    }


def routing_snapshot_hash(query_plan: dict[str, Any]) -> str:
    """Hash the offer binding that Micro can recompute without Deep types."""
    return canonical_hash(
        {
            "registry_version": query_plan["registry_version"],
            "intent": query_plan["intent"],
            "plan_id": query_plan["plan_id"],
            "queries": query_plan["queries"],
        }
    )


def decision_from_canonical_plan(query_plan: dict[str, Any]) -> IntentDecisionV1:
    """Turn a verified canonical plan into the existing execute decision.

    Raises:
        ValueError: The plan is not the current registry plan for its intent.
    """
    intent = query_plan.get("intent")
    if intent not in PAID_SEARCH_INTENTS:
        raise ValueError("pending_search_unavailable")
    expected = canonical_query_plan(intent, query_plan.get("queries") or {})
    if expected != query_plan:
        raise ValueError("pending_search_unavailable")
    queries = query_plan["queries"]
    return IntentDecisionV1(
        primary_intent=intent,
        confidence="high",
        decision_source="planner",
        action="execute",
        required_skill=query_plan["skill"],
        query_plan_ref=QueryPlanRefV1(plan_id=query_plan["plan_id"]),
        requires_plan_receipts=True,
        resource_queries=ResourceQueriesV2(
            users=queries.get("users"),
            ideas=queries.get("ideas"),
            projects=queries.get("projects"),
        ),
        planner_version="intent-planner-v2",
    )


def build_pending_search_offer(
    *,
    proposal: Any,
    decision_action: str,
    confidence: str,
    planner_status: str,
    run_id: str,
    message: str,
    conversation_id: str,
    episode_id: str,
) -> dict[str, Any] | None:
    """Return one immutable offer, or None when this turn must not create one.

    Ordinary runs use the offer while opening a drink; active unconfirmed runs
    use the same object for an independent confirmation without replacing the
    current drink. Clarification, conversation, planner failure, and an
    already-executing decision never produce an offer. The offer carries no
    token and no raw client-editable copy of the message beyond canonical plan
    queries.
    """
    if not pending_offer_enabled():
        return None
    if decision_action == "execute" or planner_status != "ok":
        return None
    if confidence not in {"high", "medium"}:
        return None
    if proposal is None or getattr(proposal, "action", None) != "query":
        return None
    intent = getattr(proposal, "intent", None)
    if intent not in PAID_SEARCH_INTENTS:
        return None
    raw_queries = getattr(proposal, "queries", None)
    if raw_queries is None:
        return None
    try:
        query_plan = canonical_query_plan(
            intent,
            {
                "users": raw_queries.users,
                "ideas": raw_queries.ideas,
                "projects": raw_queries.projects,
            },
        )
    except ValueError:
        return None
    return {
        "version": 1,
        "source_run_id": run_id,
        "source_message_sha256": message_sha256(message),
        "conversation_id": conversation_id,
        "episode_id": episode_id,
        "intent": intent,
        "routing_snapshot_hash": routing_snapshot_hash(query_plan),
        "query_plan": query_plan,
        "query_plan_hash": canonical_hash(query_plan),
    }


def continuation_from_request(req: Any) -> tuple[str | None, IntentDecisionV1 | None]:
    """Validate a Micro-attached continuation without trusting client text.

    Returns:
        A control code when the continuation cannot be used, and otherwise the
        frozen execute decision. ``(None, None)`` means this run has no
        continuation and should follow the normal confirmation matrix.
    """
    trusted = getattr(req, "trusted_search", None)
    if not isinstance(trusted, dict):
        return None, None
    if trusted.get("status") == "unavailable":
        return "pending_search_unavailable", None
    if trusted.get("status") != "ready":
        return "pending_search_unavailable", None
    query_plan = trusted.get("query_plan")
    if not isinstance(query_plan, dict):
        return "pending_search_unavailable", None
    if message_sha256(getattr(req, "message", None) or "") != trusted.get(
        "original_message_sha256"
    ):
        return "pending_search_unavailable", None
    if not _HASH.fullmatch(str(trusted.get("query_plan_hash") or "")):
        return "pending_search_unavailable", None
    if canonical_hash(query_plan) != trusted.get("query_plan_hash"):
        return "pending_search_unavailable", None
    if routing_snapshot_hash(query_plan) != trusted.get("routing_snapshot_hash"):
        return "pending_search_unavailable", None
    access = getattr(req, "chat_access", None) or {}
    if query_plan.get("intent") != access.get("confirmed_search_intent"):
        return "pending_search_unavailable", None
    try:
        return None, decision_from_canonical_plan(query_plan)
    except ValueError:
        return "pending_search_unavailable", None


def _public_id(raw: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = raw.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return str(value)
        if isinstance(value, str) and value.strip().isdigit():
            return value.strip()
    return ""


def _cards_for_resource(
    execution: IntentExecution, resource: str
) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    seen: set[str] = set()
    for receipt in execution.receipts:
        if receipt.resource != resource or receipt.status not in {"succeeded", "empty"}:
            continue
        evidence = execution.evidence_by_receipt.get(receipt.receipt_id)
        items = evidence if isinstance(evidence, list) else []
        for item in items:
            if not isinstance(item, dict):
                continue
            identity = _public_id(item, "id", "project_id", "idea_id")
            if not identity or identity in seen:
                continue
            seen.add(identity)
            title = scrub_contact_text(
                str(item.get("title") or item.get("name") or "").strip()
            )[:80]
            summary = scrub_contact_text(
                str(
                    item.get("summary")
                    or item.get("description")
                    or item.get("content")
                    or ""
                ).strip()
            )[:200]
            card: dict[str, Any] = {
                "kind": resource,
                "id": identity,
                "title": title,
                "source_url": receipt.source_urls[0] if receipt.source_urls else "",
            }
            if summary:
                card["summary"] = summary
            cards.append(card)
    return cards


def build_search_projection(
    execution: IntentExecution | None,
    people: IntentSearchProjection,
) -> dict[str, Any] | None:
    """Freeze one projection shared by the reply, cards, and callbacks.

    Bar entries are the first three items in the existing result order. They
    are not a second relevance filter. People cards stay on the existing
    candidate projection; ideas and projects stay on their primary resource.
    """
    if (
        execution is None
        or execution.decision.action != "execute"
        or not people.completed
    ):
        return None
    intent = execution.decision.primary_intent
    if intent not in PAID_SEARCH_INTENTS:
        return None
    primary = _PRIMARY_RESOURCE[intent]
    if intent == "people_search":
        cards = [
            {
                "kind": "users",
                "id": candidate["user_id"],
                "title": candidate.get("display_name") or "",
                "source_url": candidate.get("profile_url") or "",
            }
            for candidate in people.candidates
            if candidate.get("user_id")
        ]
    else:
        cards = _cards_for_resource(execution, primary)
    identities = [f"{card['kind']}:{card['id']}" for card in cards]
    receipt_ids = [receipt.receipt_id for receipt in execution.receipts]
    body = {
        "version": 1,
        "intent": intent,
        "primary_resource": primary,
        "receipt_ids": receipt_ids,
        "identities": identities,
        "cards": cards,
        "result_count": people.result_count,
    }
    body["digest"] = canonical_hash(
        {
            "intent": intent,
            "receipt_ids": receipt_ids,
            "identities": identities,
            "result_count": people.result_count,
        }
    )
    body["bar_entries"] = cards[:_BAR_LIMIT]
    return body


def search_suggestions(*, empty: bool) -> list[dict[str, str]]:
    """Stable final replies that do not reuse a confirmation or start a search."""
    rows = (
        (
            ("换个条件", "我想调整条件后再找一次"),
            ("先聊聊", "先不找了，我们继续聊聊"),
            ("说得更具体", "我补充一下更具体的要求"),
        )
        if empty
        else (
            ("看第一个", "我想先看看第一个结果"),
            ("再收窄", "我想把条件再收窄一点"),
            ("先聊聊", "先不继续找了，我们聊聊"),
        )
    )
    return [
        {"id": f"suggested_{index}", "label": label, "reply": reply}
        for index, (label, reply) in enumerate(rows, start=1)
    ]
