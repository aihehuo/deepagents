"""Server-owned deterministic execution for frozen public-read QueryPlans."""

from __future__ import annotations

import datetime
import json
import re
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from .contracts import QueryPlanStepV1
from .constraint_lexicon import expand_constraint_token
from .execution import IntentExecution, QueryPlanError, active_intent_execution

_LATIN_ALIAS = re.compile(r"^[a-z0-9][a-z0-9+.#'_\s-]*$", re.IGNORECASE)
_NEGATION_PREFIX = r"(?:不要|别|排除|去掉|不看|not(?:\s+in)?)"


def _alias_pattern(alias: str) -> str:
    escaped = re.escape(alias)
    if _LATIN_ALIAS.fullmatch(alias):
        return rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"
    return escaped


def _remove_negated_constraints(execution: IntentExecution, message: str) -> str:
    """Remove only explicit negation phrases, never unrelated alias substrings."""
    cleaned = message
    removed = False
    allowed_kinds = {"location", "industry", "role", "capability"}
    for negation in execution.decision.negations:
        kind, separator, value = negation.partition(":")
        value = value.strip()
        if separator != ":" or kind not in allowed_kinds or not value:
            continue
        for alias in expand_constraint_token(kind, value):
            if not alias.strip():
                continue
            pattern = rf"{_NEGATION_PREFIX}\s*{_alias_pattern(alias)}"
            updated, count = re.subn(pattern, "", cleaned, flags=re.IGNORECASE)
            if count:
                cleaned = updated
                removed = True
    if removed:
        cleaned = re.sub(r"\s*[,，;；]\s*", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _semantic_query(execution: IntentExecution) -> str:
    """Merge current semantics with frozen constraints, retaining both when clipped."""
    current = _remove_negated_constraints(execution, execution.original_message)
    constraints = execution.decision.constraints
    frozen = [
        *constraints.location,
        *constraints.industry,
        *constraints.role,
        *constraints.capability,
    ]
    normalized_current = current.casefold()
    missing = list(
        dict.fromkeys(
            item.strip()
            for item in frozen
            if item.strip() and item.strip().casefold() not in normalized_current
        )
    )
    if not missing:
        return current[:160]
    suffix = " ".join(missing)[:80]
    current_limit = max(0, 160 - len(suffix) - 1)
    return f"{current[:current_limit]} {suffix}".strip()


def _keyword_text(execution: IntentExecution) -> str:
    constraints = execution.decision.constraints
    values = [
        *constraints.location,
        *constraints.industry,
        *constraints.role,
        *constraints.capability,
    ]
    unique = list(dict.fromkeys(item.strip() for item in values if item.strip()))
    if unique:
        return " ".join(unique)[:160]
    return _semantic_query(execution)


def _created_since(execution: IntentExecution) -> str | None:
    value = execution.decision.constraints.time_range
    if value not in {"本周", "近期", "最近", "本月"}:
        return None
    try:
        anchor_utc = datetime.datetime.fromisoformat(
            execution.snapshot.created_at.replace("Z", "+00:00")
        ).astimezone(datetime.UTC)
    except (TypeError, ValueError):
        raise QueryPlanError("routing_snapshot_invalid") from None
    if value in {"近期", "最近"}:
        start_utc = anchor_utc - datetime.timedelta(days=30)
        return start_utc.isoformat(timespec="milliseconds").replace("+00:00", "Z")

    local_anchor = anchor_utc.astimezone(ZoneInfo("Asia/Shanghai"))
    if value == "本周":
        local_start = (
            local_anchor - datetime.timedelta(days=local_anchor.weekday())
        ).replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        local_start = local_anchor.replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )
    return (
        local_start.astimezone(datetime.UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _selected_group_event_id(execution: IntentExecution) -> str | None:
    text = _semantic_query(execution)
    lowered = text.lower()
    if not any(
        marker in lowered for marker in ("成员", "参与者", "报名名单", "member")
    ):
        return None
    explicit = re.search(
        r"(?:event[_ -]?id|群(?:组)?\s*(?:id|ID|编号)|社群\s*(?:id|ID|编号))"
        r"\s*[:：=#]?\s*([0-9]{1,18})(?![0-9])",
        text,
        flags=re.IGNORECASE,
    )
    if explicit is not None:
        return explicit.group(1)
    if not any(
        marker in lowered
        for marker in ("第一个群", "首个群", "第一个社群", "首个社群", "first group")
    ):
        return None
    for receipt in execution.receipts:
        if receipt.resource != "groups" or receipt.status != "succeeded":
            continue
        evidence = execution.evidence_by_receipt.get(receipt.receipt_id)
        if not isinstance(evidence, list):
            continue
        for item in evidence:
            if not isinstance(item, dict):
                continue
            value = item.get("event_id", item.get("id"))
            candidate = str(value) if value is not None else ""
            if candidate.isascii() and candidate.isdecimal() and len(candidate) <= 18:
                return candidate
    return None


def build_step_parameters(
    execution: IntentExecution, step: QueryPlanStepV1, *, page: int = 1
) -> dict[str, Any]:
    """Build bounded parameters from the authoritative decision and step policy."""
    if execution.plan is None or page < 1:
        raise QueryPlanError("query_plan_invalid")
    output = "md" if step.resource == "weekly" else "json"
    if step.resource == "weekly":
        return {"output_format": output}

    parameters: dict[str, Any] = {
        "page": page,
        "per": min(20, execution.plan.budget.max_items),
        "output_format": output,
    }
    created_since = _created_since(execution)
    if created_since is not None:
        parameters["created_since"] = created_since
    if step.resource in {"users", "ideas"}:
        parameters["q"] = _semantic_query(execution)
    elif step.resource in {"projects", "groups", "faqs"}:
        parameters["keyword"] = _keyword_text(execution)
    elif step.resource == "group-members":
        event_id = _selected_group_event_id(execution)
        if event_id is not None:
            parameters["event_id"] = event_id
    return parameters


def _content(result: Any) -> str:
    value = getattr(result, "content", result)
    return value if isinstance(value, str) else str(value)


def _error_code(content: str) -> str | None:
    rejected = "AIHEHUO query rejected: "
    if content.startswith(rejected):
        return content[len(rejected) :].strip() or "tool_failed"
    if content.startswith("AIHEHUO query failed:"):
        return "tool_timed_out" if "timed out" in content.lower() else "tool_failed"
    if content.startswith("Resource is not installed."):
        return "query_plan_invalid"
    return None


def _next_page(content: str, current: int) -> int | None:
    try:
        document = json.loads(content)
    except (TypeError, ValueError):
        return None
    if not isinstance(document, dict) or not isinstance(document.get("meta"), dict):
        return None
    meta = document["meta"]
    if meta.get("has_more") is not True:
        return None
    value = meta.get("next_page")
    return value if type(value) is int and value > current else current + 1


async def _execute_page(
    execution: IntentExecution,
    *,
    tool: Any,
    step: QueryPlanStepV1,
    page: int,
) -> int | None:
    parameters = build_step_parameters(execution, step, page=page)
    tool_call_id = f"server-{step.step_id}-{page}"
    before = {receipt.receipt_id for receipt in execution.receipts}
    with active_intent_execution(execution):
        result = await tool.ainvoke(
            {
                "name": "query_aihehuo_public_data",
                "args": {"resource": step.resource, **parameters},
                "id": tool_call_id,
                "type": "tool_call",
            }
        )
    content = _content(result)
    code = _error_code(content)
    if code is not None:
        raise QueryPlanError(code)
    produced = [
        receipt
        for receipt in execution.receipts
        if receipt.receipt_id not in before
        and receipt.plan_step_id == step.step_id
        and receipt.tool_call_id == tool_call_id
    ]
    if len(produced) != 1:
        raise QueryPlanError("receipt_invalid")
    receipt = produced[0]
    if (
        receipt.status == "empty"
        or execution.unique_result_count() >= execution.plan.budget.max_items
    ):
        return None
    return _next_page(content, page)


async def execute_query_plan(execution: IntentExecution, *, skills_dir: Path) -> None:
    """Execute every active plan step before the composing model is invoked."""
    if execution.decision.action != "execute":
        return
    if execution.plan is None:
        raise QueryPlanError("query_plan_invalid")

    # Validate the selected skill before any public-network side effect. The
    # prompt is rebuilt after execution so it also contains the new receipts.
    execution.route_prompt(skills_dir=skills_dir)

    from skillpacks.aihehuo.integration import create_aihehuo_tool

    tool = create_aihehuo_tool(skills_dir=skills_dir)
    requests_used = len(execution.receipts) + len(execution.audits)
    pending_pages: dict[str, tuple[QueryPlanStepV1, int]] = {}

    # First satisfy every active step once. Pagination is deliberately deferred
    # so one resource cannot starve a later conditional step of the shared budget.
    for step in execution.plan.steps:
        if any(receipt.plan_step_id == step.step_id for receipt in execution.receipts):
            continue
        parameters = build_step_parameters(execution, step)
        if step.kind != "required" and not execution.step_active(
            step, parameters=parameters
        ):
            continue
        requests_left = execution.plan.budget.max_requests - requests_used
        if requests_left <= 0:
            raise QueryPlanError("query_budget_exhausted")
        next_page = await _execute_page(execution, tool=tool, step=step, page=1)
        requests_used += 1
        if next_page is not None:
            pending_pages[step.step_id] = (step, next_page)

    if not execution.required_steps_satisfied():
        raise QueryPlanError("required_tool_not_called")

    while pending_pages and requests_used < execution.plan.budget.max_requests:
        step_id = next(iter(pending_pages))
        step, page = pending_pages.pop(step_id)
        try:
            next_page = await _execute_page(execution, tool=tool, step=step, page=page)
        except QueryPlanError as exc:
            if exc.code == "query_budget_exhausted":
                break
            raise
        requests_used += 1
        if next_page is not None:
            pending_pages[step_id] = (step, next_page)
