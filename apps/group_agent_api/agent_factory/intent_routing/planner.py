"""Authoritative semantic and query planner with one retry and fail-closed status."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

from .prompts import PLANNER_SYSTEM_PROMPT, planner_user_prompt
from .schema import PlannerContextV2, PlannerProposalV2

_logger = logging.getLogger("uvicorn.error")
_KNOWN_FIELDS = frozenset(
    {
        "version",
        "action",
        "intent",
        "queries",
        "users",
        "ideas",
        "projects",
        "groups",
        "faqs",
        "clarification",
        "time_range",
        "group_member_scope",
        "group_event_id",
        "planner_version",
    }
)
_BINDING_REASONS = {
    "query action requires a registered read intent": "query_intent_missing",
    "query action cannot use non-read intents": "query_intent_not_read",
    "query action cannot include clarification text": "query_has_clarification",
    "group member scope is only valid for community_search": "group_scope_wrong_intent",
    "non-query actions cannot request group members": "group_scope_without_query",
    "clarify action requires clarification intent": "clarify_intent_mismatch",
    "clarify action cannot include resource queries": "clarify_has_queries",
    "reply action must be conversation or profile_update": "reply_intent_mismatch",
    "reply action cannot include resource queries": "reply_has_queries",
}


def _validation_details(exc: ValidationError) -> list[dict[str, object]]:
    """Return bounded schema diagnostics without model-provided values or keys."""
    details: list[dict[str, object]] = []
    for error in exc.errors(include_input=False, include_url=False)[:8]:
        path = [
            part if part in _KNOWN_FIELDS else "<other>"
            for part in error.get("loc", ())
            if isinstance(part, str)
        ]
        message = error.get("msg", "")
        if isinstance(message, str):
            message = message.removeprefix("Value error, ")
        reason = _BINDING_REASONS.get(message)
        if isinstance(message, str) and message.startswith("query action for "):
            reason = "query_missing_primary_resource"
        details.append(
            {
                "path": path or ["<root>"],
                "type": str(error.get("type", "unknown"))[:64],
                "reason": reason,
            }
        )
    return details


def _failure_diagnostic(
    exc: Exception, *, stage: str, attempt: int
) -> dict[str, object]:
    """Describe failure structure while excluding prompts, outputs, and exception text."""
    diagnostic: dict[str, object] = {
        "stage": stage,
        "attempt": attempt,
        "exception_type": type(exc).__name__,
    }
    if isinstance(exc, ValidationError):
        diagnostic["validation_error_count"] = exc.error_count()
        diagnostic["validation_errors"] = _validation_details(exc)
    _logger.warning("intent_planner_failure diagnostic=%s", diagnostic)
    return diagnostic


@dataclass(frozen=True)
class PlannerResult:
    """Validated planner proposal plus a non-sensitive execution status."""

    proposal: PlannerProposalV2 | None
    status: str
    attempts: int
    diagnostics: tuple[dict[str, object], ...] = ()


def _structured(model: Any) -> Any:
    if model is None or not hasattr(model, "with_structured_output"):
        return None
    candidate = model
    if hasattr(model, "model_copy") and hasattr(model, "temperature"):
        candidate = model.model_copy(update={"temperature": 0})
    return candidate.with_structured_output(PlannerProposalV2)


def _prompts(context: PlannerContextV2) -> list[Any]:
    return [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=planner_user_prompt(context)),
    ]


def plan_intent(
    *,
    model: Any,
    context: PlannerContextV2,
) -> PlannerResult:
    """Invoke a tool-free structured planner at most twice.

    Args:
        model: Bare chat model supporting `with_structured_output`.
        context: Closed planner context; values are data, never instructions.

    Returns:
        Validated proposal or a failure status for clarification fallback.
    """
    try:
        structured = _structured(model)
    except Exception as exc:  # noqa: BLE001 - model setup must not break observe mode.
        diagnostic = _failure_diagnostic(exc, stage="model_setup", attempt=0)
        return PlannerResult(None, "error", 0, (diagnostic,))
    if structured is None:
        diagnostic: dict[str, object] = {
            "stage": "model_setup",
            "attempt": 0,
            "exception_type": "ModelUnavailable",
        }
        _logger.warning("intent_planner_failure diagnostic=%s", diagnostic)
        return PlannerResult(None, "unavailable", 0, (diagnostic,))
    prompts = _prompts(context)
    diagnostics: list[dict[str, object]] = []
    for attempt in (1, 2):
        try:
            raw = structured.invoke(prompts)
        except (
            ValidationError,
            OutputParserException,
            TypeError,
            ValueError,
            KeyError,
        ) as exc:
            diagnostics.append(
                _failure_diagnostic(exc, stage="model_invoke_or_parse", attempt=attempt)
            )
            if attempt == 2:
                return PlannerResult(None, "invalid", attempt, tuple(diagnostics))
            continue
        except Exception as exc:  # noqa: BLE001 - provider errors become a safe status.
            diagnostics.append(
                _failure_diagnostic(exc, stage="model_invoke", attempt=attempt)
            )
            return PlannerResult(None, "error", attempt, tuple(diagnostics))
        try:
            parsed = (
                raw
                if isinstance(raw, PlannerProposalV2)
                else PlannerProposalV2.model_validate(raw)
            )
            return PlannerResult(parsed, "ok", attempt, tuple(diagnostics))
        except (ValidationError, TypeError, ValueError, KeyError) as exc:
            diagnostics.append(
                _failure_diagnostic(exc, stage="proposal_validation", attempt=attempt)
            )
            if attempt == 2:
                return PlannerResult(None, "invalid", attempt, tuple(diagnostics))
    return PlannerResult(None, "invalid", 2, tuple(diagnostics))


async def aplan_intent(
    *,
    model: Any,
    context: PlannerContextV2,
    timeout_s: float = 1.5,
) -> PlannerResult:
    """Asynchronously plan with a total timeout and one schema retry."""
    try:
        structured = _structured(model)
    except Exception as exc:  # noqa: BLE001 - model setup must not break observe mode.
        diagnostic = _failure_diagnostic(exc, stage="model_setup", attempt=0)
        return PlannerResult(None, "error", 0, (diagnostic,))
    if structured is None:
        diagnostic: dict[str, object] = {
            "stage": "model_setup",
            "attempt": 0,
            "exception_type": "ModelUnavailable",
        }
        _logger.warning("intent_planner_failure diagnostic=%s", diagnostic)
        return PlannerResult(None, "unavailable", 0, (diagnostic,))
    prompts = _prompts(context)
    diagnostics: list[dict[str, object]] = []

    async def run() -> PlannerResult:
        for attempt in (1, 2):
            try:
                raw = await structured.ainvoke(prompts)
            except (
                ValidationError,
                OutputParserException,
                TypeError,
                ValueError,
                KeyError,
            ) as exc:
                diagnostics.append(
                    _failure_diagnostic(
                        exc, stage="model_invoke_or_parse", attempt=attempt
                    )
                )
                if attempt == 2:
                    return PlannerResult(None, "invalid", attempt, tuple(diagnostics))
                continue
            except Exception as exc:  # noqa: BLE001 - expose only the status.
                diagnostics.append(
                    _failure_diagnostic(exc, stage="model_invoke", attempt=attempt)
                )
                return PlannerResult(None, "error", attempt, tuple(diagnostics))
            try:
                parsed = (
                    raw
                    if isinstance(raw, PlannerProposalV2)
                    else PlannerProposalV2.model_validate(raw)
                )
                return PlannerResult(parsed, "ok", attempt, tuple(diagnostics))
            except (ValidationError, TypeError, ValueError, KeyError) as exc:
                diagnostics.append(
                    _failure_diagnostic(
                        exc, stage="proposal_validation", attempt=attempt
                    )
                )
                if attempt == 2:
                    return PlannerResult(None, "invalid", attempt, tuple(diagnostics))
        return PlannerResult(None, "invalid", 2, tuple(diagnostics))

    try:
        return await asyncio.wait_for(run(), timeout=max(0.05, timeout_s))
    except TimeoutError:
        diagnostic: dict[str, object] = {
            "stage": "model_invoke_or_parse",
            "attempt": min(2, len(diagnostics) + 1),
            "exception_type": "TimeoutError",
        }
        _logger.warning("intent_planner_failure diagnostic=%s", diagnostic)
        return PlannerResult(
            None, "timed_out", min(2, len(diagnostics) + 1), (*diagnostics, diagnostic)
        )
