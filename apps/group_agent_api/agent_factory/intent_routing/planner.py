"""Authoritative semantic and query planner with one retry and fail-closed status."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

from .prompts import PLANNER_SYSTEM_PROMPT, planner_user_prompt
from .schema import PlannerContextV2, PlannerProposalV2


@dataclass(frozen=True)
class PlannerResult:
    """Validated planner proposal plus a non-sensitive execution status."""

    proposal: PlannerProposalV2 | None
    status: str
    attempts: int


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
    except Exception:  # noqa: BLE001 - model setup must not break observe mode.
        return PlannerResult(None, "error", 0)
    if structured is None:
        return PlannerResult(None, "unavailable", 0)
    prompts = _prompts(context)
    for attempt in (1, 2):
        try:
            raw = structured.invoke(prompts)
            parsed = (
                raw
                if isinstance(raw, PlannerProposalV2)
                else PlannerProposalV2.model_validate(raw)
            )
            return PlannerResult(parsed, "ok", attempt)
        except (
            ValidationError,
            OutputParserException,
            TypeError,
            ValueError,
            KeyError,
        ):
            if attempt == 2:
                return PlannerResult(None, "invalid", attempt)
        except Exception:  # noqa: BLE001 - provider errors become a safe status.
            return PlannerResult(None, "error", attempt)
    return PlannerResult(None, "invalid", 2)


async def aplan_intent(
    *,
    model: Any,
    context: PlannerContextV2,
    timeout_s: float = 1.5,
) -> PlannerResult:
    """Asynchronously plan with a total timeout and one schema retry."""
    try:
        structured = _structured(model)
    except Exception:  # noqa: BLE001 - model setup must not break observe mode.
        return PlannerResult(None, "error", 0)
    if structured is None:
        return PlannerResult(None, "unavailable", 0)
    prompts = _prompts(context)

    async def run() -> PlannerResult:
        for attempt in (1, 2):
            try:
                raw = await structured.ainvoke(prompts)
                parsed = (
                    raw
                    if isinstance(raw, PlannerProposalV2)
                    else PlannerProposalV2.model_validate(raw)
                )
                return PlannerResult(parsed, "ok", attempt)
            except (
                ValidationError,
                OutputParserException,
                TypeError,
                ValueError,
                KeyError,
            ):
                if attempt == 2:
                    return PlannerResult(None, "invalid", attempt)
            except Exception:  # noqa: BLE001 - expose only the status.
                return PlannerResult(None, "error", attempt)
        return PlannerResult(None, "invalid", 2)

    try:
        return await asyncio.wait_for(run(), timeout=max(0.05, timeout_s))
    except TimeoutError:
        return PlannerResult(None, "timed_out", 1)
