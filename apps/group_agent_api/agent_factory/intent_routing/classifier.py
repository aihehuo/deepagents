"""L2 structured classifier with one retry and deterministic failure status."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

from .prompts import CLASSIFIER_SYSTEM_PROMPT, classifier_user_prompt
from .schema import IntentClassificationV1


@dataclass(frozen=True)
class ClassifierResult:
    """Validated classification plus a non-sensitive execution status."""

    classification: IntentClassificationV1 | None
    status: str
    attempts: int


def _structured(model: Any) -> Any:
    if model is None or not hasattr(model, "with_structured_output"):
        return None
    candidate = model
    if hasattr(model, "model_copy") and hasattr(model, "temperature"):
        candidate = model.model_copy(update={"temperature": 0})
    return candidate.with_structured_output(IntentClassificationV1)


def classify_intent(
    message: str,
    *,
    model: Any,
    prior_intent: str | None = None,
) -> ClassifierResult:
    """Invoke a tool-free structured model at most twice.

    Args:
        message: Current user message, bounded by the prompt builder.
        model: Bare chat model supporting `with_structured_output`.
        prior_intent: Optional trusted prior intent name, never free text.

    Returns:
        Validated output or a failure status for clarification fallback.
    """
    try:
        structured = _structured(model)
    except Exception:  # noqa: BLE001 - model setup must not break observe mode.
        return ClassifierResult(None, "error", 0)
    if structured is None:
        return ClassifierResult(None, "unavailable", 0)
    prompts = [
        SystemMessage(content=CLASSIFIER_SYSTEM_PROMPT),
        HumanMessage(content=classifier_user_prompt(message, prior_intent)),
    ]
    for attempt in (1, 2):
        try:
            raw = structured.invoke(prompts)
            parsed = (
                raw
                if isinstance(raw, IntentClassificationV1)
                else IntentClassificationV1.model_validate(raw)
            )
            return ClassifierResult(parsed, "ok", attempt)
        except (
            ValidationError,
            OutputParserException,
            TypeError,
            ValueError,
            KeyError,
        ):
            if attempt == 2:
                return ClassifierResult(None, "invalid", attempt)
        except Exception:  # noqa: BLE001 - provider errors become a safe status, never raw logs.
            return ClassifierResult(None, "error", attempt)
    return ClassifierResult(None, "invalid", 2)


async def aclassify_intent(
    message: str,
    *,
    model: Any,
    prior_intent: str | None = None,
    timeout_s: float = 1.5,
) -> ClassifierResult:
    """Asynchronously classify with a total timeout and one schema retry."""
    try:
        structured = _structured(model)
    except Exception:  # noqa: BLE001 - model setup must not break observe mode.
        return ClassifierResult(None, "error", 0)
    if structured is None:
        return ClassifierResult(None, "unavailable", 0)
    prompts = [
        SystemMessage(content=CLASSIFIER_SYSTEM_PROMPT),
        HumanMessage(content=classifier_user_prompt(message, prior_intent)),
    ]

    async def run() -> ClassifierResult:
        for attempt in (1, 2):
            try:
                raw = await structured.ainvoke(prompts)
                parsed = (
                    raw
                    if isinstance(raw, IntentClassificationV1)
                    else IntentClassificationV1.model_validate(raw)
                )
                return ClassifierResult(parsed, "ok", attempt)
            except (
                ValidationError,
                OutputParserException,
                TypeError,
                ValueError,
                KeyError,
            ):
                if attempt == 2:
                    return ClassifierResult(None, "invalid", attempt)
            except Exception:  # noqa: BLE001 - expose only the status.
                return ClassifierResult(None, "error", attempt)
        return ClassifierResult(None, "invalid", 2)

    try:
        return await asyncio.wait_for(run(), timeout=max(0.05, timeout_s))
    except TimeoutError:
        return ClassifierResult(None, "timed_out", 1)
