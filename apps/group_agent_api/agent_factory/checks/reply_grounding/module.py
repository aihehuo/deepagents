"""Single-shot check entry for mod.brain.reply_grounding."""

from __future__ import annotations

import time
from typing import Any

from apps.group_agent_api.agent_factory.checks.reply_grounding.ids import (
    CHECK_ID,
    MODULE_ID,
    PROTOCOL_NAME,
)
from apps.group_agent_api.agent_factory.checks.reply_grounding.judge import run_l1
from apps.group_agent_api.agent_factory.checks.reply_grounding.l0 import run_l0
from apps.group_agent_api.agent_factory.checks.reply_grounding.protocol import (
    ReplyGroundingInput,
    ReplyGroundingOutput,
    RepairableBy,
    Verdict,
    schema_invalid_output,
)

__all__ = [
    "MODULE_ID",
    "CHECK_ID",
    "PROTOCOL_NAME",
    "check_reply_grounding",
]


def check_reply_grounding(
    payload: ReplyGroundingInput | dict[str, Any],
    *,
    model: Any | None = None,
    l0_only: bool = False,
) -> ReplyGroundingOutput:
    """Run L0 then L1. Never raises for judge failures — fail closed instead.

    ``l0_only=True`` is for orchestrator-authored abandon drafts that must not
    depend on the semantic judge after repair is exhausted.
    """
    started = time.perf_counter()
    try:
        if isinstance(payload, dict):
            parsed = ReplyGroundingInput.model_validate(payload)
        else:
            parsed = payload
    except Exception:  # noqa: BLE001
        out = schema_invalid_output(message="输入不符合 check.reply_grounding.v1")
        _log_span(out, started, skipped=False)
        return out

    l0 = run_l0(parsed)
    if l0 is not None:
        _log_span(l0, started, skipped=False)
        return l0

    if l0_only:
        out = ReplyGroundingOutput(
            verdict=Verdict.pass_,
            codes=[],
            spans=[],
            repairable_by=RepairableBy.llm,
            message="",
            layer="l0",
        )
        _log_span(out, started, skipped=False)
        return out

    out = run_l1(payload=parsed, model=model)
    _log_span(out, started, skipped=False)
    return out


def _log_span(result: ReplyGroundingOutput, started: float, *, skipped: bool) -> None:
    import logging

    logging.getLogger("uvicorn.error").info(
        "action=module_span module_id=%s check_id=%s skipped=%s "
        "elapsed_ms=%s verdict=%s codes=%s layer=%s",
        MODULE_ID,
        CHECK_ID,
        skipped,
        int((time.perf_counter() - started) * 1000),
        result.verdict.value,
        ",".join(result.codes),
        result.layer,
    )
