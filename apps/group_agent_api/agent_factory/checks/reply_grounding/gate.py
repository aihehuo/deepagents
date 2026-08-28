"""Orchestrator-facing gate with repair loop (TSD-14 §4.6.4 / §4.6.6)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from apps.group_agent_api.agent_factory.debug_trace import record_decision_point
from apps.group_agent_api.agent_factory.checks.reply_grounding.deny import (
    format_check_deny,
)
from apps.group_agent_api.agent_factory.checks.reply_grounding.ground import (
    build_ground_from_turn,
)
from apps.group_agent_api.agent_factory.checks.reply_grounding.ids import MODULE_ID
from apps.group_agent_api.agent_factory.checks.reply_grounding.module import (
    check_reply_grounding,
)
from apps.group_agent_api.agent_factory.checks.reply_grounding.protocol import (
    GroundBlock,
    ReplyGroundingInput,
    ReplyGroundingOutput,
    Verdict,
)
from apps.group_agent_api.agent_factory.profile_schema import GroupProfile

_logger = logging.getLogger("uvicorn.error")

RepairFn = Callable[[str, str, GroundBlock], str]


@dataclass
class ReplyGroundingGateResult:
    reply: str
    passed: bool
    skipped: bool = False
    attempts: int = 0
    abandoned: bool = False
    last_result: ReplyGroundingOutput | None = None
    check_deny: str | None = None
    results: list[ReplyGroundingOutput] = field(default_factory=list)


def apply_reply_grounding_gate(
    *,
    reply: str,
    reply_mode: str,
    candidates: list[dict[str, Any]] | None = None,
    profile: GroupProfile | None = None,
    receipts: list[dict[str, Any]] | None = None,
    candidate_count: int | None = None,
    model: Any | None = None,
    repair_fn: RepairFn | None = None,
    max_attempts: int | None = None,
    locale: str = "zh-CN",
    enabled: bool | None = None,
    user_id: int | str | None = None,
    run_id: str | None = None,
    thread_id: str | None = None,
) -> ReplyGroundingGateResult:
    """When enabled, do not let a failing reply reach the user.

    Flow: check → (fail + llm repair) × (max_attempts-1) → abandon draft → recheck.
    When disabled, returns the original reply unchanged.
    """
    from apps.group_agent_api.agent_factory.module_config import (
        reply_grounding_enabled_for_user,
        reply_grounding_max_attempts,
    )

    if enabled is None:
        enabled = reply_grounding_enabled_for_user(user_id)
    if not enabled:
        record_decision_point(
            phase="reply_grounding",
            detail={
                "initial_draft": reply or "",
                "reply_mode": reply_mode,
                "enabled": False,
                "passed": True,
                "skipped": True,
                "verdict": "skipped",
                "attempts": 0,
                "rewrite_attempts": 0,
                "violated_codes": [],
                "violated_spans": [],
                "abandoned": False,
                "final_text": reply or "",
            },
            run_id=run_id,
            thread_id=thread_id,
        )
        return ReplyGroundingGateResult(
            reply=reply or "",
            passed=True,
            skipped=True,
            attempts=0,
        )

    ground = build_ground_from_turn(
        candidates=candidates,
        profile=profile,
        receipts=receipts,
        candidate_count=candidate_count,
    )
    attempts_limit = max(
        1,
        int(max_attempts if max_attempts is not None else reply_grounding_max_attempts()),
    )
    current = (reply or "").strip()
    results: list[ReplyGroundingOutput] = []
    last_deny: str | None = None
    # Without a judge model, still enforce L0; do not schema-fail the whole turn.
    l0_only = model is None

    for attempt in range(1, attempts_limit + 1):
        payload = ReplyGroundingInput(
            reply=current,
            reply_mode=reply_mode or "dialogue",
            ground=ground,
            locale=locale,
        )
        result = check_reply_grounding(payload, model=model, l0_only=l0_only)
        results.append(result)
        if result.verdict == Verdict.pass_:
            record_decision_point(
                phase="reply_grounding",
                detail={
                    "initial_draft": reply or "",
                    "reply_mode": reply_mode,
                    "enabled": True,
                    "passed": True,
                    "skipped": False,
                    "verdict": result.verdict.value,
                    "attempts": attempt,
                    "rewrite_attempts": max(0, attempt - 1),
                    "violated_codes": result.codes,
                    "violated_spans": [
                        s.model_dump(mode="json") if hasattr(s, "model_dump") else s
                        for s in (getattr(result, "spans", []) or [])
                    ],
                    "abandoned": False,
                    "final_text": current,
                },
                run_id=run_id,
                thread_id=thread_id,
            )
            return ReplyGroundingGateResult(
                reply=current,
                passed=True,
                skipped=False,
                attempts=attempt,
                last_result=result,
                results=results,
            )

        last_deny = format_check_deny(result)
        _logger.info(
            "action=reply_grounding_fail module=%s attempt=%s/%s codes=%s",
            MODULE_ID,
            attempt,
            attempts_limit,
            ",".join(result.codes),
        )

        if attempt >= attempts_limit or l0_only:
            break

        # Must repair via LLM rewrite path; gate must not invent user copy.
        if repair_fn is not None:
            try:
                rewritten = (repair_fn(last_deny, current, ground) or "").strip()
            except Exception as exc:  # noqa: BLE001
                _logger.warning(
                    "action=reply_grounding_repair_failed error=%s", exc
                )
                rewritten = ""
            if rewritten:
                current = rewritten
                continue

        # No usable repair → fall through to abandon after loop.
        break

    abandoned = build_abandon_reply(
        reply_mode=reply_mode,
        candidate_count=ground.candidate_count,
    )
    abandon_payload = ReplyGroundingInput(
        reply=abandoned,
        reply_mode=reply_mode or "dialogue",
        ground=ground,
        locale=locale,
    )
    # Orchestrator-authored safe draft: L0 only (no dependency on a bad judge).
    abandon_result = check_reply_grounding(
        abandon_payload, model=model, l0_only=True
    )
    results.append(abandon_result)

    if abandon_result.verdict != Verdict.pass_:
        abandoned = _hard_safe_reply(ground.candidate_count)
        abandon_result = check_reply_grounding(
            ReplyGroundingInput(
                reply=abandoned,
                reply_mode="no_match" if ground.candidate_count <= 0 else "recommendation",
                ground=ground,
                locale=locale,
            ),
            model=model,
            l0_only=True,
        )
        results.append(abandon_result)

    final_abandon_text = (
        abandoned if abandon_result.verdict == Verdict.pass_ else _hard_safe_reply(
            ground.candidate_count
        )
    )
    record_decision_point(
        phase="reply_grounding",
        detail={
            "initial_draft": reply or "",
            "reply_mode": reply_mode,
            "enabled": True,
            "passed": False,
            "skipped": False,
            "verdict": results[0].verdict.value if results else "fail",
            "attempts": len(results),
            "rewrite_attempts": max(0, len(results) - 1),
            "violated_codes": results[0].codes if results else [],
            "violated_spans": [
                s.model_dump(mode="json") if hasattr(s, "model_dump") else s
                for s in (getattr(results[0], "spans", []) or [])
            ] if results else [],
            "abandoned": True,
            "final_text": final_abandon_text,
        },
        run_id=run_id,
        thread_id=thread_id,
    )
    return ReplyGroundingGateResult(
        reply=final_abandon_text,
        passed=True,
        skipped=False,
        attempts=len(results),
        abandoned=True,
        last_result=abandon_result,
        check_deny=last_deny,
        results=results,
    )


def build_abandon_reply(*, reply_mode: str, candidate_count: int) -> str:
    """Safe draft with no invented bios; must be able to pass L0."""
    count = max(0, int(candidate_count or 0))
    mode = (reply_mode or "").strip().lower()
    if count > 0 and mode in {"recommendation", "dialogue", "profile_confirmation"}:
        return (
            f"本群已有 {count} 位公开信息与需求有交集的人选；"
            "具体经历以对方公开资料为准，是否匹配仍需沟通确认。"
        )
    if mode == "no_match" or count <= 0:
        return (
            "这次暂未展开具体人选履历。"
            "你可以继续补充需求，或再说「先匹配」。"
        )
    return (
        "我先不展开具体人选细节。"
        "你可以继续补充，或再说「先匹配」。"
    )


def _hard_safe_reply(candidate_count: int) -> str:
    count = max(0, int(candidate_count or 0))
    if count > 0:
        return f"本群已有 {count} 位公开信息与需求有交集的人选；是否匹配仍需沟通确认。"
    return "这次暂未展开具体人选履历。你可以继续补充需求。"


def default_repair_fn(model: Any | None) -> RepairFn | None:
    """Build a lightweight rewrite repair using the same judge-capable model."""
    if model is None:
        return None

    def _repair(deny_block: str, previous_reply: str, ground: GroundBlock) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        import json

        system = (
            "你是群智能体主对话模型的改口步骤。上一份 reply 未通过事实对照，"
            "必须按 check_deny 修改后只输出新的用户可见正文。\n"
            "硬约束：只能使用 ground 中的 facts；不得编造经历/数字/项目；"
            "不得声称已发群/已通知；"
            "不得写「非常适合/高度契合/完美匹配」等无 evidence 支持的主观拔高；"
            "不确定就写「是否合拍仍需沟通确认」；"
            "不要输出 JSON 或标签。"
        )
        human = (
            f"{deny_block}\n\n"
            f"ground: {json.dumps(ground.model_dump(mode='json'), ensure_ascii=False)}\n\n"
            f"previous_reply:\n{previous_reply}\n\n"
            "请只输出改口后的完整正文。"
        )
        msg = model.invoke(
            [SystemMessage(content=system), HumanMessage(content=human)]
        )
        content = getattr(msg, "content", None)
        if content is None:
            return ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text") or ""))
            return "".join(parts).strip()
        return str(content).strip()

    return _repair
