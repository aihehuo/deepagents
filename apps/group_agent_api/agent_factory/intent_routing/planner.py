"""Authoritative semantic and query planner with one retry and fail-closed status."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from .prompts import PLANNER_SYSTEM_PROMPT, planner_user_prompt
from .schema import (
    PlannerContextV2,
    PlannerDraftV2,
    PlannerProposalV2,
    ResourceQueriesV2,
)

_logger = logging.getLogger("uvicorn.error")
_PRIMARY_RESOURCE = {
    "people_search": "users",
    "opportunity_search": "ideas",
    "project_partnership_search": "projects",
    "community_search": "groups",
    "platform_help": "faqs",
}
_TEMPORAL_EVIDENCE = {
    "本周": re.compile(r"本周|这周|这个星期|这一周"),
    "本月": re.compile(r"本月|这个月|这一月"),
    "最近": re.compile(r"最近|近期|新近|(?:近|过去)[一二三四五六七八九十\d]+[天周月年]"),
    "近期": re.compile(r"最近|近期|新近|(?:近|过去)[一二三四五六七八九十\d]+[天周月年]"),
}
_EXPLICIT_OPPORTUNITY = re.compile(r"创业机会|商业机会|创业方向|创业想法")
_EXPLICIT_PROJECT = re.compile(r"项目|招募|招聘|加盟|招商|合作席位")
_EXPLICIT_PROJECT_REQUEST = re.compile(
    r"(?:找|搜|搜索|查询|查找|看看|推荐|介绍|发现).{0,16}"
    r"(?:项目|招募|加盟|招商|合作席位)"
    r"|(?:项目|招募|加盟|招商|合作席位).{0,12}(?:找|搜|搜索|看看|推荐)"
)
_ADVICE_REQUEST = re.compile(
    r"怎么|如何|怎样|该怎么|应该怎么|方法|步骤|策略|建议|思路|定位|规划|流程|怎么判断"
)
_EXPLICIT_SEARCH_EXECUTION = re.compile(
    r"(?:帮我|请帮我|麻烦帮我|给我|能否|可以|可不可以).{0,10}"
    r"(?:找|搜|搜索|查询|查找|推荐|列出|看看)"
    r"|我想(?:要)?(?:找|搜|搜索|查询|查找|推荐|列出|看看)"
    r"|我要(?:找|搜|搜索|查询|查找|推荐|列出|看看)"
    r"|^(?:找|搜|搜索|查询|查找|推荐|列出|看看)"
    r"|有哪些|有什么|有没有|哪里有"
)
_EXPLICIT_PEOPLE = re.compile(r"找人|找.{0,8}(?:合伙人|候选人|搭档|伙伴|人才)")
_EXPLICIT_COMMUNITY = re.compile(r"社群|群组|社区|活动|群里|群成员|成员名单")
_NEWS_CONTENT_REQUEST = re.compile(r"新闻|资讯|周报|简报|博客|文章|最新消息|新动态")
_AIHEHUO_REAL_NAME_HELP = re.compile(
    r"爱合伙.{0,24}(?:实名认证|实名验证)|(?:实名认证|实名验证).{0,24}爱合伙"
)
_NO_PLATFORM_READ = re.compile(r"不要(?:搜索|查询|查)|不用(?:搜索|查询|查)|别(?:搜索|查询|查)")
_EXISTING_RESULT_REFERENCE = re.compile(
    r"刚才|之前|上次|已有|找到的|找到了|搜索结果|这些|这几个|这几位|他们|她们|前两个|第[\s一二三\d]*个"
)
_NEW_SEARCH_REQUEST = re.compile(
    r"重新搜|再搜|继续搜|重新找|再找|找更多|换一批|换条件|扩大范围|另外找"
)
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
        "time_range_evidence",
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


def _profile_people_query(context: PlannerContextV2) -> str:
    """Build a bounded people query when the planner selected the wrong resource.

    Explicit positive match constraints describe the desired candidate most
    directly. The authorized need and doing fields provide semantic context
    when structured constraints have not been saved yet. The user's generic
    command (for example "I want to find someone") is deliberately excluded.
    """
    parts: list[str] = []
    for line in context.profile.match_constraints:
        field, separator, remainder = line.partition(":")
        operator, value_separator, values = remainder.partition(":")
        if (
            separator
            and value_separator
            and field in {"city", "location", "industry", "role", "capability"}
            and operator in {"eq", "in", "all", "any"}
        ):
            parts.extend(item.strip() for item in values.split(",") if item.strip())
    for value in (context.profile.need, context.profile.doing):
        text = str(value or "").strip()
        if text:
            parts.append(text)
    unique: list[str] = []
    for part in parts:
        if part not in unique:
            unique.append(part)
    return (" ".join(unique) or "合伙人")[:160]


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


def _provider_failure_status(exc: Exception) -> str:
    """Map provider timeout classes to the planner's retryable timeout status.

    OpenAI-compatible clients expose different timeout exception classes, so
    importing one SDK-specific type would miss other configured providers.
    The bounded class-name check keeps exception messages and model output out
    of routing decisions and diagnostics.
    """
    return "timed_out" if isinstance(exc, TimeoutError) or type(exc).__name__.endswith("TimeoutError") else "error"


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
    return candidate.with_structured_output(PlannerDraftV2)


def _prompts(context: PlannerContextV2) -> list[Any]:
    return [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=planner_user_prompt(context)),
    ]


def _proposal_from_raw(raw: Any, context: PlannerContextV2) -> PlannerProposalV2:
    """Discard only fields irrelevant to the model-selected action, then validate."""
    draft = raw if isinstance(raw, PlannerDraftV2) else PlannerDraftV2.model_validate(raw)
    fields = draft.model_dump()
    action = draft.action
    existing_result_followup = bool(
        context.prior_searches
        and _EXISTING_RESULT_REFERENCE.search(context.current_message)
        and not _NEW_SEARCH_REQUEST.search(context.current_message)
    )
    advice_without_search_request = bool(
        _ADVICE_REQUEST.search(context.current_message)
        and not _EXPLICIT_SEARCH_EXECUTION.search(context.current_message)
    )
    # The model sometimes mistakes a concrete App workflow for generic
    # entrepreneurial advice. This narrow repair covers the platform's own
    # identity-verification help without expanding paid search routing.
    if (
        _AIHEHUO_REAL_NAME_HELP.search(context.current_message)
        and not _NO_PLATFORM_READ.search(context.current_message)
        and not existing_result_followup
    ):
        fields["action"] = "query"
        fields["intent"] = "platform_help"
        fields["queries"] = ResourceQueriesV2(faqs="实名认证").model_dump()
        fields["clarification"] = None
        action = "query"
    if (
        action == "query"
        and fields["intent"]
        in {
            "people_search",
            "opportunity_search",
            "project_partnership_search",
            "community_search",
            "startup_digest",
        }
        and advice_without_search_request
    ):
        _logger.info(
            "intent_planner_action_corrected from=query to=reply "
            "reason=advice_without_search_request"
        )
        action = "reply"
        fields["action"] = "reply"
        fields["intent"] = "conversation"
    if (
        action != "query"
        and _EXPLICIT_PROJECT_REQUEST.search(context.current_message)
        and not _EXPLICIT_OPPORTUNITY.search(context.current_message)
        and not existing_result_followup
        and not advice_without_search_request
    ):
        _logger.info(
            "intent_planner_action_corrected from=%s to=query "
            "reason=explicit_project_search_request",
            action,
        )
        action = "query"
        fields["action"] = "query"
        fields["intent"] = "project_partnership_search"
        fields["queries"] = ResourceQueriesV2(
            projects=context.current_message[:160]
        ).model_dump()
        fields["clarification"] = None
    if (
        _NEWS_CONTENT_REQUEST.search(context.current_message)
        and _EXPLICIT_SEARCH_EXECUTION.search(context.current_message)
        and not existing_result_followup
    ):
        _logger.info(
            "intent_planner_resource_corrected from=%s to=startup_digest "
            "reason=explicit_news_content_request",
            fields.get("intent"),
        )
        action = "query"
        fields["action"] = "query"
        fields["intent"] = "startup_digest"
        fields["queries"] = ResourceQueriesV2().model_dump()
        fields["clarification"] = None
    if (
        action == "query"
        and existing_result_followup
    ):
        _logger.info("intent_planner_existing_results_corrected from=query to=reply")
        action = "reply"
        fields["action"] = "reply"
        fields["intent"] = "conversation"
    evidence = (draft.time_range_evidence or "").strip()
    human_turns = [context.current_message, *(
        turn.content for turn in context.episode_turns if turn.role == "human"
    )]
    if (
        not draft.time_range
        or not evidence
        or not _TEMPORAL_EVIDENCE[draft.time_range].search(evidence)
        or not any(evidence in message for message in human_turns)
    ):
        if draft.time_range:
            _logger.info(
                "intent_planner_time_filter_dropped range=%s reason=missing_user_temporal_evidence",
                draft.time_range,
            )
        fields["time_range"] = None
        fields["time_range_evidence"] = None
    if action == "query":
        fields["clarification"] = None
        if (
            fields["intent"] != "people_search"
            and _EXPLICIT_PEOPLE.search(context.current_message)
            and not _EXPLICIT_COMMUNITY.search(context.current_message)
        ):
            _logger.info(
                "intent_planner_resource_corrected from=%s to=users "
                "reason=explicit_people_request",
                draft.intent,
            )
            fields["intent"] = "people_search"
            fields["queries"] = ResourceQueriesV2(
                users=_profile_people_query(context)
            ).model_dump()
        elif (
            fields["intent"] == "project_partnership_search"
            and _EXPLICIT_OPPORTUNITY.search(context.current_message)
            and not _EXPLICIT_PROJECT.search(context.current_message)
        ):
            _logger.info(
                "intent_planner_resource_corrected from=projects to=ideas "
                "reason=explicit_opportunity_request"
            )
            fields["intent"] = "opportunity_search"
            fields["queries"] = ResourceQueriesV2(
                ideas=context.current_message[:160]
            ).model_dump()
        # Resource correction above may have changed a community draft into a
        # people/opportunity proposal. Normalize bindings against the corrected
        # intent, not the model's original intent, so stale group-only fields
        # cannot make the repaired proposal fail closed.
        if fields["intent"] != "community_search":
            fields["group_member_scope"] = "none"
            fields["group_event_id"] = None
    else:
        fields["queries"] = ResourceQueriesV2().model_dump()
        fields["group_member_scope"] = "none"
        fields["group_event_id"] = None
    return PlannerProposalV2.model_validate(fields)


def _repair_prompts(
    prompts: list[Any], diagnostic: dict[str, object], raw: Any = None
) -> list[Any]:
    """Provide safe validation feedback without reusing raw model output as instructions."""
    reasons = [
        str(error.get("reason") or error.get("type"))
        for error in diagnostic.get("validation_errors", [])
        if isinstance(error, dict)
    ]
    resource = None
    try:
        draft = raw if isinstance(raw, PlannerDraftV2) else PlannerDraftV2.model_validate(raw)
        if draft.action == "query":
            resource = _PRIMARY_RESOURCE.get(draft.intent or "")
    except (ValidationError, TypeError, ValueError):
        pass
    requirement = (
        f"若选择 query，必须生成非空的 queries.{resource} 检索词。"
        if resource is not None
        else "若选择 query，必须为所选意图生成非空的主要资源检索词。"
    )
    feedback = (
        "上一次结构化草案未通过服务器校验，错误码："
        + ", ".join(reasons[:8] or ["structured_output_invalid"])
        + "。请结合最初的 planner_context 重新独立生成完整草案。"
        + requirement
        + "query 时 clarification 留空；非 query 时社群成员范围为 none。"
        "不能从对话和画像形成明确检索对象时才选择 clarify；不要编造条件。"
    )
    return [*prompts, HumanMessage(content=feedback)]


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
    retry_prompts = prompts
    for attempt in (1, 2):
        try:
            raw = structured.invoke(retry_prompts)
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
            retry_prompts = _repair_prompts(prompts, diagnostics[-1])
            continue
        except Exception as exc:  # noqa: BLE001 - provider errors become a safe status.
            diagnostics.append(
                _failure_diagnostic(exc, stage="model_invoke", attempt=attempt)
            )
            return PlannerResult(
                None, _provider_failure_status(exc), attempt, tuple(diagnostics)
            )
        try:
            parsed = _proposal_from_raw(raw, context)
            return PlannerResult(parsed, "ok", attempt, tuple(diagnostics))
        except (ValidationError, TypeError, ValueError, KeyError) as exc:
            diagnostics.append(
                _failure_diagnostic(exc, stage="proposal_validation", attempt=attempt)
            )
            if attempt == 2:
                return PlannerResult(None, "invalid", attempt, tuple(diagnostics))
            retry_prompts = _repair_prompts(prompts, diagnostics[-1], raw)
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
    retry_prompts = prompts

    async def run() -> PlannerResult:
        nonlocal retry_prompts
        for attempt in (1, 2):
            try:
                raw = await structured.ainvoke(retry_prompts)
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
                retry_prompts = _repair_prompts(prompts, diagnostics[-1])
                continue
            except Exception as exc:  # noqa: BLE001 - expose only the status.
                diagnostics.append(
                    _failure_diagnostic(exc, stage="model_invoke", attempt=attempt)
                )
                return PlannerResult(
                    None, _provider_failure_status(exc), attempt, tuple(diagnostics)
                )
            try:
                parsed = _proposal_from_raw(raw, context)
                return PlannerResult(parsed, "ok", attempt, tuple(diagnostics))
            except (ValidationError, TypeError, ValueError, KeyError) as exc:
                diagnostics.append(
                    _failure_diagnostic(
                        exc, stage="proposal_validation", attempt=attempt
                    )
                )
                if attempt == 2:
                    return PlannerResult(None, "invalid", attempt, tuple(diagnostics))
                retry_prompts = _repair_prompts(prompts, diagnostics[-1], raw)
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
