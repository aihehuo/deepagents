"""Cafe Host model-selected public reads.

The independent planner remains the default. A new v4 run freezes
``planner`` or ``model_tools`` into its encrypted envelope at admission.
Later attempts read that frozen link and do not consult the current flag.
"""

from __future__ import annotations

import json
import os
import re
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal

from pydantic import BaseModel, ConfigDict, Field

from apps.group_agent_api.agent_factory.debug_trace import record_decision_point
from apps.group_agent_api.agent_factory.intent_routing.execution import QueryPlanError
from apps.group_agent_api.agent_factory.intent_routing.registry import registry_entry
from apps.group_agent_api.agent_factory.intent_routing.schema import (
    IntentConstraintsV1,
    IntentDecisionV1,
    PlannerProposalV2,
    QueryPlanRefV1,
    ResourceQueriesV2,
)

LINK_PLANNER = "planner"
LINK_MODEL_TOOLS = "model_tools"
_LINKS = frozenset({LINK_PLANNER, LINK_MODEL_TOOLS})

# Match the existing Micro v4 ChatAccess policy: only FAQ is free.
DRINK_PAID_INTENTS = frozenset(
    {
        "people_search",
        "opportunity_search",
        "project_partnership_search",
        "community_search",
        "startup_digest",
    }
)
FREE_READ_INTENTS = frozenset({"platform_help"})

TOOL_INTENTS = {
    "search_people": "people_search",
    "search_opportunities": "opportunity_search",
    "search_projects": "project_partnership_search",
    "search_communities": "community_search",
    "read_startup_digest": "startup_digest",
    "search_platform_faq": "platform_help",
}
TOOL_NAMES = frozenset(TOOL_INTENTS)
_QUERY_FIELD = {
    "people_search": "users",
    "opportunity_search": "ideas",
    "project_partnership_search": "projects",
    "community_search": "groups",
    "startup_digest": None,
    "platform_help": "faqs",
}
_TIME_RANGES = ("本周", "本月", "最近", "近期")
_SKILL_PLAN = {
    "people_search": ("get-ai-cofounders", "people-search-v1"),
    "opportunity_search": ("get-ai-opportunities", "opportunity-search-v1"),
    "project_partnership_search": (
        "get-ai-project-partnerships",
        "project-partnership-search-v1",
    ),
    "community_search": ("get-ai-communities", "community-search-v1"),
    "startup_digest": ("get-ai-startup-digest", "startup-digest-v1"),
    "platform_help": ("get-ai-platform-help", "platform-help-v1"),
}

_LINK: ContextVar[str] = ContextVar("cafe_host_link", default=LINK_PLANNER)
_RUNTIME: ContextVar[ModelToolRuntime | None] = ContextVar(
    "cafe_host_model_tool_runtime", default=None
)
_VERIFIED: ContextVar[bool] = ContextVar("cafe_host_model_tool_verified", default=False)


class _QueryArgs(BaseModel):
    """Closed query text plus an optional planner time range."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=160)
    time_range: Literal["本周", "本月", "最近", "近期"] | None = None


class _DigestArgs(BaseModel):
    """Digest has no keyword. The registry plan owns its steps."""

    model_config = ConfigDict(extra="forbid")

    time_range: Literal["本周", "本月", "最近", "近期"] | None = None


def model_selected_tools_enabled() -> bool:
    """Return the admission-time switch. Execution must not read this again."""
    return os.environ.get("CAFE_HOST_MODEL_SELECTED_TOOLS", "0").strip() == "1"


def cafe_host_link_for_admission(req: Any) -> str | None:
    """Freeze the link for a newly accepted v4 cafe run.

    Other schemas omit the field so their envelopes stay unchanged. Missing
    on an older v4 envelope means ``planner``.
    """
    if getattr(req, "request_schema_version", None) != 4:
        return None
    if getattr(req, "execution_mode", None) != "cafe_host":
        return None
    if model_selected_tools_enabled():
        return LINK_MODEL_TOOLS
    return LINK_PLANNER


def current_cafe_host_link() -> str:
    """Return the link frozen for this attempt."""
    link = _LINK.get()
    if link not in _LINKS:
        return LINK_PLANNER
    return link


def model_tools_link_active() -> bool:
    """Return whether this attempt must use model-selected reads."""
    return current_cafe_host_link() == LINK_MODEL_TOOLS


def is_model_tool_free_read(intent: str) -> bool:
    """Return whether this link may run the read without a drink."""
    return intent in FREE_READ_INTENTS


@contextmanager
def active_cafe_host_link(link: str | None) -> Iterator[None]:
    """Install the frozen link for one attempt and restore the previous one."""
    chosen = link if link in _LINKS else LINK_PLANNER
    token = _LINK.set(chosen)
    try:
        yield
    finally:
        _LINK.reset(token)


@dataclass
class ModelToolRuntime:
    """Server context for one model-tool attempt. Not visible to the model."""

    original_message: str
    run_id: str | None
    request_fingerprint: str | None
    skills_dir: Any
    user_token: str | None
    emit: Any
    progress_started: float
    emit_search_logs: bool
    emit_search_results: bool
    allow_candidates: bool
    shown_messages: list[Any] = field(default_factory=list)
    attempt_id: str | None = None
    slot: threading.Lock = field(default_factory=threading.Lock)
    claimed: bool = False
    execution: Any = None
    resolution: Any = None
    access_control: str | None = None
    verified_titles: tuple[str, ...] = ()

    def claim(self) -> bool:
        """Reserve the single read for this run."""
        with self.slot:
            if self.claimed:
                return False
            self.claimed = True
            return True


def _record(runtime: ModelToolRuntime, phase: str, detail: dict[str, Any]) -> None:
    """Record one model-tool decision joined by run and attempt."""
    record_decision_point(
        phase=phase,
        detail=detail,
        run_id=runtime.run_id,
        attempt_id=runtime.attempt_id,
    )


def current_model_tool_runtime() -> ModelToolRuntime | None:
    """Return the runtime installed around the tool-agent call."""
    return _RUNTIME.get()


def verified_model_read() -> bool:
    """Return whether this attempt already has a verified read receipt."""
    runtime = _RUNTIME.get()
    execution = runtime.execution if runtime is not None else None
    return _VERIFIED.get() or bool(
        execution is not None
        and execution.decision.action == "execute"
        and execution.receipts
    )


@contextmanager
def model_tool_runtime(runtime: ModelToolRuntime) -> Iterator[ModelToolRuntime]:
    """Scope tool execution state to one graph invocation."""
    runtime_token = _RUNTIME.set(runtime)
    verified_token = _VERIFIED.set(False)
    try:
        yield runtime
    finally:
        _VERIFIED.reset(verified_token)
        _RUNTIME.reset(runtime_token)


def peek_frozen_decision() -> IntentDecisionV1 | None:
    """Return the run's already frozen decision, if this attempt has one."""
    from apps.group_agent_api.agent_factory.intent_routing.snapshot import (
        decrypt_routing_snapshot,
    )
    from apps.group_agent_api.execution.active_fence import get_active_fence

    active = get_active_fence()
    if active is None:
        return None
    record = active.store.get(active.claim.run_id)
    if record is None or not record.routing_snapshot_hash:
        return None
    keys, _version = active.store.routing_crypto_material()
    snapshot = decrypt_routing_snapshot(record, keys=keys)
    return snapshot.intent_decision


def decision_from_tool_args(tool_name: str, args: dict[str, Any]) -> IntentDecisionV1:
    """Map one closed tool payload onto an execute decision.

    Raises:
        ValueError: The payload is not a legal query for that tool.
    """
    intent = TOOL_INTENTS.get(tool_name)
    if intent is None or registry_entry(intent) is None:
        raise ValueError("invalid_arguments")
    field_name = _QUERY_FIELD[intent]
    allowed = {"time_range"} if field_name is None else {"query", "time_range"}
    if set(args) - allowed:
        raise ValueError("invalid_arguments")
    time_range = args.get("time_range")
    if time_range is not None and time_range not in _TIME_RANGES:
        raise ValueError("invalid_arguments")
    queries: dict[str, str | None] = {
        "users": None,
        "ideas": None,
        "projects": None,
        "groups": None,
        "faqs": None,
    }
    if field_name is not None:
        raw = args.get("query")
        if not isinstance(raw, str):
            raise ValueError("invalid_arguments")
        text = raw.strip()[:160]
        if not text:
            raise ValueError("invalid_arguments")
        queries[field_name] = text
    elif "query" in args:
        raise ValueError("invalid_arguments")
    skill, plan_id = _SKILL_PLAN[intent]
    return IntentDecisionV1(
        primary_intent=intent,  # type: ignore[arg-type]
        confidence="high",
        decision_source="rules",
        action="execute",
        required_skill=skill,
        query_plan_ref=QueryPlanRefV1(plan_id=plan_id),
        requires_plan_receipts=True,
        resource_queries=ResourceQueriesV2(**queries),
        constraints=IntentConstraintsV1(time_range=time_range),
    )


def rejection_decision(decision: IntentDecisionV1) -> IntentDecisionV1:
    """Freeze a denied read without resource queries or a plan."""
    query = ""
    field_name = _QUERY_FIELD.get(decision.primary_intent)
    if field_name is not None:
        query = decision.resource_queries.for_resource(field_name) or ""
    spans = (query,) if query else ()
    return IntentDecisionV1(
        primary_intent=decision.primary_intent,  # type: ignore[arg-type]
        confidence="high",
        decision_source="rules",
        evidence_spans=spans,
        action="reject",
        constraints=decision.constraints,
    )


def proposal_from_decision(decision: IntentDecisionV1) -> PlannerProposalV2 | None:
    """Rebuild the offer input from a frozen tool decision, not a planner call."""
    if decision.primary_intent not in DRINK_PAID_INTENTS:
        return None
    field_name = _QUERY_FIELD[decision.primary_intent]
    query = ""
    if decision.action == "execute" and field_name is not None:
        query = decision.resource_queries.for_resource(field_name) or ""
    elif decision.evidence_spans:
        query = decision.evidence_spans[0]
    if not query and decision.primary_intent != "startup_digest":
        return None
    queries = {name: None for name in ("users", "ideas", "projects", "groups", "faqs")}
    if field_name is not None:
        queries[field_name] = query[:160]
    return PlannerProposalV2(
        action="query",
        intent=decision.primary_intent,  # type: ignore[arg-type]
        queries=ResourceQueriesV2(**queries),
        time_range=decision.constraints.time_range,  # type: ignore[arg-type]
    )


def _resolution(decision: IntentDecisionV1, *, proposal: Any = None, signal: str) -> Any:
    from apps.group_agent_api.agent_factory.intent_routing.resolver import IntentResolution

    return IntentResolution(
        decision=decision,
        latency_ms=0,
        signal_ids=(signal,),
        classifier_status="ok",
        classifier_attempts=0,
        context_inherited=False,
        proposal=proposal,
    )


def _safe_result(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _bounded_results(execution: Any) -> tuple[str, list[dict[str, str]], tuple[str, ...]]:
    """Return status, model-visible rows, and titles from the frozen projection."""
    from apps.group_agent_api.agent_factory.contact_scrub import scrub_contact_text
    from apps.group_agent_api.agent_factory.intent_routing.projection import (
        project_intent_search,
    )
    from apps.group_agent_api.app.cafe_host_search import (
        grounded_digest_reply,
        grounded_platform_help_reply,
    )

    intent = execution.decision.primary_intent
    if intent == "startup_digest":
        text = grounded_digest_reply(execution)
        return "ok", [{"index": "1", "title": text[:500]}], (text[:80],)
    if intent == "platform_help":
        text = grounded_platform_help_reply(execution)
        empty = "没有返回足以核实" in text
        return ("empty" if empty else "ok"), [{"index": "1", "title": text[:500]}], ()
    projected = project_intent_search(execution)
    rows: list[dict[str, str]] = []
    titles: list[str] = []
    if intent == "people_search":
        for index, candidate in enumerate(projected.candidates[:3], start=1):
            title = scrub_contact_text(str(candidate.get("display_name") or ""))[:80]
            if not title:
                continue
            titles.append(title)
            rows.append({"index": str(index), "title": title})
    else:
        from apps.group_agent_api.app.cafe_host_search import _cards_for_resource

        resource = {
            "opportunity_search": "ideas",
            "project_partnership_search": "projects",
            "community_search": "groups",
        }.get(intent, "")
        for index, card in enumerate(_cards_for_resource(execution, resource)[:3], start=1):
            title = scrub_contact_text(str(card.get("title") or ""))[:80]
            summary = scrub_contact_text(str(card.get("summary") or ""))[:160]
            if not title:
                continue
            titles.append(title)
            row = {"index": str(index), "title": title}
            if summary:
                row["summary"] = summary
            rows.append(row)
    status = "empty" if not rows else "ok"
    return status, rows, tuple(titles)


def _query_text(decision: IntentDecisionV1) -> str:
    field_name = _QUERY_FIELD.get(decision.primary_intent)
    if field_name is None:
        return ""
    return decision.resource_queries.for_resource(field_name) or ""


def frozen_public_query(execution: Any) -> str:
    """Return the query frozen into the plan, not the raw user sentence."""
    text = _query_text(execution.decision)
    return text or "（无关键词）"


_FRESH_SEARCH = re.compile(
    r"(?:重新|再)(?:搜|查|找)|刷新|扩大(?:范围|条件)?|放宽(?:条件|范围)?|"
    r"换(?:个|一)?(?:条件|关键词|城市)|另(?:外)?(?:搜|找|查)|换一批|"
    r"调用.{0,8}(?:搜索|查找|查询|工具)|"
    r"(?:帮我|请|能否|可以).{0,8}(?:找|查|搜)"
)
_RESULT_REFERENCE = re.compile(r"第[一二三123]|刚才|上次|前面|这些|这几个|那几个|对比|比较")
_SEARCH_FOLLOWUP = re.compile(
    r"第[一二三123].{0,8}(?:结果|人|位|个|项目)|"
    r"(?:刚才|上次|前面|这些|这几个|那几个).{0,20}(?:结果|候选|人|项目|搜索|找到)|"
    r"(?:对比|比较).{0,25}(?:候选|结果|项目|人)"
)
_EXPLICIT_PUBLIC_LOOKUP = re.compile(
    r"(?:找|查|搜|推荐|想进入|想加入|加入).{0,40}"
    r"(?:社群|创业群|群组|合伙人|候选人|人选|合作项目|创业机会|博客|平台新闻)|"
    r"(?:社群|创业群|群组|合伙人|候选人|合作项目|创业机会|博客|平台新闻)"
    r".{0,30}(?:有哪些|找|查|搜|推荐)|调用.{0,8}(?:搜索|查找|查询)"
)
_UNVERIFIED_GROUP_CLAIM = re.compile(
    r"(?:平台|爱合伙).{0,30}(?:确实有|有几个|有不少|有.*社群|官方孵化)|"
    r"(?:真实活跃|月均\d+场|开放\d+个新席位|已开放申请)"
)
_SAFE_NO_READ_REPLY = re.compile(r"(?:没法|无法|不能|没有|尚未|还没).{0,12}(?:查|搜|查询|搜索)")


def user_requests_fresh_search(message: str) -> bool:
    """Return whether the user explicitly asked to search again."""
    return _FRESH_SEARCH.search(message or "") is not None


def explicit_public_lookup_request(messages: list[Any]) -> bool:
    """Identify a direct public lookup for one bounded model retry, not routing."""
    from langchain_core.messages import HumanMessage

    from apps.group_agent_api.agent_factory.cafe_output import validated_search_history

    message = next(
        (str(item.content or "") for item in reversed(messages) if isinstance(item, HumanMessage)),
        "",
    )
    if not _EXPLICIT_PUBLIC_LOOKUP.search(message):
        return False
    if not user_requests_fresh_search(message) and _SEARCH_FOLLOWUP.search(message):
        return not bool(validated_search_history(messages))
    return True


def should_reuse_shown_results(
    messages: list[Any], message: str, tool_name: str
) -> bool:
    """Reuse only a referenced result group of the same resource as the tool."""
    from apps.group_agent_api.agent_factory.cafe_output import validated_search_history

    if user_requests_fresh_search(message) or not _RESULT_REFERENCE.search(message or ""):
        return False
    intent = TOOL_INTENTS.get(tool_name)
    resource = _QUERY_FIELD.get(intent or "")
    if resource is None:
        return False
    return any(
        (group.get("resource") or "users") == resource
        for group in validated_search_history(messages)
    )


def grounded_unverified_reply(messages: list[Any], draft: str) -> str:
    """Ground no-tool follow-ups and refuse unsupported public-data claims.

    This is an output guard, not a keyword router: it never starts a search.
    Only server-authored historical card markers may supply profile facts.
    """
    from langchain_core.messages import HumanMessage

    from apps.group_agent_api.agent_factory.cafe_output import validated_search_history

    user_message = next(
        (str(message.content or "") for message in reversed(messages) if isinstance(message, HumanMessage)),
        "",
    )
    if (
        user_requests_fresh_search(user_message)
        and len(draft) <= 160
        and _SAFE_NO_READ_REPLY.search(draft)
        and not _UNVERIFIED_GROUP_CLAIM.search(draft)
    ):
        return draft
    if user_requests_fresh_search(user_message):
        return "这次还没有完成新的公开数据查询，我不能把旧结果当作新结果。"
    history = validated_search_history(messages)
    if history and (
        _SEARCH_FOLLOWUP.search(user_message)
        or any(
            str(card.get("display_name") or card.get("title") or "") in user_message
            for group in history
            for card in group["cards"]
            if card.get("display_name") or card.get("title")
        )
    ):
        groups = history if re.search(r"对比|比较", user_message) else history[-1:]
        ordinal = re.search(r"第([一二三123])", user_message)
        position = {"一": 0, "二": 1, "三": 2, "1": 0, "2": 1, "3": 2}.get(
            ordinal.group(1), -1
        ) if ordinal else -1
        lines = ["我只能依据之前展示的公开卡片说明，不能补写未核实的资料："]
        for group in groups:
            cards = group["cards"]
            selected = cards[position : position + 1] if 0 <= position < len(cards) else cards
            for card in selected:
                title = str(card.get("display_name") or card.get("title") or "")
                facts = [
                    str(card.get(key) or "")
                    for key in ("doing", "need", "offer", "short_intro", "summary")
                    if card.get(key)
                ]
                lines.append(f"- {title}" + ("：" + "；".join(facts) if facts else "（卡片未提供更多详情）"))
        lines.append("更多履历、活动或合作条件未在这些卡片中核实；可以点开对应卡片查看详情。")
        return "\n".join(lines)
    if _EXPLICIT_PUBLIC_LOOKUP.search(user_message):
        return "这次还没有完成公开数据查询，我不能凭猜测给你具体名单或结果。"
    if _UNVERIFIED_GROUP_CLAIM.search(draft):
        return "刚才提到的具体社群并未经过公开查询核实，我不应该把它们说成真实可加入的群组。"
    return draft


async def _emit_ready(runtime: ModelToolRuntime, execution: Any) -> None:
    if runtime.emit is None:
        return
    import time

    payload = {
        "progress_schema_version": 1,
        "phase": "intent_resolved",
        "intent": execution.decision.primary_intent,
        "display_code": "intent_ready",
        "routing_snapshot_hash": execution.snapshot.canonical_content_hash,
        "plan_id": execution.plan.plan_id if execution.plan is not None else None,
        "plan_step_id": None,
        "request_index": None,
        "elapsed_ms": max(0, min(int((time.monotonic() - runtime.progress_started) * 1000), 600_000)),
        "step_result_count": None,
        "aggregate_result_count": None,
    }
    delivered = await runtime.emit("progress", payload)
    if delivered is not True:
        raise QueryPlanError("callback_ack_invalid")


async def execute_authorized_decision(
    decision: IntentDecisionV1,
    runtime: ModelToolRuntime,
) -> Any:
    """Freeze the decision, then execute the existing QueryPlan."""
    from apps.group_agent_api.agent_factory.intent_routing.execution import (
        prepare_intent_execution,
    )
    from apps.group_agent_api.agent_factory.intent_routing.plan_executor import (
        execute_query_plan,
    )
    from apps.group_agent_api.agent_factory.intent_routing.projection import (
        freeze_intent_search_projection,
    )
    from apps.group_agent_api.app.cafe_host_search import bound_search_credential

    resolution = _resolution(decision, signal="model_tool")
    execution = prepare_intent_execution(
        resolution,
        original_message=runtime.original_message,
        run_id=runtime.run_id,
        request_fingerprint=runtime.request_fingerprint,
        execution_mode="cafe_host",
        can_search=True,
    )
    _record(
        runtime,
        "cafe_model_tool_frozen",
        {
            "intent": decision.primary_intent,
            "routing_snapshot_hash": execution.snapshot.canonical_content_hash,
            "plan_id": execution.plan.plan_id if execution.plan is not None else None,
            "run_id": runtime.run_id,
        },
    )
    execution.bind_progress_emitter(
        runtime.emit,
        started_monotonic=runtime.progress_started,
        emit_search_logs=runtime.emit_search_logs,
        emit_search_results=runtime.emit_search_results,
    )
    await _emit_ready(runtime, execution)
    with bound_search_credential(runtime.user_token):
        await execute_query_plan(execution, skills_dir=runtime.skills_dir)
    freeze_intent_search_projection(
        execution,
        allow_candidates=runtime.allow_candidates,
    )
    runtime.execution = execution
    runtime.resolution = resolution
    return execution


async def execute_model_read(tool_name: str, args: dict[str, Any]) -> str:
    """Authorize one model read and return a bounded tool result."""
    runtime = current_model_tool_runtime()
    if runtime is None or not model_tools_link_active():
        return _safe_result(
            {
                "status": "rejected",
                "code": "search_capability_unavailable",
                "reason": "这次不能查询公开数据。",
            }
        )
    if not runtime.claim():
        _record(
            runtime,
            "cafe_model_tool_limited",
            {"tool": tool_name, "run_id": runtime.run_id},
        )
        return _safe_result(
            {
                "status": "rejected",
                "code": "read_limited",
                "reason": "本轮只能进行一次公开读取。",
            }
        )
    if should_reuse_shown_results(
        runtime.shown_messages, runtime.original_message, tool_name
    ):
        _record(
            runtime,
            "cafe_model_tool_reused",
            {"tool": tool_name, "run_id": runtime.run_id},
        )
        return _safe_result(
            {
                "status": "reused",
                "code": "existing_results",
                "reason": "本轮沿用已经展示的公开结果，没有重新查询。",
            }
        )
    try:
        decision = decision_from_tool_args(tool_name, args)
    except ValueError:
        _record(
            runtime,
            "cafe_model_tool_rejected",
            {"tool": tool_name, "code": "invalid_arguments", "run_id": runtime.run_id},
        )
        return _safe_result(
            {
                "status": "rejected",
                "code": "invalid_arguments",
                "reason": "查询条件无法使用，请换一种说法。",
            }
        )
    from apps.group_agent_api.app.chat_access import current_chat_access

    access = current_chat_access()
    if access is None or registry_entry(decision.primary_intent) is None:
        code = "search_capability_unavailable"
    elif decision.primary_intent in FREE_READ_INTENTS:
        code = None
    else:
        code = access.control_for_intent(decision.primary_intent)
    _record(
        runtime,
        "cafe_model_tool_gate",
        {
            "tool": tool_name,
            "intent": decision.primary_intent,
            "code": code or "allow",
            "run_id": runtime.run_id,
        },
    )
    if code is not None:
        denied = rejection_decision(decision)
        proposal = proposal_from_decision(decision) if code == "chat_session_required" else None
        runtime.access_control = code
        runtime.resolution = _resolution(denied, proposal=proposal, signal="model_tool_denied")
        try:
            from apps.group_agent_api.agent_factory.intent_routing.execution import (
                prepare_intent_execution,
            )

            runtime.execution = prepare_intent_execution(
                runtime.resolution,
                original_message=runtime.original_message,
                run_id=runtime.run_id,
                request_fingerprint=runtime.request_fingerprint,
                execution_mode="cafe_host",
                can_search=True,
            )
        except QueryPlanError:
            runtime.execution = None
        return _safe_result(
            {
                "status": "rejected",
                "code": code,
                "reason": _REASONS.get(code, "这次不能查询公开数据。"),
            }
        )
    execution = await execute_authorized_decision(decision, runtime)
    status, rows, titles = _bounded_results(execution)
    runtime.verified_titles = titles
    _VERIFIED.set(True)
    return _safe_result(
        {
            "status": status,
            "intent": decision.primary_intent,
            "query": _query_text(decision),
            "results": rows,
        }
    )


_REASONS = {
    "chat_session_required": "找人、找项目、找创业机会、查社群或看近期创业资讯需要先开始畅聊。这次没有查询。",
    "search_capability_unavailable": "当前不能查询这类公开数据。",
    "search_confirmation_required": "这次搜索还不能开始。",
    "confirmed_search_intent_mismatch": "确认的搜索类型和这次请求不一致。",
    "pending_search_unavailable": "上次记下的查找已经不能继续。",
    "invalid_arguments": "查询条件无法使用，请换一种说法。",
    "read_limited": "本轮只能进行一次公开读取。",
    "tool_not_allowed": "这个动作不能执行。",
    "existing_results": "本轮沿用已经展示的公开结果，没有重新查询。",
}


def conversation_resolution() -> Any:
    """Freeze a no-tool turn so a retry cannot grow a new search."""
    decision = IntentDecisionV1(
        primary_intent="conversation",
        confidence="high",
        decision_source="rules",
        action="conversation",
    )
    return _resolution(decision, signal="model_tool_chat")


def cafe_read_tools() -> list[Any]:
    """Return the six model-visible read tools backed by QueryPlan."""
    from langchain_core.tools import StructuredTool

    specs = (
        (
            "search_people",
            "提出一次找人请求。只提供查询语句，不提供资源、凭证或计划。服务端决定是否执行。",
            _query_tool("search_people"),
            _QueryArgs,
        ),
        (
            "search_opportunities",
            "提出一次找创业机会请求。只提供查询语句。服务端决定是否执行。",
            _query_tool("search_opportunities"),
            _QueryArgs,
        ),
        (
            "search_projects",
            "提出一次找合作项目请求。只提供查询语句。服务端决定是否执行。",
            _query_tool("search_projects"),
            _QueryArgs,
        ),
        (
            "search_communities",
            "提出一次找社群请求。只提供查询语句。服务端决定是否执行。",
            _query_tool("search_communities"),
            _QueryArgs,
        ),
        (
            "read_startup_digest",
            "提出一次读取近期创业博客资讯的请求。服务端决定是否执行。",
            _digest_tool(),
            _DigestArgs,
        ),
        (
            "search_platform_faq",
            "提出一次查询平台 FAQ 的请求。只提供查询语句。服务端决定是否执行。",
            _query_tool("search_platform_faq"),
            _QueryArgs,
        ),
    )
    return [
        StructuredTool.from_function(
            coroutine=function,
            name=name,
            description=description,
            args_schema=schema,
        )
        for name, description, function, schema in specs
    ]


def _query_tool(name: str) -> Any:
    async def _run(query: str, time_range: str | None = None) -> str:
        return await execute_model_read(name, {"query": query, "time_range": time_range})

    _run.__name__ = name
    return _run


def _digest_tool() -> Any:
    async def read_startup_digest(time_range: str | None = None) -> str:
        return await execute_model_read(
            "read_startup_digest", {"time_range": time_range}
        )

    return read_startup_digest


_FUNCTION_CHARS = set(
    "我你他她它们的了吗呢吧啊是还在里和与或也就都不没这那个些"
    "点开看卡片公开资料查到找已经可以另外合适变结果次询搜索"
    "展示详细介绍比较第一建议一下请帮要人项目合伙先给你看这些"
    "共返回条记录但暂时没有可应用内调整后再试本次完成符合条件"
)


def reply_conflicts_with_results(reply: str, titles: tuple[str, ...]) -> bool:
    """Return whether the draft names an entity outside the frozen titles."""
    leftover = reply or ""
    for title in titles:
        if title:
            leftover = leftover.replace(title, "")
    unknown = [
        char
        for char in leftover
        if "\u4e00" <= char <= "\u9fff" and char not in _FUNCTION_CHARS
    ]
    if len(unknown) >= 2:
        return True
    allowed = {title.casefold() for title in titles if title}
    for token in re.findall(r"\b[A-Za-z][A-Za-z0-9._-]{1,40}\b", leftover):
        if token.casefold() not in allowed and not any(token.casefold() in title for title in allowed):
            return True
    return False
