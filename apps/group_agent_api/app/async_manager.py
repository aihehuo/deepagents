"""Async task manager for group_agent_api (REQ-009 / RESP-009-FIX5).

Manages canonical request fingerprinting derived from TrustedSession, single-pass critical section slot reservation & decision gate synchronization,
task spawn failure compensation, active task locks, desensitized error logging, and callback delivery.
"""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import json
import logging
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage

from apps.group_agent_api.agent_factory.admin_ops_tools import (
    is_admin_debug_source,
)
from apps.group_agent_api.agent_factory.agent import (
    FORCE_SAVE_PROMPT,
    PROFILE_SUPERSEDED_RESULT_PREFIX,
    UC34Observer,
    save_group_profile,
)
from apps.group_agent_api.agent_factory.capability import unlocks_network
from apps.group_agent_api.agent_factory.content_quality import (
    finalize_and_guard_user_visible_reply,
)
from apps.group_agent_api.agent_factory.context import (
    known_profile_system_message as _known_profile_system_message,
    prior_recommendation_system_content,
    referral_context_system_message as _referral_context_system_message,
)
from apps.group_agent_api.agent_factory.debug_trace import (
    end_decision_trace,
    record_decision_point,
    start_decision_trace,
)
from apps.group_agent_api.agent_factory.guard import enforce_capability_guard
from apps.group_agent_api.agent_factory.integrations.callback_client import (
    MouthIngressRejected,
    send_callback_event,
    validate_and_normalize_callback_url,
)
from apps.group_agent_api.agent_factory.integrations.config import async_run_timeout_s, read_mode
from apps.group_agent_api.agent_factory.intent_routing.context import (
    IntentContextScope,
    completed_intent_context_message,
)
from apps.group_agent_api.agent_factory.intent_routing.execution import (
    IntentExecution,
    QueryPlanError,
    active_intent_execution,
    prepare_intent_execution,
    search_log_schema_enabled,
    search_result_schema_enabled,
)
from apps.group_agent_api.agent_factory.intent_routing.plan_executor import (
    execute_query_plan,
)
from apps.group_agent_api.agent_factory.intent_routing.resolver import (
    IntentResolution,
    aresolve_intent,
    intent_routing_mode,
    record_intent_outcome,
    record_intent_prediction,
)
from apps.group_agent_api.agent_factory.invite import (
    generate_invite_with_optional_llm,
    should_emit_invite_artifact,
)
from apps.group_agent_api.agent_factory.module_config import (
    search_relax_enabled,
    search_relax_enabled_for_user,
)
from apps.group_agent_api.agent_factory.profile_quality import (
    looks_like_clarifying_reply,
)
from apps.group_agent_api.agent_factory.profile_store import assert_profile_persisted, load_profile
from apps.group_agent_api.agent_factory.revisit import (
    parse_revisit_from_metadata,
)
from apps.group_agent_api.agent_factory.search_relax import search_relax_system_addon
from apps.group_agent_api.agent_factory.stream_buffer import StreamChunkBuffer
from apps.group_agent_api.agent_factory.suggested_replies import (
    extract_suggested_replies,
)
from apps.group_agent_api.agent_factory.write_skills import profile_write_allowed_for_turn
from apps.group_agent_api.app.endpoints.chat import (
    MAX_PERSIST_ATTEMPTS,
    _extract_reply,
    _invoke_config,
    _merge_force_save_reply,
    _profile_usable_this_turn,
    _should_force_profile_save,
)
from apps.group_agent_api.app.models import AsyncCallRequest, AsyncCallResponse, CallbackEnvelope, SearchLogEntry
from apps.group_agent_api.app.session import TrustedSession
from apps.group_agent_api.app.state import AppState
from apps.group_agent_api.app.utils import aget_agent_state, get_agent_checkpointer
from apps.group_agent_api.app.venue_protocol import calculate_v2_fingerprint, callback_bindings
from skillpacks.aihehuo.evidence import collect_public_evidence

_logger = logging.getLogger("uvicorn.error")

_MAX_IDEMPOTENCY_CACHE = 5_000
_DETERMINISTIC_SAVE_TOOL_CALL_ID = "harness_deterministic_save"


def _trusted_rollout_context(req: AsyncCallRequest) -> tuple[str, str]:
    """Read only the typed top-level rollout context issued by Micro."""
    context = req.rollout_context
    if context is None:
        return "legacy_v1", "ga-v2-canary-v1"
    return context.protocol_mode, context.rollout_version


def _match_contract_for_run(protocol_mode: str) -> tuple[str | None, str | None]:
    """Resolve the match wire contract without downgrading a v2 Run to v1."""
    if protocol_mode != "grounded_v2":
        return None, None

    from apps.group_agent_api.agent_factory.integrations.config import match_v2_enabled

    if not match_v2_enabled():
        return None, "v2_capability_unavailable"
    return "ga-match-v2", None


def _referral_payload_from_metadata(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    """Extract and build structured referral payload dict for final TaskChannel event (REQ-042)."""
    if not isinstance(metadata, dict):
        return None
    ref_ctx = metadata.get("referral_context")
    if not isinstance(ref_ctx, dict) or not ref_ctx.get("referral_id") or not ref_ctx.get("applicant_id"):
        return None

    return {
        "referral_id": ref_ctx["referral_id"],
        "applicant_id": ref_ctx["applicant_id"],
        "applicant_name": str(ref_ctx.get("applicant_name") or ""),
        "applicant_doing": str(ref_ctx.get("applicant_doing") or ""),
        "applicant_need": str(ref_ctx.get("applicant_need") or ""),
        "applicant_offer": str(ref_ctx.get("applicant_offer") or ""),
        "match_highlights": list(ref_ctx.get("match_highlights") or []),
        "status": str(ref_ctx.get("status") or "pending"),
    }


@dataclass
class IdempotencySlot:
    idempotency_key: str
    run_id: str
    fingerprint: str
    status: Literal["reserved", "completed", "rolled_back"]
    response: AsyncCallResponse | None
    created_at: float
    decision_event: asyncio.Event = field(default_factory=asyncio.Event)


_idempotency_store: OrderedDict[str, IdempotencySlot] = OrderedDict()
_run_id_store: dict[str, str] = {}
_idempotency_lock = asyncio.Lock()


def _profile_save_succeeded_this_turn(messages: list[Any], start: int) -> bool:
    """True when save_group_profile returned a non-error, non-superseded ack this turn."""
    save_tool_call_ids: set[str] = set()
    for message in messages[start:]:
        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls and hasattr(message, "additional_kwargs"):
            tool_calls = (message.additional_kwargs or {}).get("tool_calls")
        if not tool_calls:
            continue
        for tc in tool_calls:
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            if name != "save_group_profile":
                continue
            tool_call_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
            if tool_call_id:
                save_tool_call_ids.add(str(tool_call_id))

    for message in messages[start:]:
        if not isinstance(message, ToolMessage):
            continue
        tool_name = getattr(message, "name", None)
        tool_call_id = str(getattr(message, "tool_call_id", "") or "")
        is_save = tool_name == "save_group_profile" or (bool(tool_call_id) and tool_call_id in save_tool_call_ids)
        if not is_save:
            continue
        content = str(getattr(message, "content", "") or "").strip()
        if content.startswith("error:"):
            continue
        if content.startswith(PROFILE_SUPERSEDED_RESULT_PREFIX):
            continue
        if content.startswith("ok:"):
            return True
    return False


def _profile_save_attempted_this_turn(messages: list[Any], start: int) -> bool:
    """True when the model emitted a save_group_profile tool call this turn."""
    for message in messages[start:]:
        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls and hasattr(message, "additional_kwargs"):
            tool_calls = (message.additional_kwargs or {}).get("tool_calls")
        if not tool_calls:
            continue
        for tc in tool_calls:
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            if name == "save_group_profile":
                return True
    return False


def _profile_was_superseded(messages: list[Any], start: int) -> bool:
    """Return whether a new tool result reports a stale/superseded profile."""
    for message in messages[start:]:
        if not isinstance(message, ToolMessage):
            continue
        content = str(message.content or "").strip()
        if content.startswith(PROFILE_SUPERSEDED_RESULT_PREFIX):
            return True
    return False


def determine_persistence_failure_reason(
    messages: list[Any],
    start: int,
    attempt: int = 1,
    last_assertion_reason: str | None = None,
) -> str:
    """Extract a desensitized, actionable failure reason from execution message traces."""
    save_tool_called = False
    save_tool_call_ids: set[str] = set()
    last_tool_error: str | None = None

    for message in messages[start:]:
        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls and hasattr(message, "additional_kwargs"):
            tool_calls = message.additional_kwargs.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                if name == "save_group_profile":
                    save_tool_called = True
                    tool_call_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                    if tool_call_id:
                        save_tool_call_ids.add(str(tool_call_id))

    for message in messages[start:]:
        if not isinstance(message, ToolMessage):
            continue
        tool_name = getattr(message, "name", None)
        tool_call_id = str(getattr(message, "tool_call_id", "") or "")
        is_save_result = tool_name == "save_group_profile" or (bool(tool_call_id) and tool_call_id in save_tool_call_ids)
        if is_save_result:
            save_tool_called = True
            content = str(getattr(message, "content", "") or "").strip()
            if content.startswith("error:"):
                last_tool_error = content

    if not save_tool_called:
        if attempt >= 2:
            return "force_save_failed:tool_not_called"
        return "tool_not_called"

    if last_tool_error:
        if "semantic_projection" in last_tool_error or "missing user_id" in last_tool_error:
            return "validation_error"
        if "profile_database" in last_tool_error:
            return "remote_ack_failed"
        return "tool_execution_error"

    if attempt >= 2:
        return "force_save_failed"

    if last_assertion_reason:
        if "invalid_schema" in last_assertion_reason or "incomplete" in last_assertion_reason:
            return "validation_error"

    return "tool_execution_error"


_PROFILE_LABEL = re.compile(
    r"(?:^|[\s,，;；])(?:[-*]\s*)?(doing|need|offer|正在推进|在做|做|项目|业务|希望|需要|需求|寻找|求|缺|可以提供|可提供|提供|资源|技能)[：:]\s*",
    re.IGNORECASE,
)
_PROFILE_FIELDS = {
    **dict.fromkeys(("doing", "正在推进", "在做", "做", "项目", "业务"), "doing"),
    **dict.fromkeys(("need", "希望", "需要", "需求", "寻找", "求", "缺"), "need"),
    **dict.fromkeys(("offer", "可以提供", "可提供", "提供", "资源", "技能"), "offer"),
}


def _profile_write_declined(message: str) -> bool:
    """Honor explicit read-only requests before any fallback or history scan."""
    normalized = message.lower()
    return any(
        term in normalized
        for term in (
            "只查询",
            "只搜索",
            "不更新画像",
            "不要更新画像",
            "不保存画像",
            "不要保存画像",
            "不记录画像",
            "不要记录画像",
            "read only",
            "read-only",
            "do not save",
            "don't save",
            "do not update",
            "don't update",
        )
    )


def _profile_save_requested(message: str) -> bool:
    return any(
        term in message.lower()
        for term in ("保存画像", "保存本人画像", "更新画像", "记录画像", "保存我的画像", "更新我的画像", "save my profile", "update my profile")
    )


def _explicit_dimensions(text: str) -> dict[str, str] | None:
    """Slice between bounded labels instead of backtracking across whole documents."""
    labels = list(_PROFILE_LABEL.finditer(text))
    dimensions: dict[str, str] = {}
    for index, label in enumerate(labels):
        end = labels[index + 1].start() if index + 1 < len(labels) else len(text)
        value = re.split(r"[\n,，;；。.!！]", text[label.end() : end], maxsplit=1)[0].strip()
        if value:
            dimensions[_PROFILE_FIELDS[label.group(1).lower()]] = value
    if set(dimensions) == {"doing", "need", "offer"}:
        return dimensions
    # Retain the old three-clause Chinese form without cross-document wildcards.
    clauses = re.split(r"[，,；;\n]", text.strip(), maxsplit=3)
    if len(clauses) == 3:
        dimensions = {}
        for field, clause, prefix in zip(
            ("doing", "need", "offer"),
            clauses,
            (r"(?:正在推进|在做|做|项目|业务)", r"(?:希望|需要|需求|寻找|求|缺)", r"(?:可以提供|可提供|提供|资源|技能)"),
            strict=True,
        ):
            label = re.match(prefix + r"[：:]?\s*", clause.strip())
            if label is None:
                return None
            value = clause.strip()[label.end() :].strip(" 。.!！")
            if not value:
                return None
            dimensions[field] = value
        return dimensions
    return None


_NATURAL_DOING = re.compile(r"(?:正在做|在做|做一个|最近在做|做的是)(?P<doing>.+?)(?=[。.!！]|，目前|，我|。目前)")
_NATURAL_NEED = re.compile(r"(?:需要找|想找|找一个|找一位|缺一个)(?P<need>.+?)(?=[。.!！～~]|你这边|有合适|，最好)")
_NATURAL_OFFER = re.compile(r"(?:目前我有|我有|能提供|可以提供|手上有)(?P<offer>.+?)(?=[。.!！]|我需要|需要找|想找)")


def _clip_dimension(value: str | None, limit: int = 200) -> str:
    text = re.sub(r"\s+", " ", (value or "").strip())
    return text[:limit].strip(" ，,；;。")


def _dimensions_from_natural_chinese(text: str) -> dict[str, str] | None:
    """Pull doing/need/offer from a spoken dump without `doing:` labels."""
    if not text or len(text) > 2_000:
        return None
    doing_m = _NATURAL_DOING.search(text)
    need_m = _NATURAL_NEED.search(text)
    offer_m = _NATURAL_OFFER.search(text)
    if not (doing_m and need_m and offer_m):
        return None
    dimensions = {
        "doing": _clip_dimension(doing_m.group("doing")),
        "need": _clip_dimension(need_m.group("need")),
        "offer": _clip_dimension(offer_m.group("offer")),
    }
    if not all(len(value) >= 4 for value in dimensions.values()):
        return None
    return dimensions


def extract_explicit_profile_dimensions(message: str, messages: list[Any] | None = None) -> dict[str, str] | None:
    """Extract a complete profile from explicit user statement or recent conversation history.

    This provides a deterministic last-resort fallback path if LLM omitted save_group_profile
    tool call even when doing/need/offer were explicitly stated or listed in conversation.
    """
    if _profile_write_declined(message):
        return None
    if message and len(message) <= 2_000:
        dimensions = _explicit_dimensions(message)
        if dimensions is not None:
            return dimensions
        natural = _dimensions_from_natural_chinese(message)
        if natural is not None:
            return natural

    if messages and _profile_save_requested(message):
        recent = [msg for msg in messages[-24:] if isinstance(msg, HumanMessage)][-6:]
        for msg in reversed(recent):
            text = msg.content.strip() if isinstance(msg.content, str) else ""
            if _profile_write_declined(text):
                break
            if not text or len(text) > 3_000:
                continue
            dimensions = _explicit_dimensions(text)
            if dimensions is not None:
                return dimensions
            natural = _dimensions_from_natural_chinese(text)
            if natural is not None:
                return natural
    return None


async def _attempt_deterministic_profile_save(
    *,
    message: str,
    config: dict[str, Any],
    messages: list[Any],
) -> bool:
    """Invoke the real profile tool once for an explicitly complete statement or historical profile."""
    from apps.group_agent_api.agent_factory.agent import default_runtime_dir
    from apps.group_agent_api.agent_factory.write_skills import member_write_configuration, profile_write_mode

    metadata = config.get("metadata") or {}
    try:
        permission = metadata.get("can_write_profile", profile_write_mode() == "legacy")
        allowed = permission is True if isinstance(permission, bool) else str(permission).strip().lower() in {"true", "1", "yes", "on", "enabled"}
        if not allowed:
            return False
        writers, _, _ = member_write_configuration(
            runtime=Path(str(metadata.get("base_dir") or default_runtime_dir())),
            enabled=True,
            writer=save_group_profile,
        )
    except (ValueError, OSError):
        return False
    if not writers:
        return False
    try:
        dimensions = extract_explicit_profile_dimensions(message, messages)
    except Exception:  # Optional extraction must not discard an otherwise valid reply.
        _logger.exception("profile_fallback_extraction_failed")
        return False
    if dimensions is None:
        return False

    tool_args = {
        **dimensions,
        "doing_disclosure": "inferred_unconfirmed",
        "need_disclosure": "inferred_unconfirmed",
        "offer_disclosure": "inferred_unconfirmed",
    }
    messages.append(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "save_group_profile",
                    "args": tool_args,
                    "id": _DETERMINISTIC_SAVE_TOOL_CALL_ID,
                }
            ],
        )
    )
    tool_result = await asyncio.to_thread(
        writers[0].invoke,
        tool_args,
        config,
    )
    messages.append(
        ToolMessage(
            content=str(tool_result),
            name="save_group_profile",
            tool_call_id=_DETERMINISTIC_SAVE_TOOL_CALL_ID,
        )
    )
    return True


def calculate_request_fingerprint(req: AsyncCallRequest, session: TrustedSession) -> str:
    """Construct complete canonical request fingerprint derived from TrustedSession & canonical callback URL."""
    if req.request_schema_version in {2, 3}:
        if (
            req.user_id != session.principal.user_id
            or req.unionid != session.principal.unionid
            or req.group_id != session.group_id
            or (req.user_token or "").strip() != (session.principal.user_token or "").strip()
            or (req.group_token or "").strip() != (session.group_token or "").strip()
        ):
            raise ValueError("binding_mismatch: request fingerprint requires trusted principal")
        return calculate_v2_fingerprint(req.model_dump())
    canonical_url = validate_and_normalize_callback_url(req.callback_url)
    group_token_sha = hashlib.sha256((session.group_token or "").encode("utf-8")).hexdigest()
    user_token_sha = hashlib.sha256((session.principal.user_token or "").encode("utf-8")).hexdigest()

    canon = {
        "user_id": session.principal.user_id,
        "unionid": session.principal.unionid,
        "group_id": session.group_id,
        "conversation_id": req.conversation_id,
        "run_id": req.run_id,
        "callback_url": canonical_url,
        "message": req.message,
        "membership": session.membership.tier.value,
        "membership_source": session.membership.source,
        "run_match": req.run_match,
        "run_invite": req.run_invite,
        "willing_to_at": req.willing_to_at,
        "group_token_sha256": group_token_sha,
        "user_token_sha256": user_token_sha,
        "metadata": req.metadata or {},
        "rollout_context": (req.rollout_context.model_dump(mode="json") if req.rollout_context is not None else None),
    }
    raw = json.dumps(canon, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def reserve_idempotency_slot(
    req: AsyncCallRequest,
    session: TrustedSession,
) -> tuple[Literal["RESERVED", "HIT", "INITIALIZING", "CONFLICT"], AsyncCallResponse | None, IdempotencySlot | None]:
    """Atomic check-and-reserve lock with TrustedSession fingerprint binding in a single critical section."""
    fp = calculate_request_fingerprint(req, session)
    is_pending = False
    pending_slot: IdempotencySlot | None = None

    async with _idempotency_lock:
        if req.idempotency_key in _idempotency_store:
            slot = _idempotency_store[req.idempotency_key]
            if slot.fingerprint != fp or slot.run_id != req.run_id:
                _logger.warning("Idempotency conflict: key=%s fingerprint mismatch", req.idempotency_key)
                return ("CONFLICT", None, None)

            if slot.status == "completed" and slot.response:
                return ("HIT", slot.response, slot)

            if slot.status == "rolled_back":
                return ("INITIALIZING", None, None)

            is_pending = True
            pending_slot = slot

        else:
            if req.run_id in _run_id_store:
                existing_key = _run_id_store[req.run_id]
                if existing_key != req.idempotency_key:
                    _logger.warning("Idempotency conflict: run_id=%s bound to key=%s", req.run_id, existing_key)
                    return ("CONFLICT", None, None)

            # Eviction: ONLY evict oldest COMPLETED slot, NEVER reserved/pending slots!
            while len(_idempotency_store) >= _MAX_IDEMPOTENCY_CACHE:
                completed_key_to_evict = None
                for k, s in _idempotency_store.items():
                    if s.status == "completed":
                        completed_key_to_evict = k
                        break

                if completed_key_to_evict is not None:
                    old_slot = _idempotency_store.pop(completed_key_to_evict)
                    _run_id_store.pop(old_slot.run_id, None)
                else:
                    _logger.warning(
                        "Idempotency store cache full (%d) with all reserved slots; rejecting new slot key=%s",
                        _MAX_IDEMPOTENCY_CACHE,
                        req.idempotency_key,
                    )
                    return ("INITIALIZING", None, None)

            new_slot = IdempotencySlot(
                idempotency_key=req.idempotency_key,
                run_id=req.run_id,
                fingerprint=fp,
                status="reserved",
                response=None,
                created_at=time.time(),
            )
            _idempotency_store[req.idempotency_key] = new_slot
            _run_id_store[req.run_id] = req.idempotency_key
            return ("RESERVED", None, new_slot)

    if is_pending and pending_slot is not None:
        try:
            await asyncio.wait_for(pending_slot.decision_event.wait(), timeout=2.0)
        except TimeoutError:
            _logger.warning("Waiter timed out for idempotency_key=%s", req.idempotency_key)
            return ("INITIALIZING", None, None)

        async with _idempotency_lock:
            cur_slot = _idempotency_store.get(req.idempotency_key)
            if cur_slot is pending_slot and cur_slot.status == "completed" and cur_slot.response:
                if cur_slot.fingerprint == fp:
                    return ("HIT", cur_slot.response, cur_slot)
                return ("CONFLICT", None, None)
            return ("INITIALIZING", None, None)

    return ("INITIALIZING", None, None)


async def complete_idempotency_reservation(slot: IdempotencySlot, response: AsyncCallResponse) -> bool:
    """Commit idempotency slot with exact slot identity matching. Returns True on success."""
    async with _idempotency_lock:
        stored_slot = _idempotency_store.get(slot.idempotency_key)
        if stored_slot is slot and stored_slot.status == "reserved":
            stored_slot.status = "completed"
            stored_slot.response = response
            stored_slot.decision_event.set()
            return True
        _logger.warning("Complete idempotency reservation failed: slot mismatch or not reserved key=%s", slot.idempotency_key)
        return False


async def rollback_idempotency_reservation(slot: IdempotencySlot) -> bool:
    """Rollback idempotency slot with exact slot identity matching. Always signals decision_event on slot."""
    async with _idempotency_lock:
        slot.status = "rolled_back"
        slot.decision_event.set()

        stored_slot = _idempotency_store.get(slot.idempotency_key)
        if stored_slot is slot:
            _idempotency_store.pop(slot.idempotency_key, None)
            _run_id_store.pop(slot.run_id, None)
            return True
        _logger.warning("Rollback idempotency reservation mismatch: slot key=%s not active in store", slot.idempotency_key)
        return False


def clear_async_idempotency_cache() -> None:
    """Test helper."""
    _idempotency_store.clear()
    _run_id_store.clear()


async def execute_async_run(
    *,
    req: AsyncCallRequest,
    session: TrustedSession,
    state: AppState,
    tid: str,
    slot: IdempotencySlot | None = None,
) -> None:
    """Legacy API adapter: decision gate + timeout + shared core + process-local cleanup."""
    user_id = session.principal.user_id
    group_id = session.group_id

    # Wait for decision_event gate before initiating core execution
    if slot is not None:
        try:
            await asyncio.wait_for(slot.decision_event.wait(), timeout=5.0)
        except (TimeoutError, asyncio.CancelledError):
            _logger.warning("Task decision gate interrupted run_id=%s", req.run_id)
            state.finish_agent_run(tid, "call_async")
            return

        # Check decision result: ONLY proceed to progress callback & core LLM if status == completed!
        if slot.status != "completed":
            _logger.warning("Task decision gate rolled back, aborting execution run_id=%s status=%s", req.run_id, slot.status)
            state.finish_agent_run(tid, "call_async")
            return

    seq = 0
    terminal_created = False
    terminal_delivered = False

    async def _emit_event(event_type: str, payload: dict[str, Any]) -> bool:
        nonlocal seq, terminal_created, terminal_delivered
        if terminal_created:
            _logger.warning("Attempted to emit callback after terminal state created run_id=%s event=%s", req.run_id, event_type)
            return False

        seq += 1
        env = CallbackEnvelope(
            version="GA-CALLBACK-V1",
            run_id=req.run_id,
            **({"idempotency_key": req.idempotency_key} if req.request_schema_version not in {2, 3} else {}),
            seq=seq,
            event=event_type,  # type: ignore[arg-type]
            occurred_at=datetime.datetime.now(datetime.UTC).isoformat(timespec="microseconds").replace("+00:00", "Z"),
            user_id=user_id,
            group_id=group_id,
            conversation_id=req.conversation_id,
            payload=payload,
            **callback_bindings(req),
        )

        try:
            delivered = await send_callback_event(callback_url=req.callback_url, envelope_dict=env.model_dump(exclude_none=True))
        except MouthIngressRejected:
            # Mouth did not commit seq — rewind for same-seq repair re-emit.
            seq = max(0, seq - 1)
            raise
        except Exception:
            if event_type in {"final", "error"}:
                seq = max(0, seq - 1)
            raise

        if event_type in {"final", "error"} and delivered:
            terminal_created = True
            terminal_delivered = True
        elif event_type in {"final", "error"} and not delivered:
            # Transport fail without accept — rewind so error path can use next seq.
            seq = max(0, seq - 1)

        _logger.info(
            "Callback event status run_id=%s seq=%d event=%s delivered=%s",
            req.run_id,
            seq if delivered else seq,
            event_type,
            delivered,
        )
        return delivered

    try:
        # Step 1: Send initial progress event
        await _emit_event("progress", {"phase": "started", "message": "processing_started"})

        # Step 2: Perform execution with run timeout
        timeout = async_run_timeout_s()
        await asyncio.wait_for(
            execute_async_run_core(
                req=req,
                session=session,
                state=state,
                tid=tid,
                emit_callback=_emit_event,
            ),
            timeout=timeout,
        )

    except TimeoutError:
        _logger.error("Async run timed out run_id=%s error_type=TimeoutError", req.run_id)
        UC34Observer.error(f"action=async_run_timeout run_id={req.run_id} error_type=TimeoutError")
        if not terminal_created:
            await _emit_event(
                "error",
                {
                    "error_code": "AsyncRunTimeout",
                    "message": f"Task execution timed out after {async_run_timeout_s()}s",
                },
            )

    except asyncio.CancelledError:
        _logger.warning("Async run cancelled run_id=%s error_type=CancelledError", req.run_id)
        UC34Observer.error(f"action=async_run_cancelled run_id={req.run_id} error_type=CancelledError")
        if not terminal_created:
            await _emit_event(
                "error",
                {
                    "error_code": "AsyncRunCancelled",
                    "message": "Task execution was cancelled during application shutdown",
                },
            )
        raise

    except Exception as exc:  # noqa: BLE001
        err_type = type(exc).__name__
        # Desensitized: never log traceback / raw exception (may embed secrets, tokens, PII).
        _logger.error("async_run_error run_id=%s error_type=%s", req.run_id, err_type)
        UC34Observer.error(f"action=async_run_error run_id={req.run_id} error_type={err_type}")
        if not terminal_created:
            await _emit_event(
                "error",
                {
                    "error_code": "AsyncRunFailed",
                    "message": "Task execution failed",
                },
            )

    finally:
        state.finish_agent_run(tid, "call_async")


async def execute_async_run_core(
    *,
    req: AsyncCallRequest,
    session: TrustedSession,
    state: AppState,
    tid: str,
    emit_callback: Any,
    side_effect_gate: Any | None = None,
) -> None:
    """Shared execution core used by legacy API adapter and Celery worker (REQ-032).

    Independent of FastAPI request lifecycle, process-local idempotency store,
    and ``AppState.active_tasks``.

    Args:
        side_effect_gate: Optional callable ``(action: str) -> None`` that raises
            when the attempt lost lease ownership (worker fence).
    """
    from apps.group_agent_api.agent_factory.host_context import HostContext, active_host_context
    from apps.group_agent_api.agent_factory.venue_greeting import execute_venue_greeting

    if req.input_kind == "venue_event" and state.greeting_model is None:
        raise VenueEventExecutorUnavailable("venue_event_executor_unavailable")
    host = (
        HostContext.from_request(req, principal=session.principal, base_dir=state.base_dir)
        if req.request_schema_version == 2 or (req.request_schema_version == 3 and req.execution_mode == "cafe_host")
        else None
    )
    with active_host_context(host):
        if req.input_kind == "venue_event":
            await execute_venue_greeting(host=host, model=state.greeting_model, emit_callback=emit_callback, side_effect_gate=side_effect_gate)
            return
        await _execute_core_agent(req=req, session=session, state=state, tid=tid, emit_callback=emit_callback, side_effect_gate=side_effect_gate)


class VenueEventExecutorUnavailable(RuntimeError):
    """Permanent explicit hold until the DA-002 greeting executor is ready."""


def _execution_metadata(req: AsyncCallRequest) -> dict[str, Any]:
    """Derive trusted episode scope without changing the signed request body."""
    metadata = dict(req.metadata or {})
    if req.request_schema_version in {2, 3}:
        metadata["episode_id"] = req.episode_id
    if req.request_schema_version == 3:
        from apps.group_agent_api.app.execution_modes import execution_binding

        metadata["execution_binding"] = execution_binding(req)
    return metadata


@collect_public_evidence
async def _execute_core_agent(
    *,
    req: AsyncCallRequest,
    session: TrustedSession,
    state: AppState,
    tid: str,
    emit_callback: Any,
    side_effect_gate: Any | None = None,
) -> None:
    progress_started = time.monotonic()

    def _gate(action: str) -> None:
        if side_effect_gate is not None:
            side_effect_gate(action)

    async def _intent_progress(
        phase: str,
        *,
        intent: str | None = None,
        routing_hash: str | None = None,
        display_code: str,
        plan_id: str | None = None,
        plan_step_id: str | None = None,
        request_index: int | None = None,
        step_result_count: int | None = None,
        aggregate_result_count: int | None = None,
    ) -> None:
        if side_effect_gate is None:
            return
        await emit_callback(
            "progress",
            {
                "progress_schema_version": 1,
                "phase": phase,
                "intent": intent,
                "display_code": display_code,
                "routing_snapshot_hash": routing_hash,
                "plan_id": plan_id,
                "plan_step_id": plan_step_id,
                "request_index": request_index,
                "elapsed_ms": max(
                    0, min(int((time.monotonic() - progress_started) * 1000), 600_000)
                ),
                "step_result_count": step_result_count,
                "aggregate_result_count": aggregate_result_count,
            },
        )

    user_id = session.principal.user_id
    group_id = session.group_id
    run_protocol_mode, run_rollout_version = _trusted_rollout_context(req)
    tier = session.membership.tier
    user_token = session.principal.user_token
    can_write_profile = session.capabilities.can_write_profile and profile_write_allowed_for_turn(req.message)
    runtime_metadata = _execution_metadata(req)
    admin_debug = req.request_schema_version != 3 and is_admin_debug_source(req.metadata or {})
    from apps.group_agent_api.agent_factory.host_policy import is_readonly_host

    cafe_host = is_readonly_host()
    agent = (
        state.agent
        if cafe_host
        else (
            state.admin_agent
            if admin_debug and getattr(state, "admin_agent", None) is not None
            else state.member_agent_for_capability(
                can_search=session.capabilities.can_search,
                can_write_profile=can_write_profile,
            )
        )
    )
    if cafe_host:
        from apps.group_agent_api.agent_factory.cafe_agent import create_cafe_agent

        if getattr(state, "cafe_agent", None) is None:
            state.cafe_agent = create_cafe_agent(
                model=state.greeting_model, checkpointer=get_agent_checkpointer(state.agent), quality_model=state.quality_model or state.polish_model
            )
        agent = state.cafe_agent
    lock = state.thread_locks.setdefault(tid, asyncio.Lock())
    reply = ""
    profile_ok = False
    profile_status = "failed"
    persistence_failure_reason: str | None = None
    messages: list[Any] = []
    msg_count_before = 0
    intent_resolution: IntentResolution | None = None
    intent_execution: IntentExecution | None = None

    _trace_buf, _trace_token = start_decision_trace()
    async with lock:
        try:
            test_lvl = os.environ.get("GROUP_AGENT_TEST_LEVEL")
            if not cafe_host and test_lvl and load_profile(state.base_dir, user_id, group_id) is None:
                try:
                    from apps.group_agent_api.agent_factory.profile_schema import GroupProfile, ProfileField
                    from apps.group_agent_api.agent_factory.profile_store import save_profile
                    from apps.group_agent_api.fixtures.loader import load_fixture

                    ds = load_fixture(test_lvl)
                    m_key = f"{group_id}:{user_id}"
                    m = ds.members.get(m_key)
                    if m and m.profile:
                        p = GroupProfile(
                            user_id=user_id,
                            group_id=group_id,
                            doing=ProfileField(value=m.profile.get("doing", "Building AI")),
                            need=ProfileField(value=m.profile.get("need", "Co-founder")),
                            offer=ProfileField(value=m.profile.get("offer", "Python")),
                        )
                        save_profile(state.base_dir, p)
                except Exception as exc:
                    _logger.error("Fixture seed error: %s", exc)
            agent_state = await aget_agent_state(agent, {"configurable": {"thread_id": tid}})
            prior_messages = (
                list(agent_state.values.get("messages", []))
                if agent_state and agent_state.values
                else []
            )
            msg_count_before = len(prior_messages)
            routing_mode = intent_routing_mode()
            if not admin_debug and routing_mode in {"observe", "enforce"}:
                await _intent_progress(
                    "intent_resolving",
                    display_code="resolving_intent",
                )
                execution_mode = "cafe_host" if cafe_host else "group_agent"
                episode_id = (
                    req.episode_id
                    if req.request_schema_version == 3
                    else runtime_metadata.get("episode_id")
                )
                intent_resolution = await aresolve_intent(
                    req.message,
                    execution_mode=execution_mode,
                    can_search=session.capabilities.can_search and not cafe_host,
                    prior_messages=prior_messages,
                    scope=IntentContextScope(
                        thread_id=tid,
                        user_id=user_id,
                        execution_mode=execution_mode,
                        episode_id=str(episode_id) if episode_id else None,
                    ),
                    planner_model=state.quality_model or state.polish_model,
                    profile=load_profile(state.base_dir, user_id, group_id),
                )
                record_intent_prediction(
                    intent_resolution,
                    run_id=req.run_id,
                    thread_id=tid,
                )
                if routing_mode == "enforce":
                    intent_execution = prepare_intent_execution(
                        intent_resolution,
                        original_message=req.message,
                        run_id=req.run_id,
                        request_fingerprint=calculate_request_fingerprint(req, session),
                        execution_mode=execution_mode,
                        can_search=session.capabilities.can_search and not cafe_host,
                    )
                    intent_execution.bind_progress_emitter(
                        emit_callback,
                        started_monotonic=progress_started,
                        emit_search_logs=(
                            req.request_schema_version == 3
                            and execution_mode == "group_agent"
                            and search_log_schema_enabled()
                        ),
                        emit_search_results=(
                            req.request_schema_version == 3
                            and execution_mode == "group_agent"
                            and search_result_schema_enabled()
                        ),
                    )
                    intent_execution.restore_adopted_evidence()
                    if (
                        intent_resolution.planner_status
                        in {"invalid", "error", "unavailable", "timed_out"}
                        and intent_execution.decision.action != "execute"
                    ):
                        code = (
                            "intent_classifier_timeout"
                            if intent_resolution.planner_status == "timed_out"
                            else "intent_classifier_invalid"
                        )
                        raise QueryPlanError(code)
                    await _intent_progress(
                        "intent_resolved",
                        intent=intent_execution.decision.primary_intent,
                        routing_hash=intent_execution.snapshot.canonical_content_hash,
                        display_code="intent_ready",
                    )
                    if intent_execution.decision.action == "execute":
                        await execute_query_plan(
                            intent_execution,
                            skills_dir=state.base_dir / "skills" / "aihehuo",
                        )
                        agent = state.member_agent_for_capability(
                            can_search=False,
                            can_write_profile=can_write_profile,
                        )
                    if (
                        not cafe_host
                        and intent_execution.decision.action != "execute"
                    ):
                        agent = state.member_agent_for_capability(
                            can_search=False,
                            can_write_profile=can_write_profile,
                        )
            config = _invoke_config(
                tid=tid,
                user_id=user_id,
                group_id=group_id,
                base_dir=str(state.base_dir),
                membership=tier.value,
                metadata=runtime_metadata,
                run_id=req.run_id,
                conversation_id=req.conversation_id,
                run_match=req.run_match,
                group_token=None if cafe_host else session.group_token,
                user_token=None if cafe_host else user_token,
                can_write_profile=can_write_profile if not cafe_host else False,
            )
            turn_messages: list[Any] = []
            if admin_debug:
                # Keep ops tools fresh: history often contains stale 401 narratives
                # that otherwise cause the model to skip tool calls.
                from apps.group_agent_api.agent_factory.admin_ops_tools import (
                    ADMIN_TURN_REMINDER,
                )

                turn_messages.append(SystemMessage(content=ADMIN_TURN_REMINDER))
            elif not cafe_host:
                known = (
                    None
                    if req.request_schema_version == 2
                    else _known_profile_system_message(
                        base_dir=state.base_dir,
                        user_id=user_id,
                        group_id=group_id,
                        can_write_profile=can_write_profile,
                    )
                )
                if known is not None:
                    turn_messages.append(known)
                ref_msg = _referral_context_system_message(req.metadata or {})
                if ref_msg is not None:
                    turn_messages.append(ref_msg)
                _, revisit_for_prompt = parse_revisit_from_metadata(req.metadata or {})
                from apps.group_agent_api.agent_factory.revisit import parse_prior_recommendation

                prior_rec = parse_prior_recommendation((req.metadata or {}).get("prior_recommendation"))
                match_reminder = prior_recommendation_system_content(revisit_for_prompt, prior_rec=prior_rec)
                if match_reminder and session.capabilities.can_search and read_mode() == "legacy":
                    turn_messages.append(SystemMessage(content=match_reminder))

                # If search_relax is enabled for this user via canary but off globally in YAML,
                # dynamically inject the search relaxation addon into turn context.
                if (
                    session.capabilities.can_search
                    and read_mode() == "legacy"
                    and search_relax_enabled_for_user(user_id)
                    and not search_relax_enabled()
                ):
                    relax_addon = search_relax_system_addon(user_id=user_id)
                    if relax_addon:
                        turn_messages.append(SystemMessage(content=relax_addon))
            if intent_execution is not None:
                turn_messages.append(
                    SystemMessage(
                        content=intent_execution.route_prompt(
                            skills_dir=state.base_dir / "skills" / "aihehuo"
                        )
                    )
                )
                if intent_execution.decision.action == "execute":
                    episode_id = (
                        req.episode_id
                        if req.request_schema_version == 3
                        else runtime_metadata.get("episode_id")
                    )
                    turn_messages.append(
                        completed_intent_context_message(
                            decision=intent_execution.decision,
                            scope=IntentContextScope(
                                thread_id=tid,
                                user_id=user_id,
                                execution_mode="group_agent",
                                episode_id=str(episode_id) if episode_id else None,
                            ),
                            routing_snapshot_hash=(
                                intent_execution.snapshot.canonical_content_hash
                            ),
                            plan_id=intent_execution.plan.plan_id,
                            receipt_ids=[
                                receipt.receipt_id
                                for receipt in intent_execution.receipts
                            ],
                            user_message=req.message,
                        )
                    )
            turn_messages.append(HumanMessage(content=req.message))
            _logger.info(
                "Core agent ainvoke start run_id=%s thread_id=%s msg_len=%d admin_debug=%s",
                req.run_id,
                tid,
                len(req.message or ""),
                admin_debug,
            )
            streaming_enabled = (
                intent_execution is None
                and os.environ.get("GROUP_AGENT_STREAMING_ENABLED", "0").lower()
                in {"1", "true", "yes"}
                and hasattr(agent, "astream")
            )
            _ainvoke_t0 = time.monotonic()
            try:
                if cafe_host:
                    _gate("host_model")
                if streaming_enabled and not cafe_host:
                    buffer = StreamChunkBuffer(emit_callback=emit_callback, min_interval_s=0.1)
                    tool_triggered_this_turn = False
                    async for chunk in agent.astream(
                        {"messages": turn_messages},
                        config,
                        stream_mode=["messages", "updates"],
                    ):
                        if isinstance(chunk, tuple) and len(chunk) == 3:
                            subgraph_path, mode, data = chunk
                        elif isinstance(chunk, tuple) and len(chunk) == 2:
                            mode, data = chunk
                            subgraph_path = ()
                        else:
                            continue

                        if mode == "updates":
                            if isinstance(data, dict):
                                for node_name in data:
                                    if node_name == "tools":
                                        tool_triggered_this_turn = True
                                        await emit_callback(
                                            "progress",
                                            {"phase": "searching", "step": "正在匹配合适的人选..."},
                                        )
                        elif mode == "messages":
                            msg_chunk = data[0] if isinstance(data, (tuple, list)) and data else data
                            if isinstance(msg_chunk, AIMessageChunk):
                                if getattr(msg_chunk, "tool_calls", None) or getattr(msg_chunk, "tool_call_chunks", None):
                                    tool_triggered_this_turn = True
                                elif msg_chunk.content and not tool_triggered_this_turn:
                                    await buffer.feed_token(str(msg_chunk.content))
                    await buffer.flush()
                    agent_state = await aget_agent_state(agent, {"configurable": {"thread_id": tid}})
                    messages = agent_state.values.get("messages", []) if agent_state and agent_state.values else []
                else:
                    with active_intent_execution(intent_execution):
                        result = await agent.ainvoke(
                            {"messages": turn_messages},
                            config,
                        )
                    messages = result.get("messages", [])
                if (
                    intent_execution is not None
                    and intent_execution.decision.action == "execute"
                    and not intent_execution.required_steps_satisfied()
                ):
                    raise QueryPlanError("required_tool_not_called")
            except Exception:
                _logger.exception(
                    "Core agent execution failed run_id=%s thread_id=%s elapsed_s=%.2f",
                    req.run_id,
                    tid,
                    time.monotonic() - _ainvoke_t0,
                )
                raise
            _logger.info(
                "Core agent execution done run_id=%s thread_id=%s elapsed_s=%.2f n_msgs=%d streaming=%s",
                req.run_id,
                tid,
                time.monotonic() - _ainvoke_t0,
                len(messages or []),
                streaming_enabled,
            )
            # Host middleware already produced the complete validated
            # message. Legacy prefix-peeling would diverge from its saver.
            reply = str(messages[-1].content) if cafe_host else _extract_reply(messages, msg_count_before)

            # chk.action_claim (YAML): unauthorized group-send / @ / admin claims.
            from apps.group_agent_api.agent_factory.checks.action_claim import (
                apply_action_claim_guard,
            )

            guarded_dialogue_text, action_claim_blocked = (reply, False) if cafe_host else apply_action_claim_guard(reply)
            if action_claim_blocked:
                reply = guarded_dialogue_text
                UC34Observer.warn(f"action=action_claim_blocked run_id={req.run_id}")

            if admin_debug:
                profile_ok = False
                profile_status = "admin_skip"
            elif cafe_host:
                profile_ok = False
                profile_status = "host_read_only"
            else:
                assertion = assert_profile_persisted(state.base_dir, user_id, group_id)

                if _profile_was_superseded(messages, msg_count_before):
                    profile_status = "superseded"
                    profile_ok = False
                else:
                    usable, _path = _profile_usable_this_turn(
                        base_dir=state.base_dir,
                        user_id=user_id,
                        group_id=group_id,
                        messages=messages,
                        msg_count_before=msg_count_before,
                        metadata=runtime_metadata,
                    )
                    if usable:
                        profile_ok = True
                        profile_status = "persisted"
                    else:
                        profile_ok = False
                        if assertion.ok:
                            # Prior episode profile exists but is not bound to this episode.
                            pass
                if (
                    can_write_profile
                    and not _profile_write_declined(req.message)
                    and (_profile_save_requested(req.message) or extract_explicit_profile_dimensions(req.message) is not None)
                    and _should_force_profile_save(
                        profile_ok=profile_ok,
                        profile_status=profile_status,
                        persist_alert=("profile_stale_for_episode" if (not profile_ok and assertion.ok and profile_status != "superseded") else None),
                        messages=messages,
                        msg_count_before=msg_count_before,
                        user_message=req.message,
                        reply=reply,
                    )
                ):
                    from apps.group_agent_api.agent_factory.checks.deterministic_profile_save import (
                        deterministic_profile_save_enabled,
                    )
                    from apps.group_agent_api.agent_factory.checks.force_save_retry import (
                        force_save_retry_enabled,
                    )

                    _gate("profile_force_save")
                    if force_save_retry_enabled():
                        for attempt in range(1, MAX_PERSIST_ATTEMPTS + 1):
                            _gate("profile_force_save_iter")
                            if _profile_was_superseded(messages, msg_count_before):
                                profile_status = "superseded"
                                profile_ok = False
                                break

                            usable, _path = _profile_usable_this_turn(
                                base_dir=state.base_dir,
                                user_id=user_id,
                                group_id=group_id,
                                messages=messages,
                                msg_count_before=msg_count_before,
                                metadata=runtime_metadata,
                            )
                            if usable:
                                profile_ok = True
                                profile_status = "persisted"
                                break

                            if attempt >= MAX_PERSIST_ATTEMPTS:
                                break

                            before_retry = len(messages)
                            result = await agent.ainvoke(
                                {"messages": [HumanMessage(content=FORCE_SAVE_PROMPT)]},
                                config,
                            )
                            messages = result.get("messages", [])
                            start_idx = before_retry if before_retry < len(messages) else 0
                            retry_reply = _extract_reply(messages, start_idx)
                            if retry_reply:
                                reply = _merge_force_save_reply(reply, retry_reply)

                    if not profile_ok and profile_status != "superseded":
                        profile_status = "failed"
                        fallback_attempted = False
                        if deterministic_profile_save_enabled():
                            _gate("profile_deterministic_save")
                            fallback_attempted = await _attempt_deterministic_profile_save(
                                message=req.message,
                                config=config,
                                messages=messages,
                            )
                        if fallback_attempted:
                            if _profile_was_superseded(messages, msg_count_before):
                                profile_status = "superseded"
                                profile_ok = False
                            else:
                                usable, _path = _profile_usable_this_turn(
                                    base_dir=state.base_dir,
                                    user_id=user_id,
                                    group_id=group_id,
                                    messages=messages,
                                    msg_count_before=msg_count_before,
                                    metadata=runtime_metadata,
                                )
                                if usable:
                                    profile_ok = True
                                    profile_status = "persisted"

                    if not profile_ok and profile_status == "failed":
                        last_reason = assertion.reason if not assertion.ok else "profile_stale_for_episode"
                        persistence_failure_reason = determine_persistence_failure_reason(
                            messages,
                            msg_count_before,
                            attempt=MAX_PERSIST_ATTEMPTS,
                            last_assertion_reason=last_reason,
                        )
                        UC34Observer.warn(
                            f"action=profile_persistence_failed user_id={user_id} "
                            f"group_id={group_id} run_id={req.run_id} "
                            f"reason={persistence_failure_reason}"
                        )
                elif not profile_ok and profile_status != "superseded":
                    # Clarifying turn on a new episode: keep the first reply, skip match.
                    profile_status = "stale_episode"

        except BaseException:
            end_decision_trace(_trace_token)
            raise
        finally:
            checkpointer = get_agent_checkpointer(agent)
            if checkpointer is not None and hasattr(checkpointer, "flush"):
                checkpointer.flush()

    # Harvest this turn's search_candidates tool result (model-called; not orchestrator).
    if not cafe_host:
        _gate("match")
    match_status = "skipped"
    candidates: list[dict[str, Any]] = []
    if cafe_host:
        match_reason = "host_read_only"
    elif admin_debug:
        match_reason = "admin_ops_mode"
        profile_ok = False
    else:
        match_reason = "profile_superseded" if profile_status == "superseded" else ("profile_persistence_failed" if not profile_ok else None)

    if profile_status == "superseded":
        reply = "这次画像更新已被较新的运行结果取代。我没有覆盖当前权威画像，也没有继续匹配或生成邀请。"

    if cafe_host:
        latest = messages[-1].additional_kwargs
        suggested_replies = latest.get("cafe_suggested_replies", [])
        host_violations = latest.get("cafe_guard_violations", [])
    else:
        reply, suggested_replies = extract_suggested_replies(reply)

    revisit_hint = None
    if not cafe_host:
        _, revisit_hint = parse_revisit_from_metadata(req.metadata or {})
    from apps.group_agent_api.agent_factory.search_tool import SearchTurnResult, extract_search_this_turn

    search_turn = SearchTurnResult(reason="host_read_only") if cafe_host else extract_search_this_turn(messages, msg_count_before)
    from apps.group_agent_api.agent_factory.public_evidence import extract_public_reads

    public_reads = (
        extract_public_reads(messages, msg_count_before)
        if not cafe_host and read_mode() == "public_skills"
        else None
    )
    from apps.group_agent_api.agent_factory.intent_routing import project_intent_search
    from apps.group_agent_api.agent_factory.intent_routing.projection import empty_search_reply

    intent_search = project_intent_search(intent_execution)
    no_match_reply = (
        empty_search_reply(intent_search)
        if intent_execution is not None
        and intent_execution.decision.primary_intent == "people_search"
        else None
    )
    if no_match_reply is not None:
        reply = no_match_reply
        suggested_replies = []
    search_completed = search_turn.called
    if intent_search.completed:
        search_completed = True
        public_reads = intent_search.public_reads
        candidates = intent_search.candidates
        match_status = "matched" if candidates else "empty"
        match_reason = "intent_query_plan"
    quality_gaps: list[str] = []
    search_log: SearchLogEntry | None = None
    if intent_search.completed:
        search_log = SearchLogEntry(
            search_id=f"search_{int(time.time() * 1000)}",
            timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
            query=intent_execution.original_message if intent_execution else "",
            rank_query=intent_execution.original_message if intent_execution else "",
            match_status=match_status,
            match_reason=match_reason,
            candidate_count=len(candidates),
            candidate_names=[
                str(c.get("name") or c.get("display_name") or "")
                for c in candidates
                if c.get("name") or c.get("display_name")
            ],
        )
    elif search_turn.called:
        match_status = search_turn.status if search_turn.status in {"matched", "weak", "empty", "skipped", "rejected", "error"} else "skipped"
        candidates = search_turn.candidates
        match_reason = search_turn.reason or "search_candidates_tool"
        search_log = SearchLogEntry(
            search_id=f"search_{int(time.time() * 1000)}",
            timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
            query=search_turn.query,
            rank_query=search_turn.rank_query,
            match_status=match_status,
            match_reason=match_reason,
            candidate_count=len(candidates),
            candidate_names=[str(c.get("name") or c.get("display_name") or "") for c in candidates if c.get("name") or c.get("display_name")],
        )
    elif match_reason is None:
        match_reason = "model_did_not_search"
    _logger.info(
        "Core match debug run_id=%s tier=%s profile_ok=%s match_status=%s candidates_count=%d",
        req.run_id,
        tier.value,
        profile_ok,
        match_status,
        len(candidates),
    )

    from apps.group_agent_api.agent_factory.guard import GuardResult

    # Café has no candidate/action surface. Its independent output middleware
    # already enforces text safety before checkpoint, regardless of YAML tier.
    guarded = (
        GuardResult(ok=not host_violations, tier=tier, reply=reply, violations=host_violations, blocked=bool(host_violations))
        if cafe_host
        else enforce_capability_guard(
            tier=tier,
            reply=reply,
            candidates=candidates,
            caller_group_id=group_id,
            user_id=user_id,
        )
    )
    if not cafe_host and guarded.blocked and not unlocks_network(tier):
        match_status = "skipped"
        match_reason = f"capability_{tier.value}_guard_blocked"
    elif match_status == "matched" and not guarded.candidates:
        match_status = "empty"
        match_reason = "no_auditable_public_match_basis"

    delivery_kind = None
    invite_text = None
    topic = None
    mentioned_user_ids: list[str] = []
    invite_ok = None

    if (
        req.run_invite
        and not intent_search.completed
        and profile_ok
        and unlocks_network(tier)
        and search_completed
        and should_emit_invite_artifact(
            match_status=match_status,
            match_reason=match_reason,
            candidate_count=len(guarded.candidates),
        )
    ):
        _gate("invite")
        from apps.group_agent_api.execution.active_fence import assert_write_allowed

        assert_write_allowed("invite")
        profile = load_profile(state.base_dir, user_id, group_id)
        if profile is not None:
            invite_status = match_status
            invite_candidates = guarded.candidates
            willing = req.willing_to_at

            def _invite_job():
                return generate_invite_with_optional_llm(
                    profile=profile,
                    candidates=invite_candidates,
                    match_status=invite_status,
                    willing_to_at=willing,
                    user_id=user_id,
                    group_id=group_id,
                    model=state.polish_model,
                )

            invite_res = await asyncio.to_thread(_invite_job)
            delivery_kind = invite_res.kind
            invite_text = invite_res.text
            topic = invite_res.topic
            mentioned_user_ids = invite_res.mentioned_user_ids
            invite_ok = invite_res.ok
            _logger.info(
                "Invite debug run_id=%s willing=%s kind=%s text_len=%d mentioned_count=%d",
                req.run_id,
                willing,
                delivery_kind,
                len(invite_text or ""),
                len(mentioned_user_ids),
            )
    elif req.run_invite and profile_ok and search_completed and match_status == "empty":
        _logger.info(
            "Invite skipped run_id=%s reason=empty_match_no_artifact",
            req.run_id,
        )

    final_profile = load_profile(state.base_dir, user_id, group_id) if profile_ok else None
    saved_this_turn = False if cafe_host else _profile_save_succeeded_this_turn(messages, msg_count_before)
    final_guarded = (
        guarded
        if cafe_host
        else finalize_and_guard_user_visible_reply(
            tier=tier,
            caller_group_id=group_id,
            user_id=user_id,
            original_reply=guarded.reply,
            profile=final_profile,
            profile_persisted=profile_ok,
            match_status=match_status,
            candidates=guarded.candidates,
            delivery_kind=delivery_kind,
            invite_ok=invite_ok,
            revisit_hint=revisit_hint,
            match_reason=match_reason,
            quality_gaps=quality_gaps,
            profile_saved_this_turn=saved_this_turn,
            public_reads=public_reads,
        )
    )
    combined_guard_violations = list(dict.fromkeys([*guarded.violations, *final_guarded.violations]))
    combined_guard_blocked = guarded.blocked or final_guarded.blocked

    out_candidates = invite_res.candidates if (req.run_invite and "invite_res" in locals() and invite_res.candidates) else final_guarded.candidates
    if final_profile is not None and out_candidates and not intent_search.completed:
        from apps.group_agent_api.agent_factory.invite import (
            enrich_candidates_with_single_copy,
        )

        out_candidates = enrich_candidates_with_single_copy(out_candidates, final_profile)

    # Determine reply_mode strictly in orchestrator code (TSD-13 §5.1)
    from apps.group_agent_api.agent_factory.grounding_protocol import (
        DialogueKind,
        GroundedFinalV1,
        GroundingBlock,
        InviteBlock,
        MatchSummaryBlock,
        ProfileSummaryBlock,
        ReplyMode,
        calculate_candidate_facts_digest,
        canonical_sha256,
    )
    from apps.group_agent_api.agent_factory.integrations.config import (
        grounded_final_enabled,
    )
    from apps.group_agent_api.agent_factory.integrations.profile_client import canonical_profile_digest

    if admin_debug:
        determined_reply_mode = ReplyMode.dialogue
    elif persistence_failure_reason or match_reason == "grounding_validation_failed":
        determined_reply_mode = ReplyMode.error
    elif search_completed and out_candidates and match_status == "matched":
        determined_reply_mode = ReplyMode.recommendation
    elif search_completed:
        determined_reply_mode = ReplyMode.no_match
    elif saved_this_turn and profile_ok:
        determined_reply_mode = ReplyMode.profile_confirmation
    else:
        determined_reply_mode = ReplyMode.dialogue

    is_clarifying = looks_like_clarifying_reply(guarded.reply)
    if search_completed:
        intent_route = "search"
    elif saved_this_turn and profile_ok:
        intent_route = "profile_save"
    elif is_clarifying:
        intent_route = "clarification"
    else:
        intent_route = "chit_chat"

    record_decision_point(
        phase="intent_route",
        detail={
            "route": intent_route,
            "reply_mode": determined_reply_mode.value,
            "search_called": search_completed,
            "server_query_plan_completed": intent_search.completed,
            "profile_saved_this_turn": saved_this_turn,
            "profile_persisted": profile_ok,
            "is_clarification": is_clarifying,
            "admin_debug": admin_debug,
            "message_len": len(req.message),
        },
        run_id=req.run_id,
        thread_id=tid,
    )
    if intent_resolution is not None:
        record_intent_outcome(
            intent_resolution,
            observed_route=intent_route,
            run_id=req.run_id,
            thread_id=tid,
        )

    is_v2_run = run_protocol_mode == "grounded_v2"

    # A profile confirmation is a persisted-state assertion.  In grounded
    # mode it is only legal when the local cache carries the version returned
    # by Micro for that exact write.  Stub/legacy caches deliberately have no
    # version and must not manufacture one.
    if is_v2_run and (
        not grounded_final_enabled()
        or (
            determined_reply_mode == ReplyMode.profile_confirmation
            and (final_profile is None or getattr(final_profile, "profile_version", None) is None)
        )
    ):
        determined_reply_mode = ReplyMode.error

    visible_reply = final_guarded.reply
    final_payload = {
        "reply": visible_reply,
        "reply_mode": determined_reply_mode.value,
        "suggested_replies": [] if combined_guard_blocked else suggested_replies,
        "profile_persisted": profile_ok,
        "profile_status": profile_status,
        "persistence_failure_reason": persistence_failure_reason,
        "capability": tier.value,
        "capability_source": session.membership.source,
        "match_status": match_status if out_candidates or match_status in {"empty", "skipped", "weak", "rejected", "error"} else "empty",
        "candidates": out_candidates,
        "match_reason": match_reason,
        # v3 QueryPlan emits one durable event per public request. Its old
        # final-summary log is the user message, not the actual query.
        "search_log": (
            search_log.model_dump()
            if search_log and not (
                req.request_schema_version == 3
                and req.execution_mode == "group_agent"
                and intent_search.completed
                and search_log_schema_enabled()
            )
            else None
        ),
        "guard_blocked": combined_guard_blocked,
        "guard_violations": combined_guard_violations,
        "delivery_kind": delivery_kind,
        "invite_text": invite_text,
        "topic": topic,
        "mentioned_user_ids": mentioned_user_ids,
        "at_users": mentioned_user_ids,
        "invite_ok": invite_ok,
        "willing_to_at": req.willing_to_at,
        "protocol_mode": run_protocol_mode,
        "rollout_version": run_rollout_version,
    }

    if is_v2_run:
        profile_summary = None
        if final_profile is not None:
            doing_val = getattr(final_profile, "doing", None)
            need_val = getattr(final_profile, "need", None)
            offer_val = getattr(final_profile, "offer", None)
            p_source_gid = getattr(final_profile, "group_id", None) or group_id or "global"

            def _clean_claim(val: Any) -> dict[str, Any] | None:
                if val is None:
                    return None
                if hasattr(val, "value"):
                    return {
                        "value": str(val.value or "").strip(),
                        "claim_type": getattr(val, "claim_type", "fact")
                        if isinstance(getattr(val, "claim_type", "fact"), str)
                        else getattr(val.claim_type, "value", "fact"),
                    }
                if isinstance(val, dict):
                    return {
                        "value": str(val.get("value") or "").strip(),
                        "claim_type": str(val.get("claim_type") or "fact"),
                    }
                return {"value": str(val).strip(), "claim_type": "fact"}

            profile_summary = ProfileSummaryBlock(
                schema_version=getattr(final_profile, "schema_version", 1),
                profile_version=getattr(final_profile, "profile_version", None),
                digest=f"sha256:{canonical_profile_digest(final_profile)}",
                source_group_id=p_source_gid,
                doing=_clean_claim(doing_val),
                need=_clean_claim(need_val),
                offer=_clean_claim(offer_val),
            )

        match_status_val = "matched" if out_candidates else "empty"
        match_summary = MatchSummaryBlock(
            contract_version="ga-match-v2",
            status=match_status_val,
            reason_code=match_reason or ("matched_1" if out_candidates else "empty"),
            candidates=out_candidates or [],
            candidate_count=len(out_candidates or []),
        )

        invite_block = None
        if out_candidates and invite_text:
            invite_block = InviteBlock(
                status="ready",
                delivery_kind="manual_copy",
                text=invite_text,
            )
        else:
            invite_block = InviteBlock(
                status="not_available",
                delivery_kind=None,
                text=None,
            )

        cand_facts_digest = calculate_candidate_facts_digest(out_candidates) if out_candidates else ""
        const_digest = (
            canonical_sha256(final_profile.match_constraints)
            if final_profile and getattr(final_profile, "match_constraints", None)
            else "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )
        grounding_block = GroundingBlock(
            candidate_facts_digest=cand_facts_digest,
            constraint_digest=const_digest,
        )

        d_kind: DialogueKind | None = None
        if determined_reply_mode == ReplyMode.dialogue:
            if combined_guard_blocked:
                d_kind = DialogueKind.capability_boundary
            elif looks_like_clarifying_reply(guarded.reply):
                d_kind = DialogueKind.clarification_question
            else:
                d_kind = DialogueKind.general_help

        typed_final = GroundedFinalV1(
            protocol_version="ga-grounding-v1",
            run_id=req.run_id,
            reply_mode=determined_reply_mode,
            dialogue_kind=d_kind,
            dialogue_text=visible_reply if determined_reply_mode == ReplyMode.dialogue else None,
            candidate_count=len(out_candidates or []),
            candidates=out_candidates or [],
            profile=profile_summary,
            match=match_summary,
            invite=invite_block,
            grounding=grounding_block,
            reply=visible_reply,
        )
        typed_dict = typed_final.model_dump(mode="json")
        typed_dict["rollout_version"] = run_rollout_version
        typed_dict["protocol_mode"] = run_protocol_mode
        # Ensure all top-level keys required by Micro FinalResultValidator are exposed directly at the top level
        final_payload.update(typed_dict)
        final_payload["grounded_final"] = typed_dict

    # Admin ops-brain: never surface member match/search artifacts to the client.
    if admin_debug:
        final_payload["match_status"] = None
        final_payload["match_reason"] = None
        final_payload["search_log"] = None
        final_payload["candidates"] = []
        final_payload["delivery_kind"] = None
        final_payload["invite_text"] = None
        final_payload["topic"] = None
        final_payload["mentioned_user_ids"] = []
        final_payload["at_users"] = []
        final_payload["invite_ok"] = None
        final_payload["admin_ops_mode"] = True

    ref_payload = None if cafe_host else _referral_payload_from_metadata(req.metadata)
    if ref_payload is not None:
        final_payload["referral"] = ref_payload

    from apps.group_agent_api.agent_factory.debug_trace import write_turn_trace
    from apps.group_agent_api.agent_factory.profile_quality import episode_key_from_metadata

    meta = runtime_metadata
    trace_path = write_turn_trace(
        base_dir=state.base_dir,
        run_id=req.run_id,
        thread_id=tid,
        user_id=user_id,
        group_id=group_id,
        conversation_id=req.conversation_id,
        episode_id=episode_key_from_metadata(meta),
        user_message=req.message,
        messages=messages,
        msg_count_before=msg_count_before,
        reply=visible_reply,
        profile_status=profile_status,
        match_status=final_payload["match_status"],
        match_reason=match_reason,
        extra={
            "revisit_hint": meta.get("revisit_hint"),
            "prior_candidate_ids_count": len(meta.get("prior_candidate_ids") or []),
            "delivery_kind": delivery_kind,
            "invite_ok": invite_ok,
            **(
                {
                    "execution_binding": runtime_metadata["execution_binding"],
                    "execution_mode": req.execution_mode,
                    "executor": "cafe_agent" if cafe_host else "member_agent",
                }
                if req.request_schema_version == 3
                else {}
            ),
        },
    )
    if trace_path:
        final_payload["debug_trace_path"] = trace_path

    if intent_execution is not None:
        await _intent_progress(
            "composing",
            intent=intent_execution.decision.primary_intent,
            routing_hash=intent_execution.snapshot.canonical_content_hash,
            display_code="composing_reply",
        )
    _gate("final_callback")
    from apps.group_agent_api.agent_factory.brain_repair import (
        emit_final_with_mouth_repair,
    )

    # BSD-01 P1/P2: bb.brain.repair — first final + at most one same-seq re-emit.
    try:
        if cafe_host:
            # Never route host dialogue through candidate-only mouth rewrite.
            # A rejected final fails, not an unrelated no-match draft.
            if streaming_enabled:
                buffer = StreamChunkBuffer(emit_callback=emit_callback, min_interval_s=0.1)
                await buffer.feed_token(visible_reply)
                await buffer.flush()
            final_ok = await emit_callback("final", final_payload)
            if final_ok is False:
                raise RuntimeError("final_callback_failed")
        else:
            await emit_final_with_mouth_repair(
                emit_callback=emit_callback,
                final_payload=final_payload,
                model=state.quality_model or state.polish_model,
                run_id=req.run_id,
                logger=_logger,
            )
    finally:
        end_decision_trace(_trace_token)
