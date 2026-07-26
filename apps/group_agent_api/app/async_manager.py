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
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.messages import HumanMessage

from apps.group_agent_api.agent_factory.agent import FORCE_SAVE_PROMPT, UC34Observer
from apps.group_agent_api.agent_factory.capability import unlocks_network
from apps.group_agent_api.agent_factory.guard import enforce_capability_guard
from apps.group_agent_api.agent_factory.integrations.callback_client import (
    send_callback_event,
    validate_and_normalize_callback_url,
)
from apps.group_agent_api.agent_factory.integrations.config import async_run_timeout_s
from apps.group_agent_api.agent_factory.integrations.group_bind import align_match_to_trusted_group
from apps.group_agent_api.agent_factory.integrations.match_backend import run_match
from apps.group_agent_api.agent_factory.invite_llm import generate_invite_with_optional_llm
from apps.group_agent_api.agent_factory.match_stub import build_query_from_profile
from apps.group_agent_api.agent_factory.profile_store import assert_profile_persisted, load_profile
from apps.group_agent_api.app.endpoints.chat import MAX_PERSIST_ATTEMPTS, _extract_reply, _invoke_config
from apps.group_agent_api.app.models import AsyncCallRequest, AsyncCallResponse, CallbackEnvelope
from apps.group_agent_api.app.session import TrustedSession
from apps.group_agent_api.app.state import AppState
from apps.group_agent_api.app.utils import aget_agent_state, get_agent_checkpointer, thread_id

_logger = logging.getLogger("uvicorn.error")

_MAX_IDEMPOTENCY_CACHE = 5_000


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


def calculate_request_fingerprint(req: AsyncCallRequest, session: TrustedSession) -> str:
    """Construct complete canonical request fingerprint derived from TrustedSession & canonical callback URL."""
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
                    _logger.warning("Idempotency store cache full (%d) with all reserved slots; rejecting new slot key=%s", _MAX_IDEMPOTENCY_CACHE, req.idempotency_key)
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
        except asyncio.TimeoutError:
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
    """Background task: waits for decision_event gate, runs core agent execution if completed, and delivers callbacks."""
    user_id = session.principal.user_id
    group_id = session.group_id

    # Wait for decision_event gate before initiating core execution
    if slot is not None:
        try:
            await asyncio.wait_for(slot.decision_event.wait(), timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
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
        if event_type in {"final", "error"}:
            terminal_created = True

        env = CallbackEnvelope(
            version="GA-CALLBACK-V1",
            run_id=req.run_id,
            idempotency_key=req.idempotency_key,
            seq=seq,
            event=event_type,  # type: ignore[arg-type]
            occurred_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            user_id=user_id,
            group_id=group_id,
            conversation_id=req.conversation_id,
            payload=payload,
        )

        delivered = await send_callback_event(callback_url=req.callback_url, envelope_dict=env.model_dump())
        if event_type in {"final", "error"} and delivered:
            terminal_delivered = True

        _logger.info(
            "Callback event status run_id=%s seq=%d event=%s delivered=%s",
            req.run_id,
            seq,
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
            _execute_core_agent(
                req=req,
                session=session,
                state=state,
                tid=tid,
                emit_callback=_emit_event,
            ),
            timeout=timeout,
        )

    except asyncio.TimeoutError:
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


async def _execute_core_agent(
    *,
    req: AsyncCallRequest,
    session: TrustedSession,
    state: AppState,
    tid: str,
    emit_callback: Any,
) -> None:
    user_id = session.principal.user_id
    group_id = session.group_id
    tier = session.membership.tier
    user_token = session.principal.user_token

    agent = state.agent
    lock = state.thread_locks.setdefault(tid, asyncio.Lock())
    reply = ""
    profile_ok = False

    async with lock:
        try:
            test_lvl = os.environ.get("GROUP_AGENT_TEST_LEVEL")
            if test_lvl and load_profile(state.base_dir, user_id, group_id) is None:
                try:
                    from apps.group_agent_api.fixtures.loader import load_fixture
                    from apps.group_agent_api.agent_factory.profile_schema import GroupProfile, ProfileField
                    from apps.group_agent_api.agent_factory.profile_store import save_profile
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
            msg_count_before = (
                len(agent_state.values.get("messages", []))
                if agent_state and agent_state.values
                else 0
            )
            config = _invoke_config(
                tid=tid,
                user_id=user_id,
                group_id=group_id,
                base_dir=str(state.base_dir),
                membership=tier.value,
                metadata=req.metadata or {},
            )
            before_profile = assert_profile_persisted(state.base_dir, user_id, group_id)
            before_updated_at = before_profile.profile.updated_at if before_profile.ok else None

            result = await agent.ainvoke(
                {"messages": [HumanMessage(content=req.message)]},
                config,
            )
            messages = result.get("messages", [])
            reply = _extract_reply(messages, msg_count_before)

            def _turn_persist_ok(assertion) -> bool:
                if not assertion.ok:
                    return False
                if before_updated_at is None:
                    return True
                return assertion.profile is not None and (assertion.profile.updated_at != before_updated_at)

            for attempt in range(1, MAX_PERSIST_ATTEMPTS + 1):
                assertion = assert_profile_persisted(state.base_dir, user_id, group_id)
                if _turn_persist_ok(assertion):
                    profile_ok = True
                    break

                if attempt >= MAX_PERSIST_ATTEMPTS:
                    break

                before_retry = len(messages)
                result = await agent.ainvoke(
                    {"messages": [HumanMessage(content=FORCE_SAVE_PROMPT)]},
                    config,
                )
                messages = result.get("messages", [])
                retry_reply = _extract_reply(messages, before_retry)
                if retry_reply:
                    reply = f"{reply}\n\n{retry_reply}".strip()

        finally:
            checkpointer = get_agent_checkpointer(agent)
            if checkpointer is not None and hasattr(checkpointer, "flush"):
                checkpointer.flush()

    # Step: Match pipeline
    match_status = "skipped"
    candidates: list[dict[str, Any]] = []
    match_reason: str | None = None

    if req.run_match and unlocks_network(tier) and profile_ok:
        assertion = assert_profile_persisted(state.base_dir, user_id, group_id)
        if assertion.ok and assertion.profile is not None:
            query = build_query_from_profile(assertion.profile)
            def _match_job():
                return run_match(
                    query=query,
                    group_id=group_id,
                    excluded_ids=[user_id],
                    group_token=session.group_token,
                    user_bearer=user_token,
                )
            match_res = await asyncio.to_thread(_match_job)
            aligned = align_match_to_trusted_group(match_res, trusted_group_id=group_id)
            match_status, candidates, match_reason = aligned.status, aligned.candidates, aligned.reason
    _logger.info("Core match debug run_id=%s tier=%s profile_ok=%s match_status=%s candidates_count=%d", req.run_id, tier.value, profile_ok, match_status, len(candidates))

    guarded = enforce_capability_guard(
        tier=tier,
        reply=reply,
        candidates=candidates,
        caller_group_id=group_id,
        user_id=user_id,
    )
    if guarded.blocked and not unlocks_network(tier):
        match_status = "skipped"
        match_reason = f"capability_{tier.value}_guard_blocked"

    delivery_kind = None
    invite_text = None
    topic = None
    mentioned_user_ids: list[str] = []
    invite_ok = None

    if req.run_invite and profile_ok:
        profile = load_profile(state.base_dir, user_id, group_id)
        if profile is not None:
            invite_status = match_status
            invite_candidates = guarded.candidates
            willing = req.willing_to_at
            if not unlocks_network(tier):
                invite_status = "empty"
                invite_candidates = []
                willing = False

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
            _logger.info("Invite debug run_id=%s willing=%s kind=%s text_len=%d mentioned_count=%d", req.run_id, willing, delivery_kind, len(invite_text or ""), len(mentioned_user_ids))

    final_payload = {
        "reply": guarded.reply,
        "capability": tier.value,
        "capability_source": session.membership.source,
        "match_status": match_status if guarded.candidates or match_status in {"empty", "skipped", "weak"} else "empty",
        "candidates": guarded.candidates,
        "match_reason": match_reason,
        "guard_blocked": guarded.blocked,
        "guard_violations": guarded.violations,
        "delivery_kind": delivery_kind,
        "invite_text": invite_text,
        "topic": topic,
        "mentioned_user_ids": mentioned_user_ids,
        "at_users": mentioned_user_ids,
        "invite_ok": invite_ok,
        "willing_to_at": req.willing_to_at,
    }

    await emit_callback("final", final_payload)
