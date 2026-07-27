"""Chat endpoint with FR-06 persist assert + REQ-005/007 capability/match/guard."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from fastapi import HTTPException, Request
from langchain_core.messages import AIMessage, HumanMessage

from apps.group_agent_api.agent_factory.agent import FORCE_SAVE_PROMPT, UC34Observer
from apps.group_agent_api.agent_factory.capability import unlocks_network
from apps.group_agent_api.agent_factory.guard import enforce_capability_guard
from apps.group_agent_api.agent_factory.integrations.group_bind import (
    align_match_to_trusted_group,
)
from apps.group_agent_api.agent_factory.integrations.match_backend import run_match
from apps.group_agent_api.agent_factory.invite_llm import generate_invite_with_optional_llm
from apps.group_agent_api.agent_factory.content_quality import (
    finalize_and_guard_user_visible_reply,
)
from apps.group_agent_api.agent_factory.match_stub import build_query_from_profile
from apps.group_agent_api.agent_factory.profile_store import (
    alert_persist_failure,
    assert_profile_persisted,
    load_profile,
)
from apps.group_agent_api.app.models import ChatRequest, ChatResponse
from apps.group_agent_api.app.session import resolve_trusted_session
from apps.group_agent_api.app.state import AppState
from apps.group_agent_api.app.utils import (
    aget_agent_state,
    get_agent_checkpointer,
    thread_id,
)

_logger = logging.getLogger("uvicorn.error")

MAX_PERSIST_ATTEMPTS = 2  # initial turn + 1 forced retry

_RESERVED_META_KEYS = frozenset(
    {
        "user_id",
        "group_id",
        "base_dir",
        "membership",
        "capability",
        "unionid",
        "group_token",
        "user_token",
        "run_id",
        "conversation_id",
    }
)


def _extract_reply(messages: list, msg_count_before: int) -> str:
    parts: list[str] = []
    for msg in messages[msg_count_before:]:
        if not isinstance(msg, AIMessage):
            continue
        content = str(msg.content).strip() if msg.content else ""
        if not content or content.startswith("Updated todo list"):
            continue
        parts.append(content)
    return "\n\n".join(parts)


def _invoke_config(
    *,
    tid: str,
    user_id: str,
    group_id: str,
    base_dir: str,
    membership: str,
    metadata: dict,
    run_id: str | None = None,
    conversation_id: str | None = None,
) -> dict:
    turn_id = f"{tid}::{uuid.uuid4().hex}"
    safe_meta = {
        k: v for k, v in (metadata or {}).items() if k not in _RESERVED_META_KEYS
    }
    return {
        "configurable": {
            "thread_id": tid,
            "deepagents_turn_id": turn_id,
        },
        "metadata": {
            **safe_meta,
            "user_id": user_id,
            "group_id": group_id,
            "base_dir": base_dir,
            "membership": membership,
            **({"run_id": run_id} if run_id else {}),
            **({"conversation_id": conversation_id} if conversation_id else {}),
        },
    }


def _run_match_pipeline(
    *,
    state: AppState,
    user_id: str,
    group_id: str,
    tier,
    profile_ok: bool,
    run_match_flag: bool,
    group_token: str | None,
    user_token: str | None,
) -> tuple[str, list[dict[str, Any]], str | None]:
    """Returns (match_status, candidates, match_reason). caller_group_id = trusted group_id."""
    if not run_match_flag:
        return "skipped", [], "run_match_disabled"
    if not unlocks_network(tier):
        return "skipped", [], f"capability_{tier.value}_no_network"
    if not profile_ok:
        return "skipped", [], "profile_not_ready"

    assertion = assert_profile_persisted(state.base_dir, user_id, group_id)
    if not assertion.ok or assertion.profile is None:
        return "skipped", [], "profile_not_ready"

    query = build_query_from_profile(assertion.profile)
    result = run_match(
        query=query,
        group_id=group_id,
        excluded_ids=[user_id],
        group_token=group_token,
        user_bearer=user_token,
    )
    aligned = align_match_to_trusted_group(result, trusted_group_id=group_id)
    return aligned.status, aligned.candidates, aligned.reason


def _empty_request() -> Request:
    return Request(
        {"type": "http", "method": "POST", "path": "/chat", "headers": []}
    )


async def chat(
    req: ChatRequest, state: AppState, request: Request | None = None
) -> ChatResponse:
    start = time.time()
    session = await resolve_trusted_session(
        request or _empty_request(),
        body_user_id=req.user_id,
        body_group_id=req.group_id,
        body_membership=req.membership,
        body_unionid=req.unionid,
        body_group_token=req.group_token,
        body_user_token=req.user_token,
    )
    user_id = session.principal.user_id
    group_id = session.group_id
    tier = session.membership.tier
    user_token = session.principal.user_token

    tid = thread_id(
        user_id=user_id, group_id=group_id, conversation_id=req.conversation_id
    )
    UC34Observer.info(
        f"action=chat_start user_id={user_id} group_id={group_id} "
        f"conversation_id={req.conversation_id} thread_id={tid} "
        f"membership={tier.value} principal={session.principal.source} "
        f"message_len={len(req.message)}"
    )

    if not state.try_start_agent_run(tid, "chat"):
        raise HTTPException(
            status_code=409,
            detail={
                "error": "run_in_progress",
                "message": "Agent run already in progress",
                "thread_id": tid,
            },
        )

    agent = state.agent
    lock = state.thread_locks.setdefault(tid, asyncio.Lock())
    reply = ""
    assert_attempts = 0
    persist_alert: str | None = None
    profile_path: str | None = None
    profile_ok = False

    try:
        async with lock:
            try:
                agent_state = await aget_agent_state(
                    agent, {"configurable": {"thread_id": tid}}
                )
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
                before_profile = assert_profile_persisted(
                    state.base_dir, user_id, group_id
                )
                before_updated_at = (
                    before_profile.profile.updated_at if before_profile.ok else None
                )

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
                    return assertion.profile is not None and (
                        assertion.profile.updated_at != before_updated_at
                    )

                for attempt in range(1, MAX_PERSIST_ATTEMPTS + 1):
                    assert_attempts = attempt
                    assertion = assert_profile_persisted(
                        state.base_dir, user_id, group_id
                    )
                    if _turn_persist_ok(assertion):
                        profile_ok = True
                        profile_path = assertion.path
                        persist_alert = None
                        break

                    reason = (
                        assertion.reason
                        if not assertion.ok
                        else "stale_profile_not_updated"
                    )
                    alert_persist_failure(
                        user_id=user_id,
                        group_id=group_id,
                        attempt=attempt,
                        reason=reason,
                    )
                    persist_alert = reason
                    UC34Observer.warn(
                        f"action=profile_assert_failed user_id={user_id} "
                        f"group_id={group_id} attempt={attempt} "
                        f"reason={reason}"
                    )

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
                        reply = f"{reply}\n\n{retry_reply}".strip()

                persistence_failure_reason: str | None = None
                if not profile_ok:
                    from apps.group_agent_api.app.async_manager import (
                        _attempt_deterministic_profile_save,
                        _profile_was_superseded,
                        determine_persistence_failure_reason,
                    )
                    fallback_attempted = await _attempt_deterministic_profile_save(
                        message=req.message,
                        config=config,
                        messages=messages,
                    )
                    if fallback_attempted:
                        if _profile_was_superseded(messages, msg_count_before):
                            profile_status_val = "superseded"
                        else:
                            assertion = assert_profile_persisted(
                                state.base_dir, user_id, group_id
                            )
                            if _turn_persist_ok(assertion):
                                profile_ok = True
                                profile_path = assertion.path
                                profile_status_val = "persisted"
                                persist_alert = None
                    if not profile_ok:
                        if _profile_was_superseded(messages, msg_count_before):
                            profile_status_val = "superseded"
                        else:
                            profile_status_val = "failed"
                            persistence_failure_reason = determine_persistence_failure_reason(
                                messages,
                                msg_count_before,
                                attempt=assert_attempts,
                                last_assertion_reason=persist_alert,
                            )
                else:
                    profile_status_val = "persisted"

            except Exception as exc:  # noqa: BLE001
                import traceback

                _logger.error(
                    "POST /chat failed thread_id=%s\n%s", tid, traceback.format_exc()
                )
                UC34Observer.error(
                    f"action=chat_error user_id={user_id} group_id={group_id} "
                    f"thread_id={tid} error_type={type(exc).__name__} "
                    f"error_message={exc}"
                )
                raise HTTPException(
                    status_code=502,
                    detail={
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "thread_id": tid,
                    },
                ) from exc
            finally:
                checkpointer = get_agent_checkpointer(agent)
                if checkpointer is not None and hasattr(checkpointer, "flush"):
                    checkpointer.flush()
    finally:
        state.finish_agent_run(tid, "chat")

    match_status, candidates, match_reason = await asyncio.to_thread(
        _run_match_pipeline,
        state=state,
        user_id=user_id,
        group_id=group_id,
        tier=tier,
        profile_ok=profile_ok,
        run_match_flag=req.run_match,
        group_token=session.group_token,
        user_token=user_token,
    )

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
    elif match_status == "matched" and not guarded.candidates:
        match_status = "empty"
        match_reason = "no_auditable_public_match_basis"

    delivery_kind = None
    invite_text = None
    topic = None
    mentioned_user_ids: list[str] = []
    invite_ok = None
    invite_violations: list[str] = []

    if req.run_invite and profile_ok and unlocks_network(tier):
        profile = load_profile(state.base_dir, user_id, group_id)
        if profile is not None:
            invite_status = match_status
            invite_candidates = guarded.candidates
            willing = req.willing_to_at

            def _gen_invite():
                return generate_invite_with_optional_llm(
                    profile=profile,
                    candidates=invite_candidates,
                    match_status=invite_status,
                    willing_to_at=willing,
                    user_id=user_id,
                    group_id=group_id,
                    model=state.polish_model,
                )

            invite_res = await asyncio.to_thread(_gen_invite)
            delivery_kind = invite_res.kind
            invite_text = invite_res.text
            topic = invite_res.topic
            mentioned_user_ids = invite_res.mentioned_user_ids
            invite_ok = invite_res.ok
            invite_violations = invite_res.violations

    latency_ms = int((time.time() - start) * 1000)
    UC34Observer.info(
        f"action=chat_success user_id={user_id} group_id={group_id} "
        f"thread_id={tid} capability={tier.value} "
        f"profile_persisted={profile_ok} match_status={match_status} "
        f"delivery_kind={delivery_kind} invite_ok={invite_ok} "
        f"candidates={len(guarded.candidates)} guard_blocked={guarded.blocked} "
        f"assert_attempts={assert_attempts} latency_ms={latency_ms}"
    )

    final_profile = load_profile(state.base_dir, user_id, group_id) if profile_ok else None
    final_guarded = finalize_and_guard_user_visible_reply(
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
    )
    combined_guard_violations = list(
        dict.fromkeys([*guarded.violations, *final_guarded.violations])
    )
    combined_guard_blocked = guarded.blocked or final_guarded.blocked

    return ChatResponse(
        user_id=user_id,
        group_id=group_id,
        conversation_id=req.conversation_id,
        thread_id=tid,
        reply=final_guarded.reply,
        profile_persisted=profile_ok,
        profile_path=profile_path,
        profile_status=profile_status_val,
        persistence_failure_reason=persistence_failure_reason,
        assert_attempts=assert_attempts,
        persist_alert=None if profile_ok else persist_alert,
        capability=tier.value,  # type: ignore[arg-type]
        capability_source=session.membership.source,
        match_status=match_status if final_guarded.candidates or match_status in {"empty", "skipped", "weak"} else "empty",  # noqa: E501
        candidates=final_guarded.candidates,
        match_reason=match_reason,
        guard_blocked=combined_guard_blocked,
        guard_violations=combined_guard_violations,
        delivery_kind=delivery_kind,
        invite_text=invite_text,
        topic=topic,
        mentioned_user_ids=mentioned_user_ids,
        invite_ok=invite_ok,
        invite_violations=invite_violations,
        willing_to_at=req.willing_to_at,
    )
