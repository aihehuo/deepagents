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
from apps.group_agent_api.agent_factory.invite_copy import should_emit_invite_artifact
from apps.group_agent_api.agent_factory.content_quality import (
    finalize_and_guard_user_visible_reply,
)
from apps.group_agent_api.agent_factory.profile_quality import decide_match_gate
from apps.group_agent_api.agent_factory.match_stub import build_query_from_profile
from apps.group_agent_api.agent_factory.profile_store import (
    alert_persist_failure,
    assert_profile_persisted,
    load_profile,
)
from apps.group_agent_api.agent_factory.revisit import (
    excluded_ids_for_match,
    parse_revisit_from_metadata,
    should_skip_auto_match,
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


def _episode_id_from_metadata(metadata: dict[str, Any] | None) -> str | None:
    meta = metadata or {}
    for key in ("episode_id", "episodeId"):
        val = str(meta.get(key) or "").strip()
        if val:
            return val
    return None


def _profile_usable_this_turn(
    *,
    base_dir,
    user_id: str,
    group_id: str,
    messages: list[Any],
    msg_count_before: int,
    metadata: dict[str, Any] | None,
) -> tuple[bool, str | None]:
    """Whether the persisted profile may be used for match/finalize this turn.

    Prior-episode profiles must not be treated as「本轮已更新」after 开新一轮.
    """
    from apps.group_agent_api.app.async_manager import _profile_save_succeeded_this_turn
    from apps.group_agent_api.agent_factory.profile_quality import (
        bind_profile_to_episode,
        profile_bound_to_episode,
    )

    assertion = assert_profile_persisted(base_dir, user_id, group_id)
    if not assertion.ok:
        return False, assertion.path if assertion.path else None

    saved = _profile_save_succeeded_this_turn(messages, msg_count_before)
    if saved:
        bind_profile_to_episode(
            base_dir, user_id, group_id, metadata=metadata
        )
        return True, assertion.path
    if profile_bound_to_episode(
        base_dir, user_id, group_id, metadata=metadata
    ):
        return True, assertion.path
    return False, assertion.path


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


def _ai_text(msg: AIMessage) -> str:
    content = str(msg.content).strip() if msg.content else ""
    if not content or content.startswith("Updated todo list"):
        return ""
    return content


def _peel_stacked_prior_replies(content: str, prior_texts: list[str]) -> str:
    """Remove earlier user-visible AI bubbles re-stacked at the front of ``content``.

    Observed failure (REQ-030): each turn's reply grew as
    ``prior_reply + "\\n\\n" + new_reply`` when either (a) ``_extract_reply``
    joined all AIMessages after a bad ``msg_count_before``, or (b) the model
    echoed the previous assistant bubble. Always prefer the newest segment.
    """
    text = (content or "").strip()
    if not text or not prior_texts:
        return text

    priors = sorted(
        {p.strip() for p in prior_texts if p and str(p).strip()},
        key=len,
        reverse=True,
    )
    changed = True
    while changed and text:
        changed = False
        for prior in priors:
            if not prior or prior == text:
                continue
            for sep in ("\n\n", "\n"):
                prefix = f"{prior}{sep}"
                if text.startswith(prefix):
                    text = text[len(prefix) :].strip()
                    changed = True
                    break
            if changed:
                break
    return text


def _extract_reply(messages: list, msg_count_before: int) -> str:
    """User-visible reply = last AIMessage in this turn, never a join of history.

    Joining AIMessages with ``\\n\\n`` caused cumulative stacking across turns
    (frontend export: Yes. → Yes.\\n\\nI'm ready → Yes.\\n\\nI'm ready\\n\\nGot it…).
    Intermediate pre-tool text is discarded; FORCE_SAVE merge is separate.
    """
    start = max(0, int(msg_count_before or 0))
    history = messages[:start]
    turn = messages[start:]

    turn_ai = [msg for msg in turn if isinstance(msg, AIMessage) and _ai_text(msg)]
    if not turn_ai:
        return ""

    final_content = _ai_text(turn_ai[-1])
    prior_texts = [_ai_text(m) for m in history if isinstance(m, AIMessage)]
    prior_texts.extend(_ai_text(m) for m in turn_ai[:-1])
    return _peel_stacked_prior_replies(final_content, [p for p in prior_texts if p])


def _normalize_reply_for_dedupe(text: str) -> str:
    return "".join((text or "").lower().split())


def _replies_substantially_same(a: str, b: str) -> bool:
    """True when force-save retry or multi-turn AIMessage would look like a duplicated bubble to the user."""
    na = _normalize_reply_for_dedupe(a)
    nb = _normalize_reply_for_dedupe(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    if na in nb or nb in na:
        return True
    prefix_len = min(15, len(na), len(nb))
    return prefix_len >= 5 and na[:prefix_len] == nb[:prefix_len]


def _merge_force_save_reply(reply: str, retry_reply: str) -> str:
    """Merge FORCE_SAVE user text without stacking near-duplicate confirmations."""
    a = (reply or "").strip()
    b = (retry_reply or "").strip()
    if not b:
        return a
    if not a:
        return b
    if _replies_substantially_same(a, b):
        return a if len(a) >= len(b) else b
    na = _normalize_reply_for_dedupe(a)
    nb = _normalize_reply_for_dedupe(b)
    if na in nb:
        return b
    if nb in na:
        return a
    return f"{a}\n\n{b}"


def _should_force_profile_save(
    *,
    profile_ok: bool,
    profile_status: str,
    persist_alert: str | None,
    messages: list[Any],
    msg_count_before: int,
) -> bool:
    """Skip FORCE_SAVE when a prior-episode profile is merely stale and the agent
    intentionally did not overwrite yet (e.g. clarifying need/offer).
    """
    if profile_ok or profile_status == "superseded":
        return False
    if persist_alert == "profile_stale_for_episode":
        from apps.group_agent_api.app.async_manager import (
            _profile_save_attempted_this_turn,
        )

        if not _profile_save_attempted_this_turn(messages, msg_count_before):
            return False
    return True


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
    metadata: dict[str, Any] | None = None,
    message: str | None = None,
) -> tuple[str, list[dict[str, Any]], str | None, list[str]]:
    """Returns (match_status, candidates, match_reason, quality_gaps)."""
    if not run_match_flag:
        return "skipped", [], "run_match_disabled", []
    if not unlocks_network(tier):
        return "skipped", [], f"capability_{tier.value}_no_network", []
    if not profile_ok:
        return "skipped", [], "profile_not_ready", []

    assertion = assert_profile_persisted(state.base_dir, user_id, group_id)
    if not assertion.ok or assertion.profile is None:
        return "skipped", [], "profile_not_ready", []

    decision = decide_match_gate(
        profile=assertion.profile,
        model=state.quality_model or state.polish_model,
        base_dir=state.base_dir,
        message=message,
        metadata=metadata,
    )
    gaps = list(decision.quality.gaps or [])
    if not decision.allow_match:
        return "skipped", [], decision.match_reason or "profile_too_thin", gaps

    query = build_query_from_profile(assertion.profile)
    result = run_match(
        query=query,
        group_id=group_id,
        excluded_ids=excluded_ids_for_match(user_id, metadata),
        group_token=group_token,
        user_bearer=user_token,
    )
    aligned = align_match_to_trusted_group(result, trusted_group_id=group_id)
    reason = decision.match_reason or aligned.reason
    return aligned.status, aligned.candidates, reason, gaps


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
        user_id=user_id,
        group_id=group_id,
        conversation_id=req.conversation_id,
        episode_id=_episode_id_from_metadata(req.metadata),
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
    assert_attempts = 1
    persist_alert: str | None = None
    profile_path: str | None = None
    profile_ok = False
    profile_status_val = "failed"
    persistence_failure_reason: str | None = None

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
                from apps.group_agent_api.app.async_manager import (
                    _attempt_deterministic_profile_save,
                    _known_profile_system_message,
                    _profile_was_superseded,
                    determine_persistence_failure_reason,
                )

                turn_messages: list[Any] = []
                known = _known_profile_system_message(
                    base_dir=state.base_dir,
                    user_id=user_id,
                    group_id=group_id,
                )
                if known is not None:
                    turn_messages.append(known)
                turn_messages.append(HumanMessage(content=req.message))
                result = await agent.ainvoke(
                    {"messages": turn_messages},
                    config,
                )
                messages = result.get("messages", [])
                reply = _extract_reply(messages, msg_count_before)

                assertion = assert_profile_persisted(
                    state.base_dir, user_id, group_id
                )

                if _profile_was_superseded(messages, msg_count_before):
                    profile_status_val = "superseded"
                    profile_ok = False
                else:
                    usable, path = _profile_usable_this_turn(
                        base_dir=state.base_dir,
                        user_id=user_id,
                        group_id=group_id,
                        messages=messages,
                        msg_count_before=msg_count_before,
                        metadata=req.metadata or {},
                    )
                    if usable:
                        profile_ok = True
                        profile_path = path
                        profile_status_val = "persisted"
                        persist_alert = None
                        assert_attempts = 1
                    else:
                        profile_ok = False
                        if assertion.ok:
                            persist_alert = "profile_stale_for_episode"
                if _should_force_profile_save(
                    profile_ok=profile_ok,
                    profile_status=profile_status_val,
                    persist_alert=persist_alert,
                    messages=messages,
                    msg_count_before=msg_count_before,
                ):
                    for attempt in range(1, MAX_PERSIST_ATTEMPTS + 1):
                        assert_attempts = attempt
                        if _profile_was_superseded(messages, msg_count_before):
                            profile_status_val = "superseded"
                            profile_ok = False
                            break

                        usable, path = _profile_usable_this_turn(
                            base_dir=state.base_dir,
                            user_id=user_id,
                            group_id=group_id,
                            messages=messages,
                            msg_count_before=msg_count_before,
                            metadata=req.metadata or {},
                        )
                        if usable:
                            profile_ok = True
                            profile_path = path
                            profile_status_val = "persisted"
                            persist_alert = None
                            break

                        if attempt >= MAX_PERSIST_ATTEMPTS:
                            break

                        reason = persist_alert or (
                            assertion.reason
                            if not assertion.ok
                            else "profile_stale_for_episode"
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

                    if not profile_ok and profile_status_val != "superseded":
                        fallback_attempted = await _attempt_deterministic_profile_save(
                            message=req.message,
                            config=config,
                            messages=messages,
                        )
                        if fallback_attempted:
                            if _profile_was_superseded(messages, msg_count_before):
                                profile_status_val = "superseded"
                                profile_ok = False
                            else:
                                usable, path = _profile_usable_this_turn(
                                    base_dir=state.base_dir,
                                    user_id=user_id,
                                    group_id=group_id,
                                    messages=messages,
                                    msg_count_before=msg_count_before,
                                    metadata=req.metadata or {},
                                )
                                if usable:
                                    profile_ok = True
                                    profile_path = path
                                    profile_status_val = "persisted"
                                    persist_alert = None
                        if not profile_ok:
                            if _profile_was_superseded(messages, msg_count_before):
                                profile_status_val = "superseded"
                                profile_ok = False
                            else:
                                profile_status_val = "failed"
                                persistence_failure_reason = determine_persistence_failure_reason(
                                    messages,
                                    msg_count_before,
                                    attempt=assert_attempts,
                                    last_assertion_reason=persist_alert,
                                )
                    elif profile_ok:
                        profile_status_val = "persisted"
                elif not profile_ok and profile_status_val != "superseded":
                    profile_status_val = "stale_episode"

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

    _, revisit_hint = parse_revisit_from_metadata(req.metadata or {})
    effective_run_match = req.run_match and not should_skip_auto_match(
        revisit_hint=revisit_hint,
        message=req.message,
    )
    match_status, candidates, match_reason, quality_gaps = await asyncio.to_thread(
        _run_match_pipeline,
        state=state,
        user_id=user_id,
        group_id=group_id,
        tier=tier,
        profile_ok=profile_ok,
        run_match_flag=effective_run_match,
        group_token=session.group_token,
        user_token=user_token,
        metadata=req.metadata or {},
        message=req.message,
    )
    if (
        req.run_match
        and not effective_run_match
        and match_status == "skipped"
        and match_reason == "run_match_disabled"
    ):
        match_reason = "revisit_awaiting_user_branch"

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

    # Invite only when match found people — empty match must not auto-emit
    # undirected topic/invite cards while dialogue is still clarifying.
    if (
        req.run_invite
        and profile_ok
        and unlocks_network(tier)
        and effective_run_match
        and should_emit_invite_artifact(
            match_status=match_status,
            match_reason=match_reason,
            candidate_count=len(guarded.candidates),
        )
    ):
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
        revisit_hint=revisit_hint,
        match_reason=match_reason,
        quality_gaps=quality_gaps,
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
