"""Async callback endpoint for Wu Tanchang conversations."""

from __future__ import annotations

import logging
import os

from apps.wu_tanchang_api.app.callbacks import (
    CallbackUrlError,
    build_callback_thread,
    validate_callback_url,
)
from apps.wu_tanchang_api.app.endpoints.chat import resolve_dynamic_agent
from apps.wu_tanchang_api.app.models import (
    CallWuTanchangAsyncRequest,
    CallWuTanchangAsyncResponse,
)
from apps.wu_tanchang_api.app.state import AppState
from apps.wu_tanchang_api.app.utils import thread_id

_logger = logging.getLogger("uvicorn.error")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        _logger.warning("Invalid %s=%r; using default %d", name, raw, default)
        return default
    if value < 1:
        _logger.warning("Invalid %s=%r; using default %d", name, raw, default)
        return default
    return value


async def call_async(
    req: CallWuTanchangAsyncRequest, state: AppState
) -> CallWuTanchangAsyncResponse:
    """Start a Wu Tanchang stream and POST chunks to the provided callback URL."""
    agent_name, agent = await resolve_dynamic_agent(
        state, req.user_id, req.metadata or {}, req.agent_name
    )
    tid = thread_id(
        agent_name=agent_name, user_id=req.user_id, conversation_id=req.conversation_id
    )

    try:
        callback_url = validate_callback_url(req.callback or "")
    except CallbackUrlError as exc:
        return CallWuTanchangAsyncResponse(
            success=False,
            session_id=tid,
            message=f"Invalid callback URL: {exc}",
        )

    max_active = _env_int("WU_CALLBACK_MAX_ACTIVE_STREAMS", 16)

    def _cleanup(completed_thread_id: str) -> None:
        with state.active_callback_threads_lock:
            state.active_callback_threads.pop(completed_thread_id, None)
        state.finish_agent_run(completed_thread_id, "call_async")

    try:
        if not state.try_start_agent_run(tid, "call_async"):
            return CallWuTanchangAsyncResponse(
                success=False,
                session_id=tid,
                message="Agent run already in progress for this conversation",
            )

        with state.active_callback_threads_lock:
            active_thread = state.active_callback_threads.get(tid)
            if active_thread is not None and active_thread.is_alive():
                state.finish_agent_run(tid, "call_async")
                return CallWuTanchangAsyncResponse(
                    success=False,
                    session_id=tid,
                    message="Stream already in progress for this conversation",
                )
            stale_thread_ids = [
                thread_id_key
                for thread_id_key, thread in state.active_callback_threads.items()
                if not thread.is_alive()
            ]
            for thread_id_key in stale_thread_ids:
                state.active_callback_threads.pop(thread_id_key, None)
            if len(state.active_callback_threads) >= max_active:
                state.finish_agent_run(tid, "call_async")
                return CallWuTanchangAsyncResponse(
                    success=False,
                    session_id=tid,
                    message="Too many active Wu Tanchang streams",
                )

            thread = build_callback_thread(
                agent=agent,
                user_message=req.message,
                thread_id=tid,
                user_id=req.user_id,
                conversation_id=req.conversation_id,
                agent_name=agent_name,
                metadata=req.metadata or {},
                callback_url=callback_url,
                on_complete=_cleanup,
            )
            state.active_callback_threads[tid] = thread
            thread.start()

        _logger.info(
            "POST /call_async - background stream started user_id=%s conversation_id=%s agent=%s thread_id=%s",
            req.user_id,
            req.conversation_id,
            agent_name,
            tid,
        )
        return CallWuTanchangAsyncResponse(
            success=True,
            session_id=tid,
            message="Stream started successfully",
        )
    except Exception as exc:  # noqa: BLE001
        with state.active_callback_threads_lock:
            state.active_callback_threads.pop(tid, None)
        state.finish_agent_run(tid, "call_async")
        _logger.exception("POST /call_async failed thread_id=%s", tid)
        return CallWuTanchangAsyncResponse(
            success=False,
            session_id=tid,
            message=f"Failed to start stream: {type(exc).__name__}: {exc}",
        )
