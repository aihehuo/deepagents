"""Deep agent async endpoint for background streaming with callbacks."""

from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path

from apps.business_cofounder_api.app.callbacks import run_async_stream_with_callback
from apps.business_cofounder_api.app.models import CallDeepAgentAsyncRequest, CallDeepAgentAsyncResponse
from apps.business_cofounder_api.app.state import AppState
from apps.business_cofounder_api.app.utils import thread_id

# Use uvicorn's configured logger so output reliably shows up in the terminal.
_logger = logging.getLogger("uvicorn.error")


async def call_deep_agent_async(req: CallDeepAgentAsyncRequest, state: AppState) -> CallDeepAgentAsyncResponse:
    """Start an async agent stream in a background thread and return immediately.
    
    This endpoint accepts a callback URL and starts the agent streaming in a background thread.
    Each update from the stream will be POSTed to the callback URL with the message parameter
    containing the update data.
    
    Returns immediately with a session_id (same as thread_id) and success status.
    The actual streaming happens asynchronously in a separate thread.
    
    Args:
        req: Request with user_id, message, conversation_id, callback URL, and metadata
        state: Application state
        
    Returns:
        CallDeepAgentAsyncResponse with session_id and success status
    """
    # Log immediately when function is called (validation passed)
    _logger.info("=== call_deep_agent_async CALLED ===")
    _logger.info(
        "POST /deep_agent/call_async - received request (user_id=%s, conversation_id=%s, message_len=%d, callback=%s, metadata=%s)",
        req.user_id,
        req.conversation_id,
        len(req.message),
        req.callback,
        req.metadata
    )
    tid = thread_id(user_id=req.user_id, conversation_id=req.conversation_id)
    _logger.debug("POST /deep_agent/call_async - thread_id=%s", tid)

    # Get or create lock for this thread_id (for consistency with other endpoints)
    # We don't hold it during execution since we're running in background
    lock = state.thread_locks.get(tid)
    if lock is None:
        lock = asyncio.Lock()
        state.thread_locks[tid] = lock

    try:
        # Get paths for file resolution in callback messages
        docs_dir = state.docs_dir if state else None
        backend_root_dir = state.backend_root if state else None
        
        # Get expertise_type from metadata if available
        expertise_type_from_req = (req.metadata or {}).get("expertise_type", "not specified")
        
        _logger.info(
            "POST /deep_agent/call_async - starting background thread (thread_id=%s, callback_url=%s, docs_dir=%s)",
            tid,
            req.callback,
            docs_dir,
        )
        _logger.info(
            "[DeepAgent] Expert agent info: available=%s, use_dual_agent=%s, expertise_type_from_request=%s",
            state.expert_agent is not None if state else False,
            state.use_dual_agent if state else False,
            expertise_type_from_req,
        )
        # Start background thread to run the async stream
        thread = threading.Thread(
            target=run_async_stream_with_callback,
            args=(
                state.agent,
                req.message,
                tid,
                req.user_id,
                req.metadata or {},
                req.callback,
                state.fallback_agent,  # Pass fallback_agent as parameter
                docs_dir,
                backend_root_dir,
                state.expert_agent,  # Pass expert_agent for dual-agent mode
                state.use_dual_agent,  # Pass use_dual_agent flag
                state.expertise_dir,  # Pass expertise_dir
            ),
            daemon=True,
            name=f"bc-async-{tid}",
        )
        thread.start()
        _logger.info("POST /deep_agent/call_async - background thread started (thread_id=%s)", tid)
        
        return CallDeepAgentAsyncResponse(
            success=True,
            session_id=tid,
            message="Stream started successfully",
        )
    
    except Exception as e:  # noqa: BLE001
        _logger.exception(
            "POST /callDeepAgentAsync failed user_id=%s conversation_id=%s thread_id=%s error_type=%s error_message=%s",
            req.user_id,
            req.conversation_id,
            tid,
            type(e).__name__,
            str(e),
        )
        return CallDeepAgentAsyncResponse(
            success=False,
            session_id=tid,
            message=f"Failed to start stream: {type(e).__name__}: {str(e)}",
        )
