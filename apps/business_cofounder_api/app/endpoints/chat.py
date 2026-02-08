"""Chat endpoints for synchronous and streaming conversations."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import traceback
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from apps.business_cofounder_api.expert_sync import should_trigger_expert, trigger_and_update_expert
from apps.business_cofounder_api.app.models import ChatRequest, ChatResponse
from apps.business_cofounder_api.app.state import AppState
from apps.business_cofounder_api.app.utils import (
    env_flag,
    extract_text_chunks_from_ai_message,
    format_tool_call_progress,
    log_chat_io,
    log_debug_state,
    thread_id,
)

# Use uvicorn's configured logger so output reliably shows up in the terminal.
_logger = logging.getLogger("uvicorn.error")


async def chat(req: ChatRequest, state: AppState) -> ChatResponse:
    """Synchronous chat endpoint.
    
    Args:
        req: Chat request with user message
        state: Application state
        
    Returns:
        ChatResponse with assistant reply
    """
    _logger.info(
        "POST /chat - received request (user_id=%s, conversation_id=%s, message_len=%d)",
        req.user_id,
        req.conversation_id,
        len(req.message),
    )
    tid = thread_id(user_id=req.user_id, conversation_id=req.conversation_id)

    lock = state.thread_locks.get(tid)
    if lock is None:
        lock = asyncio.Lock()
        state.thread_locks[tid] = lock

    async with lock:
        # Check if expert sync is needed BEFORE calling facilitator agent (dual-agent mode only)
        # This ensures the facilitator has fresh expert guidance when it processes the request
        if state.use_dual_agent and state.expert_agent is not None:
            try:
                # Get current state to check if expert sync is needed
                checkpointer = state.agent.checkpointer
                config = {"configurable": {"thread_id": tid}}
                checkpoint = await checkpointer.aget(config)
                
                if checkpoint:
                    state_values = checkpoint.get("channel_values", {})
                    # Build state dict for should_trigger_expert check.
                    # Request expertise_type overrides persisted state (so new default applies to existing threads).
                    current_state = {
                        "messages": state_values.get("messages", []),
                        "conversation_round": state_values.get("conversation_round", 0),
                        **state_values,
                        "expertise_type": req.expertise_type,
                    }
                    
                    if should_trigger_expert(current_state):
                        _logger.info("[DualAgent] Expert sync needed BEFORE facilitator call for thread %s", tid)
                        expertise_dir = Path(state.expertise_dir) if state.expertise_dir else None
                        
                        # Run expert sync synchronously (wait for it) so facilitator has fresh guidance
                        try:
                            await asyncio.wait_for(
                                trigger_and_update_expert(
                                    thread_id=tid,
                                    state=current_state,
                                    expert_agent=state.expert_agent,
                                    checkpointer=checkpointer,
                                    expertise_dir=expertise_dir,
                                    facilitator_agent=state.agent,  # Pass facilitator agent for state updates
                                ),
                                timeout=60.0,  # 60 second timeout
                            )
                            _logger.info("[DualAgent] Expert sync completed BEFORE facilitator call")
                        except asyncio.TimeoutError:
                            _logger.warning("[DualAgent] Expert sync timed out (60s) for thread %s", tid)
                        except Exception as e:
                            _logger.error("[DualAgent] Error during expert sync: %s", str(e), exc_info=True)
                        except asyncio.TimeoutError:
                            _logger.error("[DualAgent] Expert sync timed out before facilitator call (continuing anyway)")
                        except Exception as e:
                            _logger.error("[DualAgent] Expert sync error before facilitator call: %s", str(e), exc_info=True)
                            # Continue anyway - don't block the request
            except Exception as e:  # noqa: BLE001
                _logger.error("[DualAgent] Error checking/triggering expert sync before facilitator: %s", str(e))
                # Continue anyway - don't block the request
        
        try:
            result = await state.agent.ainvoke(
                {
                    "messages": [HumanMessage(content=req.message)],
                    "expertise_type": req.expertise_type,
                },
                {
                    "configurable": {"thread_id": tid},
                    "metadata": {
                        "user_id": req.user_id,
                        "expertise_type": req.expertise_type,
                        **(req.metadata or {}),
                    },
                },
            )
        except Exception as e:  # noqa: BLE001
            _logger.warning(
                "[ModelFallback] Primary agent (qwen) failed: %s: %s",
                type(e).__name__,
                str(e),
            )
            _logger.info("[ModelFallback] Falling back to deepseek...")
            try:
                result = await state.fallback_agent.ainvoke(
                    {
                        "messages": [HumanMessage(content=req.message)],
                        "expertise_type": req.expertise_type,
                    },
                    {
                        "configurable": {"thread_id": tid},
                        "metadata": {
                            "user_id": req.user_id,
                            "expertise_type": req.expertise_type,
                            **(req.metadata or {}),
                        },
                    },
                )
            except Exception as fallback_error:  # noqa: BLE001
                _logger.error(
                    "[ModelFallback] Fallback agent (deepseek) also failed: %s: %s",
                    type(fallback_error).__name__,
                    str(fallback_error),
                )
                # Print a full traceback to server logs for local debugging.
                _logger.exception(
                    "POST /chat failed user_id=%s conversation_id=%s thread_id=%s error_type=%s error_message=%s",
                    req.user_id,
                    req.conversation_id,
                    tid,
                    type(fallback_error).__name__,
                    str(fallback_error),
                )

                # Optionally include traceback in HTTP response (useful for local dev; avoid enabling in prod).
                detail: dict[str, Any] = {
                    "error_type": type(fallback_error).__name__,
                    "error_message": str(fallback_error),
                    "thread_id": tid,
                }
                if env_flag("BC_API_RETURN_TRACEBACK", default=False):
                    detail["traceback"] = traceback.format_exc()

                # Internal API: return a helpful message for debugging.
                raise HTTPException(
                    status_code=502,
                    detail=detail,
                ) from fallback_error

    messages = result.get("messages", [])
    ai_messages = [m for m in messages if getattr(m, "type", None) == "ai"]
    reply = str(ai_messages[-1].content) if ai_messages else ""

    log_chat_io(
        user_id=req.user_id,
        conversation_id=req.conversation_id,
        thread_id=tid,
        user_message=req.message,
        reply=reply,
    )
    log_debug_state(result=result, thread_id=tid)

    return ChatResponse(
        user_id=req.user_id,
        conversation_id=req.conversation_id,
        thread_id=tid,
        reply=reply,
    )


async def chat_stream(req: ChatRequest, state: AppState) -> StreamingResponse:
    """Stream the assistant response as Server-Sent Events (SSE).

    This avoids client/proxy timeouts for long generations (e.g. HTML via coder subagent).

    SSE event format:
    - data: {"type":"delta","text":"..."}  (many) - text chunks from assistant
    - data: {"type":"progress","message":"..."}  (many) - progress updates during execution
    - data: {"type":"final","text":"..."}  (once) - final complete response
    - data: {"type":"error","detail":{...}} (once, if error)
    
    Progress updates are sent when:
    - Tool calls are being prepared or executed
    - Nodes in the agent graph are being processed
    - Tool execution completes
    
    Args:
        req: Chat request with user message
        state: Application state
        
    Returns:
        StreamingResponse with SSE events
    """
    _logger.info(
        "POST /chat/stream - received request (user_id=%s, conversation_id=%s, message_len=%d)",
        req.user_id,
        req.conversation_id,
        len(req.message),
    )
    tid = thread_id(user_id=req.user_id, conversation_id=req.conversation_id)

    lock = state.thread_locks.get(tid)
    if lock is None:
        lock = asyncio.Lock()
        state.thread_locks[tid] = lock

    async def _gen():
        final_parts: list[str] = []
        delta_count = 0
        seen_types: dict[str, int] = {}
        last_written_html_path: str | None = None
        last_progress_update: float = 0.0
        # Track tool calls by ID to match with ToolMessages
        tool_call_args_cache: dict[str, dict[str, Any]] = {}
        # Track model call stats - start timing from the beginning of the request
        request_start_time = time.time()
        model_call_start_time: float | None = None
        async with lock:
            # Check if expert sync is needed BEFORE calling facilitator agent (dual-agent mode only)
            # This ensures the facilitator has fresh expert guidance when it processes the request
            if state.use_dual_agent and state.expert_agent is not None:
                try:
                    # Get current state to check if expert sync is needed
                    checkpointer = state.agent.checkpointer
                    config = {"configurable": {"thread_id": tid}}
                    checkpoint = await checkpointer.aget(config)
                    
                    if checkpoint:
                        state_values = checkpoint.get("channel_values", {})
                        # Build state dict for should_trigger_expert check.
                        # Request expertise_type overrides persisted state (so new default applies to existing threads).
                        current_state = {
                            "messages": state_values.get("messages", []),
                            "conversation_round": state_values.get("conversation_round", 0),
                            **state_values,
                            "expertise_type": req.expertise_type,
                        }
                        
                        if should_trigger_expert(current_state):
                            _logger.info("[DualAgent] Expert sync needed BEFORE facilitator stream for thread %s", tid)
                            expertise_dir = Path(state.expertise_dir) if state.expertise_dir else None
                            
                            # Run expert sync synchronously (wait for it) so facilitator has fresh guidance
                            try:
                                await asyncio.wait_for(
                                    trigger_and_update_expert(
                                        thread_id=tid,
                                        state=current_state,
                                        expert_agent=state.expert_agent,
                                        checkpointer=checkpointer,
                                        expertise_dir=expertise_dir,
                                        facilitator_agent=state.agent,  # Pass facilitator agent for state updates
                                    ),
                                    timeout=60.0,  # 60 second timeout
                                )
                                _logger.info("[DualAgent] Expert sync completed BEFORE facilitator stream")
                            except asyncio.TimeoutError:
                                _logger.error("[DualAgent] Expert sync timed out before facilitator stream (continuing anyway)")
                            except Exception as e:
                                _logger.error("[DualAgent] Expert sync error before facilitator stream: %s", str(e), exc_info=True)
                                # Continue anyway - don't block the request
                except Exception as e:  # noqa: BLE001
                    _logger.error("[DualAgent] Error checking/triggering expert sync before facilitator stream: %s", str(e))
                    # Continue anyway - don't block the request
            
            try:
                async for chunk in state.agent.astream(
                    {
                        "messages": [HumanMessage(content=req.message)],
                        "expertise_type": req.expertise_type,
                    },
                    config={
                        "configurable": {"thread_id": tid},
                        "metadata": {
                            "user_id": req.user_id,
                            "expertise_type": req.expertise_type,
                            **(req.metadata or {}),
                        },
                    },
                    stream_mode=["messages", "updates"],
                    subgraphs=True,
                    durability="exit",
                ):
                    # With subgraphs=True and multiple stream modes, chunks are:
                    # (namespace, stream_mode, data)
                    if not isinstance(chunk, tuple) or len(chunk) != 3:
                        continue

                    _namespace, current_stream_mode, data = chunk

                    # Handle UPDATES stream mode - provides progress information
                    if current_stream_mode == "updates":
                        # Updates stream structure: dict where keys are node names or special markers
                        # Example: {"node_name": {"tool_calls": [...]}} or {"__interrupt__": [...]}
                        if isinstance(data, dict):
                            # Skip special markers like "__interrupt__"
                            for key, update_data in data.items():
                                if key.startswith("__") and key.endswith("__"):
                                    continue  # Skip special markers
                                
                                node_name = key
                                print(f"node_name: {node_name}")
                                # update_data is the actual update content for this node
                                if not isinstance(update_data, dict):
                                    continue
                                # Send progress update (throttle to avoid spam)
                                now = time.time()
                                if now - last_progress_update > 0:  # Max once per 0.5 seconds
                                    # Special handling for "tools" node - contains ToolMessages with results
                                    if node_name == "tools":
                                        messages = update_data.get("messages", [])
                                        for msg in messages:
                                            # Check if it's a ToolMessage (tool execution result)
                                            if isinstance(msg, ToolMessage) or (isinstance(msg, dict) and msg.get("type") == "tool"):
                                                tool_name = msg.get("name", "") if isinstance(msg, dict) else getattr(msg, "name", "")
                                                tool_call_id = msg.get("tool_call_id", "") if isinstance(msg, dict) else getattr(msg, "tool_call_id", "")
                                                
                                                # Look up cached tool args using tool_call_id
                                                cached_tool_info = tool_call_args_cache.get(tool_call_id, {}) if tool_call_id else {}
                                                cached_args = cached_tool_info.get("args", {})
                                                
                                                if tool_name:
                                                    # Format progress message with file path from cached args
                                                    docs_dir = state.docs_dir if state else None
                                                    backend_root_dir = state.backend_root if state else None
                                                    progress_msg = format_tool_call_progress(tool_name, cached_args, docs_dir, backend_root_dir)
                                                    payload = {"type": "progress", "message": progress_msg}
                                                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                                                    last_progress_update = now
                                        continue
                                    
                                    # For "model" node, tool_calls are in the AIMessage within messages
                                    if node_name == "model":
                                        # Track start time for this model call if not already set
                                        # Note: The model node appears AFTER completion in the updates stream,
                                        # so we use request_start_time as an approximation
                                        # (this includes middleware processing time, not just model call time)
                                        if model_call_start_time is None:
                                            model_call_start_time = request_start_time
                                        
                                        messages = update_data.get("messages", [])
                                        for msg in messages:
                                            # Check if it's an AIMessage with tool_calls
                                            if isinstance(msg, dict):
                                                msg_type = msg.get("type", "")
                                                if msg_type == "ai":
                                                    # Extract token usage and stats - try multiple locations
                                                    response_metadata = msg.get("response_metadata", {}) or {}
                                                    usage_metadata = msg.get("usage_metadata") or response_metadata.get("usage_metadata") or {}
                                                    
                                                    # Extract token counts from various possible locations
                                                    input_tokens = 0
                                                    output_tokens = 0
                                                    
                                                    # Try usage_metadata dict
                                                    if isinstance(usage_metadata, dict):
                                                        input_tokens = usage_metadata.get("input_tokens") or usage_metadata.get("prompt_tokens") or 0
                                                        output_tokens = usage_metadata.get("output_tokens") or usage_metadata.get("completion_tokens") or 0
                                                    
                                                    # Try response_metadata directly
                                                    if isinstance(response_metadata, dict):
                                                        if not input_tokens:
                                                            input_tokens = response_metadata.get("input_tokens") or response_metadata.get("prompt_tokens") or 0
                                                        if not output_tokens:
                                                            output_tokens = response_metadata.get("output_tokens") or response_metadata.get("completion_tokens") or 0
                                                    
                                                    # Try top-level message fields
                                                    if not input_tokens:
                                                        input_tokens = msg.get("input_tokens") or msg.get("prompt_tokens") or 0
                                                    if not output_tokens:
                                                        output_tokens = msg.get("output_tokens") or msg.get("completion_tokens") or 0
                                                    
                                                    # Calculate processing time
                                                    # Use request_start_time as the baseline since model node appears after completion
                                                    processing_time = time.time() - request_start_time
                                                    # Reset model_call_start_time for potential next model call in same request
                                                    model_call_start_time = None
                                                    
                                                    # Print stats (even if zero, for debugging)
                                                    _logger.info(
                                                        "[LLM Call Stats] input_tokens=%d, output_tokens=%d, processing_time=%.2fs, response_metadata_keys=%s",
                                                        input_tokens,
                                                        output_tokens,
                                                        processing_time,
                                                        list(response_metadata.keys()) if isinstance(response_metadata, dict) else [],
                                                    )
                                                    
                                                    tool_calls = msg.get("tool_calls", [])
                                                    if tool_calls:
                                                        for tc in tool_calls[:1]:  # Just first tool call
                                                            if isinstance(tc, dict):
                                                                tool_name = tc.get("name", "")
                                                                tool_call_id = tc.get("id", "") or tc.get("tool_call_id", "")
                                                                # Try multiple ways to get args
                                                                tool_args = tc.get("args", {}) or tc.get("arguments", {})
                                                                # Handle case where args might be nested under "function"
                                                                if not tool_args and "function" in tc:
                                                                    func_data = tc.get("function", {})
                                                                    if isinstance(func_data, dict):
                                                                        args_str = func_data.get("arguments", "")
                                                                        if isinstance(args_str, str):
                                                                            try:
                                                                                tool_args = json.loads(args_str)
                                                                            except Exception:
                                                                                tool_args = {}
                                                                        else:
                                                                            tool_args = func_data.get("arguments", {})
                                                                # If args is a string (JSON), parse it
                                                                elif isinstance(tool_args, str):
                                                                    try:
                                                                        tool_args = json.loads(tool_args)
                                                                    except Exception:
                                                                        tool_args = {}
                                                                
                                                                # Cache tool call args by ID for later use with ToolMessages
                                                                if tool_call_id and tool_name:
                                                                    tool_call_args_cache[tool_call_id] = {"name": tool_name, "args": tool_args}
                                                                
                                                                if tool_name:
                                                                    docs_dir = state.docs_dir if state else None
                                                                    backend_root_dir = state.backend_root if state else None
                                                                    progress_msg = format_tool_call_progress(tool_name, tool_args, docs_dir, backend_root_dir)
                                                                    payload = {"type": "progress", "message": progress_msg}
                                                                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                                                                    last_progress_update = now
                                                        break  # Only process first message with tool_calls
                                            # Handle AIMessage objects (not dicts)
                                            elif hasattr(msg, "tool_calls") and msg.tool_calls:
                                                # Extract token usage and stats from AIMessage object
                                                # Use request_start_time as the baseline since model node appears after completion
                                                processing_time = time.time() - request_start_time
                                                # Reset model_call_start_time for potential next model call in same request
                                                model_call_start_time = None
                                                
                                                # Try to get usage_metadata from the message
                                                input_tokens = 0
                                                output_tokens = 0
                                                
                                                usage_metadata = getattr(msg, "usage_metadata", None)
                                                if usage_metadata:
                                                    if isinstance(usage_metadata, dict):
                                                        input_tokens = usage_metadata.get("input_tokens") or usage_metadata.get("prompt_tokens") or 0
                                                        output_tokens = usage_metadata.get("output_tokens") or usage_metadata.get("completion_tokens") or 0
                                                    else:
                                                        # Try as object with attributes
                                                        input_tokens = getattr(usage_metadata, "input_tokens", None) or getattr(usage_metadata, "prompt_tokens", None) or 0
                                                        output_tokens = getattr(usage_metadata, "output_tokens", None) or getattr(usage_metadata, "completion_tokens", None) or 0
                                                
                                                # Try response_metadata if usage_metadata didn't work
                                                if not input_tokens and not output_tokens:
                                                    response_metadata = getattr(msg, "response_metadata", None)
                                                    if response_metadata:
                                                        if isinstance(response_metadata, dict):
                                                            input_tokens = response_metadata.get("input_tokens") or response_metadata.get("prompt_tokens") or 0
                                                            output_tokens = response_metadata.get("output_tokens") or response_metadata.get("completion_tokens") or 0
                                                        else:
                                                            input_tokens = getattr(response_metadata, "input_tokens", None) or getattr(response_metadata, "prompt_tokens", None) or 0
                                                            output_tokens = getattr(response_metadata, "output_tokens", None) or getattr(response_metadata, "completion_tokens", None) or 0
                                                
                                                # Print stats (with debug info)
                                                _logger.info(
                                                    "[LLM Call Stats] input_tokens=%d, output_tokens=%d, processing_time=%.2fs, has_usage_metadata=%s, has_response_metadata=%s",
                                                    input_tokens,
                                                    output_tokens,
                                                    processing_time,
                                                    usage_metadata is not None,
                                                    hasattr(msg, "response_metadata"),
                                                )
                                                
                                                for tc in msg.tool_calls[:1]:
                                                    if isinstance(tc, dict):
                                                        tool_name = tc.get("name", "")
                                                        tool_call_id = tc.get("id", "") or tc.get("tool_call_id", "")
                                                        tool_args = tc.get("args", {}) or tc.get("arguments", {})
                                                        if isinstance(tool_args, str):
                                                            try:
                                                                tool_args = json.loads(tool_args)
                                                            except Exception:
                                                                tool_args = {}
                                                        
                                                        # Cache tool call args
                                                        if tool_call_id and tool_name:
                                                            tool_call_args_cache[tool_call_id] = {"name": tool_name, "args": tool_args}
                                                        
                                                        if tool_name:
                                                            docs_dir = state.docs_dir if state else None
                                                            backend_root_dir = state.backend_root if state else None
                                                            progress_msg = format_tool_call_progress(tool_name, tool_args, docs_dir, backend_root_dir)
                                                            payload = {"type": "progress", "message": progress_msg}
                                                            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                                                            last_progress_update = now
                                                break
                                    else:
                                        # For other nodes, try to extract tool call info from the update data
                                        tool_calls = update_data.get("tool_calls", [])
                                        if tool_calls:
                                            for tc in tool_calls[:1]:  # Just first tool call
                                                if isinstance(tc, dict):
                                                    tool_name = tc.get("name", "")
                                                    # Try multiple ways to get args - different providers structure this differently
                                                    tool_args = tc.get("args", {}) or tc.get("arguments", {})
                                                    # Handle case where args might be nested under "function"
                                                    if not tool_args and "function" in tc:
                                                        func_data = tc.get("function", {})
                                                        if isinstance(func_data, dict):
                                                            args_str = func_data.get("arguments", "")
                                                            if isinstance(args_str, str):
                                                                try:
                                                                    tool_args = json.loads(args_str)
                                                                except Exception:
                                                                    tool_args = {}
                                                            else:
                                                                tool_args = func_data.get("arguments", {})
                                                    # If args is a string (JSON), parse it
                                                    elif isinstance(tool_args, str):
                                                        try:
                                                            tool_args = json.loads(tool_args)
                                                        except Exception:
                                                            tool_args = {}
                                                    
                                                    if tool_name:
                                                        docs_dir = state.docs_dir if state else None
                                                        backend_root_dir = state.backend_root if state else None
                                                        progress_msg = format_tool_call_progress(tool_name, tool_args, docs_dir, backend_root_dir)
                                                        payload = {"type": "progress", "message": progress_msg}
                                                        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                                                        last_progress_update = now
                                        else:
                                            # Generic node execution (no tool calls, just node processing)
                                            progress_msg = f"Processing {node_name}..."
                                            payload = {"type": "progress", "message": progress_msg}
                                            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                                            last_progress_update = now
                        continue

                    # Handle MESSAGES stream mode
                    if current_stream_mode != "messages":
                        continue
                    # Messages stream returns (message, metadata)
                    if not isinstance(data, tuple) or len(data) != 2:
                        continue
                    message, _metadata = data

                    cls = type(message).__name__
                    seen_types[cls] = seen_types.get(cls, 0) + 1

                    # Track HTML file writes so we can fallback if the model writes a file but doesn't return text.
                    if isinstance(message, ToolMessage):
                        tool_name = getattr(message, "name", "") or ""
                        tool_content = getattr(message, "content", "") or ""
                        tool_call_id = getattr(message, "tool_call_id", "") or ""
                        
                        # Try to get tool args from cache using tool_call_id
                        cached_tool_info = tool_call_args_cache.get(tool_call_id, {}) if tool_call_id else {}
                        cached_args = cached_tool_info.get("args", {})
                        if cached_tool_info.get("name"):
                            tool_name = cached_tool_info["name"]  # Use cached name if available
                        
                        # Send progress update when tool execution completes
                        if tool_name:
                            # Try to extract file path from tool content or cached args
                            file_path = None
                            
                            # First, try to get file_path from cached args (most reliable)
                            if cached_args:
                                file_path = cached_args.get("file_path", "") or cached_args.get("path", "")
                            
                            # Fallback: try to extract from tool content
                            if not file_path:
                                if tool_name == "write_file" and isinstance(tool_content, str):
                                    # Filesystem tool returns: "Updated file <path>"
                                    prefix = "Updated file "
                                    if tool_content.startswith(prefix):
                                        file_path = tool_content[len(prefix) :].strip()
                                        if file_path.lower().endswith(".html"):
                                            last_written_html_path = file_path
                                elif tool_name == "read_file" and isinstance(tool_content, str):
                                    # Try to extract file path from read_file content
                                    # read_file content might contain file path info, or we can look for patterns
                                    # For now, try to find file path in the content if it's a short error message
                                    if len(tool_content) < 200:
                                        # Look for common patterns that might indicate file path
                                        # Try to find absolute paths in the content
                                        path_match = re.search(r'/(?:[^/\s]+/)*[^/\s]+', tool_content)
                                        if path_match:
                                            file_path = path_match.group(0)
                            
                            # Format completion message with file path if available
                            if file_path:
                                progress_msg = f"Completed {tool_name}: {file_path}"
                            else:
                                progress_msg = f"Completed {tool_name}"
                            payload = {"type": "progress", "message": progress_msg}
                            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                        continue

                    # Ignore human echoes
                    if isinstance(message, HumanMessage):
                        continue

                    # Stream assistant output: handle both full messages and streaming chunks
                    if not isinstance(message, (AIMessage, AIMessageChunk)):
                        # Some providers may not use these exact classes; fall back on type=="ai" when present.
                        if getattr(message, "type", None) != "ai":
                            continue

                    # Extract token usage from AIMessage if available (messages stream has more complete metadata)
                    if isinstance(message, (AIMessage, AIMessageChunk)) or getattr(message, "type", None) == "ai":
                        # Try to extract token usage - this is often more complete in messages stream
                        usage_metadata = getattr(message, "usage_metadata", None)
                        response_metadata = getattr(message, "response_metadata", None)
                        
                        input_tokens = 0
                        output_tokens = 0
                        
                        # Try usage_metadata first
                        if usage_metadata:
                            if isinstance(usage_metadata, dict):
                                input_tokens = usage_metadata.get("input_tokens") or usage_metadata.get("prompt_tokens") or 0
                                output_tokens = usage_metadata.get("output_tokens") or usage_metadata.get("completion_tokens") or 0
                            else:
                                input_tokens = getattr(usage_metadata, "input_tokens", None) or getattr(usage_metadata, "prompt_tokens", None) or 0
                                output_tokens = getattr(usage_metadata, "output_tokens", None) or getattr(usage_metadata, "completion_tokens", None) or 0
                        
                        # Try response_metadata if usage_metadata didn't work
                        if (not input_tokens and not output_tokens) and response_metadata:
                            if isinstance(response_metadata, dict):
                                input_tokens = response_metadata.get("input_tokens") or response_metadata.get("prompt_tokens") or 0
                                output_tokens = response_metadata.get("output_tokens") or response_metadata.get("completion_tokens") or 0
                            else:
                                input_tokens = getattr(response_metadata, "input_tokens", None) or getattr(response_metadata, "prompt_tokens", None) or 0
                                output_tokens = getattr(response_metadata, "output_tokens", None) or getattr(response_metadata, "completion_tokens", None) or 0
                        
                        # Log token usage if found
                        if input_tokens or output_tokens:
                            _logger.info(
                                "[LLM Call Stats from messages stream] input_tokens=%d, output_tokens=%d",
                                input_tokens,
                                output_tokens,
                            )
                    
                    # Check for tool calls in AI messages and send progress updates
                    tool_calls = getattr(message, "tool_calls", None)
                    if tool_calls:
                        for tc in tool_calls[:3]:  # Limit to first 3 tool calls
                            if isinstance(tc, dict):
                                tool_name = tc.get("name", "")
                                tool_call_id = tc.get("id", "") or tc.get("tool_call_id", "")
                                # Try multiple ways to get args - different providers structure this differently
                                tool_args = tc.get("args", {}) or tc.get("arguments", {})
                                # Handle case where args might be nested under "function"
                                if not tool_args and "function" in tc:
                                    func_data = tc.get("function", {})
                                    if isinstance(func_data, dict):
                                        args_str = func_data.get("arguments", "")
                                        if isinstance(args_str, str):
                                            try:
                                                tool_args = json.loads(args_str)
                                            except Exception:
                                                tool_args = {}
                                        else:
                                            tool_args = func_data.get("arguments", {})
                                # If args is a string (JSON), parse it
                                elif isinstance(tool_args, str):
                                    try:
                                        tool_args = json.loads(tool_args)
                                    except Exception:
                                        tool_args = {}
                                
                                # Cache tool call args by ID for later use with ToolMessages
                                if tool_call_id and tool_name:
                                    tool_call_args_cache[tool_call_id] = {"name": tool_name, "args": tool_args}
                                
                                # Debug logging if enabled
                                if env_flag("BC_API_STREAM_DEBUG", default=False):
                                    _logger.debug(
                                        "Tool call in AI message: name=%s, id=%s, args=%s, tc_keys=%s",
                                        tool_name,
                                        tool_call_id,
                                        tool_args,
                                        list(tc.keys()) if isinstance(tc, dict) else [],
                                    )
                                
                                if tool_name:
                                    docs_dir = state.docs_dir if state else None
                                    backend_root_dir = str(Path.cwd()) if state else None
                                    progress_msg = format_tool_call_progress(tool_name, tool_args, docs_dir, backend_root_dir)
                                    payload = {"type": "progress", "message": progress_msg}
                                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

                    for text in extract_text_chunks_from_ai_message(message):
                        final_parts.append(text)
                        delta_count += 1
                        payload = {"type": "delta", "text": text}
                        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

                final_text = "".join(final_parts).strip()

                # Fallback: if no text was streamed but an HTML file was written, read it and return its contents.
                if not final_text and last_written_html_path:
                    try:
                        pth = Path(last_written_html_path)
                        if pth.exists() and pth.is_file() and pth.stat().st_size <= 2 * 1024 * 1024:
                            final_text = pth.read_text(encoding="utf-8", errors="replace").strip()
                    except Exception:  # noqa: BLE001
                        pass

                if env_flag("BC_API_STREAM_DEBUG", default=False):
                    _logger.info(
                        "chat_stream_debug thread_id=%s delta_count=%s seen_message_types=%s last_written_html=%s final_len=%s",
                        tid,
                        delta_count,
                        seen_types,
                        last_written_html_path,
                        len(final_text),
                    )
                log_chat_io(
                    user_id=req.user_id,
                    conversation_id=req.conversation_id,
                    thread_id=tid,
                    user_message=req.message,
                    reply=final_text,
                )
                yield f"data: {json.dumps({'type':'final','text':final_text}, ensure_ascii=False)}\n\n"
            except Exception as e:  # noqa: BLE001
                _logger.warning(
                    "[ModelFallback] Primary agent (qwen) failed during stream: %s: %s",
                    type(e).__name__,
                    str(e),
                )
                _logger.info("[ModelFallback] Falling back to deepseek...")
                try:
                    # Reset state for fallback stream
                    final_parts = []
                    delta_count = 0
                    seen_types = {}
                    last_written_html_path = None
                    last_progress_update = 0.0
                    tool_call_args_cache = {}
                    request_start_time = time.time()
                    model_call_start_time = None
                    
                    async for chunk in state.fallback_agent.astream(
                        {
                            "messages": [HumanMessage(content=req.message)],
                            "expertise_type": req.expertise_type,
                        },
                        config={
                            "configurable": {"thread_id": tid},
                            "metadata": {
                                "user_id": req.user_id,
                                "expertise_type": req.expertise_type,
                                **(req.metadata or {}),
                            },
                        },
                        stream_mode=["messages", "updates"],
                        subgraphs=True,
                        durability="exit",
                    ):
                        # With subgraphs=True and multiple stream modes, chunks are:
                        # (namespace, stream_mode, data)
                        if not isinstance(chunk, tuple) or len(chunk) != 3:
                            continue

                        _namespace, current_stream_mode, data = chunk

                        # Handle UPDATES stream mode - provides progress information
                        if current_stream_mode == "updates":
                            # Updates stream structure: dict where keys are node names or special markers
                            # Example: {"node_name": {"tool_calls": [...]}} or {"__interrupt__": [...]}
                            if isinstance(data, dict):
                                # Skip special markers like "__interrupt__"
                                for key, update_data in data.items():
                                    if key.startswith("__") and key.endswith("__"):
                                        continue  # Skip special markers
                                    
                                    node_name = key
                                    print(f"node_name: {node_name}")
                                    # update_data is the actual update content for this node
                                    if not isinstance(update_data, dict):
                                        continue
                                    # Send progress update (throttle to avoid spam)
                                    now = time.time()
                                    if now - last_progress_update > 0:  # Max once per 0.5 seconds
                                        # Special handling for "tools" node - contains ToolMessages with results
                                        if node_name == "tools":
                                            messages = update_data.get("messages", [])
                                            for msg in messages:
                                                # Check if it's a ToolMessage (tool execution result)
                                                if isinstance(msg, ToolMessage) or (isinstance(msg, dict) and msg.get("type") == "tool"):
                                                    tool_name = msg.get("name", "") if isinstance(msg, dict) else getattr(msg, "name", "")
                                                    tool_call_id = msg.get("tool_call_id", "") if isinstance(msg, dict) else getattr(msg, "tool_call_id", "")
                                                    
                                                    # Look up cached tool args using tool_call_id
                                                    cached_tool_info = tool_call_args_cache.get(tool_call_id, {}) if tool_call_id else {}
                                                    cached_args = cached_tool_info.get("args", {})
                                                    
                                                    if tool_name:
                                                        # Format progress message with file path from cached args
                                                        docs_dir = state.docs_dir if state else None
                                                        backend_root_dir = state.backend_root if state else None
                                                        progress_msg = format_tool_call_progress(tool_name, cached_args, docs_dir, backend_root_dir)
                                                        payload = {"type": "progress", "message": progress_msg}
                                                        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                                                        last_progress_update = now
                                            continue
                                        
                                        # For "model" node, tool_calls are in the AIMessage within messages
                                        if node_name == "model":
                                            # Track start time for this model call if not already set
                                            # Note: The model node appears AFTER completion in the updates stream,
                                            # so we use request_start_time as an approximation
                                            # (this includes middleware processing time, not just model call time)
                                            if model_call_start_time is None:
                                                model_call_start_time = request_start_time
                                            
                                            messages = update_data.get("messages", [])
                                            for msg in messages:
                                                # Check if it's an AIMessage with tool_calls
                                                if isinstance(msg, dict):
                                                    msg_type = msg.get("type", "")
                                                    if msg_type == "ai":
                                                        # Extract token usage and stats - try multiple locations
                                                        response_metadata = msg.get("response_metadata", {}) or {}
                                                        usage_metadata = msg.get("usage_metadata") or response_metadata.get("usage_metadata") or {}
                                                        
                                                        # Extract token counts from various possible locations
                                                        input_tokens = 0
                                                        output_tokens = 0
                                                        
                                                        # Try usage_metadata dict
                                                        if isinstance(usage_metadata, dict):
                                                            input_tokens = usage_metadata.get("input_tokens") or usage_metadata.get("prompt_tokens") or 0
                                                            output_tokens = usage_metadata.get("output_tokens") or usage_metadata.get("completion_tokens") or 0
                                                        
                                                        # Try response_metadata directly
                                                        if isinstance(response_metadata, dict):
                                                            if not input_tokens:
                                                                input_tokens = response_metadata.get("input_tokens") or response_metadata.get("prompt_tokens") or 0
                                                            if not output_tokens:
                                                                output_tokens = response_metadata.get("output_tokens") or response_metadata.get("completion_tokens") or 0
                                                        
                                                        # Try top-level message fields
                                                        if not input_tokens:
                                                            input_tokens = msg.get("input_tokens") or msg.get("prompt_tokens") or 0
                                                        if not output_tokens:
                                                            output_tokens = msg.get("output_tokens") or msg.get("completion_tokens") or 0
                                                        
                                                        # Calculate processing time
                                                        # Use request_start_time as the baseline since model node appears after completion
                                                        processing_time = time.time() - request_start_time
                                                        # Reset model_call_start_time for potential next model call in same request
                                                        model_call_start_time = None
                                                        
                                                        # Print stats (even if zero, for debugging)
                                                        _logger.info(
                                                            "[LLM Call Stats] input_tokens=%d, output_tokens=%d, processing_time=%.2fs, response_metadata_keys=%s",
                                                            input_tokens,
                                                            output_tokens,
                                                            processing_time,
                                                            list(response_metadata.keys()) if isinstance(response_metadata, dict) else [],
                                                        )
                                                        
                                                        tool_calls = msg.get("tool_calls", [])
                                                        if tool_calls:
                                                            for tc in tool_calls[:1]:  # Just first tool call
                                                                if isinstance(tc, dict):
                                                                    tool_name = tc.get("name", "")
                                                                    tool_call_id = tc.get("id", "") or tc.get("tool_call_id", "")
                                                                    # Try multiple ways to get args
                                                                    tool_args = tc.get("args", {}) or tc.get("arguments", {})
                                                                    # Handle case where args might be nested under "function"
                                                                    if not tool_args and "function" in tc:
                                                                        func_data = tc.get("function", {})
                                                                        if isinstance(func_data, dict):
                                                                            args_str = func_data.get("arguments", "")
                                                                            if isinstance(args_str, str):
                                                                                try:
                                                                                    tool_args = json.loads(args_str)
                                                                                except Exception:
                                                                                    tool_args = {}
                                                                            else:
                                                                                tool_args = func_data.get("arguments", {})
                                                                    # If args is a string (JSON), parse it
                                                                    elif isinstance(tool_args, str):
                                                                        try:
                                                                            tool_args = json.loads(tool_args)
                                                                        except Exception:
                                                                            tool_args = {}
                                                                    
                                                                    # Cache tool call args by ID for later use with ToolMessages
                                                                    if tool_call_id and tool_name:
                                                                        tool_call_args_cache[tool_call_id] = {"name": tool_name, "args": tool_args}
                                                                    
                                                                    if tool_name:
                                                                        docs_dir = state.docs_dir if state else None
                                                                        backend_root_dir = state.backend_root if state else None
                                                                        progress_msg = format_tool_call_progress(tool_name, tool_args, docs_dir, backend_root_dir)
                                                                        payload = {"type": "progress", "message": progress_msg}
                                                                        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                                                                        last_progress_update = now
                                                            break  # Only process first message with tool_calls
                                                # Handle AIMessage objects (not dicts)
                                                elif hasattr(msg, "tool_calls") and msg.tool_calls:
                                                    # Extract token usage and stats from AIMessage object
                                                    # Use request_start_time as the baseline since model node appears after completion
                                                    processing_time = time.time() - request_start_time
                                                    # Reset model_call_start_time for potential next model call in same request
                                                    model_call_start_time = None
                                                    
                                                    # Try to get usage_metadata from the message
                                                    input_tokens = 0
                                                    output_tokens = 0
                                                    
                                                    usage_metadata = getattr(msg, "usage_metadata", None)
                                                    if usage_metadata:
                                                        if isinstance(usage_metadata, dict):
                                                            input_tokens = usage_metadata.get("input_tokens") or usage_metadata.get("prompt_tokens") or 0
                                                            output_tokens = usage_metadata.get("output_tokens") or usage_metadata.get("completion_tokens") or 0
                                                        else:
                                                            # Try as object with attributes
                                                            input_tokens = getattr(usage_metadata, "input_tokens", None) or getattr(usage_metadata, "prompt_tokens", None) or 0
                                                            output_tokens = getattr(usage_metadata, "output_tokens", None) or getattr(usage_metadata, "completion_tokens", None) or 0
                                                    
                                                    # Try response_metadata if usage_metadata didn't work
                                                    if not input_tokens and not output_tokens:
                                                        response_metadata = getattr(msg, "response_metadata", None)
                                                        if response_metadata:
                                                            if isinstance(response_metadata, dict):
                                                                input_tokens = response_metadata.get("input_tokens") or response_metadata.get("prompt_tokens") or 0
                                                                output_tokens = response_metadata.get("output_tokens") or response_metadata.get("completion_tokens") or 0
                                                            else:
                                                                input_tokens = getattr(response_metadata, "input_tokens", None) or getattr(response_metadata, "prompt_tokens", None) or 0
                                                                output_tokens = getattr(response_metadata, "output_tokens", None) or getattr(response_metadata, "completion_tokens", None) or 0
                                                    
                                                    # Print stats (with debug info)
                                                    _logger.info(
                                                        "[LLM Call Stats] input_tokens=%d, output_tokens=%d, processing_time=%.2fs, has_usage_metadata=%s, has_response_metadata=%s",
                                                        input_tokens,
                                                        output_tokens,
                                                        processing_time,
                                                        usage_metadata is not None,
                                                        hasattr(msg, "response_metadata"),
                                                    )
                                                    
                                                    for tc in msg.tool_calls[:1]:
                                                        if isinstance(tc, dict):
                                                            tool_name = tc.get("name", "")
                                                            tool_call_id = tc.get("id", "") or tc.get("tool_call_id", "")
                                                            tool_args = tc.get("args", {}) or tc.get("arguments", {})
                                                            if isinstance(tool_args, str):
                                                                try:
                                                                    tool_args = json.loads(tool_args)
                                                                except Exception:
                                                                    tool_args = {}
                                                            
                                                            # Cache tool call args
                                                            if tool_call_id and tool_name:
                                                                tool_call_args_cache[tool_call_id] = {"name": tool_name, "args": tool_args}
                                                            
                                                            if tool_name:
                                                                docs_dir = state.docs_dir if state else None
                                                                backend_root_dir = state.backend_root if state else None
                                                                progress_msg = format_tool_call_progress(tool_name, tool_args, docs_dir, backend_root_dir)
                                                                payload = {"type": "progress", "message": progress_msg}
                                                                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                                                                last_progress_update = now
                                                    break
                                        else:
                                            # For other nodes, try to extract tool call info from the update data
                                            tool_calls = update_data.get("tool_calls", [])
                                            if tool_calls:
                                                for tc in tool_calls[:1]:  # Just first tool call
                                                    if isinstance(tc, dict):
                                                        tool_name = tc.get("name", "")
                                                        # Try multiple ways to get args - different providers structure this differently
                                                        tool_args = tc.get("args", {}) or tc.get("arguments", {})
                                                        # Handle case where args might be nested under "function"
                                                        if not tool_args and "function" in tc:
                                                            func_data = tc.get("function", {})
                                                            if isinstance(func_data, dict):
                                                                args_str = func_data.get("arguments", "")
                                                                if isinstance(args_str, str):
                                                                    try:
                                                                        tool_args = json.loads(args_str)
                                                                    except Exception:
                                                                        tool_args = {}
                                                                else:
                                                                    tool_args = func_data.get("arguments", {})
                                                        # If args is a string (JSON), parse it
                                                        elif isinstance(tool_args, str):
                                                            try:
                                                                tool_args = json.loads(tool_args)
                                                            except Exception:
                                                                tool_args = {}
                                                        
                                                        if tool_name:
                                                            docs_dir = state.docs_dir if state else None
                                                            backend_root_dir = state.backend_root if state else None
                                                            progress_msg = format_tool_call_progress(tool_name, tool_args, docs_dir, backend_root_dir)
                                                            payload = {"type": "progress", "message": progress_msg}
                                                            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                                                            last_progress_update = now
                                            else:
                                                # Generic node execution (no tool calls, just node processing)
                                                progress_msg = f"Processing {node_name}..."
                                                payload = {"type": "progress", "message": progress_msg}
                                                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                                                last_progress_update = now
                            continue

                        # Handle MESSAGES stream mode
                        if current_stream_mode != "messages":
                            continue
                        # Messages stream returns (message, metadata)
                        if not isinstance(data, tuple) or len(data) != 2:
                            continue
                        message, _metadata = data

                        cls = type(message).__name__
                        seen_types[cls] = seen_types.get(cls, 0) + 1

                        # Track HTML file writes so we can fallback if the model writes a file but doesn't return text.
                        if isinstance(message, ToolMessage):
                            tool_name = getattr(message, "name", "") or ""
                            tool_content = getattr(message, "content", "") or ""
                            tool_call_id = getattr(message, "tool_call_id", "") or ""
                            
                            # Try to get tool args from cache using tool_call_id
                            cached_tool_info = tool_call_args_cache.get(tool_call_id, {}) if tool_call_id else {}
                            cached_args = cached_tool_info.get("args", {})
                            if cached_tool_info.get("name"):
                                tool_name = cached_tool_info["name"]  # Use cached name if available
                            
                            # Send progress update when tool execution completes
                            if tool_name:
                                # Try to extract file path from tool content or cached args
                                file_path = None
                                
                                # First, try to get file_path from cached args (most reliable)
                                if cached_args:
                                    file_path = cached_args.get("file_path", "") or cached_args.get("path", "")
                                
                                # Fallback: try to extract from tool content
                                if not file_path:
                                    if tool_name == "write_file" and isinstance(tool_content, str):
                                        # Filesystem tool returns: "Updated file <path>"
                                        prefix = "Updated file "
                                        if tool_content.startswith(prefix):
                                            file_path = tool_content[len(prefix) :].strip()
                                            if file_path.lower().endswith(".html"):
                                                last_written_html_path = file_path
                                    elif tool_name == "read_file" and isinstance(tool_content, str):
                                        # Try to extract file path from read_file content
                                        # read_file content might contain file path info, or we can look for patterns
                                        # For now, try to find file path in the content if it's a short error message
                                        if len(tool_content) < 200:
                                            # Look for common patterns that might indicate file path
                                            # Try to find absolute paths in the content
                                            path_match = re.search(r'/(?:[^/\s]+/)*[^/\s]+', tool_content)
                                            if path_match:
                                                file_path = path_match.group(0)
                                
                                # Format completion message with file path if available
                                if file_path:
                                    progress_msg = f"Completed {tool_name}: {file_path}"
                                else:
                                    progress_msg = f"Completed {tool_name}"
                                payload = {"type": "progress", "message": progress_msg}
                                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                            continue

                        # Ignore human echoes
                        if isinstance(message, HumanMessage):
                            continue

                        # Stream assistant output: handle both full messages and streaming chunks
                        if not isinstance(message, (AIMessage, AIMessageChunk)):
                            # Some providers may not use these exact classes; fall back on type=="ai" when present.
                            if getattr(message, "type", None) != "ai":
                                continue

                        # Extract token usage from AIMessage if available (messages stream has more complete metadata)
                        if isinstance(message, (AIMessage, AIMessageChunk)) or getattr(message, "type", None) == "ai":
                            # Try to extract token usage - this is often more complete in messages stream
                            usage_metadata = getattr(message, "usage_metadata", None)
                            response_metadata = getattr(message, "response_metadata", None)
                            
                            input_tokens = 0
                            output_tokens = 0
                            
                            # Try usage_metadata first
                            if usage_metadata:
                                if isinstance(usage_metadata, dict):
                                    input_tokens = usage_metadata.get("input_tokens") or usage_metadata.get("prompt_tokens") or 0
                                    output_tokens = usage_metadata.get("output_tokens") or usage_metadata.get("completion_tokens") or 0
                                else:
                                    input_tokens = getattr(usage_metadata, "input_tokens", None) or getattr(usage_metadata, "prompt_tokens", None) or 0
                                    output_tokens = getattr(usage_metadata, "output_tokens", None) or getattr(usage_metadata, "completion_tokens", None) or 0
                            
                            # Try response_metadata if usage_metadata didn't work
                            if (not input_tokens and not output_tokens) and response_metadata:
                                if isinstance(response_metadata, dict):
                                    input_tokens = response_metadata.get("input_tokens") or response_metadata.get("prompt_tokens") or 0
                                    output_tokens = response_metadata.get("output_tokens") or response_metadata.get("completion_tokens") or 0
                                else:
                                    input_tokens = getattr(response_metadata, "input_tokens", None) or getattr(response_metadata, "prompt_tokens", None) or 0
                                    output_tokens = getattr(response_metadata, "output_tokens", None) or getattr(response_metadata, "completion_tokens", None) or 0
                            
                            # Log token usage if found
                            if input_tokens or output_tokens:
                                _logger.info(
                                    "[LLM Call Stats from messages stream] input_tokens=%d, output_tokens=%d",
                                    input_tokens,
                                    output_tokens,
                                )
                        
                        # Check for tool calls in AI messages and send progress updates
                        tool_calls = getattr(message, "tool_calls", None)
                        if tool_calls:
                            for tc in tool_calls[:3]:  # Limit to first 3 tool calls
                                if isinstance(tc, dict):
                                    tool_name = tc.get("name", "")
                                    tool_call_id = tc.get("id", "") or tc.get("tool_call_id", "")
                                    # Try multiple ways to get args - different providers structure this differently
                                    tool_args = tc.get("args", {}) or tc.get("arguments", {})
                                    # Handle case where args might be nested under "function"
                                    if not tool_args and "function" in tc:
                                        func_data = tc.get("function", {})
                                        if isinstance(func_data, dict):
                                            args_str = func_data.get("arguments", "")
                                            if isinstance(args_str, str):
                                                try:
                                                    tool_args = json.loads(args_str)
                                                except Exception:
                                                    tool_args = {}
                                            else:
                                                tool_args = func_data.get("arguments", {})
                                    # If args is a string (JSON), parse it
                                    elif isinstance(tool_args, str):
                                        try:
                                            tool_args = json.loads(tool_args)
                                        except Exception:
                                            tool_args = {}
                                    
                                    # Cache tool call args by ID for later use with ToolMessages
                                    if tool_call_id and tool_name:
                                        tool_call_args_cache[tool_call_id] = {"name": tool_name, "args": tool_args}
                                    
                                    # Debug logging if enabled
                                    if env_flag("BC_API_STREAM_DEBUG", default=False):
                                        _logger.debug(
                                            "Tool call in AI message: name=%s, id=%s, args=%s, tc_keys=%s",
                                            tool_name,
                                            tool_call_id,
                                            tool_args,
                                            list(tc.keys()) if isinstance(tc, dict) else [],
                                        )
                                    
                                    if tool_name:
                                        docs_dir = state.docs_dir if state else None
                                        backend_root_dir = str(Path.cwd()) if state else None
                                        progress_msg = format_tool_call_progress(tool_name, tool_args, docs_dir, backend_root_dir)
                                        payload = {"type": "progress", "message": progress_msg}
                                        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

                        for text in extract_text_chunks_from_ai_message(message):
                            final_parts.append(text)
                            delta_count += 1
                            payload = {"type": "delta", "text": text}
                            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

                    final_text = "".join(final_parts).strip()

                    # Fallback: if no text was streamed but an HTML file was written, read it and return its contents.
                    if not final_text and last_written_html_path:
                        try:
                            pth = Path(last_written_html_path)
                            if pth.exists() and pth.is_file() and pth.stat().st_size <= 2 * 1024 * 1024:
                                final_text = pth.read_text(encoding="utf-8", errors="replace").strip()
                        except Exception:  # noqa: BLE001
                            pass

                    if env_flag("BC_API_STREAM_DEBUG", default=False):
                        _logger.info(
                            "chat_stream_debug thread_id=%s delta_count=%s seen_message_types=%s last_written_html=%s final_len=%s",
                            tid,
                            delta_count,
                            seen_types,
                            last_written_html_path,
                            len(final_text),
                        )
                    log_chat_io(
                        user_id=req.user_id,
                        conversation_id=req.conversation_id,
                        thread_id=tid,
                        user_message=req.message,
                        reply=final_text,
                    )
                    yield f"data: {json.dumps({'type':'final','text':final_text}, ensure_ascii=False)}\n\n"
                    
                    # Check if expert sync is needed (dual-agent mode only)
                    if state.use_dual_agent and state.expert_agent is not None:
                        try:
                            # Read final state from checkpoint after stream completes
                            checkpointer = state.agent.checkpointer
                            config = {"configurable": {"thread_id": tid}}
                            checkpoint = await checkpointer.aget(config)
                            
                            if checkpoint:
                                state_values = checkpoint.get("channel_values", {})
                                # Create a state dict similar to what ainvoke returns
                                result_state = {
                                    "messages": state_values.get("messages", []),
                                    "conversation_round": state_values.get("conversation_round", 0),
                                    "expertise_type": req.expertise_type,
                                    **state_values,
                                }
                        except Exception as e:  # noqa: BLE001
                            _logger.error("[DualAgent] Error checking/triggering expert sync in stream: %s", str(e))
                            # Don't fail the request if expert sync fails
                except Exception as fallback_error:  # noqa: BLE001
                    _logger.error(
                        "[ModelFallback] Fallback agent (deepseek) also failed during stream: %s: %s",
                        type(fallback_error).__name__,
                        str(fallback_error),
                    )
                    _logger.exception(
                        "POST /chat/stream failed user_id=%s conversation_id=%s thread_id=%s error_type=%s error_message=%s",
                        req.user_id,
                        req.conversation_id,
                        tid,
                        type(fallback_error).__name__,
                        str(fallback_error),
                    )
                    detail: dict[str, Any] = {
                        "error_type": type(fallback_error).__name__,
                        "error_message": str(fallback_error),
                        "thread_id": tid,
                    }
                    if env_flag("BC_API_RETURN_TRACEBACK", default=False):
                        detail["traceback"] = traceback.format_exc()
                    yield f"data: {json.dumps({'type':'error','detail':detail}, ensure_ascii=False)}\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream; charset=utf-8")
