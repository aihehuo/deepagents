"""Callback functions for streaming and async operations."""

from __future__ import annotations

import asyncio
import json
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from apps.business_cofounder_api.app.utils import (
    env_flag,
    format_tool_call_progress,
    log_message_history_for_debugging,
)

# Use uvicorn's configured logger so output reliably shows up in the terminal.
_logger = logging.getLogger("uvicorn.error")


def serialize_for_json(obj: Any) -> Any:
    """Recursively serialize an object to be JSON-serializable.
    
    Handles LangChain message objects and other complex types by converting them to dicts.
    """
    # Handle LangChain message objects
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:  # noqa: BLE001
            pass
    
    if hasattr(obj, "dict"):
        try:
            return obj.dict()
        except Exception:  # noqa: BLE001
            pass
    
    # Handle dicts
    if isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    
    # Handle lists
    if isinstance(obj, list):
        return [serialize_for_json(item) for item in obj]
    
    # Handle tuples
    if isinstance(obj, tuple):
        return tuple(serialize_for_json(item) for item in obj)
    
    # Handle basic JSON-serializable types
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    
    # For objects with __dict__, try to serialize it
    if hasattr(obj, "__dict__"):
        try:
            return {k: serialize_for_json(v) for k, v in obj.__dict__.items()}
        except Exception:  # noqa: BLE001
            pass
    
    # Last resort: convert to string
    return str(obj)


def compose_concise_callback_message(
    namespace: Any,
    stream_mode: str,
    data: Any,
    docs_dir: str | None = None,
    backend_root_dir: str | None = None,
) -> str | None:
    """Compose a concise human-readable message from stream chunk data.
    
    Args:
        namespace: The namespace from the chunk
        stream_mode: Either "messages" or "updates"
        data: The data from the chunk
        docs_dir: Docs directory for file path resolution
        backend_root_dir: Backend root directory for file path resolution
        
    Returns:
        A concise string message describing what's happening, or None to skip
    """
    
    if stream_mode == "messages":
        # For messages stream, data is a tuple: (message, metadata)
        if isinstance(data, tuple) and len(data) >= 1:
            message = data[0]
            metadata = data[1] if len(data) > 1 else {}
            
            # Get the actual class name for better identification
            class_name = type(message).__name__
            
            # Check message type (semantic type from LangChain: "ai", "tool", "human")
            msg_type = getattr(message, "type", None)
            if not msg_type:
                msg_type = message.get("type") if isinstance(message, dict) else None
            
            if msg_type in ("ai", "AIMessageChunk"):
                # Check for finish_reason to identify completed messages
                response_metadata = getattr(message, "response_metadata", None) or (message.get("response_metadata") if isinstance(message, dict) else None)
                finish_reason = None
                if isinstance(response_metadata, dict):
                    finish_reason = response_metadata.get("finish_reason")
                
                # Check for invalid_tool_calls - these are partial streaming fragments of tool call arguments
                # They're marked "invalid" because they're incomplete JSON until the full tool call is received
                # Example: fragments like '": "/Users/yc' that will become a complete file path
                # Skip these intermediate chunks - they're not meaningful for callbacks
                invalid_tool_calls = getattr(message, "invalid_tool_calls", None) or (message.get("invalid_tool_calls") if isinstance(message, dict) else None)
                if invalid_tool_calls and isinstance(invalid_tool_calls, (list, tuple)) and len(invalid_tool_calls) > 0:
                    # These are partial streaming chunks - skip them unless we have a finish_reason indicating completion
                    if finish_reason != "tool_calls":
                        # Return None to skip this callback entirely (filter out intermediate streaming chunks)
                        return None
                
                # Check for tool calls - prioritize completed tool calls
                tool_calls = getattr(message, "tool_calls", None) or (message.get("tool_calls") if isinstance(message, dict) else None)
                
                # If finish_reason is 'tool_calls', this is a completed tool call message
                if finish_reason == "tool_calls" and tool_calls and isinstance(tool_calls, (list, tuple)):
                    tool_info_list = []
                    for tc in tool_calls[:3]:  # Limit to first 3 tool calls
                        tool_name = None
                        tool_args = {}
                        
                        if isinstance(tc, dict):
                            tool_name = tc.get("name", "")
                            tool_args = tc.get("args", {}) or tc.get("arguments", {})
                            # Try to parse args if it's a string
                            if isinstance(tool_args, str):
                                try:
                                    tool_args = json.loads(tool_args)
                                except Exception:  # noqa: BLE001
                                    tool_args = {}
                        else:
                            tool_name = getattr(tc, "name", None)
                            tool_args = getattr(tc, "args", None) or getattr(tc, "arguments", None) or {}
                        
                        if tool_name:
                            # Use the progress formatter for better messages
                            try:
                                progress_msg = format_tool_call_progress(
                                    tool_name, tool_args, docs_dir, backend_root_dir
                                )
                                tool_info_list.append(progress_msg)
                            except Exception:  # noqa: BLE001
                                tool_info_list.append(f"Calling {tool_name}...")
                    
                    if tool_info_list:
                        # Join multiple tool calls with semicolons
                        return "; ".join(tool_info_list) if len(tool_info_list) > 1 else tool_info_list[0]
                
                # Check for tool calls even if not finished (streaming chunks)
                elif tool_calls and isinstance(tool_calls, (list, tuple)):
                    tool_names = []
                    for tc in tool_calls[:3]:
                        if isinstance(tc, dict):
                            tool_name = tc.get("name", "")
                        else:
                            tool_name = getattr(tc, "name", "")
                        if tool_name:
                            tool_names.append(tool_name)
                    if tool_names:
                        # Only show tool call names during streaming if we have valid names
                        # (avoid showing empty or invalid tool calls)
                        valid_names = [n for n in tool_names if n]
                        if valid_names:
                            return f"Calling {', '.join(valid_names)}..."
                
                # Extract text content - only show meaningful chunks
                content = getattr(message, "content", None) or (message.get("content") if isinstance(message, dict) else None)
                if isinstance(content, str) and len(content.strip()) > 0:  # Only show substantial text chunks
                    preview = content.strip()[:150] + "..." if len(content.strip()) > 150 else content.strip()
                    return f"Assistant: {preview}"
                elif isinstance(content, list) and content:
                    # Extract text from content blocks
                    # DeepSeek and some providers send content as list of dicts: [{'text': '...', 'type': 'text', 'index': 0}]
                    text_parts = []
                    for item in content:
                        if isinstance(item, str):
                            text = item.strip()
                            # For streaming chunks, accept shorter text (at least 1 char)
                            if len(text) > 0:
                                text_parts.append(text)
                        elif isinstance(item, dict):
                            text = item.get("text", "")
                            if isinstance(text, str):
                                text = text.strip()
                                # For streaming chunks, accept shorter text (at least 1 char)
                                if len(text) > 0:
                                    text_parts.append(text)
                    if text_parts:
                        # Join all text parts and create preview
                        combined_text = "".join(text_parts)  # Join without spaces for better handling of streaming text
                        preview = combined_text[:150] + "..." if len(combined_text) > 150 else combined_text
                        return f"Assistant: {preview}"
                
                # If we have finish_reason but no tool calls and no content, skip it
                # (don't send "Assistant processing..." status - it's not useful)
                if finish_reason:
                    return None
                
                return None
            
            elif msg_type in ("tool", "ToolMessageChunk"):
                tool_name = getattr(message, "name", None) or (message.get("name") if isinstance(message, dict) else None)
                if tool_name:
                    return f"Tool {tool_name} completed"
                return "Tool execution completed"
            
            elif msg_type in ("human", "HumanMessageChunk"):
                return "User message received"
            
            # If we couldn't determine the type from the type attribute, use the class name
            if not msg_type or msg_type not in ("ai", "tool", "human"):
                return f"Processing {class_name}..."
        
        # Fallback: try to get class name from data
        if isinstance(data, tuple) and len(data) >= 1:
            class_name = type(data[0]).__name__
            return f"Processing {class_name}..."
        
        return f"Processing message for messages type {msg_type}..."
    elif stream_mode == "updates":
        # For updates stream, extract node name and action
        if isinstance(data, dict):
            # Skip special markers
            for key, update_data in data.items():
                if key.startswith("__") and key.endswith("__"):
                    continue
                
                node_name = key
                if not isinstance(update_data, dict):
                    continue
                
                # Check for tool calls in update data
                tool_calls = update_data.get("tool_calls", [])
                if tool_calls:
                    tool_names = []
                    for tc in tool_calls[:3]:
                        if isinstance(tc, dict):
                            tool_name = tc.get("name", "")
                            if tool_name:
                                tool_args = tc.get("args", {}) or tc.get("arguments", {})
                                # Use the existing progress formatter if available
                                try:
                                    progress_msg = format_tool_call_progress(
                                        tool_name, tool_args, docs_dir, backend_root_dir
                                    )
                                    tool_names.append(progress_msg)
                                except Exception:  # noqa: BLE001
                                    tool_names.append(f"Calling {tool_name}...")
                    if tool_names:
                        return "; ".join(tool_names)
                
                # Check for messages (tool results)
                # Note: messages might be an Overwrite object or other non-iterable type
                messages = update_data.get("messages", [])
                if messages and isinstance(messages, (list, tuple)):
                    for msg in messages:
                        if isinstance(msg, dict):
                            if msg.get("type") == "tool":
                                tool_name = msg.get("name", "")
                                if tool_name:
                                    return f"Tool {tool_name} completed"
                        elif hasattr(msg, "type") and getattr(msg, "type") == "tool":
                            tool_name = getattr(msg, "name", "")
                            if tool_name:
                                return f"Tool {tool_name} completed"
                
                # Generic node processing
                return f"Processing {node_name}..."
        
        return "Processing update..."
    
    # Fallback
    return f"Stream update: {stream_mode}"


def invoke_callback(callback_url: str, message: dict[str, Any]) -> bool:
    """Invoke the callback URL with the given message payload.
    
    Args:
        callback_url: The URL to POST to
        message: The message payload to send (will be JSON serialized)
    
    Returns:
        True if interrupt signal was detected in response, False otherwise
    """
    try:
        import requests
        
        # Serialize the message to ensure it's JSON-serializable
        serialized_message = serialize_for_json(message)
        
        # Print callback payload for debugging/monitoring
        callback_type = serialized_message.get("type", "unknown")
        
        _logger.debug(
            "_invoke_callback - sending to %s (type=%s, payload_keys=%s, message_id=%s)",
            callback_url,
            callback_type,
            list(serialized_message.keys()),
            serialized_message.get("message_id"),
        )
        _logger.debug(
            "_invoke_callback - request payload: %s",
            serialized_message,
        )
        
        response = requests.post(
            callback_url,
            json=serialized_message,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        
        # Log response data - always print full response
        response_text = response.text if hasattr(response, "text") else "N/A"
        _logger.debug(
            "_invoke_callback - response received: status_code=%d, url=%s",
            response.status_code,
            callback_url,
        )
        _logger.debug(
            "_invoke_callback - response text (full): %s",
            response_text,
        )
        
        # Try to parse and log response body, and check for interrupt signal
        interrupted = False
        try:
            response_data = response.json()
            _logger.debug(
                "_invoke_callback - response data (parsed JSON): %s",
                response_data,
            )
            _logger.debug(
                "_invoke_callback - response data type: %s, keys: %s",
                type(response_data).__name__,
                list(response_data.keys()) if isinstance(response_data, dict) else "N/A",
            )
            
            # Check for interrupt signal
            if isinstance(response_data, dict) and response_data.get("action") == "interrupt":
                interrupted = True
                _logger.info(
                    "_invoke_callback - Interrupt signal detected in callback response: url=%s",
                    callback_url,
                )
        except (ValueError, json.JSONDecodeError) as json_err:
            # Response is not JSON - log text instead
            _logger.warning(
                "_invoke_callback - response is not valid JSON: error=%s, response_text=%s",
                str(json_err),
                response_text,
            )
        except Exception as parse_err:  # noqa: BLE001
            _logger.warning(
                "_invoke_callback - error parsing response: error=%s, response_text=%s",
                str(parse_err),
                response_text,
            )
        
        response.raise_for_status()
        return interrupted
    except Exception as e:  # noqa: BLE001
        # Log error but don't raise - we don't want callback failures to stop the stream
        _logger.error(
            "Failed to invoke callback URL %s: %s: %s (payload_keys=%s)",
            callback_url,
            type(e).__name__,
            str(e),
            list(message.keys()) if isinstance(message, dict) else "N/A",
        )
        return False


def send_artifacts_callback(callback_url: str, session_id: str, artifacts: list[dict[str, Any]]) -> None:
    """Send an artifacts callback to notify the front end about uploaded artifacts.
    
    Args:
        callback_url: The callback URL to POST to
        session_id: The session ID (thread_id)
        artifacts: List of artifact metadata dictionaries
    """
    if not callback_url:
        return
    
    callback_payload: dict[str, Any] = {
        "session_id": session_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "type": "artifacts",
        "artifacts": artifacts,
    }
    
    invoke_callback(callback_url, callback_payload)  # Ignore return value for artifacts callbacks


def send_canvas_callback(
    callback_url: str,
    session_id: str,
    canvas: dict[str, Any] | None,
    expert_guidance: str | None = None,
    canvas_update_summary: str | None = None,
    current_round: int = 0,
    last_sync_round: int = 0,
    analysis_timestamp: str | None = None,
) -> None:
    """Send a canvas data callback to notify the frontend about updated canvas data.
    
    This callback is sent when the expert agent synchronizes and updates the canvas.
    
    Args:
        callback_url: The callback URL to POST to
        session_id: The session ID (thread_id)
        canvas: Canvas data dictionary (domain-agnostic JSON structure)
        expert_guidance: Strategic guidance from expert agent (optional)
        canvas_update_summary: Summary of canvas updates in user's language (optional)
        current_round: Current conversation round number
        last_sync_round: Round number when expert last performed analysis
        analysis_timestamp: ISO 8601 timestamp of last expert analysis (optional)
    """
    if not callback_url:
        return
    
    callback_payload: dict[str, Any] = {
        "session_id": session_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "type": "canvas",
        "canvas": canvas,
        "current_round": current_round,
        "last_sync_round": last_sync_round,
    }
    
    # Add optional fields if provided
    if expert_guidance is not None:
        callback_payload["expert_guidance"] = expert_guidance
    if canvas_update_summary is not None:
        callback_payload["canvas_update_summary"] = canvas_update_summary
    if analysis_timestamp is not None:
        callback_payload["analysis_timestamp"] = analysis_timestamp
    
    _logger.info(
        "send_canvas_callback - sending canvas update (session_id=%s, has_canvas=%s, has_summary=%s, round=%d)",
        session_id,
        canvas is not None,
        canvas_update_summary is not None,
        current_round,
    )
    
    invoke_callback(callback_url, callback_payload)  # Ignore return value for canvas callbacks


async def fetch_and_send_canvas_callback(
    callback_url: str,
    session_id: str,
    agent: Any,
    config: dict[str, Any],
    analysis: dict[str, Any] | None = None,
) -> None:
    """Fetch canvas data from agent state and send callback to frontend.
    
    This helper function is called after expert sync completes to send the updated
    canvas data to the frontend via callback.
    
    Args:
        callback_url: The callback URL to POST to
        session_id: The session ID (thread_id)
        agent: The agent instance to fetch state from
        config: Agent configuration dict with thread_id
        analysis: Optional analysis dict from expert sync (preferred - avoids state read timing issues)
    """
    if not callback_url:
        return
    
    try:
        # Prefer using analysis dict directly if provided (avoids state read timing issues)
        if analysis is not None:
            canvas = analysis.get("canvas")
            expert_guidance = analysis.get("expert_guidance")
            canvas_update_summary = analysis.get("canvas_update_summary")
            current_round = analysis.get("last_expert_sync", 0)  # Use last_sync_round as current_round
            last_sync_round = analysis.get("last_expert_sync", 0)
            analysis_timestamp = analysis.get("analysis_timestamp")
            
            # Get current round from state if available
            try:
                current_state_snapshot = await agent.aget_state(config)
                if current_state_snapshot and hasattr(current_state_snapshot, "values"):
                    current_round = current_state_snapshot.values.get("conversation_round", current_round)
            except Exception:  # noqa: BLE001
                pass  # Use last_sync_round as fallback
            
            # Check if canvas indicates an error/fallback state - don't send callback in that case
            if canvas is not None:
                # Skip callback if canvas indicates expert analysis failed (fallback/error state)
                if isinstance(canvas, dict) and canvas.get("status") == "analysis_unavailable":
                    _logger.info(
                        "[CanvasCallback] Expert sync failed (analysis_unavailable), skipping canvas callback (session_id=%s)",
                        session_id,
                    )
                    return
                
                _logger.info(
                    "[CanvasCallback] Sending canvas update from analysis dict (session_id=%s, round=%d, has_summary=%s)",
                    session_id,
                    current_round,
                    canvas_update_summary is not None,
                )
                send_canvas_callback(
                    callback_url=callback_url,
                    session_id=session_id,
                    canvas=canvas,
                    expert_guidance=expert_guidance,
                    canvas_update_summary=canvas_update_summary,
                    current_round=current_round,
                    last_sync_round=last_sync_round,
                    analysis_timestamp=analysis_timestamp,
                )
            else:
                _logger.info(
                    "[CanvasCallback] No canvas in analysis dict (session_id=%s), skipping callback",
                    session_id,
                )
            return
        
        # Fallback: Fetch from state (may have timing issues)
        current_state_snapshot = await agent.aget_state(config)
        if current_state_snapshot and hasattr(current_state_snapshot, "values"):
            state_values = current_state_snapshot.values
            
            # Extract canvas-related fields
            canvas = state_values.get("canvas")
            expert_guidance = state_values.get("expert_guidance")
            canvas_update_summary = state_values.get("canvas_update_summary")
            current_round = state_values.get("conversation_round", 0)
            last_sync_round = state_values.get("last_expert_sync", 0)
            analysis_timestamp = state_values.get("analysis_timestamp")
            
            # Only send callback if canvas exists (expert sync has occurred)
            if canvas is not None:
                # Skip callback if canvas indicates expert analysis failed (fallback/error state)
                if isinstance(canvas, dict) and canvas.get("status") == "analysis_unavailable":
                    _logger.info(
                        "[CanvasCallback] Expert sync failed (analysis_unavailable), skipping canvas callback (session_id=%s)",
                        session_id,
                    )
                    return
                
                _logger.info(
                    "[CanvasCallback] Sending canvas update from state (session_id=%s, round=%d, has_summary=%s)",
                    session_id,
                    current_round,
                    canvas_update_summary is not None,
                )
                send_canvas_callback(
                    callback_url=callback_url,
                    session_id=session_id,
                    canvas=canvas,
                    expert_guidance=expert_guidance,
                    canvas_update_summary=canvas_update_summary,
                    current_round=current_round,
                    last_sync_round=last_sync_round,
                    analysis_timestamp=analysis_timestamp,
                )
            else:
                _logger.info(
                    "[CanvasCallback] No canvas data found in state (session_id=%s), skipping callback",
                    session_id,
                )
        else:
            _logger.info(
                "[CanvasCallback] Could not retrieve state snapshot (session_id=%s), skipping callback",
                session_id,
            )
    except Exception as e:  # noqa: BLE001
        # Log error but don't fail - canvas callback is not critical
        _logger.warning(
            "[CanvasCallback] Error fetching/sending canvas callback (session_id=%s): %s: %s",
            session_id,
            type(e).__name__,
            str(e),
        )


def send_heartbeat(callback_url: str, session_id: str) -> None:
    """Send a heartbeat to the Rails server.
    
    Args:
        callback_url: The base callback URL (will append /heartbeat)
        session_id: The session ID (thread_id) for this conversation
    """
    heartbeat_url = f"{callback_url.rstrip('/')}/heartbeat"
    try:
        import requests
        
        heartbeat_payload = {
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        
        _logger.debug(
            "_send_heartbeat - sending to %s (session_id=%s)",
            heartbeat_url,
            session_id,
        )
        
        response = requests.post(
            heartbeat_url,
            json=heartbeat_payload,
            headers={"Content-Type": "application/json"},
            timeout=10,  # Shorter timeout for heartbeats
        )
        _logger.debug(
            "_send_heartbeat - response status=%d (heartbeat_url=%s)",
            response.status_code,
            heartbeat_url,
        )
        response.raise_for_status()
    except Exception as e:  # noqa: BLE001
        # Log error but don't raise - we don't want heartbeat failures to stop the stream
        _logger.warning(
            "Failed to send heartbeat to %s: %s: %s",
            heartbeat_url,
            type(e).__name__,
            str(e),
        )


def run_async_stream_with_callback(
    agent: Any,
    user_message: str,
    thread_id: str,
    user_id: str,
    metadata: dict[str, Any],
    callback_url: str,
    fallback_agent: Any | None = None,
    docs_dir: str | None = None,
    backend_root_dir: str | None = None,
    expert_agent: Any | None = None,
    use_dual_agent: bool = False,
    expertise_dir: str | None = None,
) -> None:
    """Run the agent stream in a background thread and invoke callback for each update.
    
    This function provides automatic callbacks from stream chunks:
    - Status updates from tool calls and processing
    - Assistant messages
    - Artifacts updates (when artifacts are added to state)
    
    This function runs in a separate thread and creates its own event loop.
    Note: Locking is handled at the endpoint level before starting this thread.
    
    Args:
        agent: The agent instance to stream from
        user_message: The user's message
        thread_id: The thread ID for the conversation
        user_id: The user ID
        metadata: Optional metadata
        callback_url: The callback URL to POST updates to
        fallback_agent: Optional fallback agent to use if primary agent fails
        docs_dir: Docs directory for file path resolution in callback messages
        backend_root_dir: Backend root directory for file path resolution in callback messages
        expert_agent: Optional expert agent for dual-agent mode
        use_dual_agent: Whether dual-agent mode is enabled
        expertise_dir: Directory containing expertise templates
    """
    # Create a new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        async def _stream_and_callback():
            # Initialize error variables at the start so they're always in scope
            primary_error: Exception | None = None
            original_error: Exception | None = None
            
            # Heartbeat configuration
            HEARTBEAT_INTERVAL_SECONDS = 10  # Send heartbeat every 10 seconds
            heartbeat_task: asyncio.Task | None = None
            heartbeat_stop_event = asyncio.Event()
            
            async def _heartbeat_loop():
                """Background task that sends heartbeats periodically."""
                try:
                    while not heartbeat_stop_event.is_set():
                        # Send heartbeat
                        # Use asyncio.to_thread to run the synchronous requests.post in a thread
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(
                            None,
                            send_heartbeat,
                            callback_url,
                            thread_id,
                        )
                        
                        # Wait for the interval or until stop event is set
                        try:
                            await asyncio.wait_for(
                                heartbeat_stop_event.wait(),
                                timeout=HEARTBEAT_INTERVAL_SECONDS,
                            )
                            # If we get here, stop event was set
                            break
                        except asyncio.TimeoutError:
                            # Timeout means interval elapsed, continue loop
                            continue
                except Exception as e:  # noqa: BLE001
                    _logger.warning(
                        "Heartbeat loop error (thread_id=%s): %s: %s",
                        thread_id,
                        type(e).__name__,
                        str(e),
                    )
            
            try:
                _logger.info(
                    "run_async_stream_with_callback - starting (thread_id=%s, callback_url=%s, message_len=%d)",
                    thread_id,
                    callback_url,
                    len(user_message),
                )
                
                # Start heartbeat task
                heartbeat_task = asyncio.create_task(_heartbeat_loop())
                _logger.debug(
                    "run_async_stream_with_callback - started heartbeat task (thread_id=%s, interval=%ds)",
                    thread_id,
                    HEARTBEAT_INTERVAL_SECONDS,
                )
                
                # Set initial state for the agent
                initial_state = {
                    "messages": [HumanMessage(content=user_message)],
                }
                # Include expertise_type in initial_state if provided in metadata
                if "expertise_type" in metadata:
                    initial_state["expertise_type"] = metadata["expertise_type"]
                    _logger.info("[DeepAgent] Including expertise_type in initial_state: %s", metadata["expertise_type"])
                _logger.debug("run_async_stream_with_callback - initial_state keys: %s", list(initial_state.keys()))
                
                config = {
                    "configurable": {"thread_id": thread_id},
                    "metadata": {"user_id": user_id, **metadata},
                }
                _logger.debug("run_async_stream_with_callback - config: %s", config)
                
                # Check if expert sync is needed BEFORE calling facilitator agent (dual-agent mode only)
                # This ensures the facilitator has fresh expert guidance when it processes the request
                if use_dual_agent and expert_agent is not None:
                    try:
                        # Get current state to check if expert sync is needed
                        checkpoint = await agent.checkpointer.aget(config)
                        
                        if checkpoint:
                            state_values = checkpoint.get("channel_values", {})
                            # Build state dict for should_trigger_expert check
                            current_state = {
                                "messages": state_values.get("messages", []),
                                "conversation_round": state_values.get("conversation_round", 0),
                                "expertise_type": metadata.get("expertise_type", "business_cofounder"),
                                **state_values,
                            }
                            
                            # Import here to avoid circular dependency
                            from apps.business_cofounder_api.expert_sync import should_trigger_expert, trigger_and_update_expert
                            
                            if should_trigger_expert(current_state):
                                _logger.info("[DualAgent] Expert sync needed BEFORE facilitator call (async callback, thread_id=%s)", thread_id)
                                expertise_dir_path = Path(expertise_dir) if expertise_dir else None
                                
                                # Run expert sync synchronously (wait for it) so facilitator has fresh guidance
                                try:
                                    analysis = await asyncio.wait_for(
                                        trigger_and_update_expert(
                                            thread_id=thread_id,
                                            state=current_state,
                                            expert_agent=expert_agent,
                                            checkpointer=agent.checkpointer,
                                            expertise_dir=expertise_dir_path,
                                            facilitator_agent=agent,  # Pass facilitator agent for state updates
                                        ),
                                        timeout=60.0,  # 60 second timeout
                                    )
                                    _logger.info("[DualAgent] Expert sync completed BEFORE facilitator call (async callback)")
                                    
                                    # Send canvas callback after expert sync completes (pass analysis directly to avoid state read timing issues)
                                    await fetch_and_send_canvas_callback(
                                        callback_url=callback_url,
                                        session_id=thread_id,
                                        agent=agent,
                                        config=config,
                                        analysis=analysis,  # Pass analysis directly
                                    )
                                except asyncio.TimeoutError:
                                    _logger.error("[DualAgent] Expert sync timed out before facilitator call (async callback, continuing anyway)")
                                except Exception as e:
                                    _logger.error("[DualAgent] Expert sync error before facilitator call (async callback): %s", str(e), exc_info=True)
                                    # Continue anyway - don't block the request
                    except Exception as e:  # noqa: BLE001
                        _logger.error("[DualAgent] Error checking/triggering expert sync before facilitator (async callback): %s", str(e))
                        # Continue anyway - don't block the request
                
                # Get current state from checkpoint to see existing messages
                try:
                    current_state_snapshot = await agent.aget_state(config)
                    if current_state_snapshot and hasattr(current_state_snapshot, "values"):
                        state_messages = current_state_snapshot.values.get("messages", [])
                        message_count = len(state_messages) if isinstance(state_messages, list) else 0
                        first_message = None
                        if state_messages and isinstance(state_messages, list) and len(state_messages) > 0:
                            first_msg = state_messages[0]
                            if hasattr(first_msg, "content"):
                                first_content = str(first_msg.content)
                                first_message = first_content[:200] + ("..." if len(first_content) > 200 else "")
                            else:
                                first_message = str(first_msg)[:200]
                        
                        _logger.warning(
                            "[State Check] Before agent.astream - State has %d messages. First message: %s",
                            message_count,
                            first_message if first_message else "N/A",
                        )
                    else:
                        _logger.warning(
                            "[State Check] Before agent.astream - Could not retrieve state from checkpoint"
                        )
                except Exception as state_err:  # noqa: BLE001
                    _logger.warning(
                        "[State Check] Before agent.astream - Error getting state: %s: %s",
                        type(state_err).__name__,
                        str(state_err),
                    )
                
                chunk_count = 0
                interrupted_from_callback_response = False  # Track if we interrupted due to callback response
                try:
                    async for chunk in agent.astream(
                        initial_state,
                        config=config,
                        stream_mode=["messages", "updates"],
                        subgraphs=True,
                        durability="exit",
                    ):
                        chunk_count += 1
                        # Chunks are tuples: (namespace, stream_mode, data)
                        if not isinstance(chunk, tuple) or len(chunk) != 3:
                            _logger.debug("run_async_stream_with_callback - skipping invalid chunk (not tuple or wrong length): %s", type(chunk))
                            continue
                        
                        namespace, stream_mode, data = chunk
                        _logger.debug(
                            "run_async_stream_with_callback - chunk #%d (namespace=%s, stream_mode=%s, data_type=%s)",
                            chunk_count,
                            namespace,
                            stream_mode,
                            type(data).__name__,
                        )
                        
                        # Check for artifacts in state updates and send callback
                        if stream_mode == "updates" and isinstance(data, dict):
                            # The updates stream structure: {"node_name": {"artifacts": [...], ...}, ...}
                            # Check each node's update_data for artifacts field
                            artifacts_detected = False
                            for node_name, update_data in data.items():
                                # Skip special markers like "__interrupt__"
                                if node_name.startswith("__") and node_name.endswith("__"):
                                    continue
                                
                                if isinstance(update_data, dict) and "artifacts" in update_data:
                                    artifacts_detected = True
                                    break
                            
                            # If artifacts were updated, get the current state to retrieve the full artifacts list
                            if artifacts_detected:
                                try:
                                    # Get current state from agent to retrieve the complete artifacts list
                                    # get_state returns a StateSnapshot with a .values attribute
                                    current_state_snapshot = await agent.aget_state(config)
                                    if current_state_snapshot and hasattr(current_state_snapshot, "values"):
                                        artifacts_list = current_state_snapshot.values.get("artifacts", [])
                                    else:
                                        artifacts_list = []
                                    
                                    if artifacts_list and isinstance(artifacts_list, list) and len(artifacts_list) > 0:
                                        _logger.debug(
                                            "run_async_stream_with_callback - artifacts updated, sending artifacts callback (count=%d)",
                                            len(artifacts_list),
                                        )
                                        send_artifacts_callback(callback_url, thread_id, artifacts_list)
                                except Exception as e:  # noqa: BLE001
                                    # If we can't get state, log but don't fail
                                    _logger.debug(
                                        "run_async_stream_with_callback - could not get state for artifacts: %s: %s",
                                        type(e).__name__,
                                        str(e),
                                    )
                        
                        # Extract message ID from chunk data (for message concatenation in frontend)
                        message_id: str | None = None
                        if stream_mode == "messages" and isinstance(data, tuple) and len(data) >= 1:
                            message = data[0]
                            # Try to get the message ID from various possible attributes
                            if hasattr(message, "id"):
                                message_id = getattr(message, "id", None)
                                _logger.debug("run_async_stream_with_callback - extracted message_id from attribute: %s", message_id)
                            elif isinstance(message, dict):
                                message_id = message.get("id")
                                _logger.debug("run_async_stream_with_callback - extracted message_id from dict: %s", message_id)
                        
                        # Compose a concise message from the chunk data
                        concise_message = compose_concise_callback_message(
                            namespace, stream_mode, data, docs_dir, backend_root_dir
                        )
                        _logger.debug(
                            "run_async_stream_with_callback - concise_message: %s (message_id=%s)",
                            concise_message[:100] if concise_message else None,
                            message_id,
                        )
                        
                        # Skip None messages (e.g., intermediate streaming chunks we want to filter out)
                        if concise_message is None:
                            # Log the raw chunk data structure for debugging (especially for DeepSeek compatibility)
                            _logger.debug(
                                "run_async_stream_with_callback - skipping None concise_message (stream_mode=%s, namespace=%s, data_type=%s)",
                                stream_mode,
                                namespace,
                                type(data).__name__,
                            )
                            if stream_mode == "messages" and isinstance(data, tuple) and len(data) >= 1:
                                message = data[0]
                                msg_type = getattr(message, "type", None) or (message.get("type") if isinstance(message, dict) else None)
                                class_name = type(message).__name__
                                
                                # Get detailed attributes for debugging
                                content = getattr(message, "content", None) or (message.get("content") if isinstance(message, dict) else None)
                                tool_calls = getattr(message, "tool_calls", None) or (message.get("tool_calls") if isinstance(message, dict) else None)
                                response_metadata = getattr(message, "response_metadata", None) or (message.get("response_metadata") if isinstance(message, dict) else None)
                                finish_reason = None
                                if isinstance(response_metadata, dict):
                                    finish_reason = response_metadata.get("finish_reason")
                                invalid_tool_calls = getattr(message, "invalid_tool_calls", None) or (message.get("invalid_tool_calls") if isinstance(message, dict) else None)
                                
                                # Log comprehensive details
                                _logger.debug(
                                    "run_async_stream_with_callback - None concise_message details: msg_type=%s, class_name=%s, "
                                    "has_content=%s, content_type=%s, content_preview=%s, "
                                    "has_tool_calls=%s, tool_calls_type=%s, tool_calls_count=%s, "
                                    "has_invalid_tool_calls=%s, invalid_tool_calls_count=%s, "
                                    "finish_reason=%s, has_response_metadata=%s",
                                    msg_type,
                                    class_name,
                                    content is not None,
                                    type(content).__name__ if content is not None else None,
                                    str(content)[:100] if content is not None else None,
                                    tool_calls is not None,
                                    type(tool_calls).__name__ if tool_calls is not None else None,
                                    len(tool_calls) if isinstance(tool_calls, (list, tuple)) else None,
                                    invalid_tool_calls is not None,
                                    len(invalid_tool_calls) if isinstance(invalid_tool_calls, (list, tuple)) else None,
                                    finish_reason,
                                    response_metadata is not None,
                                )
                            continue
                        
                        # Determine if this is an assistant message or a status update
                        callback_payload: dict[str, Any] = {
                            "session_id": thread_id,
                            "timestamp": datetime.utcnow().isoformat() + "Z",
                        }
                        
                        # Add message_id if available (for frontend message concatenation)
                        if message_id:
                            callback_payload["message_id"] = message_id
                        
                        if concise_message and concise_message.lower().startswith("assistant:"):
                            # Extract the actual message content after "Assistant:"
                            message_content = concise_message[len("Assistant:"):]
                            if message_content:
                                callback_payload["type"] = "message"
                                callback_payload["message_id"] = message_id
                                callback_payload["message"] = message_content
                        else:
                            # This is a status update, not an assistant message
                            callback_payload["type"] = "status"
                            callback_payload["status"] = concise_message
                        
                        # Only invoke callback if we have a message or status
                        if "message" in callback_payload or "status" in callback_payload:
                            _logger.debug(
                                "run_async_stream_with_callback - invoking callback (payload_keys=%s, has_message_id=%s)",
                                list(callback_payload.keys()),
                                "message_id" in callback_payload,
                            )
                            interrupted_from_callback = invoke_callback(callback_url, callback_payload)
                            
                            # If interrupt signal detected, update agent state and break
                            if interrupted_from_callback:
                                interrupted_from_callback_response = True
                                _logger.info(
                                    "run_async_stream_with_callback - interrupt signal detected from callback (thread_id=%s), updating state and breaking stream loop",
                                    thread_id,
                                )
                                try:
                                    # Update agent state to set interrupted flag
                                    await agent.aupdate_state(
                                        config=config,
                                        values={"interrupted": True},
                                    )
                                    _logger.info(
                                        "run_async_stream_with_callback - agent state updated with interrupted=True (thread_id=%s)",
                                        thread_id,
                                    )
                                    break
                                except Exception as e:  # noqa: BLE001
                                    _logger.error(
                                        "run_async_stream_with_callback - failed to update agent state with interrupt: %s: %s (thread_id=%s)",
                                        type(e).__name__,
                                        str(e),
                                        thread_id,
                                    )
                                    # Still break even if state update failed
                                    break
                        else:
                            _logger.debug("run_async_stream_with_callback - skipping callback (no message or status)")
                        
                        # Also check for interruption signal in state (from callback tool)
                        try:
                            current_state_snapshot = await agent.aget_state(config)
                            if current_state_snapshot and hasattr(current_state_snapshot, "values"):
                                interrupted = current_state_snapshot.values.get("interrupted", False)
                                if interrupted:
                                    _logger.info(
                                        "run_async_stream_with_callback - interruption detected in state (thread_id=%s), breaking stream loop",
                                        thread_id,
                                    )
                                    break
                        except Exception as e:  # noqa: BLE001
                            # If we can't get state, log but continue
                            _logger.debug(
                                "run_async_stream_with_callback - could not check interruption state: %s: %s",
                                type(e).__name__,
                                str(e),
                            )
                    
                    # Check if we exited due to interruption
                    # Use both the direct flag and state check for reliability
                    interrupted = interrupted_from_callback_response
                    try:
                        current_state_snapshot = await agent.aget_state(config)
                        if current_state_snapshot and hasattr(current_state_snapshot, "values"):
                            state_interrupted = current_state_snapshot.values.get("interrupted", False)
                            if state_interrupted:
                                interrupted = True
                    except Exception as e:  # noqa: BLE001
                        _logger.debug(
                            "run_async_stream_with_callback - could not check final interruption state: %s: %s",
                            type(e).__name__,
                            str(e),
                        )
                    
                    if interrupted:
                        _logger.info(
                            "run_async_stream_with_callback - stream interrupted (thread_id=%s, from_callback_response=%s), preparing wrap-up",
                            thread_id,
                            interrupted_from_callback_response,
                        )
                        _logger.info("run_async_stream_with_callback - stream interrupted (thread_id=%s), preparing wrap-up", thread_id)
                        
                        # Make a final model call for wrap-up
                        try:
                            # Get current state to see what todos exist
                            current_state_snapshot = await agent.aget_state(config)
                            todos_summary = ""
                            if current_state_snapshot and hasattr(current_state_snapshot, "values"):
                                todos = current_state_snapshot.values.get("todos", [])
                                if todos and isinstance(todos, list):
                                    completed_todos = [t for t in todos if isinstance(t, dict) and t.get("status") == "completed"]
                                    in_progress_todos = [t for t in todos if isinstance(t, dict) and t.get("status") == "in_progress"]
                                    if completed_todos or in_progress_todos:
                                        todos_summary = "\n\nCurrent todo list status:\n"
                                        if completed_todos:
                                            todos_summary += f"Completed ({len(completed_todos)}):\n"
                                            for t in completed_todos:
                                                content = t.get("content", "")
                                                todos_summary += f"- {content}\n"
                                        if in_progress_todos:
                                            todos_summary += f"In progress ({len(in_progress_todos)}):\n"
                                            for t in in_progress_todos:
                                                content = t.get("content", "")
                                                todos_summary += f"- {content}\n"
                            
                            wrap_up_prompt = f"""Your execution has been interrupted by the user. Please provide a brief wrap-up summary of what you have completed so far.

Focus on:
1. What tasks have been completed
2. Any key findings or results from completed work
3. What was in progress when interrupted
4. Any important partial results or insights

Be concise and focus on the most important completed work.{todos_summary}

Provide your wrap-up summary now."""
                            
                            # Make final model call for wrap-up
                            _logger.info(
                                "run_async_stream_with_callback - making wrap-up model call (thread_id=%s)",
                                thread_id,
                            )
                            wrap_up_result = await agent.ainvoke(
                                {"messages": [HumanMessage(content=wrap_up_prompt)]},
                                config=config,
                            )
                            
                            _logger.info(
                                "run_async_stream_with_callback - wrap-up model call completed (thread_id=%s), result_type=%s",
                                thread_id,
                                type(wrap_up_result).__name__,
                            )
                            
                            # Extract wrap-up message from result - get the last AI message
                            wrap_up_message = ""
                            if isinstance(wrap_up_result, dict):
                                messages = wrap_up_result.get("messages", [])
                                _logger.debug(
                                    "run_async_stream_with_callback - wrap-up result has %d messages (thread_id=%s)",
                                    len(messages),
                                    thread_id,
                                )
                                
                                # Find the last AI message (most recent response)
                                for msg in reversed(messages):
                                    msg_type = getattr(msg, "type", None) or (msg.get("type") if isinstance(msg, dict) else None)
                                    
                                    if msg_type == "ai" or isinstance(msg, AIMessage):
                                        # Try multiple ways to get content
                                        content = None
                                        if hasattr(msg, "content"):
                                            content = msg.content
                                        elif hasattr(msg, "text"):
                                            content = msg.text
                                        elif isinstance(msg, dict):
                                            content = msg.get("content") or msg.get("text")
                                        
                                        if content:
                                            wrap_up_message = str(content).strip()
                                            _logger.info(
                                                "run_async_stream_with_callback - extracted wrap-up message (length=%d, thread_id=%s)",
                                                len(wrap_up_message),
                                                thread_id,
                                            )
                                            break
                            
                            if wrap_up_message:
                                # Send wrap-up callback with the final message
                                wrap_up_callback_payload: dict[str, Any] = {
                                    "session_id": thread_id,
                                    "timestamp": datetime.utcnow().isoformat() + "Z",
                                    "type": "message",
                                    "message": wrap_up_message,  # Don't prefix with [Interrupted] - let the model's message speak for itself
                                }
                                _logger.info(
                                    "run_async_stream_with_callback - sending wrap-up callback (thread_id=%s, message_length=%d)",
                                    thread_id,
                                    len(wrap_up_message),
                                )
                                invoke_callback(callback_url, wrap_up_callback_payload)
                            else:
                                _logger.warning(
                                    "run_async_stream_with_callback - wrap-up call completed but no message found (thread_id=%s). Result keys: %s",
                                    thread_id,
                                    list(wrap_up_result.keys()) if isinstance(wrap_up_result, dict) else "N/A",
                                )
                                # Send a fallback message if we couldn't extract the wrap-up
                                fallback_callback: dict[str, Any] = {
                                    "session_id": thread_id,
                                    "timestamp": datetime.utcnow().isoformat() + "Z",
                                    "type": "message",
                                    "message": "Execution was interrupted. Unable to generate wrap-up summary.",
                                }
                                invoke_callback(callback_url, fallback_callback)
                        except Exception as e:  # noqa: BLE001
                            _logger.error(
                                "run_async_stream_with_callback - error during wrap-up: %s: %s (thread_id=%s)",
                                type(e).__name__,
                                str(e),
                                thread_id,
                            )
                            # Send a simple interruption notification even if wrap-up failed
                            try:
                                interruption_callback: dict[str, Any] = {
                                    "session_id": thread_id,
                                    "timestamp": datetime.utcnow().isoformat() + "Z",
                                    "type": "status",
                                    "status": "Execution interrupted by user",
                                }
                                invoke_callback(callback_url, interruption_callback)
                            except Exception:  # noqa: BLE001
                                pass
                    else:
                        _logger.info("run_async_stream_with_callback - stream completed (thread_id=%s, total_chunks=%d)", thread_id, chunk_count)
                    
                    # Stop heartbeat before sending final callback
                    if heartbeat_task:
                        _logger.debug(
                            "run_async_stream_with_callback - stopping heartbeat (thread_id=%s)",
                            thread_id,
                        )
                        heartbeat_stop_event.set()
                        try:
                            await asyncio.wait_for(heartbeat_task, timeout=2.0)
                        except asyncio.TimeoutError:
                            _logger.warning(
                                "run_async_stream_with_callback - heartbeat task did not stop within timeout (thread_id=%s)",
                                thread_id,
                            )
                            heartbeat_task.cancel()
                        except Exception as e:  # noqa: BLE001
                            _logger.warning(
                                "run_async_stream_with_callback - error stopping heartbeat (thread_id=%s): %s",
                                thread_id,
                                str(e),
                            )
                    
                    # Send final callback to inform the Rails application that the stream is completed
                    # and it can accept new input from the user
                    final_callback_payload: dict[str, Any] = {
                        "session_id": thread_id,
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "type": "status",
                        "status": "stream_completed",
                    }
                    _logger.info(
                        "run_async_stream_with_callback - sending final completion callback (thread_id=%s)",
                        thread_id,
                    )
                    invoke_callback(callback_url, final_callback_payload)
                    
                    # Check if expert sync is needed (dual-agent mode only)
                    if use_dual_agent and expert_agent is not None:
                        try:
                            # Read final state from checkpoint after stream completes
                            checkpointer = agent.checkpointer
                            config = {"configurable": {"thread_id": thread_id}}
                            checkpoint = await checkpointer.aget(config)
                            
                            if checkpoint:
                                state_values = checkpoint.get("channel_values", {})
                                # Get expertise_type from metadata first (most recent), then state, then default
                                # This ensures we use the expertise_type from the current request
                                expertise_type = metadata.get("expertise_type")
                                if not expertise_type:
                                    expertise_type = state_values.get("expertise_type")
                                if not expertise_type:
                                    expertise_type = "business_cofounder"
                                
                                # Ensure expertise_type is persisted in state for future expert syncs
                                # Instead of updating checkpoint directly, just ensure it's in the result_state
                                # The expertise_type will be persisted when the agent processes the next message
                                if expertise_type and expertise_type != state_values.get("expertise_type"):
                                    _logger.info(
                                        "[DeepAgent] expertise_type mismatch: state has %s, request has %s (thread_id=%s). Will use %s for expert sync.",
                                        state_values.get("expertise_type", "not set"),
                                        expertise_type,
                                        thread_id,
                                        expertise_type,
                                    )
                                
                                _logger.info("=" * 80)
                                _logger.info("[DeepAgent] Expert Agent Expertise Type:")
                                _logger.info("  From metadata: %s", metadata.get("expertise_type", "not set"))
                                _logger.info("  From state: %s", state_values.get("expertise_type", "not set"))
                                _logger.info("  Final expertise_type: %s", expertise_type)
                                _logger.info("=" * 80)
                                
                                # Create a state dict similar to what ainvoke returns
                                result_state = {
                                    "messages": state_values.get("messages", []),
                                    "conversation_round": state_values.get("conversation_round", 0),
                                    "expertise_type": expertise_type,
                                    **state_values,
                                }
                                
                                # Import here to avoid circular dependency
                                from apps.business_cofounder_api.expert_sync import should_trigger_expert, trigger_and_update_expert
                                
                                if should_trigger_expert(result_state):
                                    _logger.info("[DualAgent] Expert sync needed for thread %s (async callback)", thread_id)
                                    # Run expert sync synchronously (await it) so it completes before event loop closes
                                    # The event loop will close after _stream_and_callback() returns, so we must await here
                                    expertise_dir_path = Path(expertise_dir) if expertise_dir else None
                                    
                                    try:
                                        analysis = await asyncio.wait_for(
                                            trigger_and_update_expert(
                                                thread_id=thread_id,
                                                state=result_state,
                                                expert_agent=expert_agent,
                                                checkpointer=checkpointer,
                                                expertise_dir=expertise_dir_path,
                                                facilitator_agent=agent,  # Pass facilitator agent for state updates
                                            ),
                                            timeout=60.0,  # 60 second timeout for the entire expert sync
                                        )
                                        _logger.info("[DualAgent] Expert sync completed successfully (async, callback)")
                                        
                                        # Send canvas callback after expert sync completes (pass analysis directly to avoid state read timing issues)
                                        await fetch_and_send_canvas_callback(
                                            callback_url=callback_url,
                                            session_id=thread_id,
                                            agent=agent,
                                            config=config,
                                            analysis=analysis,  # Pass analysis directly
                                        )
                                    except asyncio.TimeoutError:
                                        _logger.error("[DualAgent] Expert sync timed out after 60s (async, callback)")
                                    except Exception as e:
                                        _logger.error("[DualAgent] Expert sync error (async, callback): %s", str(e), exc_info=True)
                                        # Continue anyway - don't block the request
                        except Exception as e:  # noqa: BLE001
                            _logger.error("[DualAgent] Error checking/triggering expert sync in async callback: %s", str(e))
                            # Don't fail the request if expert sync fails
                    
                except Exception as e:  # noqa: BLE001
                    # Save the primary exception immediately so it's accessible even if error handling fails
                    primary_error = e
                    original_error = e
                    
                    _logger.warning(
                        "[ModelFallback] Primary agent (qwen) failed during async stream: %s: %s",
                        type(e).__name__,
                        str(e),
                    )
                    _logger.info("[ModelFallback] Falling back to deepseek...")
                    
                    # Try fallback agent if available
                    # fallback_agent is passed as parameter
                    
                    if fallback_agent is not None:
                        try:
                            # Get current state from checkpoint to use summarized messages if available
                            # This ensures the fallback agent uses the same state as the primary agent
                            try:
                                current_state_snapshot = await agent.aget_state(config)
                                if current_state_snapshot and hasattr(current_state_snapshot, "values"):
                                    fallback_state = current_state_snapshot.values
                                    _logger.info(
                                        "[ModelFallback] Using current state from checkpoint for fallback agent (messages: %d)",
                                        len(fallback_state.get("messages", [])) if isinstance(fallback_state.get("messages"), list) else 0,
                                    )
                                else:
                                    # Fallback to initial_state if we can't get current state
                                    fallback_state = initial_state
                                    _logger.warning(
                                        "[ModelFallback] Could not get current state, using initial_state for fallback"
                                    )
                            except Exception as state_error:  # noqa: BLE001
                                # If getting state fails, use initial_state
                                _logger.warning(
                                    "[ModelFallback] Error getting current state for fallback: %s, using initial_state",
                                    str(state_error),
                                )
                                fallback_state = initial_state
                            
                            # Reset chunk count for fallback stream
                            chunk_count = 0
                            async for chunk in fallback_agent.astream(
                                fallback_state,
                                config=config,
                                stream_mode=["messages", "updates"],
                                subgraphs=True,
                                durability="exit",
                            ):
                                chunk_count += 1
                                # Chunks are tuples: (namespace, stream_mode, data)
                                if not isinstance(chunk, tuple) or len(chunk) != 3:
                                    _logger.debug("run_async_stream_with_callback - skipping invalid chunk (not tuple or wrong length): %s", type(chunk))
                                    continue
                                
                                namespace, stream_mode, data = chunk
                                _logger.debug(
                                    "run_async_stream_with_callback - chunk #%d (namespace=%s, stream_mode=%s, data_type=%s)",
                                    chunk_count,
                                    namespace,
                                    stream_mode,
                                    type(data).__name__,
                                )
                                
                                # Check for artifacts in state updates and send callback
                                if stream_mode == "updates" and isinstance(data, dict):
                                    # The updates stream structure: {"node_name": {"artifacts": [...], ...}, ...}
                                    # Check each node's update_data for artifacts field
                                    artifacts_detected = False
                                    for node_name, update_data in data.items():
                                        # Skip special markers like "__interrupt__"
                                        if node_name.startswith("__") and node_name.endswith("__"):
                                            continue
                                        
                                        if isinstance(update_data, dict) and "artifacts" in update_data:
                                            artifacts_detected = True
                                            break
                                    
                                    # If artifacts were updated, get the current state to retrieve the full artifacts list
                                    if artifacts_detected:
                                        try:
                                            # Get current state from agent to retrieve the complete artifacts list
                                            # get_state returns a StateSnapshot with a .values attribute
                                            current_state_snapshot = await fallback_agent.aget_state(config)
                                            if current_state_snapshot and hasattr(current_state_snapshot, "values"):
                                                artifacts_list = current_state_snapshot.values.get("artifacts", [])
                                            else:
                                                artifacts_list = []
                                            
                                            if artifacts_list and isinstance(artifacts_list, list) and len(artifacts_list) > 0:
                                                _logger.debug(
                                                    "run_async_stream_with_callback - artifacts updated, sending artifacts callback (count=%d)",
                                                    len(artifacts_list),
                                                )
                                                send_artifacts_callback(callback_url, thread_id, artifacts_list)
                                        except Exception as e:  # noqa: BLE001
                                            # If we can't get state, log but don't fail
                                            _logger.debug(
                                                "run_async_stream_with_callback - could not get state for artifacts: %s: %s",
                                                type(e).__name__,
                                                str(e),
                                            )
                                
                                # Extract message ID from chunk data (for message concatenation in frontend)
                                message_id: str | None = None
                                if stream_mode == "messages" and isinstance(data, tuple) and len(data) >= 1:
                                    message = data[0]
                                    # Try to get the message ID from various possible attributes
                                    if hasattr(message, "id"):
                                        message_id = getattr(message, "id", None)
                                        _logger.debug("run_async_stream_with_callback - extracted message_id from attribute: %s", message_id)
                                    elif isinstance(message, dict):
                                        message_id = message.get("id")
                                        _logger.debug("run_async_stream_with_callback - extracted message_id from dict: %s", message_id)
                                
                                # Compose a concise message from the chunk data
                                concise_message = compose_concise_callback_message(
                                    namespace, stream_mode, data, docs_dir, backend_root_dir
                                )
                                _logger.debug(
                                    "run_async_stream_with_callback - concise_message: %s (message_id=%s)",
                                    concise_message[:100] if concise_message else None,
                                    message_id,
                                )
                                
                                # Skip None messages (e.g., intermediate streaming chunks we want to filter out)
                                if concise_message is None:
                                    continue
                                
                                # Determine if this is an assistant message or a status update
                                callback_payload: dict[str, Any] = {
                                    "session_id": thread_id,
                                    "timestamp": datetime.utcnow().isoformat() + "Z",
                                }
                                
                                # Add message_id if available (for frontend message concatenation)
                                if message_id:
                                    callback_payload["message_id"] = message_id
                                
                                if concise_message and concise_message.lower().startswith("assistant:"):
                                    # Extract the actual message content after "Assistant:"
                                    message_content = concise_message[len("Assistant:"):]
                                    if message_content:
                                        callback_payload["type"] = "message"
                                        callback_payload["message_id"] = message_id
                                        callback_payload["message"] = message_content
                                else:
                                    # This is a status update, not an assistant message
                                    callback_payload["type"] = "status"
                                    callback_payload["status"] = concise_message
                                
                                # Only invoke callback if we have a message or status
                                if "message" in callback_payload or "status" in callback_payload:
                                    _logger.debug(
                                        "run_async_stream_with_callback - invoking callback (payload_keys=%s, has_message_id=%s)",
                                        list(callback_payload.keys()),
                                        "message_id" in callback_payload,
                                    )
                                    invoke_callback(callback_url, callback_payload)  # Ignore return value for artifacts callbacks
                                else:
                                    _logger.debug("run_async_stream_with_callback - skipping callback (no message or status)")
                            
                            _logger.info("run_async_stream_with_callback - fallback stream completed (thread_id=%s, total_chunks=%d)", thread_id, chunk_count)
                            
                            # Stop heartbeat before sending final callback
                            if heartbeat_task:
                                _logger.debug(
                                    "run_async_stream_with_callback - stopping heartbeat (thread_id=%s)",
                                    thread_id,
                                )
                                heartbeat_stop_event.set()
                                try:
                                    await asyncio.wait_for(heartbeat_task, timeout=2.0)
                                except asyncio.TimeoutError:
                                    _logger.warning(
                                        "run_async_stream_with_callback - heartbeat task did not stop within timeout (thread_id=%s)",
                                        thread_id,
                                    )
                                    heartbeat_task.cancel()
                                except Exception as e:  # noqa: BLE001
                                    _logger.warning(
                                        "run_async_stream_with_callback - error stopping heartbeat (thread_id=%s): %s",
                                        thread_id,
                                        str(e),
                                    )
                            
                            # Send final callback to inform the Rails application that the stream is completed
                            final_callback_payload: dict[str, Any] = {
                                "session_id": thread_id,
                                "timestamp": datetime.utcnow().isoformat() + "Z",
                                "type": "status",
                                "status": "stream_completed",
                            }
                            _logger.info(
                                "run_async_stream_with_callback - sending final completion callback (thread_id=%s)",
                                thread_id,
                            )
                            invoke_callback(callback_url, final_callback_payload)
                            
                            # Check if expert sync is needed after fallback stream completes (dual-agent mode only)
                            # Note: We already checked earlier, but check again after final callback to ensure state is up to date
                            if use_dual_agent and expert_agent is not None:
                                try:
                                    # Read final state from checkpoint after fallback stream completes
                                    checkpointer = fallback_agent.checkpointer if fallback_agent else agent.checkpointer
                                    config = {"configurable": {"thread_id": thread_id}}
                                    checkpoint = await checkpointer.aget(config)
                                    
                                    if checkpoint:
                                        state_values = checkpoint.get("channel_values", {})
                                        # Get expertise_type from metadata first (most recent), then state, then default
                                        expertise_type = metadata.get("expertise_type")
                                        if not expertise_type:
                                            expertise_type = state_values.get("expertise_type")
                                        if not expertise_type:
                                            expertise_type = "business_cofounder"
                                        
                                        # Ensure expertise_type is persisted in state for future expert syncs
                                        # Instead of updating checkpoint directly, just ensure it's in the result_state
                                        # The expertise_type will be persisted when the agent processes the next message
                                        if expertise_type and expertise_type != state_values.get("expertise_type"):
                                            _logger.info(
                                                "[DeepAgent] expertise_type mismatch (fallback path): state has %s, request has %s (thread_id=%s). Will use %s for expert sync.",
                                                state_values.get("expertise_type", "not set"),
                                                expertise_type,
                                                thread_id,
                                                expertise_type,
                                            )
                                        
                                        _logger.info("=" * 80)
                                        _logger.info("[DeepAgent] Expert Agent Expertise Type (fallback path):")
                                        _logger.info("  From metadata: %s", metadata.get("expertise_type", "not set"))
                                        _logger.info("  From state: %s", state_values.get("expertise_type", "not set"))
                                        _logger.info("  Final expertise_type: %s", expertise_type)
                                        _logger.info("=" * 80)
                                        
                                        # Create a state dict similar to what ainvoke returns
                                        result_state = {
                                            "messages": state_values.get("messages", []),
                                            "conversation_round": state_values.get("conversation_round", 0),
                                            "expertise_type": expertise_type,
                                            **state_values,
                                        }
                                        
                                        # Import here to avoid circular dependency
                                        from apps.business_cofounder_api.expert_sync import should_trigger_expert, trigger_and_update_expert
                                        
                                        if should_trigger_expert(result_state):
                                            _logger.info("[DualAgent] Expert sync needed for thread %s (async callback, fallback, after final)", thread_id)
                                            # Run expert sync synchronously (await it) so it completes before event loop closes
                                            expertise_dir_path = Path(expertise_dir) if expertise_dir else None
                                            
                                            try:
                                                analysis = await asyncio.wait_for(
                                                    trigger_and_update_expert(
                                                        thread_id=thread_id,
                                                        state=result_state,
                                                        expert_agent=expert_agent,
                                                        checkpointer=checkpointer,
                                                        expertise_dir=expertise_dir_path,
                                                        facilitator_agent=agent,  # Pass facilitator agent for state updates
                                                    ),
                                                    timeout=60.0,  # 60 second timeout for the entire expert sync
                                                )
                                                _logger.info("[DualAgent] Expert sync completed successfully (async, callback, fallback, after final)")
                                                
                                                # Send canvas callback after expert sync completes (pass analysis directly to avoid state read timing issues)
                                                await fetch_and_send_canvas_callback(
                                                    callback_url=callback_url,
                                                    session_id=thread_id,
                                                    agent=agent,
                                                    config=config,
                                                    analysis=analysis,  # Pass analysis directly
                                                )
                                            except asyncio.TimeoutError:
                                                _logger.error("[DualAgent] Expert sync timed out after 60s (async, callback, fallback, after final)")
                                            except Exception as e:
                                                _logger.error("[DualAgent] Expert sync error (async, callback, fallback, after final): %s", str(e), exc_info=True)
                                                # Continue anyway - don't block the request
                                except Exception as e:  # noqa: BLE001
                                    _logger.error("[DualAgent] Error checking/triggering expert sync in async callback (fallback, after final): %s", str(e))
                                    # Don't fail the request if expert sync fails
                            
                            return  # Successfully completed with fallback agent
                        
                        except Exception as fallback_error:  # noqa: BLE001
                            _logger.error(
                                "[ModelFallback] Fallback agent (deepseek) also failed during async stream: %s: %s",
                                type(fallback_error).__name__,
                                str(fallback_error),
                            )
                            # Fall through to original error handling
                            # Use fallback_error as the original error since both agents failed
                            original_error = fallback_error
                    
                    # Original error handling (if no fallback agent or fallback also failed)
                    # original_error should already be set to primary_error above
                    
                    # Get and print message history for debugging
                    # await log_message_history_for_debugging(agent, config, thread_id)
                    
                    _logger.exception(
                        "Error in async stream with callback thread_id=%s: %s: %s",
                        thread_id,
                        type(original_error).__name__,
                        str(original_error),
                    )
                    
                    # Stop heartbeat on error
                    if heartbeat_task:
                        _logger.debug(
                            "run_async_stream_with_callback - stopping heartbeat due to error (thread_id=%s)",
                            thread_id,
                        )
                        heartbeat_stop_event.set()
                        try:
                            await asyncio.wait_for(heartbeat_task, timeout=2.0)
                        except (asyncio.TimeoutError, Exception):  # noqa: BLE001
                            heartbeat_task.cancel()
                    
                    # Send error to callback as a status update (errors are not assistant messages)
                    try:
                        error_message = f"Error: {type(original_error).__name__}: {str(original_error)}"
                        if env_flag("BC_API_RETURN_TRACEBACK", default=False):
                            error_message += f"\n{traceback.format_exc()}"
                        invoke_callback(
                            callback_url,
                            {
                                "session_id": thread_id,
                                "timestamp": datetime.utcnow().isoformat() + "Z",
                                "type": "status",
                                "status": error_message,
                            },
                        )
                    except Exception as callback_error:  # noqa: BLE001
                        _logger.error(
                            "[ModelFallback] Failed to send error callback: %s: %s",
                            type(callback_error).__name__,
                            str(callback_error),
                        )
            except Exception as outer_e:  # noqa: BLE001
                # Handle any unexpected errors that occur outside the inner try/except blocks
                # (e.g., during initialization, heartbeat setup, or in error handling code itself)
                # This can happen if an exception occurs in the error handling code (lines 1221-1340)
                # where 'e' might not be in scope
                _logger.exception(
                    "Unexpected error in _stream_and_callback (thread_id=%s): %s: %s",
                    thread_id,
                    type(outer_e).__name__,
                    str(outer_e),
                )
                
                # Send error to callback if we have callback_url
                if callback_url:
                    try:
                        error_message = f"Error: {type(outer_e).__name__}: {str(outer_e)}"
                        if env_flag("BC_API_RETURN_TRACEBACK", default=False):
                            error_message += f"\n{traceback.format_exc()}"
                        invoke_callback(
                            callback_url,
                            {
                                "session_id": thread_id,
                                "timestamp": datetime.utcnow().isoformat() + "Z",
                                "type": "status",
                                "status": error_message,
                            },
                        )
                    except Exception:  # noqa: BLE001
                        # If callback fails, just log it
                        _logger.error("Failed to send error callback for outer exception")
                # Ensure cleanup
                if heartbeat_task:
                    heartbeat_stop_event.set()
                    try:
                        await asyncio.wait_for(heartbeat_task, timeout=2.0)
                    except (asyncio.TimeoutError, Exception):  # noqa: BLE001
                        heartbeat_task.cancel()
                # Send error callback
                invoke_callback(
                    callback_url,
                    {
                        "session_id": thread_id,
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "type": "status",
                        "status": f"Error: {type(outer_e).__name__}: {str(outer_e)}",
                    },
                )
        
        loop.run_until_complete(_stream_and_callback())
    
    finally:
        loop.close()
