from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import StreamingResponse
    from fastapi.exceptions import RequestValidationError
    from pydantic import BaseModel, Field, ValidationError, field_validator
except Exception as e:  # noqa: BLE001
    raise RuntimeError(
        "FastAPI is required to run the Business Co-Founder API. "
        "Install it with: `pip install fastapi uvicorn`."
    ) from e

from apps.business_cofounder_api.agent_factory import create_business_cofounder_agent


def _thread_id(*, user_id: str, conversation_id: str) -> str:
    return f"bc::{user_id}::{conversation_id}"

# Use uvicorn's configured logger so output reliably shows up in the terminal.
_logger = logging.getLogger("uvicorn.error")
if _logger.level == logging.NOTSET:
    _logger.setLevel(logging.INFO)


def _env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v in {"1", "true", "TRUE", "yes", "YES", "on", "ON"}


def _truncate(text: str, limit: int) -> str:
    if limit <= 0:
        return text
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... (truncated, {len(text)} chars total)"


def _log_chat_io(*, user_id: str, conversation_id: str, thread_id: str, user_message: str, reply: str) -> None:
    if not _env_flag("BC_API_LOG_CHAT_IO", default=False):
        return
    limit = int(os.environ.get("BC_API_LOG_TRUNCATE_CHARS", "2000"))
    _logger.info(
        "chat_io user_id=%s conversation_id=%s thread_id=%s\nUSER:\n%s\n\nASSISTANT:\n%s",
        user_id,
        conversation_id,
        thread_id,
        _truncate(user_message, limit),
        _truncate(reply, limit),
    )


def _log_debug_state(*, result: dict[str, Any], thread_id: str) -> None:
    """Optional debug logging of milestones/todos/tool calls to diagnose stalled workflows."""
    if not _env_flag("BC_API_LOG_STATE", default=False):
        return

    milestones = {
        "business_idea_complete": bool(result.get("business_idea_complete")),
        "persona_clarified": bool(result.get("persona_clarified")),
        "painpoint_enhanced": bool(result.get("painpoint_enhanced")),
        "early_adopter_identified": bool(result.get("early_adopter_identified")),
        "pitch_created": bool(result.get("pitch_created")),
        "pricing_optimized": bool(result.get("pricing_optimized")),
        "business_model_explored": bool(result.get("business_model_explored")),
    }
    todos = result.get("todos") if isinstance(result.get("todos"), list) else []
    in_progress = None
    for t in todos:
        if isinstance(t, dict) and t.get("status") == "in_progress":
            in_progress = (t.get("content") or "")[:120]
            break

    messages = result.get("messages", [])
    last_ai_tool_calls = None
    for m in reversed(messages):
        if getattr(m, "type", None) == "ai":
            tcs = getattr(m, "tool_calls", None)
            if tcs:
                last_ai_tool_calls = [
                    {"name": tc.get("name"), "args": tc.get("args")}
                    for tc in tcs
                    if isinstance(tc, dict)
                ][:5]
            break

    _logger.info(
        "chat_state thread_id=%s milestones=%s current_todo=%s last_ai_tool_calls=%s",
        thread_id,
        milestones,
        in_progress,
        last_ai_tool_calls,
    )


def _extract_state_values_from_checkpoint(checkpoint: Any) -> dict[str, Any]:
    """Best-effort extraction of LangGraph 'values' from a checkpoint object."""
    if isinstance(checkpoint, dict):
        for k in ("channel_values", "state", "values"):
            v = checkpoint.get(k)
            if isinstance(v, dict):
                return v
        return checkpoint
    return {}

def _extract_text_chunks_from_ai_message(message: Any) -> list[str]:
    """Best-effort extraction of streamed text chunks from an AI message/chunk.

    Different providers expose different shapes:
    - Anthropic: message.content_blocks -> [{"type":"text","text":"..."}]
    - OpenAI-compatible: message.content may be a string OR a list of blocks/dicts
    """
    chunks: list[str] = []

    # Prefer content_blocks (Anthropic-style)
    content_blocks = getattr(message, "content_blocks", None)
    if isinstance(content_blocks, list) and content_blocks:
        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "text":
                continue
            text = block.get("text", "")
            if isinstance(text, str) and text:
                chunks.append(text)
        if chunks:
            return chunks

    # Fallback: content could be str or list
    content = getattr(message, "content", None)
    if isinstance(content, str) and content:
        return [content]
    if isinstance(content, list) and content:
        for item in content:
            if isinstance(item, str) and item:
                chunks.append(item)
                continue
            if isinstance(item, dict):
                # Common patterns: {"type":"text","text":"..."} or {"text":"..."}
                text = item.get("text")
                if isinstance(text, str) and text:
                    chunks.append(text)
        if chunks:
            return chunks

    # Last resort: some message types may expose .text
    text_attr = getattr(message, "text", None)
    if isinstance(text_attr, str) and text_attr:
        return [text_attr]

    return []


def _resolve_write_path(virtual_path: str, docs_dir: str | None = None) -> str:
    """Resolve a virtual filesystem path to the actual write path in docs_dir.
    
    DocsOnlyWriteBackend maps all writes to docs_dir, taking only the filename
    from the virtual path. This function mimics that behavior for display purposes.
    
    Args:
        virtual_path: Virtual path from the agent (e.g., "/path/to/file.md")
        docs_dir: Docs directory where files are actually written
        
    Returns:
        Actual write path in docs_dir, or original path if docs_dir is not available
    """
    if not virtual_path or not docs_dir:
        return virtual_path
    
    try:
        from pathlib import Path
        docs = Path(docs_dir).expanduser().resolve()
        
        # DocsOnlyWriteBackend._map_write_path extracts just the filename
        filename = Path(virtual_path).name or "output.txt"
        actual_path = (docs / filename).resolve()
        
        return str(actual_path)
    except Exception:  # noqa: BLE001
        return virtual_path


def _resolve_read_path(virtual_path: str, backend_root_dir: str | None = None) -> str:
    """Resolve a virtual filesystem path to actual read path.
    
    For reads, the path is resolved relative to the backend's root directory.
    Virtual paths start with "/" and are resolved relative to root_dir.
    
    Args:
        virtual_path: Virtual path (e.g., "/" or "/path/to/file")
        backend_root_dir: Root directory of the backend
        
    Returns:
        Resolved absolute path, or original path if resolution fails
    """
    if not virtual_path:
        return virtual_path
    
    # If path is already absolute (contains directory separators and looks like absolute path)
    # Check if it looks like an absolute path that shouldn't be resolved
    path_obj = None
    try:
        from pathlib import Path
        path_obj = Path(virtual_path)
        # If it's already an absolute path and exists or looks like a real absolute path
        if path_obj.is_absolute() and len(path_obj.parts) > 2:
            # Check if it starts with common absolute path prefixes
            if str(path_obj).startswith(("/Users/", "/home/", "/tmp/", "/var/", "/opt/", "/usr/")):
                return str(path_obj.resolve())
    except Exception:  # noqa: BLE001
        pass
    
    # If no backend_root_dir, return as-is
    if not backend_root_dir:
        return virtual_path
    
    try:
        from pathlib import Path
        root = Path(backend_root_dir).resolve()
        
        # Virtual paths start with "/" - remove it and resolve relative to root_dir
        if virtual_path == "/":
            return str(root)
        
        # Remove leading slash and resolve
        relative_path = virtual_path.lstrip("/")
        if not relative_path:
            return str(root)
        
        resolved = (root / relative_path).resolve()
        return str(resolved)
    except Exception:  # noqa: BLE001
        return virtual_path


def _format_tool_call_progress(tool_name: str, tool_args: dict[str, Any] | None = None, docs_dir: str | None = None, backend_root_dir: str | None = None) -> str:
    """Format a progress message for a tool call, including relevant parameters.
    
    Note: File paths shown are virtual filesystem paths (relative to agent's working directory),
    not absolute local filesystem paths.
    
    Args:
        tool_name: Name of the tool being called
        tool_args: Dictionary of tool call arguments
        
    Returns:
        Formatted progress message string
    """
    if not tool_args:
        return f"Calling {tool_name}..."
    
    # Extract relevant parameters based on tool name
    if tool_name == "read_file":
        file_path = tool_args.get("file_path", "")
        offset = tool_args.get("offset")
        limit = tool_args.get("limit")
        if file_path:
            # For reads, resolve relative to backend root
            actual_path = _resolve_read_path(file_path, backend_root_dir)
            parts = [f"Reading {actual_path}"]
            if offset is not None or limit is not None:
                offset_str = str(offset) if offset is not None else "0"
                limit_str = f", limit={limit}" if limit is not None else ""
                parts.append(f" (offset={offset_str}{limit_str})")
            return "".join(parts)
    
    elif tool_name == "write_file":
        file_path = tool_args.get("file_path", "")
        if file_path:
            # For writes, show actual path in docs_dir (DocsOnlyWriteBackend maps all writes there)
            actual_path = _resolve_write_path(file_path, docs_dir)
            return f"Writing {actual_path}"
    
    elif tool_name == "edit_file":
        file_path = tool_args.get("file_path", "")
        if file_path:
            # For edits, show actual path in docs_dir (DocsOnlyWriteBackend maps all edits there)
            actual_path = _resolve_write_path(file_path, docs_dir)
            return f"Editing {actual_path}"
    
    elif tool_name == "ls" or tool_name == "list_files":
        path = tool_args.get("path", "")
        if path:
            # For directory listing, resolve relative to backend root
            actual_path = _resolve_read_path(path, backend_root_dir)
            return f"Listing files in {actual_path}"
    
    elif tool_name == "glob":
        pattern = tool_args.get("pattern", "")
        path = tool_args.get("path", "")
        if pattern and path:
            # For glob, resolve relative to backend root
            actual_path = _resolve_read_path(path, backend_root_dir)
            return f"Globbing '{pattern}' in {actual_path}"
        elif pattern:
            return f"Globbing '{pattern}'"
        elif path:
            # For glob, resolve relative to backend root
            actual_path = _resolve_read_path(path, backend_root_dir)
            return f"Globbing in {actual_path}"
    
    elif tool_name == "grep":
        pattern = tool_args.get("pattern", "")
        path = tool_args.get("path", "")
        if pattern and path:
            # For grep, resolve relative to backend root
            actual_path = _resolve_read_path(path, backend_root_dir)
            return f"Searching for '{pattern[:30]}...' in {actual_path}"
        elif pattern:
            return f"Searching for '{pattern[:30]}...'"
        elif path:
            # For grep, resolve relative to backend root
            actual_path = _resolve_read_path(path, backend_root_dir)
            return f"Searching in {actual_path}"
    
    elif tool_name == "execute" or tool_name == "shell":
        command = tool_args.get("command", "")
        if command:
            # Truncate long commands
            cmd_preview = command[:50] + "..." if len(command) > 50 else command
            return f"Executing: {cmd_preview}"
    
    elif tool_name == "task":
        subagent_type = tool_args.get("subagent_type", "")
        if subagent_type:
            return f"Delegating to {subagent_type} subagent"
    
    # Default: just tool name
    return f"Calling {tool_name}..."


def _compose_concise_callback_message(
    namespace: Any,
    stream_mode: str,
    data: Any,
    docs_dir: str | None = None,
    backend_root_dir: str | None = None,
) -> str:
    """Compose a concise human-readable message from stream chunk data.
    
    Args:
        namespace: The namespace from the chunk
        stream_mode: Either "messages" or "updates"
        data: The data from the chunk
        docs_dir: Docs directory for file path resolution
        backend_root_dir: Backend root directory for file path resolution
        
    Returns:
        A concise string message describing what's happening
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
                                    import json
                                    tool_args = json.loads(tool_args)
                                except Exception:  # noqa: BLE001
                                    tool_args = {}
                        else:
                            tool_name = getattr(tc, "name", None)
                            tool_args = getattr(tc, "args", None) or getattr(tc, "arguments", None) or {}
                        
                        if tool_name:
                            # Use the progress formatter for better messages
                            try:
                                progress_msg = _format_tool_call_progress(
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
                    text_parts = []
                    for item in content[:2]:
                        if isinstance(item, str) and len(item.strip()) > 5:
                            text_parts.append(item.strip()[:80])
                        elif isinstance(item, dict):
                            text = item.get("text", "")
                            if text and len(text.strip()) > 5:
                                text_parts.append(text.strip()[:80])
                    if text_parts:
                        preview = " ".join(text_parts)
                        if len(preview) > 150:
                            preview = preview[:150] + "..."
                        return f"Assistant: {preview}"
                
                # If we have finish_reason but no tool calls and no content, it's just processing
                if finish_reason:
                    return "Assistant processing..."
                
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
                                    progress_msg = _format_tool_call_progress(
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


def _serialize_for_json(obj: Any) -> Any:
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
        return {k: _serialize_for_json(v) for k, v in obj.items()}
    
    # Handle lists
    if isinstance(obj, list):
        return [_serialize_for_json(item) for item in obj]
    
    # Handle tuples
    if isinstance(obj, tuple):
        return tuple(_serialize_for_json(item) for item in obj)
    
    # Handle basic JSON-serializable types
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    
    # For objects with __dict__, try to serialize it
    if hasattr(obj, "__dict__"):
        try:
            return {k: _serialize_for_json(v) for k, v in obj.__dict__.items()}
        except Exception:  # noqa: BLE001
            pass
    
    # Last resort: convert to string
    return str(obj)


def _invoke_callback(callback_url: str, message: dict[str, Any]) -> None:
    """Invoke the callback URL with the given message payload.
    
    Args:
        callback_url: The URL to POST to
        message: The message payload to send (will be JSON serialized)
    """
    try:
        import requests
        
        # Serialize the message to ensure it's JSON-serializable
        serialized_message = _serialize_for_json(message)
        _logger.debug(
            "_invoke_callback - sending to %s (payload_keys=%s, message_id=%s)",
            callback_url,
            list(serialized_message.keys()),
            serialized_message.get("message_id"),
        )
        
        response = requests.post(
            callback_url,
            json=serialized_message,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        _logger.debug(
            "_invoke_callback - response status=%d (callback_url=%s)",
            response.status_code,
            callback_url,
        )
        response.raise_for_status()
    except Exception as e:  # noqa: BLE001
        # Log error but don't raise - we don't want callback failures to stop the stream
        _logger.warning(
            "Failed to invoke callback URL %s: %s: %s (payload_keys=%s)",
            callback_url,
            type(e).__name__,
            str(e),
            list(message.keys()) if isinstance(message, dict) else "N/A",
        )


def _send_heartbeat(callback_url: str, session_id: str) -> None:
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


def _run_async_stream_with_callback(
    agent: Any,
    user_message: str,
    thread_id: str,
    user_id: str,
    metadata: dict[str, Any],
    callback_url: str,
    docs_dir: str | None = None,
    backend_root_dir: str | None = None,
) -> None:
    """Run the agent stream in a background thread and invoke callback for each update.
    
    This function provides two callback mechanisms:
    1. Automatic callbacks from stream chunks (processed here)
    2. LLM-driven callbacks via the callback tool (handled by CallbackMiddleware)
    
    This function runs in a separate thread and creates its own event loop.
    Note: Locking is handled at the endpoint level before starting this thread.
    
    Args:
        agent: The agent instance to stream from
        user_message: The user's message
        thread_id: The thread ID for the conversation
        user_id: The user ID
        metadata: Optional metadata
        callback_url: The callback URL to POST updates to
        docs_dir: Docs directory for file path resolution in callback messages
        backend_root_dir: Backend root directory for file path resolution in callback messages
    """
    # Create a new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        async def _stream_and_callback():
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
                            _send_heartbeat,
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
                    "_run_async_stream_with_callback - starting (thread_id=%s, callback_url=%s, message_len=%d)",
                    thread_id,
                    callback_url,
                    len(user_message),
                )
                
                # Start heartbeat task
                heartbeat_task = asyncio.create_task(_heartbeat_loop())
                _logger.debug(
                    "_run_async_stream_with_callback - started heartbeat task (thread_id=%s, interval=%ds)",
                    thread_id,
                    HEARTBEAT_INTERVAL_SECONDS,
                )
                
                # Set callback_url in initial state so middleware can access it for LLM-driven callbacks
                # The CallbackMiddleware will set session_id from thread_id in before_agent
                initial_state = {
                    "messages": [HumanMessage(content=user_message)],
                    "callback_url": callback_url,
                }
                _logger.debug("_run_async_stream_with_callback - initial_state keys: %s", list(initial_state.keys()))
                
                config = {
                    "configurable": {"thread_id": thread_id},
                    "metadata": {"user_id": user_id, **metadata},
                }
                _logger.debug("_run_async_stream_with_callback - config: %s", config)
                
                chunk_count = 0
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
                        _logger.debug("_run_async_stream_with_callback - skipping invalid chunk (not tuple or wrong length): %s", type(chunk))
                        continue
                    
                    namespace, stream_mode, data = chunk
                    _logger.debug(
                        "_run_async_stream_with_callback - chunk #%d (namespace=%s, stream_mode=%s, data_type=%s)",
                        chunk_count,
                        namespace,
                        stream_mode,
                        type(data).__name__,
                    )
                    
                    # Extract message ID from chunk data (for message concatenation in frontend)
                    message_id: str | None = None
                    if stream_mode == "messages" and isinstance(data, tuple) and len(data) >= 1:
                        message = data[0]
                        # Try to get the message ID from various possible attributes
                        if hasattr(message, "id"):
                            message_id = getattr(message, "id", None)
                            _logger.debug("_run_async_stream_with_callback - extracted message_id from attribute: %s", message_id)
                        elif isinstance(message, dict):
                            message_id = message.get("id")
                            _logger.debug("_run_async_stream_with_callback - extracted message_id from dict: %s", message_id)
                    
                    # Compose a concise message from the chunk data
                    concise_message = _compose_concise_callback_message(
                        namespace, stream_mode, data, docs_dir, backend_root_dir
                    )
                    _logger.debug(
                        "_run_async_stream_with_callback - concise_message: %s (message_id=%s)",
                        concise_message[:100] if concise_message else None,
                        message_id,
                    )
                    
                    # Skip None messages (e.g., intermediate streaming chunks we want to filter out)
                    if concise_message is None:
                        _logger.debug("_run_async_stream_with_callback - skipping None concise_message")
                        continue
                    
                    # Determine if this is an assistant message or a status update
                    callback_payload: dict[str, Any] = {
                        "session_id": thread_id,
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                    }
                    
                    # Add message_id if available (for frontend message concatenation)
                    if message_id:
                        callback_payload["message_id"] = message_id
                    
                    if concise_message.lower().startswith("assistant:"):
                        # Extract the actual message content after "Assistant:"
                        message_content = concise_message[len("Assistant:"):].strip()
                        if message_content:
                            callback_payload["message_id"] = message_id
                            callback_payload["message"] = message_content
                    else:
                        # This is a status update, not an assistant message
                        callback_payload["status"] = concise_message
                    
                    # Only invoke callback if we have a message or status
                    if "message" in callback_payload or "status" in callback_payload:
                        _logger.debug(
                            "_run_async_stream_with_callback - invoking callback (payload_keys=%s, has_message_id=%s)",
                            list(callback_payload.keys()),
                            "message_id" in callback_payload,
                        )
                        _invoke_callback(callback_url, callback_payload)
                    else:
                        _logger.debug("_run_async_stream_with_callback - skipping callback (no message or status)")
                
                _logger.info("_run_async_stream_with_callback - stream completed (thread_id=%s, total_chunks=%d)", thread_id, chunk_count)
                
                # Stop heartbeat before sending final callback
                if heartbeat_task:
                    _logger.debug(
                        "_run_async_stream_with_callback - stopping heartbeat (thread_id=%s)",
                        thread_id,
                    )
                    heartbeat_stop_event.set()
                    try:
                        await asyncio.wait_for(heartbeat_task, timeout=2.0)
                    except asyncio.TimeoutError:
                        _logger.warning(
                            "_run_async_stream_with_callback - heartbeat task did not stop within timeout (thread_id=%s)",
                            thread_id,
                        )
                        heartbeat_task.cancel()
                    except Exception as e:  # noqa: BLE001
                        _logger.warning(
                            "_run_async_stream_with_callback - error stopping heartbeat (thread_id=%s): %s",
                            thread_id,
                            str(e),
                        )
                
                # Send final callback to inform the Rails application that the stream is completed
                # and it can accept new input from the user
                final_callback_payload: dict[str, Any] = {
                    "session_id": thread_id,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "status": "stream_completed",
                }
                _logger.info(
                    "_run_async_stream_with_callback - sending final completion callback (thread_id=%s)",
                    thread_id,
                )
                _invoke_callback(callback_url, final_callback_payload)
            
            except Exception as e:  # noqa: BLE001
                _logger.exception(
                    "Error in async stream with callback thread_id=%s: %s: %s",
                    thread_id,
                    type(e).__name__,
                    str(e),
                )
                
                # Stop heartbeat on error
                if heartbeat_task:
                    _logger.debug(
                        "_run_async_stream_with_callback - stopping heartbeat due to error (thread_id=%s)",
                        thread_id,
                    )
                    heartbeat_stop_event.set()
                    try:
                        await asyncio.wait_for(heartbeat_task, timeout=2.0)
                    except (asyncio.TimeoutError, Exception):  # noqa: BLE001
                        heartbeat_task.cancel()
                
                # Send error to callback as a status update (errors are not assistant messages)
                error_message = f"Error: {type(e).__name__}: {str(e)}"
                if _env_flag("BC_API_RETURN_TRACEBACK", default=False):
                    error_message += f"\n{traceback.format_exc()}"
                _invoke_callback(
                    callback_url,
                    {
                        "session_id": thread_id,
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "status": error_message,
                    },
                )
        
        loop.run_until_complete(_stream_and_callback())
    
    finally:
        loop.close()


def _summarize_state_values(values: dict[str, Any]) -> dict[str, Any]:
    """Return a small JSON-serializable summary for debugging."""
    milestones = {
        "business_idea_complete": bool(values.get("business_idea_complete")),
        "persona_clarified": bool(values.get("persona_clarified")),
        "painpoint_enhanced": bool(values.get("painpoint_enhanced")),
        "early_adopter_identified": bool(values.get("early_adopter_identified")),
        "pitch_created": bool(values.get("pitch_created")),
        "pricing_optimized": bool(values.get("pricing_optimized")),
        "business_model_explored": bool(values.get("business_model_explored")),
    }
    todos = values.get("todos") if isinstance(values.get("todos"), list) else []
    msg_count = 0
    msgs = values.get("messages")
    if isinstance(msgs, list):
        msg_count = len(msgs)

    written_files: list[str] = []
    if isinstance(msgs, list):
        for m in reversed(msgs[-200:]):  # scan recent messages only
            if not isinstance(m, ToolMessage):
                continue
            if getattr(m, "name", "") != "write_file":
                continue
            content = getattr(m, "content", "")
            if isinstance(content, str) and content.startswith("Updated file "):
                written_files.append(content.replace("Updated file ", "", 1).strip())

    # Keep deterministic order, unique
    written_files = sorted({p for p in written_files if p})

    return {
        "milestones": milestones,
        "todos": todos,
        "message_count": msg_count,
        "written_files": written_files,
    }


class ChatRequest(BaseModel):
    user_id: str = Field(..., description="Upstream server-provided user id")
    message: str = Field(..., description="User message")
    conversation_id: str = Field("default", description="Conversation id (defaults to 'default')")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Optional metadata from upstream")


class ChatResponse(BaseModel):
    user_id: str
    conversation_id: str
    thread_id: str
    reply: str


class ResetRequest(BaseModel):
    user_id: str
    conversation_id: str = "default"


class ResetResponse(BaseModel):
    user_id: str
    conversation_id: str
    thread_id: str
    ok: bool


class CallDeepAgentAsyncRequest(BaseModel):
    user_id: str = Field(..., description="Upstream server-provided user id (accepts int or str, coerced to str)")
    message: str = Field(..., description="User message")
    conversation_id: str = Field("default", description="Conversation id (accepts int or str, coerced to str, defaults to 'default')")
    callback: str | None = Field(None, description="Callback URL to receive update messages (alias: callback_url)")
    callback_url: str | None = Field(None, alias="callback_url", description="Callback URL to receive update messages")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Optional metadata from upstream")
    
    @field_validator("user_id", mode="before")
    @classmethod
    def coerce_user_id_to_str(cls, v: Any) -> str:
        """Coerce user_id to string."""
        return str(v)
    
    @field_validator("conversation_id", mode="before")
    @classmethod
    def coerce_conversation_id_to_str(cls, v: Any) -> str:
        """Coerce conversation_id to string."""
        if v is None:
            return "default"
        return str(v)
    
    def model_post_init(self, __context: Any) -> None:
        """Ensure callback is set from callback_url if needed."""
        if not self.callback and self.callback_url:
            self.callback = self.callback_url
        if not self.callback:
            raise ValueError("Either 'callback' or 'callback_url' must be provided")


class CallDeepAgentAsyncResponse(BaseModel):
    success: bool
    session_id: str = Field(..., description="Session ID (same as thread_id)")
    message: str = Field(..., description="Success or error message")


@dataclass
class _AppState:
    agent: Any
    checkpoints_path: str
    # Ensure the same thread_id is processed serially (avoid checkpoint races).
    thread_locks: dict[str, asyncio.Lock]
    # Docs directory where agent writes files (DocsOnlyWriteBackend constraint)
    docs_dir: str | None = None


app = FastAPI(title="Business Co-Founder Agent API", version="0.1.0")
_state: _AppState | None = None


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> Any:
    """Log validation errors for debugging."""
    body_bytes = await request.body()
    body_str = body_bytes.decode("utf-8", errors="ignore")[:1000]  # Truncate long bodies
    _logger.error(
        "=== VALIDATION ERROR === %s %s",
        request.method,
        request.url.path,
    )
    _logger.error("Validation errors: %s", exc.errors())
    _logger.error("Request body (first 1000 chars): %s", body_str)
    _logger.error("Request headers: %s", dict(request.headers))
    # Return the default FastAPI validation error response
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": body_str[:500]},
    )

# Async networking in CPython uses a threadpool for DNS resolution (loop.getaddrinfo -> run_in_executor).
# Some production hosts have extremely low thread limits; to avoid "can't start new thread" under load,
# we install a tiny, fixed-size default executor and warm it up at startup.
_ASYNCIO_DEFAULT_EXECUTOR: ThreadPoolExecutor | None = None


def _patch_openai_no_thread() -> None:
    """Patch OpenAI python SDK to avoid asyncio.to_thread in ultra-restricted environments.

    Some production environments have extremely low thread limits and crash with:
      RuntimeError: can't start new thread

    The OpenAI SDK's async path calls asyncio.to_thread() for small sync helpers (e.g. platform detection).
    If thread creation is disallowed, that fails. This patch replaces that helper with a direct call.

    Enable with: BC_API_OPENAI_NO_THREAD=1
    """
    if not _env_flag("BC_API_OPENAI_NO_THREAD", default=False):
        return
    try:
        import openai._utils._sync as _openai_sync  # type: ignore
    except Exception:
        return

    async def _to_thread_noop(func, /, *args, **kwargs):  # type: ignore[no-untyped-def]
        return func(*args, **kwargs)

    try:
        _openai_sync.to_thread = _to_thread_noop  # type: ignore[attr-defined]
        _logger.info("Applied BC_API_OPENAI_NO_THREAD patch (openai._utils._sync.to_thread).")
    except Exception:
        return


async def _configure_asyncio_default_executor() -> None:
    global _ASYNCIO_DEFAULT_EXECUTOR
    if _ASYNCIO_DEFAULT_EXECUTOR is not None:
        return
    max_workers = int(os.environ.get("BC_API_ASYNCIO_EXECUTOR_WORKERS", "1"))
    if max_workers < 1:
        max_workers = 1
    _ASYNCIO_DEFAULT_EXECUTOR = ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="bc-asyncio",
    )
    loop = asyncio.get_running_loop()
    loop.set_default_executor(_ASYNCIO_DEFAULT_EXECUTOR)
    # Warm up: forces the executor to start at least one worker thread now,
    # so later DNS lookups don't try (and fail) to spawn a new thread.
    try:
        await loop.run_in_executor(None, lambda: None)
        _logger.info("Configured asyncio default executor (workers=%s).", max_workers)
    except Exception as e:  # noqa: BLE001
        _logger.warning(
            "Failed to warm up asyncio default executor (workers=%s): %s: %s. "
            "Async DNS/networking may fail with 'can't start new thread'.",
            max_workers,
            type(e).__name__,
            str(e),
        )


@app.on_event("startup")
async def _startup() -> None:
    global _state
    await _configure_asyncio_default_executor()
    _patch_openai_no_thread()
    agent, checkpoints_path = create_business_cofounder_agent(agent_id="business_cofounder_agent")
    
    # Extract docs_dir from agent configuration
    # The backend is wrapped in DocsOnlyWriteBackend which constrains all writes to docs_dir.
    from pathlib import Path
    docs_dir = str(Path.home() / ".deepagents" / "business_cofounder_api" / "docs")
    
    _state = _AppState(
        agent=agent,
        checkpoints_path=str(checkpoints_path),
        thread_locks={},
        docs_dir=docs_dir,
    )


@app.get("/health")
async def health() -> dict[str, str]:
    _logger.info("GET /health - received request")
    assert _state is not None
    return {"status": "ok", "checkpoints_path": _state.checkpoints_path}


@app.get("/state")
async def get_state(user_id: str, conversation_id: str = "default") -> dict[str, Any]:
    """Debug endpoint: return current milestone flags + todo list from checkpoint state.

    Disabled by default. Enable with:
      BC_API_ENABLE_STATE_ENDPOINT=1
    """
    _logger.info("GET /state - received request (user_id=%s, conversation_id=%s)", user_id, conversation_id)
    if not _env_flag("BC_API_ENABLE_STATE_ENDPOINT", default=False):
        raise HTTPException(status_code=404, detail="Not found")

    assert _state is not None
    tid = _thread_id(user_id=user_id, conversation_id=conversation_id)

    lock = _state.thread_locks.get(tid)
    if lock is None:
        lock = asyncio.Lock()
        _state.thread_locks[tid] = lock

    async with lock:
        checkpointer = getattr(_state.agent, "checkpointer", None)
        if checkpointer is None or not hasattr(checkpointer, "get_tuple"):
            raise HTTPException(status_code=500, detail="Agent checkpointer not available")

        try:
            ckpt_tuple = await asyncio.to_thread(
                checkpointer.get_tuple, {"configurable": {"thread_id": tid}}
            )
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Failed to read checkpoint: {e!s}") from e

    if ckpt_tuple is None:
        return {
            "thread_id": tid,
            "checkpoints_path": _state.checkpoints_path,
            "milestones": {},
            "todos": [],
            "message_count": 0,
        }

    checkpoint = ckpt_tuple.checkpoint if hasattr(ckpt_tuple, "checkpoint") else ckpt_tuple[1]
    values = _extract_state_values_from_checkpoint(checkpoint)
    summary = _summarize_state_values(values)

    # Also compute the canonical todo list derived from milestone flags.
    # This helps debug cases where `todos` in the checkpoint is stale (e.g. if an LLM rewrote it,
    # or if milestones were updated at the end of a run and todos will only be refreshed on the next run).
    try:
        from deepagents.middleware.business_idea_development import BusinessIdeaDevelopmentMiddleware

        canonical_todos = BusinessIdeaDevelopmentMiddleware()._generate_todos_from_state(values)  # noqa: SLF001
    except Exception:  # noqa: BLE001
        canonical_todos = []

    return {
        "thread_id": tid,
        "checkpoints_path": _state.checkpoints_path,
        **summary,
        "todos_canonical": canonical_todos,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    _logger.info(
        "POST /chat - received request (user_id=%s, conversation_id=%s, message_len=%d)",
        req.user_id,
        req.conversation_id,
        len(req.message),
    )
    assert _state is not None
    tid = _thread_id(user_id=req.user_id, conversation_id=req.conversation_id)

    lock = _state.thread_locks.get(tid)
    if lock is None:
        lock = asyncio.Lock()
        _state.thread_locks[tid] = lock

    async with lock:
        try:
            result = await _state.agent.ainvoke(
                {"messages": [HumanMessage(content=req.message)]},
                {
                    "configurable": {"thread_id": tid},
                    "metadata": {"user_id": req.user_id, **(req.metadata or {})},
                },
            )
        except Exception as e:  # noqa: BLE001
            # Print a full traceback to server logs for local debugging.
            _logger.exception(
                "POST /chat failed user_id=%s conversation_id=%s thread_id=%s error_type=%s error_message=%s",
                req.user_id,
                req.conversation_id,
                tid,
                type(e).__name__,
                str(e),
            )

            # Optionally include traceback in HTTP response (useful for local dev; avoid enabling in prod).
            detail: dict[str, Any] = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "thread_id": tid,
            }
            if _env_flag("BC_API_RETURN_TRACEBACK", default=False):
                detail["traceback"] = traceback.format_exc()

            # Internal API: return a helpful message for debugging.
            raise HTTPException(
                status_code=502,
                detail=detail,
            ) from e

    messages = result.get("messages", [])
    ai_messages = [m for m in messages if getattr(m, "type", None) == "ai"]
    reply = str(ai_messages[-1].content) if ai_messages else ""

    _log_chat_io(
        user_id=req.user_id,
        conversation_id=req.conversation_id,
        thread_id=tid,
        user_message=req.message,
        reply=reply,
    )
    _log_debug_state(result=result, thread_id=tid)

    return ChatResponse(
        user_id=req.user_id,
        conversation_id=req.conversation_id,
        thread_id=tid,
        reply=reply,
    )


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
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
    """
    _logger.info(
        "POST /chat/stream - received request (user_id=%s, conversation_id=%s, message_len=%d)",
        req.user_id,
        req.conversation_id,
        len(req.message),
    )
    assert _state is not None
    tid = _thread_id(user_id=req.user_id, conversation_id=req.conversation_id)

    lock = _state.thread_locks.get(tid)
    if lock is None:
        lock = asyncio.Lock()
        _state.thread_locks[tid] = lock

    async def _gen():
        import time
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
            try:
                async for chunk in _state.agent.astream(
                    {"messages": [HumanMessage(content=req.message)]},
                    config={
                        "configurable": {"thread_id": tid},
                        "metadata": {"user_id": req.user_id, **(req.metadata or {})},
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
                                import time
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
                                                    from pathlib import Path
                                                    docs_dir = _state.docs_dir if _state else None
                                                    backend_root_dir = str(Path.cwd()) if _state else None
                                                    progress_msg = _format_tool_call_progress(tool_name, cached_args, docs_dir, backend_root_dir)
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
                                                                    from pathlib import Path
                                                                    docs_dir = _state.docs_dir if _state else None
                                                                    backend_root_dir = str(Path.cwd()) if _state else None
                                                                    progress_msg = _format_tool_call_progress(tool_name, tool_args, docs_dir, backend_root_dir)
                                                                    payload = {"type": "progress", "message": progress_msg}
                                                                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                                                                    last_progress_update = now
                                                        break  # Only process first message with tool_calls
                                            # Handle AIMessage objects (not dicts)
                                            elif hasattr(msg, "tool_calls") and msg.tool_calls:
                                                import time
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
                                                            from pathlib import Path
                                                            docs_dir = _state.docs_dir if _state else None
                                                            backend_root_dir = str(Path.cwd()) if _state else None
                                                            progress_msg = _format_tool_call_progress(tool_name, tool_args, docs_dir, backend_root_dir)
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
                                                        from pathlib import Path
                                                        docs_dir = _state.docs_dir if _state else None
                                                        backend_root_dir = str(Path.cwd()) if _state else None
                                                        progress_msg = _format_tool_call_progress(tool_name, tool_args, docs_dir, backend_root_dir)
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
                                if _env_flag("BC_API_STREAM_DEBUG", default=False):
                                    _logger.debug(
                                        "Tool call in AI message: name=%s, id=%s, args=%s, tc_keys=%s",
                                        tool_name,
                                        tool_call_id,
                                        tool_args,
                                        list(tc.keys()) if isinstance(tc, dict) else [],
                                    )
                                
                                if tool_name:
                                    from pathlib import Path
                                    docs_dir = _state.docs_dir if _state else None
                                    backend_root_dir = str(Path.cwd()) if _state else None
                                    progress_msg = _format_tool_call_progress(tool_name, tool_args, docs_dir, backend_root_dir)
                                    payload = {"type": "progress", "message": progress_msg}
                                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

                    for text in _extract_text_chunks_from_ai_message(message):
                        final_parts.append(text)
                        delta_count += 1
                        payload = {"type": "delta", "text": text}
                        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

                final_text = "".join(final_parts).strip()

                # Fallback: if no text was streamed but an HTML file was written, read it and return its contents.
                if not final_text and last_written_html_path:
                    try:
                        from pathlib import Path

                        pth = Path(last_written_html_path)
                        if pth.exists() and pth.is_file() and pth.stat().st_size <= 2 * 1024 * 1024:
                            final_text = pth.read_text(encoding="utf-8", errors="replace").strip()
                    except Exception:  # noqa: BLE001
                        pass

                if _env_flag("BC_API_STREAM_DEBUG", default=False):
                    _logger.info(
                        "chat_stream_debug thread_id=%s delta_count=%s seen_message_types=%s last_written_html=%s final_len=%s",
                        tid,
                        delta_count,
                        seen_types,
                        last_written_html_path,
                        len(final_text),
                    )
                _log_chat_io(
                    user_id=req.user_id,
                    conversation_id=req.conversation_id,
                    thread_id=tid,
                    user_message=req.message,
                    reply=final_text,
                )
                yield f"data: {json.dumps({'type':'final','text':final_text}, ensure_ascii=False)}\n\n"
            except Exception as e:  # noqa: BLE001
                _logger.exception(
                    "POST /chat/stream failed user_id=%s conversation_id=%s thread_id=%s error_type=%s error_message=%s",
                    req.user_id,
                    req.conversation_id,
                    tid,
                    type(e).__name__,
                    str(e),
                )
                detail: dict[str, Any] = {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "thread_id": tid,
                }
                if _env_flag("BC_API_RETURN_TRACEBACK", default=False):
                    detail["traceback"] = traceback.format_exc()
                yield f"data: {json.dumps({'type':'error','detail':detail}, ensure_ascii=False)}\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream; charset=utf-8")


@app.post("/deep_agent/call_async", response_model=CallDeepAgentAsyncResponse)
async def call_deep_agent_async(req: CallDeepAgentAsyncRequest) -> CallDeepAgentAsyncResponse:
    """Start an async agent stream in a background thread and return immediately.
    
    This endpoint accepts a callback URL and starts the agent streaming in a background thread.
    Each update from the stream will be POSTed to the callback URL with the message parameter
    containing the update data.
    
    Returns immediately with a session_id (same as thread_id) and success status.
    The actual streaming happens asynchronously in a separate thread.
    """
    # Log immediately when function is called (validation passed)
    _logger.info("=== call_deep_agent_async CALLED ===")
    _logger.info(
        "POST /deep_agent/call_async - received request (user_id=%s, conversation_id=%s, message_len=%d, callback=%s, metadata=%s)",
        req.user_id,
        req.conversation_id,
        len(req.message),
        req.callback,
        req.metadata,
    )
    _logger.debug(
        "POST /deep_agent/call_async - full request: user_id=%r, conversation_id=%r, message=%r, callback=%r, metadata=%r",
        req.user_id,
        req.conversation_id,
        req.message[:100] if req.message else None,
        req.callback,
        req.metadata,
    )
    assert _state is not None
    tid = _thread_id(user_id=req.user_id, conversation_id=req.conversation_id)
    _logger.debug("POST /deep_agent/call_async - thread_id=%s", tid)

    # Get or create lock for this thread_id (for consistency with other endpoints)
    # We don't hold it during execution since we're running in background
    lock = _state.thread_locks.get(tid)
    if lock is None:
        lock = asyncio.Lock()
        _state.thread_locks[tid] = lock

    try:
        # Get paths for file resolution in callback messages
        from pathlib import Path
        docs_dir = _state.docs_dir if _state else None
        backend_root_dir = str(Path.cwd()) if _state else None
        
        _logger.info(
            "POST /deep_agent/call_async - starting background thread (thread_id=%s, callback_url=%s, docs_dir=%s)",
            tid,
            req.callback,
            docs_dir,
        )
        
        # Start background thread to run the async stream
        thread = threading.Thread(
            target=_run_async_stream_with_callback,
            args=(
                _state.agent,
                req.message,
                tid,
                req.user_id,
                req.metadata or {},
                req.callback,
                docs_dir,
                backend_root_dir,
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


@app.post("/reset", response_model=ResetResponse)
async def reset(req: ResetRequest) -> ResetResponse:
    """Reset a user's conversation by deleting the thread from the checkpointer."""
    _logger.info(
        "POST /reset - received request (user_id=%s, conversation_id=%s)",
        req.user_id,
        req.conversation_id,
    )
    assert _state is not None
    tid = _thread_id(user_id=req.user_id, conversation_id=req.conversation_id)

    # The checkpointer is held by the agent graph; we access it via .checkpointer (best-effort).
    checkpointer = getattr(_state.agent, "checkpointer", None)
    if checkpointer is not None and hasattr(checkpointer, "delete_thread"):
        checkpointer.delete_thread(tid)

    # Also drop the in-process lock (fresh start)
    _state.thread_locks.pop(tid, None)

    return ResetResponse(user_id=req.user_id, conversation_id=req.conversation_id, thread_id=tid, ok=True)


