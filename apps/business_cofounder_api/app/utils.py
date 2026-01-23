"""Utility functions for the API."""

from __future__ import annotations

import logging
import os
from typing import Any

# Use uvicorn's configured logger so output reliably shows up in the terminal.
_logger = logging.getLogger("uvicorn.error")
_logger.setLevel(logging.INFO)


def thread_id(*, user_id: str, conversation_id: str) -> str:
    """Generate thread ID from user_id and conversation_id."""
    return f"bc::{user_id}::{conversation_id}"


def env_flag(name: str, default: bool = False) -> bool:
    """Parse environment variable as boolean flag."""
    v = os.environ.get(name)
    if v is None:
        return default
    return v in {"1", "true", "TRUE", "yes", "YES", "on", "ON"}


def truncate(text: str, limit: int) -> str:
    """Truncate text to a character limit."""
    if limit <= 0:
        return text
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... (truncated, {len(text)} chars total)"


def log_chat_io(*, user_id: str, conversation_id: str, thread_id: str, user_message: str, reply: str) -> None:
    """Log chat input/output if enabled."""
    if not env_flag("BC_API_LOG_CHAT_IO", default=False):
        return
    limit = int(os.environ.get("BC_API_LOG_TRUNCATE_CHARS", "2000"))
    _logger.info(
        "chat_io user_id=%s conversation_id=%s thread_id=%s\nUSER:\n%s\n\nASSISTANT:\n%s",
        user_id,
        conversation_id,
        thread_id,
        truncate(user_message, limit),
        truncate(reply, limit),
    )


def log_debug_state(*, result: dict[str, Any], thread_id: str) -> None:
    """Optional debug logging of milestones/todos/tool calls to diagnose stalled workflows."""
    if not env_flag("BC_API_LOG_STATE", default=False):
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


async def log_message_history_for_debugging(
    agent: Any,
    config: dict[str, Any],
    thread_id: str,
) -> None:
    """Log detailed message history for debugging when errors occur.
    
    Args:
        agent: The agent instance to get state from
        config: The runtime config containing thread_id
        thread_id: The thread ID for logging context
    """
    try:
        current_state_snapshot = await agent.aget_state(config)
        if current_state_snapshot and hasattr(current_state_snapshot, "values"):
            messages = current_state_snapshot.values.get("messages", [])
            _logger.error(
                "Error in async stream with callback thread_id=%s - Message history (count=%d):",
                thread_id,
                len(messages),
            )
            
            # Check for duplicate consecutive human messages
            for i in range(len(messages) - 1):
                current_msg = messages[i]
                next_msg = messages[i + 1]
                current_type = getattr(current_msg, "type", None) or (current_msg.get("type") if isinstance(current_msg, dict) else "unknown")
                next_type = getattr(next_msg, "type", None) or (next_msg.get("type") if isinstance(next_msg, dict) else "unknown")
                
                if current_type == "human" and next_type == "human":
                    current_content = getattr(current_msg, "content", None) or (current_msg.get("content") if isinstance(current_msg, dict) else "")
                    next_content = getattr(next_msg, "content", None) or (next_msg.get("content") if isinstance(next_msg, dict) else "")
                    if current_content == next_content:
                        _logger.error(
                            "    ⚠️  DUPLICATE CONSECUTIVE HUMAN MESSAGES detected at Message[%d] and Message[%d]: %s",
                            i,
                            i + 1,
                            str(current_content)[:100] if current_content else None,
                        )
            
            for i, msg in enumerate(messages):
                msg_type = getattr(msg, "type", None) or (msg.get("type") if isinstance(msg, dict) else "unknown")
                msg_content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else "")
                tool_calls = getattr(msg, "tool_calls", None) or (msg.get("tool_calls") if isinstance(msg, dict) else None)
                tool_call_id = getattr(msg, "tool_call_id", None) or (msg.get("tool_call_id") if isinstance(msg, dict) else None)
                
                # Extract tool_call_ids from tool_calls if it's an AI message
                tool_call_ids_in_msg = []
                if tool_calls:
                    if isinstance(tool_calls, list):
                        for tc in tool_calls:
                            if isinstance(tc, dict):
                                tc_id = tc.get("id") or tc.get("tool_call_id")
                                if tc_id:
                                    tool_call_ids_in_msg.append(tc_id)
                            elif hasattr(tc, "id"):
                                tool_call_ids_in_msg.append(tc.id)
                
                _logger.error(
                    "  Message[%d]: type=%s, tool_call_id=%s, tool_call_ids_in_tool_calls=%s, content_preview=%s",
                    i,
                    msg_type,
                    tool_call_id,
                    tool_call_ids_in_msg if tool_call_ids_in_msg else None,
                    str(msg_content)[:100] if msg_content else None,
                )
                
                # Check if this is an AI message with tool_calls - verify each has a response
                if msg_type in ("ai", "AIMessage") and tool_calls:
                    # Find all subsequent tool messages to see which tool_call_ids are covered
                    subsequent_tool_call_ids = set()
                    for j in range(i + 1, len(messages)):
                        next_msg = messages[j]
                        next_msg_type = getattr(next_msg, "type", None) or (next_msg.get("type") if isinstance(next_msg, dict) else "unknown")
                        if next_msg_type == "tool":
                            next_tool_call_id = getattr(next_msg, "tool_call_id", None) or (next_msg.get("tool_call_id") if isinstance(next_msg, dict) else None)
                            if next_tool_call_id:
                                subsequent_tool_call_ids.add(next_tool_call_id)
                    
                    # Check which tool_call_ids from this message don't have responses
                    missing_responses = []
                    for tc_id in tool_call_ids_in_msg:
                        if tc_id not in subsequent_tool_call_ids:
                            missing_responses.append(tc_id)
                    
                    if missing_responses:
                        _logger.error(
                            "    ⚠️  Message[%d] has tool_calls with missing responses: %s",
                            i,
                            missing_responses,
                        )
    except Exception as state_err:  # noqa: BLE001
        _logger.error(
            "Failed to get message history for debugging: %s: %s",
            type(state_err).__name__,
            str(state_err),
        )


def extract_state_values_from_checkpoint(checkpoint: Any) -> dict[str, Any]:
    """Best-effort extraction of LangGraph 'values' from a checkpoint object."""
    if isinstance(checkpoint, dict):
        for k in ("channel_values", "state", "values"):
            v = checkpoint.get(k)
            if isinstance(v, dict):
                return v
        return checkpoint
    return {}


def extract_text_chunks_from_ai_message(message: Any) -> list[str]:
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


def resolve_write_path(virtual_path: str, backend_root: str | None = None) -> str:
    """Resolve a virtual filesystem path to the actual write path.
    
    With virtual_mode=True, virtual paths are resolved relative to backend_root.
    For example, /docs/file.md resolves to backend_root/docs/file.md.
    
    Args:
        virtual_path: Virtual path from the agent (e.g., "/docs/file.md")
        backend_root: Backend root directory (defaults to base_dir)
        
    Returns:
        Actual write path, or original path if backend_root is not available
    """
    if not virtual_path:
        return virtual_path
    
    if not backend_root:
        # Default to the standard base_dir location
        from pathlib import Path
        backend_root = str(Path.home() / ".deepagents" / "business_cofounder_api")
    
    try:
        from pathlib import Path
        root = Path(backend_root).expanduser().resolve()
        
        # Remove leading slash and resolve relative to backend root
        # e.g., /docs/file.md -> backend_root/docs/file.md
        relative_path = virtual_path.lstrip("/")
        actual_path = (root / relative_path).resolve()
        
        # Safety check: ensure path is within backend root
        try:
            actual_path.relative_to(root)
        except ValueError:
            # If path escapes root, just return the virtual path
            return virtual_path
        
        return str(actual_path)
    except Exception:  # noqa: BLE001
        return virtual_path


def resolve_read_path(virtual_path: str, backend_root_dir: str | None = None) -> str:
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


def format_tool_call_progress(tool_name: str, tool_args: dict[str, Any] | None = None, docs_dir: str | None = None, backend_root_dir: str | None = None) -> str:
    """Format a progress message for a tool call, including relevant parameters.
    
    Note: File paths shown are virtual filesystem paths (relative to agent's working directory),
    not absolute local filesystem paths.
    
    Args:
        tool_name: Name of the tool being called
        tool_args: Dictionary of tool call arguments
        docs_dir: Docs directory (unused, kept for compatibility)
        backend_root_dir: Backend root directory for path resolution
        
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
            actual_path = resolve_read_path(file_path, backend_root_dir)
            parts = [f"Reading {actual_path}"]
            if offset is not None or limit is not None:
                offset_str = str(offset) if offset is not None else "0"
                limit_str = f", limit={limit}" if limit is not None else ""
                parts.append(f" (offset={offset_str}{limit_str})")
            return "".join(parts)
    
    elif tool_name == "write_file":
        file_path = tool_args.get("file_path", "")
        if file_path:
            # For writes, resolve virtual path to actual path relative to backend root
            actual_path = resolve_write_path(file_path, backend_root_dir)
            return f"Writing {actual_path}"
    
    elif tool_name == "edit_file":
        file_path = tool_args.get("file_path", "")
        if file_path:
            # For edits, resolve virtual path to actual path relative to backend root
            actual_path = resolve_write_path(file_path, backend_root_dir)
            return f"Editing {actual_path}"
    
    elif tool_name == "ls" or tool_name == "list_files":
        path = tool_args.get("path", "")
        if path:
            # For directory listing, resolve relative to backend root
            actual_path = resolve_read_path(path, backend_root_dir)
            return f"Listing files in {actual_path}"
    
    elif tool_name == "glob":
        pattern = tool_args.get("pattern", "")
        path = tool_args.get("path", "")
        if pattern and path:
            # For glob, resolve relative to backend root
            actual_path = resolve_read_path(path, backend_root_dir)
            return f"Globbing '{pattern}' in {actual_path}"
        elif pattern:
            return f"Globbing '{pattern}'"
        elif path:
            # For glob, resolve relative to backend root
            actual_path = resolve_read_path(path, backend_root_dir)
            return f"Globbing in {actual_path}"
    
    elif tool_name == "grep":
        pattern = tool_args.get("pattern", "")
        path = tool_args.get("path", "")
        if pattern and path:
            # For grep, resolve relative to backend root
            actual_path = resolve_read_path(path, backend_root_dir)
            return f"Searching for '{pattern[:30]}...' in {actual_path}"
        elif pattern:
            return f"Searching for '{pattern[:30]}...'"
        elif path:
            # For grep, resolve relative to backend root
            actual_path = resolve_read_path(path, backend_root_dir)
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


def summarize_state_values(values: dict[str, Any]) -> dict[str, Any]:
    """Summarize state values for logging/debugging (truncate long content)."""
    summary: dict[str, Any] = {}
    
    # Copy basic fields
    for key in ["conversation_round", "needs_expert_sync", "last_expert_sync", "expertise_type"]:
        if key in values:
            summary[key] = values[key]
    
    # Summarize messages (limit to last 3)
    messages = values.get("messages", [])
    if messages:
        summary["message_count"] = len(messages)
        summary["last_3_messages"] = []
        for msg in messages[-3:]:
            msg_type = getattr(msg, "type", None) or (msg.get("type") if isinstance(msg, dict) else "unknown")
            content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else "")
            content_preview = str(content)[:100] + "..." if content and len(str(content)) > 100 else str(content) if content else None
            summary["last_3_messages"].append({"type": msg_type, "content_preview": content_preview})
    
    # Summarize canvas (just keys, not full content)
    canvas = values.get("canvas")
    if canvas:
        summary["canvas_keys"] = list(canvas.keys()) if isinstance(canvas, dict) else "not_a_dict"
    
    # Summarize expert_guidance (truncate)
    expert_guidance = values.get("expert_guidance")
    if expert_guidance:
        summary["expert_guidance_preview"] = str(expert_guidance)[:200] + "..." if len(str(expert_guidance)) > 200 else str(expert_guidance)
    
    return summary
