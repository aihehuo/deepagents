"""Utilities for Wu Tanchang API."""

from __future__ import annotations

from typing import Any
from langchain_core.messages import ToolMessage


def thread_id(
    *, agent_name: str = "default", user_id: str, conversation_id: str
) -> str:
    """Build LangGraph thread id with agent name prefix."""
    return f"wt::{agent_name}::{user_id}::{conversation_id}"


def get_progress_status(
    subgraph_path: Any,
    stream_mode: str,
    data: Any,
    *,
    workspace_name: str = "workspace",
) -> str | None:
    """Extract a user-friendly progress status message from a stream chunk.

    Args:
        subgraph_path: The subgraph path (empty for main agent).
        stream_mode: The stream mode ("messages" or "updates").
        data: The chunk data.
        workspace_name: The name of the active workspace.

    Returns:
        A user-friendly status message, or None if no status update.
    """
    if stream_mode != "updates" or not isinstance(data, dict):
        return None

    def has_tool_call(update_data: dict[str, Any], tool_names: set[str]) -> bool:
        # Check tool_calls field
        tool_calls = update_data.get("tool_calls") or []
        for tc in tool_calls:
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
            if name in tool_names:
                return True
        # Check messages field
        for msg in update_data.get("messages") or []:
            # Check AIMessage / AIMessageChunk tool_calls
            if hasattr(msg, "tool_calls"):
                for tc in msg.tool_calls or []:
                    if tc.get("name") in tool_names:
                        return True
            elif isinstance(msg, dict) and "tool_calls" in msg:
                for tc in msg.get("tool_calls") or []:
                    if tc.get("name") in tool_names:
                        return True
            # Check ToolMessage name
            if isinstance(msg, ToolMessage) or (
                isinstance(msg, dict) and msg.get("type") == "tool"
            ):
                name = (
                    msg.get("name")
                    if isinstance(msg, dict)
                    else getattr(msg, "name", "")
                )
                if name in tool_names:
                    return True
        return False

    is_subagent = bool(subgraph_path)

    for node_name, update_data in data.items():
        if node_name.startswith("__") and node_name.endswith("__"):
            if node_name == "__start__":
                if is_subagent:
                    return "正在启动知识库检索与分析..."
                else:
                    return "正在规划回复内容..."
            continue

        if not isinstance(update_data, dict):
            continue

        if is_subagent:
            # Subagent execution progress
            # Check if kb_semantic_search is being called
            if has_tool_call(update_data, {"kb_semantic_search"}):
                if "1" in workspace_name:
                    return "正在对创业案例与方法论进行语义检索..."
                else:
                    return "正在对餐饮案例库进行语义检索..."

            # Check if other tools are called (e.g. filesystem tools or skill tools)
            tool_calls = update_data.get("tool_calls") or []
            tool_msgs = [
                msg
                for msg in (update_data.get("messages") or [])
                if isinstance(msg, ToolMessage)
                or (isinstance(msg, dict) and msg.get("type") == "tool")
            ]
            if tool_calls or tool_msgs:
                if "1" in workspace_name:
                    return "正在调阅创业案例详情并分析商业动作..."
                else:
                    return "正在调阅餐饮案例详情并分析商业动作..."

            # If the agent node is running without tool calls, it's synthesizing the response
            if node_name == "agent":
                return "正在整理匹配的案例与商业建议..."
        else:
            # Main agent execution progress
            if has_tool_call(update_data, {"task"}):
                return "正在为您调用知识库分析师进行检索..."
            if has_tool_call(update_data, {"mark_material_delivered"}):
                return "正在为您生成会议准备材料..."

    return None
