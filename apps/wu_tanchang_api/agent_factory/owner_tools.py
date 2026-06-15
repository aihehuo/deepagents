"""Tools for Wu Tanchang Owner Agent to query consultation data."""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Tuple

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

_logger = logging.getLogger("uvicorn.error")


def _parse_thread_id(thread_id: str) -> Tuple[str, str, str] | None:
    """Parse thread ID into (agent_name, user_id, conversation_id)."""
    parts = thread_id.split("::")
    if len(parts) == 4 and parts[0] == "wt":
        return parts[1], parts[2], parts[3]
    return None


async def _get_client_threads(agent: Any) -> List[Dict[str, Any]]:
    """Scan checkpointer to extract metadata and status of all client threads.

    Args:
        agent: The current agent instance with a checkpointer.

    Returns:
        List of client thread info dicts.
    """
    checkpointer = getattr(agent, "checkpointer", None)
    if not checkpointer:
        return []

    client_threads = []
    # InMemorySaver stores checkpoints in checkpointer.storage
    storage = getattr(checkpointer, "storage", {})

    for tid in list(storage.keys()):
        parsed = _parse_thread_id(tid)
        if not parsed:
            continue
        agent_name, user_id, conv_id = parsed

        # We only process client conversations (prefix wt::default::)
        if agent_name != "default":
            continue

        ns_map = storage[tid]
        all_ckpts = []
        for ns, ckpts in ns_map.items():
            # Only process the root namespace for the main thread
            if ns != "":
                continue
            # ckpts maps checkpoint_id to serialized tuple.
            # We can use the checkpointer.list method to safely get CheckpointTuples
            # instead of parsing the raw storage dictionary.
            # But checkpointer.list(config={"configurable": {"thread_id": tid}}) is extremely clean.
            try:
                # list returns an iterator of CheckpointTuple objects for this thread
                ckpt_tuples = list(
                    checkpointer.list(config={"configurable": {"thread_id": tid}})
                )
                all_ckpts.extend(ckpt_tuples)
            except Exception as exc:  # noqa: BLE001
                _logger.error("Failed to list checkpoints for thread %s: %s", tid, exc)

        if not all_ckpts:
            continue

        # Sort checkpoint tuples by timestamp
        all_ckpts.sort(key=lambda x: x.checkpoint.get("ts", ""))

        first_ckpt = all_ckpts[0]
        latest_ckpt = all_ckpts[-1]

        delivered = False
        prep_body = None
        messages = latest_ckpt.checkpoint.get("channel_values", {}).get("messages", [])

        # Check messages for save_meeting_prep and mark_material_delivered tool calls
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.get("name") == "mark_material_delivered":
                        delivered = True
                    if tc.get("name") == "save_meeting_prep":
                        prep_body = tc.get("args", {}).get("body")

        first_ts_str = first_ckpt.checkpoint.get("ts", "")
        last_ts_str = latest_ckpt.checkpoint.get("ts", "")

        client_threads.append(
            {
                "thread_id": tid,
                "user_id": user_id,
                "conversation_id": conv_id,
                "started_at": first_ts_str,
                "last_active_at": last_ts_str,
                "status": "completed" if delivered else "chatting",
                "prep_body": prep_body,
                "messages": messages,
            }
        )

    return client_threads


@tool(parse_docstring=True)
async def get_consultation_stats(
    days: int = 7,
    *,
    config: RunnableConfig,
) -> str:
    """获取近期客户预聊统计数据，包括新增客户数、准备资料生成数及转化率。

    Args:
        days: 统计的历史天数，默认 7 天。
        config: 运行时配置对象，自动注入。
    """
    agent = config.get("metadata", {}).get("agent_instance")
    if not agent:
        return "无法获取 checkpointer，统计失败。"

    threads = await _get_client_threads(agent)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    new_clients = 0
    completed_preps = 0
    daily_stats: Dict[str, Dict[str, int]] = {}

    for t in threads:
        ts_str = t["started_at"]
        if not ts_str:
            continue
        try:
            started_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except ValueError:
            continue

        if started_dt >= cutoff:
            new_clients += 1
            date_str = started_dt.strftime("%Y-%m-%d")
            daily_stats.setdefault(date_str, {"started": 0, "completed": 0})
            daily_stats[date_str]["started"] += 1

            if t["status"] == "completed":
                completed_preps += 1
                daily_stats[date_str]["completed"] += 1

    rate = (completed_preps / new_clients * 100) if new_clients > 0 else 0.0

    report = [
        f"### 📊 近 {days} 天客户预聊统计报告",
        f"- **新增预聊客户数**: {new_clients} 人",
        f"- **已生成准备资料数**: {completed_preps} 份",
        f"- **资料生成转化率**: {rate:.1f}%",
        "\n| 日期 | 新增预聊人数 | 新增交付资料数 |",
        "| :--- | :---: | :---: |",
    ]

    for d_str in sorted(daily_stats.keys(), reverse=True):
        started = daily_stats[d_str]["started"]
        completed = daily_stats[d_str]["completed"]
        report.append(f"| {d_str} | {started} | {completed} |")

    return "\n".join(report)


@tool(parse_docstring=True)
async def list_recent_clients(
    days: int = 7,
    status: str = "all",
    *,
    config: RunnableConfig,
) -> str:
    """列出最近预聊的客户清单及其状态。

    Args:
        days: 回溯的天数，默认 7 天。
        status: 过滤状态。支持 "all", "chatting", "completed"。
        config: 运行时配置对象，自动注入。
    """
    agent = config.get("metadata", {}).get("agent_instance")
    if not agent:
        return "无法获取 checkpointer。"

    threads = await _get_client_threads(agent)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    filtered = []
    for t in threads:
        ts_str = t["started_at"]
        if not ts_str:
            continue
        try:
            started_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except ValueError:
            continue

        if started_dt >= cutoff:
            if status != "all" and t["status"] != status:
                continue
            filtered.append(t)

    if not filtered:
        return f"近 {days} 天内未找到状态为 {status} 的预聊客户。"

    lines = [
        f"### 👥 近 {days} 天客户列表 (状态: {status})",
        "| 客户ID | 开始时间 | 最后活跃时间 | 状态 |",
        "| :--- | :--- | :--- | :---: |",
    ]
    for t in sorted(filtered, key=lambda x: x["started_at"], reverse=True):
        status_label = "✅ 已交付" if t["status"] == "completed" else "💬 沟通中"
        lines.append(
            f"| `{t['user_id']}` | {t['started_at'][:19]} | {t['last_active_at'][:19]} | {status_label} |"
        )

    return "\n".join(lines)


@tool(parse_docstring=True)
async def get_client_detail(
    client_user_id: str,
    *,
    config: RunnableConfig,
) -> str:
    """获取指定客户的预聊需求细节和已生成的会议材料。

    Args:
        client_user_id: 客户的唯一 user_id。
        config: 运行时配置对象，自动注入。
    """
    agent = config.get("metadata", {}).get("agent_instance")
    if not agent:
        return "无法获取 checkpointer。"

    threads = await _get_client_threads(agent)
    target = next((t for t in threads if t["user_id"] == client_user_id), None)

    if not target:
        return f"未找到客户 ID 为 `{client_user_id}` 的预聊记录。"

    result = [
        f"# 👤 客户需求详情 (ID: {client_user_id})",
        f"- **开始预聊时间**: {target['started_at'][:19]}",
        f"- **最后互动时间**: {target['last_active_at'][:19]}",
        f"- **当前进度状态**: {'✅ 会议材料已交付' if target['status'] == 'completed' else '💬 仍在补充信息'}",
        "\n---",
    ]

    if target["prep_body"]:
        result.append("\n## 📄 生成的会议准备材料：\n")
        result.append(target["prep_body"])
    else:
        result.append("\n⚠️ **该客户的会议准备材料尚未生成。**\n")
        # Extract the last few dialogue turns
        chat_history = []
        for msg in target["messages"][-6:]:
            role = "AI助手" if msg.type == "ai" else "客户"
            content = getattr(msg, "content", "")
            if content and not str(content).startswith("Updated todo list"):
                chat_history.append(f"**{role}**: {content}")
        if chat_history:
            result.append("### 💬 最近对话片段：")
            result.extend(chat_history)

    return "\n".join(result)
