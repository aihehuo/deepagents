"""Tools for Wu Tanchang Owner Agent to query consultation data."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Tuple

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

_logger = logging.getLogger("uvicorn.error")


def mask_pii(text: str) -> str:
    """Mask sensitive PII like emails, ID card numbers, and phone numbers from the text.

    Args:
        text: The input text potentially containing PII.

    Returns:
        The masked text.
    """
    if not text:
        return text

    # Mask email addresses
    def _replace_email(match: re.Match[str]) -> str:
        email = match.group(0)
        if "@" in email:
            local, domain = email.split("@", 1)
            if len(local) > 2:
                return f"{local[0]}***{local[-1]}@{domain}"
            return f"***@{domain}"
        return email

    text = re.sub(
        r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", _replace_email, text
    )

    # Mask Chinese mobile phone numbers: 1[3-9]\d{9} -> 1xx****xxxx
    text = re.sub(r"(?<!\d)(1[3-9]\d)\d{4}(\d{4})(?!\d)", r"\1****\2", text)

    # Mask 7-12 digit numbers that look like phone numbers
    # (e.g. landline 010-12345678 or 021 1234 5678)
    text = re.sub(r"(?<!\d)(\d{3,4})[- ]?(\d{4})[- ]?(\d{4})(?!\d)", r"\1****\3", text)

    # Mask 18-digit ID card numbers: 18 digits or 17 digits followed by X/x
    text = re.sub(
        r"(?<!\d)(\d{6})\d{8}(\d{3}[\dXx])(?!\d)",
        r"\1********\2",
        text,
    )

    return text


def _parse_thread_id(thread_id: str) -> Tuple[str, str, str] | None:
    """Parse thread ID into (agent_name, user_id, conversation_id)."""
    parts = thread_id.split("::")
    if len(parts) == 4 and parts[0] == "wt":
        return parts[1], parts[2], parts[3]
    return None


async def _get_client_threads(config: RunnableConfig) -> List[Dict[str, Any]]:
    """Scan checkpointer to extract metadata and status of all client threads.

    Filters client threads by the current owner's calendar_id (Scope control).

    Args:
        config: The current RunnableConfig object.

    Returns:
        List of client thread info dicts.
    """
    tid = config.get("configurable", {}).get("thread_id")
    agent = None
    if tid:
        from apps.wu_tanchang_api.agent_factory.agent import get_active_agent

        agent = get_active_agent(tid)
    if not agent:
        agent = config.get("metadata", {}).get("agent_instance")

    checkpointer = getattr(agent, "checkpointer", None)
    if not checkpointer:
        return []

    # Extract owner's calendar_id for scoping (S2)
    metadata = config.get("metadata") or {}
    owner_calendar_id = metadata.get("calendar_id") or metadata.get("user_b_id")
    if not owner_calendar_id and tid:
        parts = tid.split("::")
        if len(parts) == 4 and parts[0] == "wt" and parts[1] == "owner":
            owner_calendar_id = parts[2]

    owner_calendar_id_str = (
        str(owner_calendar_id).strip() if owner_calendar_id else None
    )
    if not owner_calendar_id_str:
        _logger.warning(
            "[OwnerTools] Access denied: calendar_id/user_b_id not found in config context"
        )
        return []

    # Call checkpointer.list(config=None) once to fetch all checkpoints (P1/P7 Optimization)
    try:
        all_tuples = list(checkpointer.list(config=None))
    except Exception as exc:  # noqa: BLE001
        _logger.error("Failed to list all checkpoints: %s", exc)
        return []

    from collections import defaultdict

    thread_groups = defaultdict(list)
    for tup in all_tuples:
        t_id = tup.config["configurable"]["thread_id"]
        ns = tup.config["configurable"].get("checkpoint_ns", "")
        # Only process main thread client conversations (prefix wt::default::)
        if ns == "" and t_id.startswith("wt::default::"):
            thread_groups[t_id].append(tup)

    client_threads = []
    for t_id, ckpt_tuples in thread_groups.items():
        parsed = _parse_thread_id(t_id)
        if not parsed:
            continue
        agent_name, user_id, conv_id = parsed

        # Sort the checkpoint tuples for this thread by timestamp
        ckpt_tuples.sort(key=lambda x: x.checkpoint.get("ts", ""))
        first_ckpt = ckpt_tuples[0]
        latest_ckpt = ckpt_tuples[-1]

        # Check calendar_id scope (S2)
        client_calendar_id = None
        if latest_ckpt.metadata:
            client_calendar_id = latest_ckpt.metadata.get(
                "calendar_id"
            ) or latest_ckpt.metadata.get("user_b_id")

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
                        if not client_calendar_id:
                            client_calendar_id = tc.get("args", {}).get("user_b_id")

        client_calendar_id_str = (
            str(client_calendar_id).strip() if client_calendar_id else None
        )

        # Filter by owner_calendar_id (S2 Point 1): client must match owner's calendar
        if client_calendar_id_str != owner_calendar_id_str:
            continue

        first_ts_str = first_ckpt.checkpoint.get("ts", "")
        last_ts_str = latest_ckpt.checkpoint.get("ts", "")

        client_threads.append(
            {
                "thread_id": t_id,
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
    threads = await _get_client_threads(config)
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
    threads = await _get_client_threads(config)
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
    expand_prep_details: bool = False,
    config: RunnableConfig,
) -> str:
    """获取指定客户的预聊需求细节和已生成的会议材料。

    Args:
        client_user_id: 客户的唯一 user_id。
        expand_prep_details: 是否展开详细的会议材料及历史对话片段。默认为 False 以保护客户隐私。
        config: 运行时配置对象，自动注入。
    """
    threads = await _get_client_threads(config)
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

    if expand_prep_details:
        _logger.info(
            "[AUDIT] Owner agent expanded full details for client: %s", client_user_id
        )
        if target["prep_body"]:
            result.append("\n## 📄 生成的会议准备材料：\n")
            result.append(mask_pii(target["prep_body"]))
        else:
            result.append("\n⚠️ **该客户的会议准备材料尚未生成。**\n")
            # Extract the last few dialogue turns
            chat_history = []
            for msg in target["messages"][-6:]:
                role = "AI助手" if msg.type == "ai" else "客户"
                content = getattr(msg, "content", "")
                if content and not str(content).startswith("Updated todo list"):
                    chat_history.append(f"**{role}**: {mask_pii(str(content))}")
            if chat_history:
                result.append("### 💬 最近对话片段：")
                result.extend(chat_history)
    else:
        if target["prep_body"]:
            result.append(
                f"\n📄 **会议准备材料已生成** (共 {len(target['prep_body'])} 字符)。"
            )
            result.append(
                "💡 *提示：如需查阅详细的会议材料，请将参数 `expand_prep_details` 设置为 `True` 调用本工具。*"
            )
        else:
            result.append("\n⚠️ **该客户的会议准备材料尚未生成。**")
            chat_history_len = len(
                [
                    msg
                    for msg in target["messages"]
                    if getattr(msg, "content", "")
                    and not str(msg.content).startswith("Updated todo list")
                ]
            )
            result.append(f"💬 预聊对话包含 {chat_history_len} 条记录。")
            result.append(
                "💡 *提示：如需查阅详细的历史对话片段，请将参数 `expand_prep_details` 设置为 `True` 调用本工具。*"
            )

    return "\n".join(result)
