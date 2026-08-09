"""Tools for Wu Tanchang Owner Agent to query consultation data."""

from __future__ import annotations

import logging
import os
import re
import threading
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Tuple

import requests
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


def _extract_name_and_title(messages: List[Any]) -> str:
    """Extract name and title from human messages in the conversation history."""
    if not messages:
        return "未知客户"

    for msg in messages:
        # Check if the message is from the human/client
        msg_type = getattr(msg, "type", "")
        if (
            msg_type != "human"
            and msg_type != "user"
            and type(msg).__name__ != "HumanMessage"
        ):
            continue

        content = getattr(msg, "content", "")
        if not isinstance(content, str) or not content:
            continue

        # Pattern 1: Direct name + title patterns anywhere in the message
        # e.g., "我李老板准备...", "张总好..."
        title_pattern = (
            r"([^的我是你他她它有个了想要开在有就和与，。！？、\s]{1,3})"
            r"(老板|经理|总|创始人|合伙人|主理人|先生|女士|医生|老师|师傅)"
        )
        for name_part, title_part in re.findall(title_pattern, content):
            stop_words = [
                "是",
                "要",
                "想",
                "去",
                "做",
                "叫",
                "给",
                "跟",
                "和",
                "或",
                "在",
                "不",
                "有",
                "个",
                "人",
                "都",
                "此",
            ]
            if name_part not in stop_words and len(name_part) >= 1:
                return f"{name_part}{title_part}"

        # Pattern 2: Explicit introductions with name/title patterns
        # e.g., 我是面馆的李老板, 我叫王二, 我是李总, 我是王老师
        intro_matches = re.findall(
            r"(?:我是|我叫|叫我|称呼我为|本人是)\s*([^结构化，。！？、\s]{1,10})",
            content,
        )
        for match in intro_matches:
            # Clean up: if the match contains "的" (e.g., "面馆的李老板"), get the part after "的"
            candidate = (
                match.split("的")[-1].strip() if "的" in match else match.strip()
            )

            # Filter candidates to avoid matching verbs/phrases
            if candidate and len(candidate) <= 4:
                invalid_keywords = [
                    "一个",
                    "做",
                    "想",
                    "开",
                    "打算",
                    "去",
                    "要",
                    "准备",
                    "来",
                    "咨询",
                    "加",
                    "加盟",
                    "找",
                ]
                if not any(w in candidate for w in invalid_keywords):
                    return candidate

        # Pattern 3: Surname and title combinations
        # e.g., 我姓张，是开面馆的。
        surname_matches = re.findall(r"我姓\s*([^，。！？、\s]{1,3})", content)
        for s in surname_matches:
            # Look for a title in the same content
            title_match = re.search(
                r"(老板|经理|总|创始人|合伙人|主理人|先生|女士|医生|老师|师傅|阿姨|叔叔|大姐)",
                content,
            )
            if title_match:
                title = title_match.group(1)
                return f"{s}{title}"
            return f"{s}先生/女士"

    return "未知客户"


def _parse_thread_id(thread_id: str) -> Tuple[str, str, str] | None:
    """Parse thread ID into (agent_name, user_id, conversation_id)."""
    parts = thread_id.split("::")
    if len(parts) == 4 and parts[0] == "wt":
        return parts[1], parts[2], parts[3]
    return None


def _is_wu_agent_shared_owner(config: RunnableConfig) -> bool:
    """True for mode=wu-agent admin mode → shared workspace_owner (薛神)."""
    metadata = config.get("metadata") or {}
    source = str(metadata.get("source") or "").strip()
    if source == "wu_agent_owner_admin":
        return True
    return bool(metadata.get("admin_owner_mode"))


def _micro_callbacks_base_url(config: RunnableConfig) -> str | None:
    metadata = config.get("metadata") or {}
    callback_url = str(metadata.get("callback_url") or "").strip()
    if callback_url and "/wu_tanchang_callbacks/" in callback_url:
        return callback_url.split("/wu_tanchang_callbacks/")[0] + "/wu_tanchang_callbacks"

    raw_allowed = os.environ.get("WU_CALLBACK_ALLOWED_BASE_URLS")
    if raw_allowed:
        for base in raw_allowed.split(","):
            base = base.strip().rstrip("/")
            if base:
                if base.endswith("/wu_tanchang_callbacks"):
                    return base
                return f"{base}/wu_tanchang_callbacks"
    return "http://aihehuomicro-web:3001/wu_tanchang_callbacks"


def _agent_headers() -> Dict[str, str]:
    token = os.environ.get("WU_TANCHANG_CALLBACK_TOKEN") or os.environ.get(
        "WU_CALLBACK_AGENT_KEY"
    )
    return {"X-Agent-Key": token or "", "Content-Type": "application/json"}


def _simple_message(role: str, text: str) -> Any:
    """Minimal message-like object for name extraction / detail display."""

    class _Msg:
        def __init__(self, msg_type: str, content: str) -> None:
            self.type = msg_type
            self.content = content

    return _Msg("human" if role == "user" else "ai", text or "")


async def _get_wu_agent_visitors_from_micro(
    config: RunnableConfig, *, days: int = 30
) -> List[Dict[str, Any]]:
    """Load front-desk visitors from Micro wu_agent_conversations (authoritative)."""
    base = _micro_callbacks_base_url(config)
    if not base:
        _logger.warning("[OwnerTools] No Micro callbacks base URL for wu-agent visitors")
        return []

    metadata = config.get("metadata") or {}
    exclude_user_id = metadata.get("user_id") or metadata.get("user_a_id")
    params: Dict[str, Any] = {"days": max(1, int(days or 30))}
    if exclude_user_id is not None:
        params["exclude_user_id"] = exclude_user_id

    url = f"{base.rstrip('/')}/wu_agent_visitors"
    try:
        resp = requests.get(url, headers=_agent_headers(), params=params, timeout=15)
        if resp.status_code != 200:
            _logger.warning(
                "[OwnerTools] wu_agent_visitors HTTP %s body=%s",
                resp.status_code,
                (resp.text or "")[:200],
            )
            return []
        data = resp.json() if resp.content else {}
    except Exception as exc:  # noqa: BLE001
        _logger.error("[OwnerTools] wu_agent_visitors request failed: %s", exc)
        return []

    visitors = data.get("visitors") if isinstance(data, dict) else None
    if not isinstance(visitors, list):
        return []

    out: List[Dict[str, Any]] = []
    for v in visitors:
        if not isinstance(v, dict):
            continue
        user_id = str(v.get("user_id") or "").strip()
        if not user_id:
            continue
        conv_id = str(v.get("conversation_id") or f"wu_agent_{user_id}")
        out.append(
            {
                "thread_id": f"wt::default::{user_id}::{conv_id}",
                "user_id": user_id,
                "conversation_id": conv_id,
                "started_at": str(v.get("started_at") or ""),
                "last_active_at": str(v.get("last_active_at") or ""),
                "status": str(v.get("status") or "chatting"),
                "prep_body": None,
                "messages": [],
                "client_name": str(v.get("client_name") or f"访客{user_id}"),
                "source": "wu_agent_conversation",
            }
        )
    return out


async def _get_wu_agent_visitor_detail_from_micro(
    config: RunnableConfig, client_user_id: str
) -> Dict[str, Any] | None:
    base = _micro_callbacks_base_url(config)
    if not base:
        return None
    url = f"{base.rstrip('/')}/wu_agent_visitors/{client_user_id}"
    try:
        resp = requests.get(url, headers=_agent_headers(), timeout=20)
        if resp.status_code == 404:
            return None
        if resp.status_code != 200:
            _logger.warning(
                "[OwnerTools] wu_agent_visitor HTTP %s body=%s",
                resp.status_code,
                (resp.text or "")[:200],
            )
            return None
        data = resp.json() if resp.content else {}
    except Exception as exc:  # noqa: BLE001
        _logger.error("[OwnerTools] wu_agent_visitor request failed: %s", exc)
        return None

    if not isinstance(data, dict):
        return None

    user_id = str(data.get("user_id") or client_user_id)
    messages_raw = data.get("messages") or []
    messages = []
    if isinstance(messages_raw, list):
        for m in messages_raw:
            if isinstance(m, dict):
                messages.append(
                    _simple_message(str(m.get("role") or "user"), str(m.get("text") or ""))
                )

    return {
        "thread_id": f"wt::default::{user_id}::{data.get('conversation_id') or f'wu_agent_{user_id}'}",
        "user_id": user_id,
        "conversation_id": str(data.get("conversation_id") or f"wu_agent_{user_id}"),
        "started_at": str(data.get("started_at") or ""),
        "last_active_at": str(data.get("last_active_at") or ""),
        "status": str(data.get("status") or "chatting"),
        "prep_body": data.get("prep_body"),
        "messages": messages,
        "client_name": str(data.get("client_name") or f"访客{user_id}"),
        "source": "wu_agent_conversation",
    }


async def _get_client_threads(
    config: RunnableConfig, *, days: int = 30
) -> List[Dict[str, Any]]:
    """Scan checkpointer to extract metadata and status of all client threads.

    Filters client threads by the current owner's calendar_id (Scope control).
    For wu-agent admin / shared workspace_owner, load visitors from Micro instead.

    Args:
        config: The current RunnableConfig object.
        days: Lookback window (used by Micro visitor directory).

    Returns:
        List of client thread info dicts.
    """
    if _is_wu_agent_shared_owner(config):
        return await _get_wu_agent_visitors_from_micro(config, days=days)

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

    if hasattr(checkpointer, "_load_from_disk"):
        try:
            checkpointer._load_from_disk()
        except Exception as exc:  # noqa: BLE001
            _logger.error("Failed to reload checkpointer from disk: %s", exc)

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

    storage = getattr(checkpointer, "storage", {})
    lock = getattr(checkpointer, "_lock", threading.RLock())

    thread_groups = {}
    with lock:
        try:
            t_ids = list(storage.keys())
        except Exception:
            t_ids = []

        for t_id in t_ids:
            # Only process main thread client conversations (prefix wt::default::)
            if not t_id.startswith("wt::default::"):
                continue
            ns_map = storage.get(t_id, {})
            ckpts = ns_map.get("", {})
            if not ckpts:
                continue

            # Sort checkpoint IDs (which are in incremental order) to find first/latest
            sorted_ids = sorted(ckpts.keys())
            if not sorted_ids:
                continue

            first_raw = ckpts[sorted_ids[0]]
            latest_raw = ckpts[sorted_ids[-1]]
            thread_groups[t_id] = (first_raw, latest_raw)

    client_threads = []
    for t_id, (first_raw, latest_raw) in thread_groups.items():
        parsed = _parse_thread_id(t_id)
        if not parsed:
            continue
        agent_name, user_id, conv_id = parsed

        try:
            # Check for real InMemorySaver database tuple format
            if isinstance(latest_raw, tuple) and len(latest_raw) == 3:
                f_ckpt_b, f_meta_b, _ = first_raw
                _f_meta = checkpointer.serde.loads_typed(f_meta_b)
                f_ckpt = checkpointer.serde.loads_typed(f_ckpt_b)
                f_channels = checkpointer._load_blobs(
                    t_id, "", f_ckpt["channel_versions"]
                )
                first_checkpoint = {**f_ckpt, "channel_values": f_channels}

                l_ckpt_b, l_meta_b, _ = latest_raw
                l_meta = checkpointer.serde.loads_typed(l_meta_b)
                l_ckpt = checkpointer.serde.loads_typed(l_ckpt_b)
                l_channels = checkpointer._load_blobs(
                    t_id, "", l_ckpt["channel_versions"]
                )
                latest_checkpoint = {**l_ckpt, "channel_values": l_channels}
            # Fallback for CheckpointTuple objects
            elif hasattr(latest_raw, "checkpoint"):
                first_checkpoint = first_raw.checkpoint
                l_meta = latest_raw.metadata
                latest_checkpoint = latest_raw.checkpoint
            # Fallback for fake/mock checkpointer dictionaries in unit tests
            elif isinstance(latest_raw, dict):
                first_checkpoint = first_raw
                latest_checkpoint = latest_raw
                # Find matching tuple in mock list to get metadata
                tups = getattr(checkpointer, "_tuples", [])
                match_tup = next(
                    (x for x in tups if x.config["configurable"]["thread_id"] == t_id),
                    None,
                )
                l_meta = match_tup.metadata if match_tup else {}
            else:
                _logger.warning(
                    "Unknown checkpoint raw value type in storage: %s", type(latest_raw)
                )
                continue
        except Exception as exc:  # noqa: BLE001
            _logger.error(
                "Failed to deserialize checkpoint for thread %s: %s", t_id, exc
            )
            continue

        # Check calendar_id scope (S2)
        client_calendar_id = l_meta.get("calendar_id") or l_meta.get("user_b_id")
        client_calendar_id_str = (
            str(client_calendar_id).strip() if client_calendar_id else None
        )

        # Filter by owner_calendar_id (S2 Point 1)
        if client_calendar_id_str != owner_calendar_id_str:
            continue

        delivered = False
        prep_body = None
        messages = latest_checkpoint.get("channel_values", {}).get("messages", [])

        # Check messages for save_meeting_prep and mark_material_delivered tool calls
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.get("name") == "mark_material_delivered":
                        delivered = True
                    if tc.get("name") == "save_meeting_prep":
                        prep_body = tc.get("args", {}).get("body")
                        if not client_calendar_id_str:
                            client_calendar_id_str = str(
                                tc.get("args", {}).get("user_b_id")
                            ).strip()

        first_ts_str = first_checkpoint.get("ts", "")
        last_ts_str = latest_checkpoint.get("ts", "")

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
                "client_name": _extract_name_and_title(messages),
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
    threads = await _get_client_threads(config, days=days)
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
    threads = await _get_client_threads(config, days=days)
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
        "| 姓名/称呼 | 客户ID | 开始时间 | 最后活跃时间 | 状态 |",
        "| :--- | :--- | :--- | :--- | :---: |",
    ]
    for t in sorted(filtered, key=lambda x: x["started_at"], reverse=True):
        status_label = "✅ 已交付" if t["status"] == "completed" else "💬 沟通中"
        started = (t.get("started_at") or "")[:19] or "-"
        last_active = (t.get("last_active_at") or "")[:19] or "-"
        lines.append(
            f"| {t['client_name']} | `{t['user_id']}` | {started} | {last_active} | {status_label} |"
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
    if _is_wu_agent_shared_owner(config):
        target = await _get_wu_agent_visitor_detail_from_micro(config, client_user_id)
    else:
        threads = await _get_client_threads(config)
        target = next((t for t in threads if t["user_id"] == client_user_id), None)

    if not target:
        return f"未找到客户 ID 为 `{client_user_id}` 的预聊记录。"

    started = (target.get("started_at") or "")[:19] or "未知"
    last_active = (target.get("last_active_at") or "")[:19] or "未知"

    result = [
        f"# 👤 客户需求详情: {target['client_name']} (ID: {client_user_id})",
        f"- **开始预聊时间**: {started}",
        f"- **最后互动时间**: {last_active}",
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
