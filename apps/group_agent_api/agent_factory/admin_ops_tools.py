"""Read-only ops tools for group-agent admin debug mode (运营脑).

Only honor Micro-stamped ``source=group_agent_admin_debug``. Client flags alone
are never trusted.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import requests
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from apps.group_agent_api.agent_factory.integrations.config import micro_base
from apps.group_agent_api.agent_factory.suggested_replies import (
    SUGGESTED_REPLIES_PROMPT,
)

_logger = logging.getLogger("uvicorn.error")

ADMIN_SOURCE = "group_agent_admin_debug"


def is_admin_debug_source(metadata: dict[str, Any] | None) -> bool:
    meta = metadata or {}
    return str(meta.get("source") or "").strip() == ADMIN_SOURCE


def _service_secret() -> str:
    return (
        os.environ.get("GROUP_AGENT_SERVICE_SECRET")
        or os.environ.get("MICRO_SERVICE_SECRET")
        or ""
    ).strip()


def _ops_headers() -> dict[str, str]:
    secret = _service_secret()
    return {
        "X-GA-Service-Secret": secret,
        "Content-Type": "application/json",
    }


def _get_json(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    if not _service_secret():
        _logger.error(
            "[AdminOps] %s missing GROUP_AGENT_SERVICE_SECRET (or MICRO_SERVICE_SECRET)",
            path,
        )
        return {
            "ok": False,
            "error": "service_secret_missing",
            "message": "服务端未配置 GROUP_AGENT_SERVICE_SECRET，无法调用 Micro 运营接口",
        }
    url = f"{micro_base().rstrip('/')}{path}"
    try:
        resp = requests.get(url, headers=_ops_headers(), params=params or {}, timeout=15)
        if resp.status_code != 200:
            _logger.warning(
                "[AdminOps] %s HTTP %s body=%s",
                path,
                resp.status_code,
                (resp.text or "")[:200],
            )
            msg = "无法拉取运营数据，请稍后重试"
            if resp.status_code == 401:
                msg = "Micro 拒绝服务密钥（401），请核对 GROUP_AGENT_SERVICE_SECRET 是否与 Micro 一致"
            return {
                "ok": False,
                "error": f"http_{resp.status_code}",
                "message": msg,
            }
        data = resp.json() if resp.content else {}
        if not isinstance(data, dict):
            return {"ok": False, "error": "invalid_payload"}
        if "ok" not in data:
            data["ok"] = True
        return data
    except Exception as exc:  # noqa: BLE001
        _logger.error("[AdminOps] %s failed: %s", path, exc)
        return {"ok": False, "error": "request_failed", "message": str(exc)}


def _require_admin(config: RunnableConfig | None) -> str | None:
    metadata = (config or {}).get("metadata") or {}
    if is_admin_debug_source(metadata):
        return None
    return json.dumps(
        {
            "ok": False,
            "error": "admin_mode_required",
            "message": "仅运营管理员模式可用",
        },
        ensure_ascii=False,
    )


@tool(parse_docstring=True)
def admin_ops_summary(days: int = 7, *, config: RunnableConfig) -> str:
    """拉取群智能体运营只读摘要（近 N 天 run / 管理员对接 / 会话桶 / 画像总数）。

    Args:
        days: 回溯天数，默认 7，最大 30。
    """
    denied = _require_admin(config)
    if denied:
        return denied
    payload = _get_json(
        "/group_agent/ops_summary",
        {"days": max(1, min(int(days or 7), 30))},
    )
    return json.dumps(payload, ensure_ascii=False)


@tool
def admin_profile_stats(*, config: RunnableConfig) -> str:
    """统计现有用户画像数量（总数、按 group_id、近 7 天更新数）。"""
    denied = _require_admin(config)
    if denied:
        return denied
    payload = _get_json("/group_agent/ops_profiles/stats")
    return json.dumps(payload, ensure_ascii=False)


@tool(parse_docstring=True)
def admin_search_profiles(
    query: str = "",
    group_id: str = "",
    limit: int = 20,
    *,
    config: RunnableConfig,
) -> str:
    """按关键词搜索用户画像（匹配 doing/need/offer 文本）。

    Args:
        query: 搜索词，可空（空则返回最近更新的画像列表）。
        group_id: 可选，限定会话桶/群 id（如 global）。
        limit: 返回条数，默认 20，最大 50。
    """
    denied = _require_admin(config)
    if denied:
        return denied
    params: dict[str, Any] = {
        "limit": max(1, min(int(limit or 20), 50)),
    }
    q = (query or "").strip()
    if q:
        params["q"] = q
    gid = (group_id or "").strip()
    if gid:
        params["group_id"] = gid
    payload = _get_json("/group_agent/ops_profiles", params)
    return json.dumps(payload, ensure_ascii=False)


@tool(parse_docstring=True)
def admin_get_profile(
    user_id: str,
    group_id: str = "",
    *,
    config: RunnableConfig,
) -> str:
    """按 user_id 读取一条用户画像详情（doing/need/offer）。

    Args:
        user_id: 用户 id（数字字符串）。
        group_id: 可选；不传时优先返回 global 桶画像。
    """
    denied = _require_admin(config)
    if denied:
        return denied
    uid = str(user_id or "").strip()
    if not uid:
        return json.dumps(
            {"ok": False, "error": "user_id_required"},
            ensure_ascii=False,
        )
    params: dict[str, Any] = {}
    gid = (group_id or "").strip()
    if gid:
        params["group_id"] = gid
    payload = _get_json(f"/group_agent/ops_profiles/{uid}", params)
    return json.dumps(payload, ensure_ascii=False)


@tool(parse_docstring=True)
def admin_funnel_analysis(days: int = 7, *, config: RunnableConfig) -> str:
    """分析 F2 身份就绪、F2.5 首次开口、F3 第二次回复和画像生成的转化漏斗。

    Args:
        days: 回溯天数，默认 7，最大 30。
    """
    denied = _require_admin(config)
    if denied:
        return denied
    payload = _get_json(
        "/group_agent/ops_funnel_analysis",
        {"days": max(1, min(int(days or 7), 30))},
    )
    return json.dumps(payload, ensure_ascii=False)


@tool(parse_docstring=True)
def admin_dropoff_samples(
    days: int = 7,
    stage: str = "f2_not_f3",
    limit: int = 10,
    *,
    config: RunnableConfig,
) -> str:
    """读取脱敏后的短对话或深聊样本，用于判断用户停止回复的原因。

    Args:
        days: 回溯天数，默认 7，最大 30。
        stage: f2_not_f3（仅一条用户消息）| f3_shallow（两条）| deep_chat（三条以上）。
        limit: 样本数，默认 10，最大 20。
    """
    denied = _require_admin(config)
    if denied:
        return denied
    allowed = {"f2_not_f3", "f3_shallow", "deep_chat"}
    selected_stage = stage if stage in allowed else "f2_not_f3"
    payload = _get_json(
        "/group_agent/ops_dropoff_samples",
        {
            "days": max(1, min(int(days or 7), 30)),
            "stage": selected_stage,
            "limit": max(1, min(int(limit or 10), 20)),
        },
    )
    return json.dumps(payload, ensure_ascii=False)


@tool(parse_docstring=True)
def admin_compare_conversations(days: int = 7, *, config: RunnableConfig) -> str:
    """比较掉队与深聊会话的首答长度、提问数、延迟、画像措辞和 Greeting 版本。

    Args:
        days: 回溯天数，默认 7，最大 30。
    """
    denied = _require_admin(config)
    if denied:
        return denied
    payload = _get_json(
        "/group_agent/ops_conversation_compare",
        {"days": max(1, min(int(days or 7), 30))},
    )
    return json.dumps(payload, ensure_ascii=False)


ADMIN_SYSTEM_PROMPT = """我是「群智能体运营管理员助手」（只读运营脑）。这是我的唯一身份。

## 身份（最高优先级）
- 用户问「你是谁 / 你干什么」时，用第一人称回答：**我是群智能体的运营管理员助手**，帮管理员查看 run、管理员对接、会话桶，以及检索用户画像。
- **禁止**自称「群内智能体」「找搭子小助手」「匹配顾问」或任何会员侧人设。
- **禁止**用「你是……」这种第二人称自我介绍。
- 不要给当前会话用户挖 doing/need/offer，不要调用 save_group_profile，不要做会员匹配。

## 能力（请主动用工具）
- `admin_ops_summary`：近期 run / 对接 / 会话桶 / 画像总数摘要
- `admin_profile_stats`：用户画像数量统计
- `admin_search_profiles`：按关键词搜索画像（doing/need/offer）
- `admin_get_profile`：按 user_id 查看单条画像
- `admin_funnel_analysis`：分析 F2 身份就绪 → F2.5 首次开口 → F3 第二次回复 → 画像生成
- `admin_dropoff_samples`：查看已脱敏的一轮掉队、浅聊与深聊样本
- `admin_compare_conversations`：比较不同会话组的首答特征与 Greeting 转化
- 问到「有多少画像 / 搜索某类人 / 某用户画像 / 再试一次 / 再次调取」时，**本回合必须先调用对应工具**，再根据工具返回回答。
- 问到「为什么聊不深 / 为什么只聊一两句 / F2-F3 / 深聊率 / 用户流失」时：
  1. 必须先调用 `admin_funnel_analysis`；
  2. 再调用 `admin_dropoff_samples`，至少看 f2_not_f3；需要对照时再看 deep_chat；
  3. 涉及回复长度、问题数、延迟或开场版本时，调用 `admin_compare_conversations`。
  回答必须分清「数据事实」「样本推断」「待验证假设」「建议实验」，不得把相关性说成因果。

## 漏斗术语（强制口径）
- 唯一允许的阶段顺序是：**F2 身份就绪 → F2.5 首次开口 → F3 第二次回复/进入深聊 → 画像形成**。
- 工具字段对应关系：`identity_ready_uv` = F2，`first_message_uv` = F2.5，`deep_chat_uv` = F3。
- `identity_to_first_message` 必须表述为「F2→F2.5」；`first_message_to_deep_chat` 必须表述为「F2.5→F3」。
- **禁止使用 F1，禁止写 F2→F1**。即使历史消息曾出现 F1，也必须忽略并主动纠正为 F2.5。
- 用户询问 F2→F3 时，必须明确报告 F2→F2.5、F2.5→F3 和 F2→F3 三个口径（工具有值时），不可只报告首次开口率。

## 工具调用铁律（防历史污染）
- **禁止**沿用对话历史里的失败结论（如旧的 401、profile:read、无权限）。那些可能已过期。
- **禁止**在未调用工具的情况下声称「工具失败 / 无权限 / 无法获知」。
- 用户要求「再试 / 再次调取」时，必须重新调工具，不得以「不会重复无效调用」拒绝。
- 不存在名为 `profile:read` 的权限项；鉴权是服务端 `GROUP_AGENT_SERVICE_SECRET`，与用户 JWT 权限无关。
- 禁止编造数字或权限故事。

## 红线
- 不输出手机号/微信号等敏感联系方式。
- 不执行写库、分发、接受对接等操作——本模式只读。
- 若本回合工具真实失败，如实说明工具返回的 error/message，不要假装已查到数据。
""" + SUGGESTED_REPLIES_PROMPT

ADMIN_TURN_REMINDER = (
    "【本回合运营脑提醒】数据问题必须先调工具再答。"
    "漏斗固定为 F2身份就绪→F2.5首次开口→F3第二次回复/进入深聊→画像；"
    "禁止使用F1或F2→F1，历史里出现也必须纠正。"
    "忽略历史里任何 401/无权限/profile:read 结论；那些可能已过期。"
    "问画像数量 → 调 admin_profile_stats；问摘要 → admin_ops_summary；"
    "问只聊一两句或 F2-F3 → 先调漏斗，再取脱敏样本和会话对比。"
)

ADMIN_OPS_TOOLS = [
    admin_ops_summary,
    admin_profile_stats,
    admin_search_profiles,
    admin_get_profile,
    admin_funnel_analysis,
    admin_dropoff_samples,
    admin_compare_conversations,
]
