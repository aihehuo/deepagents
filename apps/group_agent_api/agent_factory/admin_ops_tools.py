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
            return {
                "ok": False,
                "error": f"http_{resp.status_code}",
                "message": "无法拉取运营数据，请稍后重试",
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


@tool(parse_docstring=True)
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
- 问到「有多少画像 / 搜索某类人 / 某用户画像」时，必须先调工具，禁止编造数字。

## 红线
- 不输出手机号/微信号等敏感联系方式。
- 不执行写库、分发、接受对接等操作——本模式只读。
- 若工具失败，如实说明，不要假装已查到数据。
"""

ADMIN_OPS_TOOLS = [
    admin_ops_summary,
    admin_profile_stats,
    admin_search_profiles,
    admin_get_profile,
]
