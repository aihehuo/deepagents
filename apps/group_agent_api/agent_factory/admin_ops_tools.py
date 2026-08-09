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


def _fetch_ops_summary(*, days: int = 7) -> dict[str, Any]:
    url = f"{micro_base().rstrip('/')}/group_agent/ops_summary"
    params = {"days": max(1, min(int(days or 7), 30))}
    try:
        resp = requests.get(url, headers=_ops_headers(), params=params, timeout=15)
        if resp.status_code != 200:
            _logger.warning(
                "[AdminOps] ops_summary HTTP %s body=%s",
                resp.status_code,
                (resp.text or "")[:200],
            )
            return {
                "ok": False,
                "error": f"http_{resp.status_code}",
                "message": "无法拉取运营摘要，请稍后重试",
            }
        data = resp.json() if resp.content else {}
        if not isinstance(data, dict):
            return {"ok": False, "error": "invalid_payload"}
        data["ok"] = True
        return data
    except Exception as exc:  # noqa: BLE001
        _logger.error("[AdminOps] ops_summary failed: %s", exc)
        return {"ok": False, "error": "request_failed", "message": str(exc)}


@tool(parse_docstring=True)
def admin_ops_summary(days: int = 7, *, config: RunnableConfig) -> str:
    """拉取群智能体运营只读摘要（近 N 天 run / 管理员对接 / 会话桶计数）。

    Args:
        days: 回溯天数，默认 7，最大 30。
    """
    metadata = (config or {}).get("metadata") or {}
    if not is_admin_debug_source(metadata):
        return json.dumps(
            {
                "ok": False,
                "error": "admin_mode_required",
                "message": "仅运营管理员模式可用",
            },
            ensure_ascii=False,
        )
    payload = _fetch_ops_summary(days=days)
    return json.dumps(payload, ensure_ascii=False)


ADMIN_SYSTEM_PROMPT = """你是「群智能体运营管理员助手」（只读运营脑）。这是你的唯一身份。

## 身份（最高优先级）
- 用户问「你是谁 / 你干什么」时，必须回答：你是群智能体的**运营管理员助手**，帮管理员查看 run、管理员对接申请、会话桶等运营状态。
- **禁止**自称「群内智能体」「找搭子小助手」「匹配顾问」或任何会员侧人设。
- 不要挖会员 doing/need/offer，不要调用 save_group_profile，不要承诺匹配人选。

## 能力
- 可用工具：`admin_ops_summary`（只读：近期 run、管理员对接申请、会话桶计数）。
- 用中文简短汇报；需要数字时先调工具，禁止编造统计。
- 可主动问管理员想看哪一类数据（今日 run / 待处理对接 / 近 7 天摘要）。

## 红线
- 不输出手机号/微信号等敏感联系方式。
- 不执行写库、分发、接受对接等操作——本模式只读。
- 若工具失败，如实说明，不要假装已查到数据。
"""
