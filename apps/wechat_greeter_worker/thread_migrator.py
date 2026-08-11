"""Thread migrator for wechat_greeter (REQ-050 验收 4 / FR-09).

当用户从 openid 升级到 user_id（注册/绑定微信）时, 旧 thread_id="openid:xxx"
的 checkpoint 必须迁移到新 thread_id="user_id:yyy" + 显式销毁旧 key（防膨胀）。

Flow:
  1. 读旧 checkpoint（可能不存在, 容错）
  2. 摘要旧 checkpoint（A 阶段: 仅消息计数; B 阶段: 接入 LLM 摘要）
  3. 写入新 checkpoint（含 migrated_from + summary + 旧消息列表）
  4. 显式 store.delete(old_key) (REQ-050 验收 4 硬要求)
  5. 上报 wechat_msg_thread_migrated 监控事件
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from wechat_greeter.observer import WechatGreeterObserver

_logger = logging.getLogger(__name__)


class CheckpointStore(Protocol):
    """Minimal protocol for checkpoint store. 真实实现: langgraph MemorySaver / Redis / DiskBackedInMemorySaver."""

    def get(self, key: str) -> dict[str, Any] | None: ...
    def put(self, key: str, value: dict[str, Any]) -> None: ...
    def delete(self, key: str) -> bool: ...


def _summarize_old_checkpoint(old: dict[str, Any] | None) -> dict[str, Any]:
    """摘要旧 checkpoint. A 阶段: 简单计数. B 阶段: 接 LLM 摘要."""
    if not old:
        return {"summary": "no_prior_history", "message_count": 0}
    messages = old.get("messages", []) or []
    return {
        "summary": f"prior_history_{len(messages)}_messages",
        "message_count": len(messages),
    }


def migrate_thread(
    *,
    store: CheckpointStore,
    old_key: str,
    new_key: str,
) -> dict[str, Any]:
    """迁移 checkpoint. 显式 del 旧 key (防膨胀).

    Returns:
        {
            "status": "migrated" | "no_migration_needed" | "no_old_checkpoint",
            "old_key": str,
            "new_key": str,
            "old_deleted": bool,
            "summary": dict,
        }
    """
    if old_key == new_key:
        WechatGreeterObserver.info(
            f"wechat_msg_thread_migrated_skip old_key==new_key={old_key}"
        )
        return {"status": "no_migration_needed", "old_key": old_key, "new_key": new_key}

    # 1. 读旧
    old = store.get(old_key)

    if old is None:
        WechatGreeterObserver.info(
            f"wechat_msg_thread_migrated old_key={old_key} new_key={new_key} no_old_checkpoint=true"
        )
        # 即便没旧 checkpoint, 也要确保旧 key 不存在（容错）
        store.delete(old_key)
        return {
            "status": "no_old_checkpoint",
            "old_key": old_key,
            "new_key": new_key,
            "old_deleted": True,
            "summary": {"summary": "no_prior_history", "message_count": 0},
        }

    # 2. 摘要
    summary = _summarize_old_checkpoint(old)

    # 3. 写新
    new_value: dict[str, Any] = {
        "migrated_from": old_key,
        "summary": summary,
        "prior_messages": old.get("messages", []) or [],
    }
    store.put(new_key, new_value)

    # 4. 显式 del 旧 (REQ-050 验收 4 硬要求 - 防 checkpoint 膨胀)
    deleted = store.delete(old_key)

    # 5. 上报监控
    WechatGreeterObserver.info(
        f"wechat_msg_thread_migrated old_key={old_key} new_key={new_key} "
        f"deleted={deleted} message_count={summary.get('message_count', 0)}"
    )

    return {
        "status": "migrated",
        "old_key": old_key,
        "new_key": new_key,
        "old_deleted": deleted,
        "summary": summary,
    }
