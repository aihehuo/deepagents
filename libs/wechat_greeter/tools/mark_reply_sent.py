"""mark_reply_sent 工具 (REQ-050 工具 4/5).

标记客服回复已发送. 无 user_id 注入.

A 阶段: stub. B 阶段: new_api 端 /internal/wechat_greeter/mark_reply_sent
"""

from __future__ import annotations

from typing import Any


def mark_reply_sent(msg_id: str) -> dict[str, Any]:
    """公开工具: 标记 msg_id 的回复已发送.

    Returns:
        {
            "msg_id": str,
            "marked": bool,
            "stub": bool (A 阶段冒烟: True)
        }
    """
    return {"msg_id": msg_id, "marked": True, "stub": True}
