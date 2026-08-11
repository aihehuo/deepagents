"""get_user_by_openid 工具 (REQ-050 工具 1/5).

openid → user_id 解析. 无 user_id 注入（user_id 是结果而非输入）.

A 阶段: stub (返回 mock user_id). B 阶段: HMAC 调 aihehuomicro /internal/wechat_greeter/user_by_openid
"""

from __future__ import annotations

from typing import Any


def get_user_by_openid(openid: str) -> dict[str, Any]:
    """公开工具: openid → user_id.

    Returns:
        {
            "user_id": int (0 if unknown),
            "openid": str,
            "stub": bool (A 阶段冒烟: True; B 阶段: False)
        }
    """
    if not openid:
        return {"user_id": 0, "openid": openid, "stub": True}

    # A 阶段冒烟 stub: deterministic mock by hash(openid)
    user_id = abs(hash(openid)) % 100_000
    return {"user_id": user_id, "openid": openid, "stub": True}
