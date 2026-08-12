"""get_user_by_openid 工具 (REQ-063 P0-3).

openid → user_id 解析。无 user_id 注入（user_id 是结果而非输入）。

REQ-063 P0-3: 删 hash stub → 接真 HMAC 调 aihehuomicro。
Fail-closed: HMAC 未配置或网络异常 → user_id=0 (guest)，不阻主流程。
"""

from __future__ import annotations

import logging
from typing import Any

from wechat_greeter.micro_client import get_user_by_openid as _hmac_get_user_by_openid

_logger = logging.getLogger(__name__)


def get_user_by_openid(openid: str) -> dict[str, Any]:
    """公开工具: openid → user_id (REQ-063: 真 HMAC, fail-closed).

    Returns:
        {
            "user_id": int (0 if not found or error),
            "openid": str,
            "source": "aihehuomicro" | "fail_closed",
        }
    """
    if not openid:
        _logger.warning("get_user_by_openid called with empty openid → guest")
        return {"user_id": 0, "openid": "", "source": "fail_closed"}

    try:
        result = _hmac_get_user_by_openid(openid)
        user_id = int(result.get("user_id") or 0)
        _logger.info(
            f"get_user_by_openid openid={openid[:12]}... user_id={user_id} "
            f"source={result.get('source', 'unknown')}"
        )
        return result
    except Exception as exc:  # noqa: BLE001
        _logger.warning(
            f"get_user_by_openid openid={openid[:12]}... failed {type(exc).__name__}: {exc} → guest (fail-closed)"
        )
        return {"user_id": 0, "openid": openid, "error": str(exc), "source": "fail_closed"}
