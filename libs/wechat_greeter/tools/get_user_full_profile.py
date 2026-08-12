"""get_user_full_profile 工具 (REQ-063 P0-4, user_id 注入层).

替代 v1 的 get_profile_status + get_project_status（2→1 收敛）。
签名（绑后）: () -> dict
user_id 由 functools.partial / 闭包注入, 不出现在 LLM tool schema 中（IDOR 防御第 2 层）。

返回 4 段结构化 JSON（profile / seeking / hiring / published_projects），
供 LLM 读取已注册用户的完整背景。

REQ-063 P0-4: 删"打样中"假数据 stub → 接真 HMAC 调 aihehuomicro。
Fail-closed: HMAC 未配置或后端不可达 → 抛 RuntimeError（worker 捕获 → SC-01 guest 兜底）。
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from wechat_greeter.micro_client import get_user_full_profile as _hmac_get_user_full_profile

_logger = logging.getLogger(__name__)


def _impl(*, user_id: int) -> dict[str, Any]:
    """真实实现 (REQ-063 P0-4): HMAC 调 aihehuomicro /internal/wechat_greeter/user_full_profile.

    Fail-closed: 任何失败 → raise RuntimeError. Worker 捕获后按 SC-01 guest 兜底.
    """
    if user_id <= 0:
        raise RuntimeError(
            f"get_user_full_profile called with invalid user_id={user_id} — "
            "guest users have no profile (fail-closed)"
        )

    try:
        data = _hmac_get_user_full_profile(user_id)
        _logger.info(
            f"get_user_full_profile user_id={user_id} ok=true "
            f"seeking={len(data.get('seeking', []))} "
            f"hiring={len(data.get('hiring', []))} "
            f"projects={len(data.get('published_projects', []))}"
        )
        return data
    except Exception as exc:  # noqa: BLE001
        _logger.error(
            f"get_user_full_profile user_id={user_id} FAILED {type(exc).__name__}: {exc} → fail-closed"
        )
        raise


def make_get_user_full_profile(user_id: int) -> Callable[..., dict[str, Any]]:
    """user_id 注入（闭包模式）.

    LLM 看到的工具签名: get_user_full_profile() -> dict
    实际执行时: _impl(user_id=<injected>)
    """
    def _bound() -> dict[str, Any]:
        return _impl(user_id=user_id)
    _bound.__name__ = "get_user_full_profile"
    _bound.__qualname__ = "get_user_full_profile"
    _bound.__doc__ = (
        "Get full profile for the current user (4 segments: profile / seeking / hiring / published_projects). "
        "user_id is implicit and cannot be set by the LLM (IDOR defense layer 2/3). "
        "Returns real data from aihehuomicro (REQ-063)."
    )
    return _bound
