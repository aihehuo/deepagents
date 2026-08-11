"""get_profile_status 工具 (REQ-050 工具 2/5, user_id 注入层 1/2).

签名（绑后）: (query: str = "") -> dict
user_id 由 functools.partial / 闭包注入, 不出现在 LLM tool schema 中（IDOR 防御第 2 层）。

A 阶段: stub. B 阶段: HMAC 调 aihehuomicro /internal/wechat_greeter/profile_status
"""

from __future__ import annotations

from typing import Any, Callable


def _impl(*, user_id: int, query: str = "") -> dict[str, Any]:
    """真实实现. A 阶段冒烟 stub."""
    return {
        "user_id": user_id,
        "has_profile": True,
        "stub": True,
        "query": query,
    }


def make_get_profile_status(user_id: int) -> Callable[..., dict[str, Any]]:
    """user_id 注入（闭包模式）.

    LLM 看到的工具签名: get_profile_status(query: str = "") -> dict
    实际执行时: _impl(user_id=<injected>, query=query)
    """
    def _bound(query: str = "") -> dict[str, Any]:
        return _impl(user_id=user_id, query=query)
    _bound.__name__ = "get_profile_status"
    _bound.__qualname__ = "get_profile_status"
    _bound.__doc__ = (
        "Get profile status for the current user. "
        "user_id is implicit and cannot be set by the LLM (IDOR defense layer 2/3)."
    )
    return _bound
