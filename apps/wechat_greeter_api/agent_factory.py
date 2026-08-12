"""3 tools for wechat_greeter (REQ-062 v2).

v2 工具集 (REQ-062): 5 → 3 工具
  - 保留: get_user_by_openid, get_user_faq
  - 新增: get_user_full_profile (替代 v1 的 get_profile_status + get_project_status)
  - 删除: get_profile_status, get_project_status, mark_reply_sent (v1)

IDOR 三层防御（user_id 不暴露 LLM）：
  1. 工具签名层剔除 user_id（get_user_full_profile 签名无参数）
  2. functools.partial/闭包 注入层（make_get_user_full_profile(user_id) → bound function with user_id captured）
  3. aihehuomicro 服务端再校验层（HMAC from 头 + X-GA-From=wechat_greeter + user_id from session）
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from libs.wechat_greeter.tools import (
    get_user_by_openid,
    make_get_user_full_profile,
)

_logger = logging.getLogger(__name__)


def get_user_faq(query: str) -> list[dict[str, Any]]:
    """公开工具 3/3：FAQ 检索 (FAISS, REQ-051).

    C 阶段: 走 libs/wechat_greeter/faq_store.py 纯 Python fallback (jaccard keyword match).
    B 阶段预留: 真 FAISS 接入时改 faq_store.search() 内部, 对外签名不变.
    """
    from libs.wechat_greeter import faq_store  # 延后导入, 避免循环依赖
    return faq_store.search(query, top_k=3, min_score=0.05)


def make_tools(*, user_id: int) -> list[Callable[..., Any]]:
    """Build the 3 tools for the agent. user_id is injected via closure.

    Returns list of 3 callables:
      [0] get_user_by_openid           signature: (openid: str) -> dict
      [1] get_user_full_profile (bound) signature: () -> dict   ← user_id injected
      [2] get_user_faq                 signature: (query: str) -> list[dict]
    """
    return [
        get_user_by_openid,
        make_get_user_full_profile(user_id),
        get_user_faq,
    ]
