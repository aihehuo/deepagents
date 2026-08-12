"""wechat_greeter tools package.

按 TSD-09 v1.0 路径: libs/wechat_greeter/tools/{get_user_by_openid,get_user_full_profile}.py
- v2 工具集: 3 工具 → get_user_faq / get_user_by_openid / get_user_full_profile
- v1 的 get_profile_status / get_project_status / mark_reply_sent 已删除 (REQ-062)
- get_user_faq 在 apps/wechat_greeter_api/agent_factory.py (与 FAISS 绑定)
- 真实 aihehuomicro HMAC 调留 B 阶段 (联调方 aihehuomicro HMAC 2 端点待实施)
"""

from __future__ import annotations

from libs.wechat_greeter.tools.get_user_by_openid import get_user_by_openid
from libs.wechat_greeter.tools.get_user_full_profile import make_get_user_full_profile

__all__ = [
    "get_user_by_openid",
    "make_get_user_full_profile",
]
