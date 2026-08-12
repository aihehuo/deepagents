"""wechat_greeter shared lib (UC-35 v2 / REQ-063 / REQ-062 / REQ-051).

跨进程复用代码（API 容器 + Worker 容器 + 评测 CI）：

- `config`       — 环境变量读取器（HMAC 密钥 / 端口 / 截断阈值 / 死信阈值）
- `observer`     — UCObserver 子类（uc_35_wechat_greeter）
- `callback`     — 给 new_api 发 HMAC 回执的客户端
- `faq_store`    — FAQ 知识库 (FAISS, REQ-051)
- `micro_client` — HMAC HTTP 客户端 → aihehuomicro (REQ-063 P0-3/4)
- `llm_client`   — LLM 统一入口 + bind_tools agent executor (REQ-063 P0-1/2)
- `tools`        — 3 工具（v2: get_user_by_openid / get_user_full_profile / get_user_faq）

v2 (REQ-062): 工具集 5→3，删 get_profile_status/get_project_status/mark_reply_sent，
新增 get_user_full_profile（4 段结构化 JSON: profile/seeking/hiring/published_projects）。

REQ-063: P0-1 bind_tools + P0-2 profile 真注入 + P0-3/4 真 HMAC fail-closed。
"""

from __future__ import annotations

__version__ = "0.2.0-req063-p0"
__all__ = [
    "config",
    "observer",
    "callback",
    "micro_client",
    "llm_client",
]
