"""wechat_greeter shared lib (UC-35 / REQ-050 / REQ-051).

跨进程复用代码（API 容器 + Worker 容器 + 评测 CI）：

- `config`    — 环境变量读取器（HMAC 密钥 / 端口 / 截断阈值 / 死信阈值）
- `observer`  — UCObserver 子类（uc_35_wechat_greeter）
- `callback`  — 给 new_api 发 HMAC 回执的客户端
- `tools`     — 5 工具（get_user_by_openid / get_profile_status / get_project_status / mark_reply_sent / get_user_faq）— A 阶段冒烟 stub

A 阶段冒烟：仅覆盖 happy path 进队→出队→callback mock 全链路。完整 5 工具 + j2 system_prompt + FAISS + thread_migrator + 50 条负向评测 + CI workflow 留 A/B/C/D 阶段正式实施。

__version__ = "0.1.0-a-stage-smoke"
"""

from __future__ import annotations

__version__ = "0.1.0-a-stage-smoke"
__all__ = [
    "config",
    "observer",
    "callback",
]
