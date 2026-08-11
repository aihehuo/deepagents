"""UCObserver subclass for wechat_greeter (UC-35).

必报事件（跨仓 P2 协调点，PRD-07 §3.3 / TSD-09 §3.3）：
  - wechat_msg_idor_blocked        (id 与 session 不匹配时上报)
  - wechat_msg_thread_migrated     (FR-09 触发时上报)
  - wechat_msg_24h_expired_worker  (24h 死信判定触发时上报)
"""

from __future__ import annotations

from deepagents.observability import UCObserver


class WechatGreeterObserver(UCObserver):
    """UC-35: wechat_greeter observability namespace = uc_35_wechat_greeter.

    复用 libs/deepagents/deepagents/observability.py:UCObserver 跨库范式（per memory: 可观测性专项）。
    写入路径：~/.deepagents/logs/uc_uc_35_wechat_greeter.log（uc_name 前缀加 "uc_"）。
    """

    uc_name = "uc_35_wechat_greeter"
