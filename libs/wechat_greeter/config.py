"""Config for wechat_greeter (UC-35 / REQ-050 / REQ-051).

环境变量读取统一入口。绝不读 .env 真实 secret 值，只读 os.environ 引用 —— 真实 secret 由部署环境注入。
"""

from __future__ import annotations

import os


# ---------------------------------------------------------------------------
# HMAC 跨仓共享密钥（绝不读真实值，仅引用 env 名）
# ---------------------------------------------------------------------------

def new_api_hmac_secret() -> str:
    """HMAC 共享密钥（new_api 端 ↔ deep agents 端）.

    对应 new_api 端 env 名：SIMPLE_HMAC_SHARED_SECRET（老板拍板统一命名）。
    跨 3 端点统一 X-GA-From=wechat_greeter（TSD-09 v0.1 DRAFT §3.2/3.3/3.4）。
    """
    return (os.environ.get("HMAC_SECRET_NEW_API") or "").strip()


def aihehuomicro_hmac_secret() -> str:
    """HMAC 共享密钥（aihehuomicro 端 ↔ deep agents 端）.

    对应 aihehuomicro 端 env 名：MICRO_HMAC_SECRET。
    用于 deep agents → aihehuomicro /internal/wechat_greeter/{user_by_openid,profile_status,project_status}。
    """
    return (os.environ.get("HMAC_SECRET_AIHEHUOMICRO") or "").strip()


# ---------------------------------------------------------------------------
# 端点 / URL
# ---------------------------------------------------------------------------

def new_api_callback_url() -> str:
    """Callback URL to new_api (REQ-061 联调方 A).

    老板 2026-08-11 拍板默认：http://new-api:3000/wechat_greeter_callbacks。
    """
    return (
        os.environ.get("DEEPAGENTS_WECHAT_GREETER_CALLBACK_URL")
        or "http://new-api:3000/wechat_greeter_callbacks"
    ).strip()


def api_port() -> int:
    """API 容器端口。"""
    return int(os.environ.get("WECHAT_GREETER_API_PORT", "8005"))


# ---------------------------------------------------------------------------
# HMAC 验签参数
# ---------------------------------------------------------------------------

def timestamp_skew_s() -> int:
    """±timestamp_skew_s seconds 接受窗口（默认 5 分钟 = 300s）.

    new_api 端 verifier 同步使用此阈值（PRD-07 §3.3 拍板）。
    """
    return int(os.environ.get("WECHAT_GREETER_HMAC_TIMESTAMP_SKEW_S", "300"))


# ---------------------------------------------------------------------------
# 模型 / 行为参数
# ---------------------------------------------------------------------------

def model_mode() -> str:
    """LLM 模式：stub | deepseek。A 阶段冒烟默认 stub，A 阶段正式实施锁 deepseek。"""
    return (os.environ.get("WECHAT_GREETER_MODEL_MODE") or "stub").strip().lower()


def hard_truncate_limit() -> int:
    """硬截断上限（REQ-050 验收 6：≤ 200 字 + 固定尾巴）。"""
    return int(os.environ.get("WECHAT_GREETER_TRUNCATE_LIMIT", "200"))


def hard_truncate_tail() -> str:
    """固定尾巴（REQ-050 验收 6）。"""
    return os.environ.get(
        "WECHAT_GREETER_TRUNCATE_TAIL",
        "〔详情见 App,扫码看完整建议〕",
    )


def dead_letter_after_s() -> int:
    """24h 死信阈值（REQ-050 验收 5 / EX-06：send_time < 24h.ago → 死信 + 不调 callback）。"""
    return int(os.environ.get("WECHAT_GREETER_DEAD_LETTER_AFTER_S", str(24 * 3600)))
