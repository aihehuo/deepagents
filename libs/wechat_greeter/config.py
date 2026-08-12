"""Config for wechat_greeter (UC-35 / REQ-050 / REQ-051).

环境变量读取统一入口。绝不读 .env 真实 secret 值，只读 os.environ 引用 —— 真实 secret 由部署环境注入。
"""

from __future__ import annotations

import hashlib
import os


def openid_hash(openid: str) -> str:
    """REQ-065 P1-12: SHA256 truncated hash for openid PII masking.

    Matches new_api openid_hash strategy: hexdigest[:16] for log safety
    while preserving enough uniqueness to correlate logs across services.
    """
    if not openid:
        return "none"
    return hashlib.sha256(openid.encode("utf-8")).hexdigest()[:16]


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
    用于 deep agents → aihehuomicro — 2 个端点:
      - /internal/wechat_greeter/user_by_openid
      - /internal/wechat_greeter/user_full_profile
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


def aihehuomicro_base_url() -> str:
    """aihehuomicro 内网地址 (REQ-063 P0-3/4 HMAC 调用).

    默认 http://aihehuomicro:3000，生产由 HOST_AIHEHUOMICRO 环境变量注入。
    """
    return (
        os.environ.get("HOST_AIHEHUOMICRO")
        or "http://aihehuomicro:3000"
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
    """LLM 模式：stub | dashscope。

    REQ-065 P0-C7: stub 不再是生产默认。不配置 → RuntimeError (fail-closed)。
    生产统一使用阿里云 DashScope；readiness 要求 model_mode=dashscope。
    stub 仅在显式设置 WECHAT_GREETER_MODEL_MODE=stub 时可用 (测试/CI)。
    """
    raw = (os.environ.get("WECHAT_GREETER_MODEL_MODE") or "").strip().lower()
    if not raw:
        raise RuntimeError(
            "WECHAT_GREETER_MODEL_MODE not set. "
            "Must be explicitly set to 'dashscope' (production/grayscale) or 'stub' (test/CI only). "
            "Ref: REQ-065 P0-C7 — stub is no longer the production default."
        )
    if raw not in ("stub", "dashscope"):
        raise RuntimeError(
            f"WECHAT_GREETER_MODEL_MODE={raw!r} is invalid. "
            f"Must be 'dashscope' (production/grayscale) or 'stub' (test/CI only)."
        )
    return raw


def dashscope_api_key() -> str:
    """DashScope API key (生产唯一允许的模型供应商)."""
    return (os.environ.get("DASHSCOPE_API_KEY") or "").strip()


def readiness_details() -> dict:
    """REQ-065 P1-1: 细粒度 readiness 检查结果。

    每个检查返回 {ok: bool, reason: str}，同时返回 overall: bool。
    部署方用 /ready 端点根据 overall 决定是否接流量。

    P0-1: 移除 aihehuomicro 实时 HTTP 探测 — 端点不存在导致必现阻断。
    P0-3: 移除 dry_run_off — dry_run 是灰度前有效冒烟模式，不应阻止接单。
    P1-1: readiness 仅检查本地静态配置，不做跨服务实时同步调用。
    """
    checks: dict[str, dict] = {}

    # 1. model_mode 已显式配置
    mode: str | None = None
    try:
        mode = model_mode()
        checks["model_mode"] = {"ok": True, "value": mode}
    except RuntimeError as exc:
        checks["model_mode"] = {"ok": False, "reason": str(exc)}

    # 2. model_mode=dashscope (not stub in production)
    if mode == "dashscope":
        checks["model_mode_is_dashscope"] = {"ok": True}
    else:
        checks["model_mode_is_dashscope"] = {
            "ok": False,
            "reason": f"model_mode={mode!r}, must be 'dashscope' for production",
        }

    # 3. DashScope API key. Do not fall back to keys for another provider.
    dashscope_key = dashscope_api_key()
    checks["dashscope_api_key"] = {
        "ok": bool(dashscope_key),
        "reason": None if dashscope_key else "DASHSCOPE_API_KEY is empty or not set",
    }

    # 4. HMAC secret (new_api)
    napi = new_api_hmac_secret()
    checks["hmac_secret_new_api"] = {
        "ok": bool(napi),
        "reason": None if napi else "HMAC_SECRET_NEW_API is empty or not set",
    }

    # 5. HMAC secret (aihehuomicro)
    micro = aihehuomicro_hmac_secret()
    checks["hmac_secret_aihehuomicro"] = {
        "ok": bool(micro),
        "reason": None if micro else "HMAC_SECRET_AIHEHUOMICRO is empty or not set",
    }

    checks["overall"] = all(c.get("ok", False) for k, c in checks.items() if k != "overall")
    return checks


def hard_truncate_limit() -> int:
    """硬截断上限（REQ-050 验收 6：≤ 200 字 + 固定尾巴）。"""
    return int(os.environ.get("WECHAT_GREETER_TRUNCATE_LIMIT", "200"))


def hard_truncate_tail() -> str:
    """固定尾巴（REQ-050 验收 6）。"""
    return os.environ.get(
        "WECHAT_GREETER_TRUNCATE_TAIL",
        "〔详情见 App，扫码看完整建议〕",
    )


def apply_tail_and_truncate(
    raw_reply: str,
    tail: str | None = None,
    limit: int | None = None,
) -> str:
    """Ensure raw_reply ends with tail exactly once.

    Deduplicates existing tail (or common variants like Chinese/English comma or newlines)
    generated by LLM prompt to prevent duplicate tails in output reply.
    If limit is explicitly provided, character slicing is applied. Otherwise, text length
    and sentence integrity are preserved (guided by LLM prompt).
    """
    if tail is None:
        tail = hard_truncate_tail()

    cleaned = raw_reply.strip()
    tail_variants = [
        tail,
        "〔详情见 App，扫码看完整建议〕",
        "〔详情见 App,扫码看完整建议〕",
        "详情见 App，扫码看完整建议",
        "详情见 App,扫码看完整建议",
    ]
    for variant in tail_variants:
        if cleaned.endswith(variant):
            cleaned = cleaned[:-len(variant)].strip()
            break

    if limit is not None:
        tail_len = len(tail)
        if len(cleaned) + tail_len > limit:
            cleaned = cleaned[: limit - tail_len]
    return cleaned + tail


def dead_letter_after_s() -> int:
    """24h 死信阈值（REQ-050 验收 5 / EX-06：send_time < 24h.ago → 死信 + 不调 callback）。"""
    return int(os.environ.get("WECHAT_GREETER_DEAD_LETTER_AFTER_S", str(24 * 3600)))


# ---------------------------------------------------------------------------
# D-1 灰度切档 (跨仓 P0)
# ---------------------------------------------------------------------------

def dry_run() -> bool:
    """D-1 dry-run 模式 (灰度切档前最后一遍 smoke).

    True 时:
      - worker 不真打 callback (post_callback 替换为 log only)
      - 用于切档前在生产流量上跑一遍验证链路 (不污染生产 callback)
    默认: False (生产路径)

    REQ-065 P0-3: dry_run 不阻塞 /ready 和 /call_async readiness gate.
    dry_run 状态不通过 /healthz 暴露 (P1-1 liveness 简化).
    部署方通过独立配置检查确认 dry_run 状态, 或 GET /ready 检查 model_mode + key.

    回滚: 设回 false 即可, 无副作用.
    """
    return os.environ.get("WECHAT_GREETER_DRY_RUN", "").strip().lower() in ("1", "true", "yes", "on")
