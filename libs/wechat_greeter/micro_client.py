"""HMAC 内部 HTTP 客户端：deepagents → aihehuomicro (REQ-063, REQ-064 P0).

给 get_user_by_openid / get_user_full_profile 两个工具共用。

REQ-064 P0: 签名契约与 aihehuomicro WechatGreeter::HmacVerifier.canonical_payload 字节一致:
  canonical = ts + "\\n" + method + "\\n" + path + "\\n" + body  (4 段, 不含 query)
  header ts = X-GA-Ts (不是 X-GA-Timestamp)
  参照: callback.py sign_callback_headers() 同样 4 段契约。
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
from typing import Any

import httpx

from wechat_greeter.config import aihehuomicro_base_url, aihehuomicro_hmac_secret

_logger = logging.getLogger(__name__)

# 指数退避 (秒). 总重试 ≤ 3 次
_RETRY_DELAYS_S = (0.3, 1.0, 2.0)


def _sign_headers(
    *,
    method: str,
    path: str,
    body: str = "",
    ts: str | None = None,
) -> dict[str, str]:
    """Build X-GA-* headers for aihehuomicro HMAC call.

    REQ-064 P0: canonical 4 段 = ts + "\\n" + method + "\\n" + path + "\\n" + body
    与 aihehuomicro HmacVerifier.canonical_payload 字节一致。
    GET 请求 body 为空串, query 参数走 URL 但不参与签名。
    """
    ts_s = ts or str(int(time.time()))
    secret = aihehuomicro_hmac_secret()
    if not secret:
        raise RuntimeError("HMAC_SECRET_AIHEHUOMICRO not set — fail-closed (REQ-063 P0-3/4)")

    canonical = f"{ts_s}\n{method.upper()}\n{path}\n{body}"
    sig = hmac.new(secret.encode("utf-8"), canonical.encode("utf-8"), hashlib.sha256).hexdigest()

    return {
        "X-GA-From": "wechat_greeter",
        "X-GA-Ts": ts_s,
        "X-GA-Signature": sig,
        "Content-Type": "application/json",
    }


def _get(
    path: str,
    query_params: dict[str, str] | None = None,
    *,
    timeout_s: float = 3.0,
) -> httpx.Response:
    """HMAC-signed GET to aihehuomicro with retry. Raises on all-retries-exhausted.

    REQ-064 P0: query 参数通过 URL 发送 (httpx params=...), 不参与 HMAC 签名。
    """
    base = aihehuomicro_base_url()
    url = f"{base}{path}"
    headers = _sign_headers(method="GET", path=path)

    last_exc: Exception | None = None
    for attempt, delay in enumerate((0.0,) + _RETRY_DELAYS_S, start=1):
        if delay > 0:
            time.sleep(delay)
        try:
            resp = httpx.get(url, params=query_params, headers=headers, timeout=timeout_s)
            if resp.status_code < 500:
                return resp
            last_exc = httpx.HTTPStatusError(
                f"5xx from aihehuomicro: {resp.status_code}",
                request=resp.request,
                response=resp,
            )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            last_exc = exc

    if last_exc:
        raise last_exc
    raise RuntimeError(f"aihehuomicro GET {path} all retries exhausted")


def get_user_by_openid(openid: str) -> dict[str, Any]:
    """HMAC GET /internal/wechat_greeter/user_by_openid?openid=xxx

    Returns: {"user_id": int, ...} — user_id=0 if not found.
    Fail-closed: HMAC secret not set or network error → {"user_id": 0, "error": "..."}
    """
    secret = aihehuomicro_hmac_secret()
    if not secret:
        _logger.warning("HMAC_SECRET_AIHEHUOMICRO not set — get_user_by_openid fail-closed → guest")
        return {"user_id": 0, "openid": openid, "error": "hmac_not_configured", "source": "fail_closed"}

    try:
        resp = _get("/internal/wechat_greeter/user_by_openid", {"openid": openid})
        if resp.status_code == 200:
            data = resp.json()
            return {
                "user_id": int(data.get("user_id") or 0),
                "openid": openid,
                "source": "aihehuomicro",
            }
        # 4xx: user not found or bad request → guest
        _logger.warning(
            f"get_user_by_openid openid={openid[:12]}... status={resp.status_code} → guest"
        )
        return {"user_id": 0, "openid": openid, "error": f"http_{resp.status_code}", "source": "fail_closed"}
    except Exception as exc:
        _logger.warning(
            f"get_user_by_openid openid={openid[:12]}... failed {type(exc).__name__}: {exc} → guest (fail-closed)"
        )
        return {"user_id": 0, "openid": openid, "error": str(exc), "source": "fail_closed"}


def get_user_full_profile(user_id: int) -> dict[str, Any]:
    """HMAC GET /internal/wechat_greeter/user_full_profile?user_id=XXX

    Returns: 4-segment structured JSON (profile/seeking/hiring/published_projects).
    Fail-closed: HMAC secret not set or network error → raises RuntimeError (worker catches → SC-01 guest).
    """
    secret = aihehuomicro_hmac_secret()
    if not secret:
        raise RuntimeError(
            "HMAC_SECRET_AIHEHUOMICRO not set — get_user_full_profile fail-closed (REQ-063 P0-4). "
            "Cannot inject profile into LLM context without backend."
        )

    resp = _get("/internal/wechat_greeter/user_full_profile", {"user_id": str(user_id)})
    if resp.status_code == 200:
        data = resp.json()
        if not data.get("ok"):
            raise RuntimeError(f"aihehuomicro user_full_profile returned ok=false for user_id={user_id}")
        return data

    raise RuntimeError(
        f"aihehuomicro user_full_profile HTTP {resp.status_code} for user_id={user_id}"
    )
