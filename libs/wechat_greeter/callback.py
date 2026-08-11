"""HMAC callback client to new_api (REQ-061 联调方 A).

Canonical: "#{ts}\n#{method}\n#{path}\n#{body}"
Headers:
  - X-GA-From: wechat_greeter        (跨 3 端点统一，TSD-09 v0.1 DRAFT §3.2/3.3/3.4)
  - X-GA-Ts: <unix epoch seconds>
  - X-GA-Signature: hmac_sha256(secret, canonical)

A 阶段冒烟：单 httpx.post 调用，不带 SSRF allowlist（new-api 内部网络，可信）。
A 阶段正式实施 (NIT-M2 修订): retry + 指数退避 + 监控埋点 wechat_msg_callback_retry_count。
  - 默认 3 次 retry, 退避 0.5s / 1s / 2s
  - 仅对网络异常/5xx 重试, 4xx 不重试 (4xx 是 new_api 业务错误, 重试无意义)
  - 监控埋点: wechat_msg_callback_retry_count (counter) + wechat_msg_callback_failed (counter)
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from typing import Any

import httpx

from wechat_greeter.config import new_api_callback_url, new_api_hmac_secret
from wechat_greeter.observer import WechatGreeterObserver

_logger = logging.getLogger("uvicorn.error")


# NIT-M2: 指数退避配置 (秒). 总耗时 ≤ 0.5 + 1 + 2 = 3.5s
_CALLBACK_RETRY_DELAYS_S = (0.5, 1.0, 2.0)
_CALLBACK_RETRYABLE_STATUSES = (500, 502, 503, 504)


def sign_callback_headers(
    *,
    body: str,
    ts: str | None = None,
    path: str = "/wechat_greeter_callbacks",
    method: str = "POST",
    secret: str | None = None,
) -> dict[str, str]:
    """Build X-GA-* headers for a callback to new_api.

    Canonical: "#{ts}\\n#{method}\\n#{path}\\n#{body}"
    """
    ts_s = ts or str(int(time.time()))
    key = (secret if secret is not None else new_api_hmac_secret()).encode("utf-8")
    if not key:
        raise ValueError("missing HMAC secret for new_api callback (set HMAC_SECRET_NEW_API)")
    canonical = f"{ts_s}\n{method}\n{path}\n{body}"
    sig = hmac.new(key, canonical.encode("utf-8"), hashlib.sha256).hexdigest()
    return {
        "X-GA-From": "wechat_greeter",
        "X-GA-Ts": ts_s,
        "X-GA-Signature": sig,
        "Content-Type": "application/json",
    }


def _is_retryable_response(resp: httpx.Response) -> bool:
    """仅 5xx 重试. 4xx 是 new_api 业务错误, 重试无意义 (HMAC 错/字段错等)."""
    return resp.status_code in _CALLBACK_RETRYABLE_STATUSES


def post_callback(
    envelope: dict[str, Any],
    *,
    timeout_s: float = 10.0,
) -> httpx.Response:
    """Send callback to new_api with HMAC signature + 指数退避 retry.

    Returns httpx.Response so caller can inspect status / body / raise_for_status.
    Raises last exception if all retries fail.
    A 阶段冒烟：被 tests/wechat_greeter/test_req050_acceptance.py mock，不真打 new_api。
    """
    body_str = json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    headers = sign_callback_headers(body=body_str)
    url = new_api_callback_url()
    msg_id = envelope.get("msg_id")

    _logger.info(
        "post_callback url=%s msg_id=%s reply_len=%s",
        url, msg_id, len(str(envelope.get("reply", ""))),
    )

    # NIT-M2: 指数退避 retry. 仅网络异常 + 5xx 重试, 4xx 立即 raise
    last_exc: Exception | None = None
    for attempt, delay in enumerate((0.0,) + _CALLBACK_RETRY_DELAYS_S, start=1):
        if delay > 0:
            time.sleep(delay)
        try:
            resp = httpx.post(url, content=body_str, headers=headers, timeout=timeout_s)
            if resp.status_code < 400:
                if attempt > 1:
                    WechatGreeterObserver.info(
                        f"wechat_msg_callback_retry_count msg_id={msg_id} "
                        f"attempt={attempt} status={resp.status_code} ok=true"
                    )
                return resp
            if not _is_retryable_response(resp):
                # 4xx: 业务错误, 不重试
                WechatGreeterObserver.warn(
                    f"wechat_msg_callback_failed msg_id={msg_id} "
                    f"status={resp.status_code} reason=4xx_no_retry"
                )
                return resp
            # 5xx: 重试
            WechatGreeterObserver.warn(
                f"wechat_msg_callback_retry_count msg_id={msg_id} "
                f"attempt={attempt} status={resp.status_code} retry=true"
            )
            last_exc = httpx.HTTPStatusError(
                f"5xx from new_api: {resp.status_code}", request=resp.request, response=resp
            )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            WechatGreeterObserver.warn(
                f"wechat_msg_callback_retry_count msg_id={msg_id} "
                f"attempt={attempt} exc={type(exc).__name__} retry=true"
            )
            last_exc = exc

    # 全部 retry 失败
    WechatGreeterObserver.warn(
        f"wechat_msg_callback_failed msg_id={msg_id} attempts={len(_CALLBACK_RETRY_DELAYS_S) + 1} reason=all_retries_exhausted"
    )
    if last_exc is not None:
        raise last_exc
    # 理论上不会到这里 (总会有一次 attempt), 兜底
    raise RuntimeError(f"post_callback failed msg_id={msg_id} no response captured")
