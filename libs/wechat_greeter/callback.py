"""HMAC callback client to new_api (REQ-061 联调方 A).

Canonical: "#{ts}\n#{method}\n#{path}\n#{body}"
Headers:
  - X-GA-From: wechat_greeter        (跨 3 端点统一，TSD-09 v0.1 DRAFT §3.2/3.3/3.4)
  - X-GA-Ts: <unix epoch seconds>
  - X-GA-Signature: hmac_sha256(secret, canonical)

A 阶段冒烟：单 httpx.post 调用，不带 SSRF allowlist（new-api 内部网络，可信）。
A 阶段正式实施：加 retry + 指数退避 + 监控埋点 wechat_msg_callback_failed。
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

_logger = logging.getLogger("uvicorn.error")


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


def post_callback(envelope: dict[str, Any], *, timeout_s: float = 10.0) -> httpx.Response:
    """Send callback to new_api with HMAC signature.

    Returns httpx.Response so caller can inspect status / body / raise_for_status.
    A 阶段冒烟：被 tests/wechat_greeter/test_req050_acceptance.py mock，不真打 new_api。
    """
    body_str = json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    headers = sign_callback_headers(body=body_str)
    url = new_api_callback_url()
    _logger.info(
        "post_callback url=%s msg_id=%s reply_len=%s",
        url,
        envelope.get("msg_id"),
        len(str(envelope.get("reply", ""))),
    )
    return httpx.post(url, content=body_str, headers=headers, timeout=timeout_s)
