"""FastAPI app for wechat_greeter_api (REQ-050).

A 阶段冒烟最小实现：
  - POST /call_async  : HMAC 验签（X-GA-From + X-GA-Ts + X-GA-Signature） + 202 Accepted + 持久化入队
  - GET  /healthz     : 健康检查（无 HMAC）
  - GET  /            : 返回 build_version（无 HMAC）

HMAC Canonical: "#{ts}\\n#{method}\\n#{path}\\n#{body}"

A 阶段冒烟：用 Celery .delay() + CELERY_TASK_ALWAYS_EAGER=1 同步跑 worker（测试用），生产用 Redis broker。
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from wechat_greeter.config import (
    api_port,
    dry_run,
    model_mode,
    new_api_hmac_secret,
    timestamp_skew_s,
)
from wechat_greeter.faq_store import get_faq_count

_logger = logging.getLogger("uvicorn.error")

# Build version (replaced at build time by CI; default is "dev")
BUILD_VERSION = "dev-wechat-greeter-0.1.0-a-stage-smoke"

app = FastAPI(
    title="wechat_greeter_api",
    description="微信公众号客服/介绍智能体 (UC-35 / REQ-050 / REQ-051)",
    version=BUILD_VERSION,
)


# ---------------------------------------------------------------------------
# HMAC 验签
# ---------------------------------------------------------------------------

def _verify_wechat_greeter_hmac(
    *,
    body: bytes,
    headers: dict,
    path: str,
    method: str = "POST",
) -> None:
    """Verify X-GA-From + X-GA-Ts + X-GA-Signature. Raises HTTPException 401 on failure.

    失败分支（REQ-050 验收 1, A 阶段正式实施要补全 3 失败单测）：
      - missing_headers     : 缺 X-GA-From / X-GA-Ts / X-GA-Signature 任一
      - unknown_from        : X-GA-From != "wechat_greeter"
      - invalid_timestamp   : |now - ts| > timestamp_skew_s
      - invalid_signature   : hmac.compare_digest 失败
    """
    # 1. X-GA-From
    from_header = (headers.get("x-ga-from") or "").strip()
    if not from_header:
        raise HTTPException(
            status_code=401,
            detail={"error": "missing_headers", "missing": "X-GA-From"},
        )
    if from_header != "wechat_greeter":
        raise HTTPException(
            status_code=401,
            detail={"error": "unknown_from", "from": from_header, "expected": "wechat_greeter"},
        )

    # 2. X-GA-Ts
    ts = (headers.get("x-ga-ts") or "").strip()
    if not ts:
        raise HTTPException(
            status_code=401,
            detail={"error": "missing_headers", "missing": "X-GA-Ts"},
        )
    try:
        ts_int = int(ts)
    except ValueError:
        raise HTTPException(
            status_code=401,
            detail={"error": "invalid_timestamp", "ts": ts, "reason": "not_an_integer"},
        )

    now = int(time.time())
    skew = timestamp_skew_s()
    if abs(now - ts_int) > skew:
        raise HTTPException(
            status_code=401,
            detail={"error": "invalid_timestamp", "ts": ts_int, "now": now, "skew_s": abs(now - ts_int)},
        )

    # 3. X-GA-Signature
    sig = (headers.get("x-ga-signature") or "").strip()
    if not sig:
        raise HTTPException(
            status_code=401,
            detail={"error": "missing_headers", "missing": "X-GA-Signature"},
        )

    # 4. Secret
    secret = new_api_hmac_secret()
    if not secret:
        raise HTTPException(
            status_code=503,
            detail={"error": "server_misconfigured", "message": "HMAC secret not set (HMAC_SECRET_NEW_API)"},
        )

    # 5. 验签
    body_str = body.decode("utf-8")
    canonical = f"{ts}\n{method}\n{path}\n{body_str}"
    expected = hmac.new(secret.encode("utf-8"), canonical.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected):
        raise HTTPException(
            status_code=401,
            detail={"error": "invalid_signature"},
        )


# ---------------------------------------------------------------------------
# 端点
# ---------------------------------------------------------------------------

@app.get("/healthz")
async def healthz() -> dict:
    """Liveness + D-1 readiness: 灰度切档决策面板.

    老板 2026-08-11 拍板 D-1:
      - status: ok | dry_run (灰度切档前用)
      - model_mode: stub | deepseek (切档前确认)
      - dry_run: bool (切档前 True, 切档后 False)
      - faq_count: int (FAQ 索引条数, 0 需警惕)
    """
    is_dry = dry_run()
    return {
        "status": "dry_run" if is_dry else "ok",
        "service": "wechat_greeter_api",
        "build_version": BUILD_VERSION,
        "port": str(api_port()),
        "model_mode": model_mode(),
        "dry_run": is_dry,
        "faq_count": get_faq_count(),
    }


@app.get("/")
async def root() -> dict[str, str]:
    """返回 build_version。"""
    return {
        "service": "wechat_greeter_api",
        "build_version": BUILD_VERSION,
    }


@app.post("/call_async")
async def call_async(request: Request) -> JSONResponse:
    """接受 wechat 客服请求 → HMAC 验签 → 持久化入队 → 202 Accepted。

    Request body (JSON):
      - openid:   str (微信 openid)
      - content:  str (用户消息)
      - send_time: int (unix seconds, 微信发来时间戳)
      - msg_id:   str (可选, 默认 auto-generated)
    """
    body_bytes = await request.body()
    body_str = body_bytes.decode("utf-8")

    # 1. HMAC 验签（失败 → 401）
    _verify_wechat_greeter_hmac(
        body=body_bytes,
        headers=dict(request.headers),
        path=request.url.path,
    )

    # 2. Parse body
    try:
        payload = json.loads(body_str) if body_str else {}
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_json", "message": str(exc)},
        )

    # 3. Build envelope
    msg_id = payload.get("msg_id") or f"auto_{uuid.uuid4().hex[:12]}"
    send_time = int(payload.get("send_time") or int(time.time()))
    envelope = {
        "msg_id": msg_id,
        "openid": payload.get("openid") or "",
        "content": payload.get("content") or "",
        "send_time": send_time,
        "received_at": int(time.time()),
    }

    # 4. Enqueue（Celery eager mode 同步跑；生产用 Redis broker）
    from apps.wechat_greeter_worker.tasks import process_greeting
    process_greeting.delay(envelope)

    _logger.info(
        "call_async accepted msg_id=%s openid=%s send_time=%s received_at=%s",
        msg_id, envelope["openid"], send_time, envelope["received_at"],
    )

    return JSONResponse(
        status_code=202,
        content={"msg_id": msg_id, "status": "accepted"},
    )
