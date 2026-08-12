"""Celery task: process a single wechat greeting (REQ-065 P0-A / REQ-063 / REQ-062).

REQ-065 P0-A: callback 端到端字段契约对齐
  P0-A1: worker 入口 msg_id → trace_id (new_api DeepAgentClient.call_async 发的是 trace_id)
  P0-A2: callback_envelope 字段对齐 new_api WechatGreeterCallbacksController:
         reply→reply_text, msg_id→trace_id, 补 branch 字段
  P0-A3: 截断长度修正: raw_reply[:limit - len(tail)] + tail, 保证 len ≤ 200 且 endswith(tail)
  P0-A4: callback 失败可重试: 非 2xx raise_for_status; 4xx 不可重试落告警

REQ-063 P0-1: tools 真传入 call_llm (不再死变量).
REQ-063 P0-2: registered 分支 4 段 profile 真进 LLM 上下文 (pre-fetch → profile_context).
  SC-01: get_user_full_profile RuntimeError → user_id=0 guest 兜底.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import httpx

# IMPORTANT: import celery_app FIRST so it sets the default Celery app
# (via `app.set_default()`) before @shared_task binds. Otherwise the task
# would be registered to a "pending" default app without task_always_eager.
from apps.wechat_greeter_worker.celery_app import app as _celery_app  # noqa: F401
from celery import shared_task

from apps.wechat_greeter_api.agent_factory import make_tools
from wechat_greeter.callback import post_callback
from wechat_greeter.config import (
    dead_letter_after_s,
    dry_run,
    hard_truncate_limit,
    hard_truncate_tail,
)
from wechat_greeter.llm_client import call_llm
from wechat_greeter.observer import WechatGreeterObserver

_logger = logging.getLogger(__name__)


# REQ-065 P0-A4: 4xx HTTP status codes that are irrecoverable (don't retry)
_IRRECOVERABLE_CALLBACK_STATUSES = frozenset({400, 401, 403, 404, 405, 422})


@shared_task(name="wechat_greeter.process_greeting", bind=True, ignore_result=True)
def process_greeting(self, envelope: dict[str, Any]) -> dict[str, Any]:
    """Process a single wechat greeting (REQ-065 P0-A v2).

    new_api dispatch worker sends: {trace_id, openid, content, callback_url}
    """
    # REQ-065 P0-A1: 读 trace_id (不是 msg_id)
    trace_id = str(envelope.get("trace_id") or "unknown")
    openid = str(envelope.get("openid") or "")
    content = str(envelope.get("content") or "")
    send_time = int(envelope.get("send_time") or 0)
    now = int(time.time())

    WechatGreeterObserver.info(
        f"process_greeting start trace_id={trace_id} content_len={len(content)}"
    )

    # 1. 24h 死信判定
    if send_time and (now - send_time) > dead_letter_after_s():
        WechatGreeterObserver.warn(
            f"wechat_msg_24h_expired_worker trace_id={trace_id} send_time={send_time} now={now} skew_s={now - send_time}"
        )
        return {
            "trace_id": trace_id,
            "status": "dead_lettered",
            "reason": "send_time_too_old",
            "skew_s": now - send_time,
        }

    # 2. get_user_by_openid → resolve user_id
    tools = make_tools(user_id=0)
    user_lookup = tools[0](openid)  # get_user_by_openid
    user_id = int(user_lookup.get("user_id") or 0)
    branch = "registered" if user_id > 0 else "guest"

    # 3. 重建 tools with real user_id（get_user_full_profile 注入 user_id）
    tools = make_tools(user_id=user_id)

    # 4. REQ-063 P0-2: registered 分支预取 4 段 profile → LLM 上下文
    profile_context: str | None = None
    if user_id > 0:
        try:
            profile_data = tools[1]()  # get_user_full_profile (user_id injected via closure)
            profile_context = json.dumps(profile_data, ensure_ascii=False, indent=2)
            WechatGreeterObserver.info(
                f"profile_fetched trace_id={trace_id} user_id={user_id} "
                f"seeking={len(profile_data.get('seeking', []))} "
                f"hiring={len(profile_data.get('hiring', []))} "
                f"projects={len(profile_data.get('published_projects', []))}"
            )
        except RuntimeError as exc:
            # SC-01: backend not connected → fallback to guest
            WechatGreeterObserver.warn(
                f"profile_fetch_failed_fallback_to_guest trace_id={trace_id} "
                f"user_id={user_id} err={type(exc).__name__}: {exc}"
            )
            user_id = 0
            branch = "guest"
            tools = make_tools(user_id=0)
            profile_context = None

    # 5. LLM (REQ-063 P0-1: tools 真传入 + P0-2: profile_context 真注入)
    raw_reply = call_llm(
        user_message=content,
        user_id=user_id,
        tools=tools,
        profile_context=profile_context,
    )

    # 6. REQ-065 P0-A3: 硬截断 ≤ 200 字 + 固定尾巴 (尾巴空间预留)
    tail = hard_truncate_tail()
    limit = hard_truncate_limit()
    tail_len = len(tail)
    if len(raw_reply) + tail_len > limit:
        truncated = raw_reply[: limit - tail_len] + tail
    else:
        truncated = raw_reply + tail
    # 防御性断言: 最终字符串必须 ≤ limit 且以 tail 结尾
    assert len(truncated) <= limit, (
        f"REQ-065 P0-A3: truncated length {len(truncated)} > limit {limit}"
    )
    assert truncated.endswith(tail), (
        f"REQ-065 P0-A3: truncated must end with tail {tail!r}, "
        f"got ...{truncated[-tail_len - 10:]}"
    )

    # 7. REQ-065 P0-A2: callback envelope 字段对齐 new_api controller
    callback_envelope = {
        "trace_id": trace_id,
        "openid": openid,
        "user_id": user_id,
        "reply_text": truncated,
        "branch": branch,
        "delivered_at": int(time.time()),
    }

    if dry_run():
        WechatGreeterObserver.info(
            f"DRY_RUN callback_skipped trace_id={trace_id} reply_len={len(truncated)} "
            f"branch={branch} "
            f"would_post_to={os.environ.get('DEEPAGENTS_WECHAT_GREETER_CALLBACK_URL', 'unset')}"
        )
        return {
            "trace_id": trace_id,
            "status": "dry_run",
            "reply_len": len(truncated),
            "branch": branch,
            "callback_skipped": True,
        }

    # REQ-065 P0-A4: callback 失败可重试
    #  - 5xx / 网络异常 → raise, Celery 重试
    #  - 4xx → 不可恢复, 落告警, 不重试
    try:
        resp = post_callback(callback_envelope)
        WechatGreeterObserver.info(
            f"callback sent trace_id={trace_id} status={resp.status_code} "
            f"reply_len={len(truncated)} branch={branch}"
        )
        return {
            "trace_id": trace_id,
            "status": "ok",
            "reply_len": len(truncated),
            "branch": branch,
            "http_status": resp.status_code,
        }
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code if hasattr(exc, "response") else 0
        if status_code in _IRRECOVERABLE_CALLBACK_STATUSES:
            # 4xx: 不可恢复 — 不重试, 落告警
            WechatGreeterObserver.warn(
                f"wechat_msg_callback_irrecoverable trace_id={trace_id} "
                f"http_status={status_code} reason=4xx_no_retry"
            )
            return {
                "trace_id": trace_id,
                "status": "callback_irrecoverable",
                "http_status": status_code,
                "err": str(exc),
            }
        # 5xx: 重试 — 让异常传播给 Celery
        raise
    except (httpx.TimeoutException, httpx.NetworkError):
        # 网络异常: 重试
        raise
