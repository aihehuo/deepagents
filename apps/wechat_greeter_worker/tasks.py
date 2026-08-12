"""Celery task: process a single wechat greeting (REQ-065 P0-A / REQ-063 / REQ-062).

REQ-065 P0-A: callback 端到端字段契约对齐
  P0-A1: worker 入口 msg_id → trace_id (new_api DeepAgentClient.call_async 发的是 trace_id)
  P0-A2: callback_envelope 字段对齐 new_api WechatGreeterCallbacksController:
         reply→reply_text, msg_id→trace_id, 补 branch 字段
  P0-A3: 截断长度修正: raw_reply[:limit - len(tail)] + tail, 保证 len ≤ 200 且 endswith(tail)
  P0-A4: callback 失败可重试: 显式 self.retry() 指数退避; 4xx 不可重试落告警

REQ-063 P0-1: tools 真传入 call_llm (不再死变量).
REQ-063 P0-2: registered 分支 4 段 profile 真进 LLM 上下文 (pre-fetch → profile_context).
  SC-01: get_user_full_profile RuntimeError → user_id=0 guest 兜底.

REQ-065 P0-A4 v2: Celery 显式重试
  原实现裸 raise 依赖 task_acks_late 自动重入队, 但 task_acks_on_failure_or_timeout=True
  导致消息被 ack 后永久丢弃, 不会重试. 修复为显式 self.retry().
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
    apply_tail_and_truncate,
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

# REQ-065 P0-A4 v2: Celery 显式 self.retry() 配置
# 指数退避: 5s → 25s → 125s (最大 3 次, 总等待 ≤ 155s)
# 24h 死信窗口内足够重试 3 次
_MAX_CALLBACK_RETRIES = 3
_CALLBACK_RETRY_BASE_S = 5  # countdown = base * (retry_backoff_factor ** retries)


def _retry_callback(self, exc: Exception, trace_id: str, http_status: int) -> None:
    """REQ-065 P0-A4 v2: 显式 self.retry() — 指数退避重新入队.

    指数退避: 5s → 25s → 125s (base=5s, factor=5).
    达到 max_retries 后不再重试, 直接 re-raise (最终失败告警).
    """
    retries = self.request.retries
    if retries >= _MAX_CALLBACK_RETRIES:
        WechatGreeterObserver.warn(
            f"wechat_msg_callback_max_retries_exceeded trace_id={trace_id} "
            f"retries={retries} max={_MAX_CALLBACK_RETRIES} "
            f"http_status={http_status}"
        )
        raise exc

    countdown = _CALLBACK_RETRY_BASE_S * (5 ** retries)
    WechatGreeterObserver.warn(
        f"wechat_msg_callback_retry trace_id={trace_id} "
        f"retries={retries}/{_MAX_CALLBACK_RETRIES} "
        f"countdown={countdown}s http_status={http_status} "
        f"exc={type(exc).__name__}: {exc}"
    )
    raise self.retry(exc=exc, countdown=countdown, max_retries=_MAX_CALLBACK_RETRIES)


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

    # 6. REQ-065 P0-A3: 软字数简短性由 Prompt 约束，此处只做尾巴去重 & 规范拼接 (不做断句破损的硬截断)
    tail = hard_truncate_tail()
    truncated = apply_tail_and_truncate(raw_reply, tail=tail, limit=None)
    # 防御性断言: 最终字符串必须以 tail 结尾
    assert truncated.endswith(tail), (
        f"REQ-065 P0-A3: truncated must end with tail {tail!r}, "
        f"got ...{truncated[-len(tail) - 10:]}"
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

    # REQ-065 P0-A4 v2: callback 失败显式 self.retry() 指数退避
    #  - 5xx / 网络异常 → self.retry(exc=..., countdown=...) 重新入队
    #  - 4xx → 不可恢复, 落告警, 不重试
    #  - 达到 max_retries → 最终失败告警
    #
    # 原实现: 裸 raise 依赖 Celery task_acks_late 自动重入队,
    #          但 task_acks_on_failure_or_timeout=True 导致消息被 ack 丢弃.
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
        # 5xx: 指数退避重试
        _retry_callback(self, exc, trace_id, status_code)
    except (httpx.TimeoutException, httpx.NetworkError) as exc:
        # 网络异常: 指数退避重试
        _retry_callback(self, exc, trace_id, 0)
