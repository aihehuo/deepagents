"""Celery task: process a single wechat greeting (REQ-050).

A 阶段 + A 阶段正式 + B 阶段 实施：
  1. 24h 死信判定（send_time < 24h.ago → 不调 callback + 埋点 wechat_msg_24h_expired_worker）
  2. 5 工具全部 stub（A 阶段）/ 真实 aihehuomicro HMAC 调（待 aihehuomicro 3 端点就绪后切换）
  3. LLM 统一入口（libs/wechat_greeter/llm_client.py）:
     - stub 模式: 固定长文本 (单测用 WECHAT_GREETER_LLM_STUB_RAW 覆盖)
     - deepseek 模式: init_chat_model + model_provider=deepseek + model=deepseek-v4-flash
  4. system_prompt v1 (libs/wechat_greeter/prompts/wechat_greeter_v1.j2): 4 红线 + 工具白名单 + 4 身份分支
  5. 硬截断 ≤ 200 字 + 固定尾巴
  6. mark_reply_sent stub
  7. callback mock（post_callback 会被 test 替换为 MagicMock）

C/D 阶段留：
  - 真实 FAISS 调（libs/wechat_greeter/faq_store.py + get_user_faq tool 注入 FAISS 索引）
  - 完整 50 条负向评测 + run_negative_eval.py + CI workflow
  - 灰度切档 + 联调 aihehuomicro 3 端点
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

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


@shared_task(name="wechat_greeter.process_greeting", bind=True, ignore_result=True)
def process_greeting(self, envelope: dict[str, Any]) -> dict[str, Any]:
    """Process a single wechat greeting (A 阶段冒烟实现)."""
    msg_id = str(envelope.get("msg_id") or "unknown")
    openid = str(envelope.get("openid") or "")
    content = str(envelope.get("content") or "")
    send_time = int(envelope.get("send_time") or 0)
    now = int(time.time())

    WechatGreeterObserver.info(
        f"process_greeting start msg_id={msg_id} openid={openid} content_len={len(content)}"
    )

    # 1. 24h 死信判定
    if send_time and (now - send_time) > dead_letter_after_s():
        WechatGreeterObserver.warn(
            f"wechat_msg_24h_expired_worker msg_id={msg_id} send_time={send_time} now={now} skew_s={now - send_time}"
        )
        return {
            "msg_id": msg_id,
            "status": "dead_lettered",
            "reason": "send_time_too_old",
            "skew_s": now - send_time,
        }

    # 2. get_user_by_openid (stub)
    tools = make_tools(user_id=0)  # user_id=0 placeholder, 真实值在下面覆写
    user_lookup = tools[0](openid)  # get_user_by_openid
    user_id = int(user_lookup.get("user_id") or 0)

    # 3. 重建 tools with real user_id（profile_status / project_status 注入 user_id）
    tools = make_tools(user_id=user_id)

    # 4. LLM (B 阶段: llm_client 统一入口, stub/deepseek 模式分支在内部)
    raw_reply = call_llm(user_message=content, user_id=user_id)

    # 5. 硬截断 ≤ 200 字 + 固定尾巴
    tail = hard_truncate_tail()
    limit = hard_truncate_limit()
    if len(raw_reply) > limit:
        truncated = raw_reply[:limit] + tail
    else:
        truncated = raw_reply + tail  # 永远加尾巴（REQ-050 验收 6 兜底）

    # 6. mark_reply_sent (stub)
    tools[3](msg_id)

    # 7. Callback to new_api (D-1: dry_run 模式仅 log 不真打)
    callback_envelope = {
        "msg_id": msg_id,
        "openid": openid,
        "user_id": user_id,
        "reply": truncated,
        "reply_len": len(truncated),
        "delivered_at": int(time.time()),
    }
    if dry_run():
        # D-1: dry-run 模式, 仅 log, 不真打 new_api (避免污染生产 callback)
        WechatGreeterObserver.info(
            f"DRY_RUN callback_skipped msg_id={msg_id} reply_len={len(truncated)} "
            f"would_post_to={os.environ.get('DEEPAGENTS_WECHAT_GREETER_CALLBACK_URL', 'unset')}"
        )
        return {
            "msg_id": msg_id,
            "status": "dry_run",
            "reply_len": len(truncated),
            "callback_skipped": True,
        }
    try:
        resp = post_callback(callback_envelope)
        WechatGreeterObserver.info(
            f"callback sent msg_id={msg_id} status={resp.status_code} reply_len={len(truncated)}"
        )
        return {"msg_id": msg_id, "status": "ok", "reply_len": len(truncated), "http_status": resp.status_code}
    except Exception as exc:  # noqa: BLE001
        WechatGreeterObserver.warn(
            f"wechat_msg_callback_failed msg_id={msg_id} err={type(exc).__name__}: {exc}"
        )
        return {"msg_id": msg_id, "status": "callback_failed", "err": str(exc)}
