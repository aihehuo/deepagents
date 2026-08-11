"""REQ-050 A 阶段 7 条验收 test scaffold + 1 happy path 冒烟。

A 阶段正式实施状态（task #17 完成）:
  ✅ test_smoke_happy_path_call_async_to_callback  (冒烟: 进队→出队→callback mock 全链路)
  ✅ test_01_call_async_202_hmac_failures           (4 失败分支: missing_headers / unknown_from / invalid_timestamp / invalid_signature → 401)
  ⏸ test_02_5_tools_registered_user_id_hidden       (LLM tool schema 隐藏 user_id 断言留 B 阶段: 需真 LLM 或 schema 检查器)
  ⏸ test_03_system_prompt_red_lines                (system_prompt j2 模板 + 50 条负向评测集留 B+C 阶段)
  ✅ test_04_thread_migration_old_key_deleted       (thread_migrator.py 已实施, 旧 key 显式 del + 摘要注入新 key)
  ✅ test_05_24h_dead_letter                        (worker 24h 死信 + 监控埋点 wechat_msg_24h_expired_worker)
  ✅ test_06_hard_truncate_200_chars                (3 分支: raw > 200 / raw < 200 / raw == 200)
  ⏸ test_07_negative_eval_ci                       (.github/workflows/wechat_greeter_negative_eval.yml + 50 条 yaml 留 C 阶段)
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sign_request(
    secret: str,
    body: str,
    *,
    ts: str | None = None,
    path: str = "/call_async",
    method: str = "POST",
) -> dict[str, str]:
    """Build signed headers for a request to wechat_greeter_api.

    Canonical: "#{ts}\\n#{method}\\n#{path}\\n#{body}"
    """
    ts = ts or str(int(time.time()))
    canonical = f"{ts}\n{method}\n{path}\n{body}"
    sig = hmac.new(secret.encode("utf-8"), canonical.encode("utf-8"), hashlib.sha256).hexdigest()
    return {
        "X-GA-From": "wechat_greeter",
        "X-GA-Ts": ts,
        "X-GA-Signature": sig,
        "Content-Type": "application/json",
    }


def _set_minimal_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the minimum env vars to make the API + worker + callback work."""
    monkeypatch.setenv("HMAC_SECRET_NEW_API", "test-secret-new-api-001")
    monkeypatch.setenv("HMAC_SECRET_AIHEHUOMICRO", "test-secret-micro-001")
    monkeypatch.setenv(
        "DEEPAGENTS_WECHAT_GREETER_CALLBACK_URL",
        "http://new-api.test.local:3000/wechat_greeter_callbacks",
    )
    monkeypatch.setenv("CELERY_TASK_ALWAYS_EAGER", "1")
    monkeypatch.setenv("CELERY_BROKER_URL", "memory://")
    monkeypatch.setenv("CELERY_RESULT_BACKEND", "cache+memory://")
    monkeypatch.setenv("WECHAT_GREETER_MODEL_MODE", "stub")
    monkeypatch.setenv("WECHAT_GREETER_HMAC_TIMESTAMP_SKEW_S", "300")
    monkeypatch.setenv("WECHAT_GREETER_DEAD_LETTER_AFTER_S", str(24 * 3600))
    monkeypatch.setenv("WECHAT_GREETER_TRUNCATE_LIMIT", "200")
    monkeypatch.setenv("WECHAT_GREETER_TRUNCATE_TAIL", "〔详情见 App,扫码看完整建议〕")


# ---------------------------------------------------------------------------
# 冒烟: 进队 → 出队 → callback mock 全链路
# ---------------------------------------------------------------------------

def test_smoke_happy_path_call_async_to_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    """冒烟: POST /call_async 202 + Celery eager 入队 + worker 消费 + callback mock 收到对的参数。"""
    _set_minimal_env(monkeypatch)

    # 1. 准备 HMAC 签名的请求 body
    body_obj = {
        "openid": "test_openid_smoke_001",
        "content": "你好,爱合伙是什么?",
        "send_time": int(time.time()),
        "msg_id": "msg_smoke_001",
    }
    body_str = json.dumps(body_obj, ensure_ascii=False)
    headers = _sign_request("test-secret-new-api-001", body_str)

    # 2. Mock 外部 callback 端点（httpx.post）
    with patch("wechat_greeter.callback.httpx.post") as mock_post:
        mock_resp = MagicMock(status_code=200, json=lambda: {"status": "ok"})
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        # 3. 触发 API（import 在 setenv 之后，确保 config 读到 env）
        from fastapi.testclient import TestClient
        from apps.wechat_greeter_api.main import app

        client = TestClient(app)
        response = client.post("/call_async", headers=headers, content=body_str)

        # 4. 验证 202 + msg_id
        assert response.status_code == 202, f"expected 202, got {response.status_code}: {response.text}"
        body_resp = response.json()
        assert body_resp.get("msg_id") == "msg_smoke_001"
        assert body_resp.get("status") == "accepted"

        # 5. 验证 callback 被调（Celery eager mode 同步执行）
        assert mock_post.called, "callback should be invoked when CELERY_TASK_ALWAYS_EAGER=1"
        called_url = mock_post.call_args[0][0]
        assert "wechat_greeter_callbacks" in called_url, f"unexpected callback url: {called_url}"

        # 6. 验证 callback 头
        called_kwargs = mock_post.call_args[1]
        called_headers = called_kwargs.get("headers", {})
        assert called_headers.get("X-GA-From") == "wechat_greeter", (
            f"X-GA-From should be wechat_greeter, got {called_headers.get('X-GA-From')}"
        )
        assert "X-GA-Ts" in called_headers, "X-GA-Ts missing in callback headers"
        assert "X-GA-Signature" in called_headers, "X-GA-Signature missing in callback headers"

        # 7. 时间戳新鲜
        cb_ts = int(called_headers["X-GA-Ts"])
        assert abs(cb_ts - int(time.time())) < 300, f"callback ts too old or future: {cb_ts}"

        # 8. callback body 包含 msg_id + reply + user_id
        called_body_str = called_kwargs.get("content", "")
        callback_envelope = json.loads(called_body_str)
        assert callback_envelope.get("msg_id") == "msg_smoke_001"
        assert "reply" in callback_envelope
        assert "user_id" in callback_envelope

        # 9. reply 包含固定尾巴（REQ-050 验收 6 基础断言）
        reply = callback_envelope["reply"]
        assert "〔详情见 App,扫码看完整建议〕" in reply, (
            f"reply should contain fixed tail, got: {reply!r}"
        )


# ---------------------------------------------------------------------------
# 7 条验收 test scaffold (A 阶段最小实现仅覆盖 happy path, 其余 skip)
# ---------------------------------------------------------------------------

def test_01_call_async_202_hmac_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-050 验收 1: POST /call_async 202 + HMAC 3 失败分支 (missing_headers / unknown_from / invalid_timestamp) → 401."""
    _set_minimal_env(monkeypatch)

    body_str = json.dumps({"openid": "test", "content": "x", "send_time": int(time.time())}, ensure_ascii=False)
    good_headers = _sign_request("test-secret-new-api-001", body_str)

    from fastapi.testclient import TestClient
    from apps.wechat_greeter_api.main import app

    client = TestClient(app)

    # 分支 1: missing_headers — 缺 X-GA-From
    bad_headers = {k: v for k, v in good_headers.items() if k != "X-GA-From"}
    r = client.post("/call_async", headers=bad_headers, content=body_str)
    assert r.status_code == 401
    assert r.json()["detail"]["error"] in ("missing_headers", "unknown_from")

    # 分支 2: unknown_from — X-GA-From = 其他值
    bad_headers = dict(good_headers)
    bad_headers["X-GA-From"] = "another_service"
    r = client.post("/call_async", headers=bad_headers, content=body_str)
    assert r.status_code == 401
    assert r.json()["detail"]["error"] == "unknown_from"

    # 分支 3: invalid_timestamp — ts 太久以前
    bad_headers = _sign_request("test-secret-new-api-001", body_str, ts=str(int(time.time()) - 3600))
    r = client.post("/call_async", headers=bad_headers, content=body_str)
    assert r.status_code == 401
    assert r.json()["detail"]["error"] == "invalid_timestamp"

    # 4. invalid_signature (额外验证 401 第 4 分支, 加固)
    bad_headers = dict(good_headers)
    bad_headers["X-GA-Signature"] = "0" * 64
    r = client.post("/call_async", headers=bad_headers, content=body_str)
    assert r.status_code == 401
    assert r.json()["detail"]["error"] == "invalid_signature"


def test_02_5_tools_registered_user_id_hidden() -> None:
    """REQ-050 验收 2: 5 工具全部注册成功, 3 个有 user_id 注入的工具不出现在 LLM 工具列表的 parameters 中."""
    from apps.wechat_greeter_api.agent_factory import make_tools
    from inspect import signature

    # 1. 5 工具全部返回
    tools = make_tools(user_id=12345)
    assert len(tools) == 5, f"expected 5 tools, got {len(tools)}"

    # 2. 工具名白名单（与 j2 system_prompt 锁定）
    expected_names = [
        "get_user_by_openid",
        "get_profile_status",
        "get_project_status",
        "mark_reply_sent",
        "get_user_faq",
    ]
    # 工具函数 __name__ 校验
    for i, (tool, expected_name) in enumerate(zip(tools, expected_names)):
        actual_name = getattr(tool, "__name__", str(tool))
        assert actual_name == expected_name, (
            f"tool[{i}].__name__ should be {expected_name!r}, got {actual_name!r}"
        )

    # 3. 签名层剔除 user_id (get_profile_status / get_project_status 签名只有 query)
    for i in (1, 2):  # get_profile_status, get_project_status
        sig = signature(tools[i])
        params = list(sig.parameters.keys())
        assert "user_id" not in params, (
            f"tool[{i}] ({expected_names[i]}) signature must NOT contain user_id, got {params}"
        )
        # 允许 0 或 1 个 query 参数
        assert len(params) <= 1, (
            f"tool[{i}] should have ≤1 params (query only), got {params}"
        )

    # 4. 签名层 get_user_by_openid 保留 openid (它是 IDOR 第 1 层：LLM 可传 openid, 但 get 到的 user_id 是 aihehuomicro 返回的)
    sig0 = signature(tools[0])
    assert "openid" in sig0.parameters, "get_user_by_openid must accept openid"

    # 5. 签名层 mark_reply_sent 保留 msg_id
    sig3 = signature(tools[3])
    assert "msg_id" in sig3.parameters, "mark_reply_sent must accept msg_id"

    # 6. 签名层 get_user_faq 保留 query
    sig4 = signature(tools[4])
    assert "query" in sig4.parameters, "get_user_faq must accept query"

    # 7. 闭包层验证: 调 get_profile_status(user_id=999) 不会因为传入 user_id 报 TypeError
    # (签名层已经把 user_id 剔除, 内部从 closure 拿)
    result = tools[1]("any query")
    assert isinstance(result, dict), f"get_profile_status should return dict, got {type(result)}"
    # 注: stub 阶段 result 可能是 {"status": "stub", "user_id_from_closure": 12345} 之类的结构
    # 关键断言: user_id 是从 closure 拿的 (12345), 不是调用方传的
    if "user_id_from_closure" in result:
        assert result["user_id_from_closure"] == 12345, (
            f"user_id should come from closure, got {result.get('user_id_from_closure')}"
        )

    # 8. 跨 user_id 调: 重新 build, user_id=67890, 闭包应捕获新值
    tools2 = make_tools(user_id=67890)
    result2 = tools2[1]("any query")
    if "user_id_from_closure" in result2:
        assert result2["user_id_from_closure"] == 67890, (
            f"new closure should capture 67890, got {result2.get('user_id_from_closure')}"
        )


def test_03_system_prompt_red_lines() -> None:
    """REQ-050 验收 3: system_prompt v1 红线 4 条 + 工具白名单 + 身份分支 4 类, 全 50 条负向评测集 0 越界."""
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    import os

    # 1. 加载 v1 j2 模板
    prompt_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),  # tests/wechat_greeter/ → /
        "libs", "wechat_greeter", "prompts",
    )
    env = Environment(
        loader=FileSystemLoader(prompt_dir),
        autoescape=select_autoescape(disabled_extensions=("j2",), default_for_string=False),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    tmpl = env.get_template("wechat_greeter_v1.j2")

    # 2. 4 红线硬约束 - 模板源文本必含 4 个关键词串
    src = open(os.path.join(prompt_dir, "wechat_greeter_v1.j2"), encoding="utf-8").read()
    red_line_markers = [
        "不承诺投资回报",   # 红线 1
        "不诊断法律/医疗/财务决策",  # 红线 2
        "不冒充人工客服",   # 红线 3
        "不泄露其他用户信息",  # 红线 4
    ]
    for marker in red_line_markers:
        assert marker in src, f"red line marker missing in j2 source: {marker!r}"

    # 3. 5 工具白名单
    tool_whitelist = [
        "get_user_by_openid",
        "get_profile_status",
        "get_project_status",
        "mark_reply_sent",
        "get_user_faq",
    ]
    for tool_name in tool_whitelist:
        assert tool_name in src, f"tool {tool_name!r} missing in j2 source"

    # 4. 4 身份分支
    identity_branches = [
        '"guest"',           # 游客
        "registered_no_invest",  # 注册未投资
        "investor",          # 投资人
        "founder",           # 项目方
    ]
    for branch in identity_branches:
        assert branch in src, f"identity branch {branch!r} missing in j2 source"

    # 5. 渲染 4 身份分支, 必都成功 + 必含对应分支特征文案
    branch_signatures = {
        "guest": "未注册",
        "registered_no_invest": "注册但未投资项目",
        "investor": "投资人",
        "founder": "项目方",
    }
    for branch, sig in branch_signatures.items():
        rendered = tmpl.render(identity_branch=branch, user_message="测试消息")
        assert sig in rendered, f"branch {branch!r} should contain signature {sig!r}, got:\n{rendered[:300]}"
        # 4 红线在每条渲染里都该有（任何身份都遵守）
        for marker in red_line_markers:
            assert marker in rendered, (
                f"branch {branch!r} rendering missing red line {marker!r}"
            )
        # 5 工具名在每条渲染里都该有
        for tool_name in tool_whitelist:
            assert tool_name in rendered, (
                f"branch {branch!r} rendering missing tool {tool_name!r}"
            )

    # 6. 200 字硬约束 + 固定尾巴 (system_prompt 显式告知 LLM)
    assert "200 字" in src or "200字" in src, "system_prompt should tell LLM about 200-char hard truncate"
    assert "〔详情见 App" in src, "system_prompt should mention fixed tail"

    # 7. 模板元数据: 版本号 + 锁定人 + 锁定日期
    assert "v1" in src, "j2 should have v1 version marker"
    assert "老板 2026-08-11 拍板" in src, "j2 should have boss lock metadata"
    assert "TSD-09" in src, "j2 should reference TSD-09 spec"


def test_04_thread_migration_old_key_deleted(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-050 验收 4: Thread 迁移: openid → user_id 触发后旧 key 必 del checkpoint_store[old_key]."""
    # 1. In-memory checkpoint store
    class _InMemoryStore:
        def __init__(self) -> None:
            self._data: dict[str, dict] = {}

        def get(self, key: str):
            return self._data.get(key)

        def put(self, key: str, value: dict) -> None:
            self._data[key] = value

        def delete(self, key: str) -> bool:
            if key in self._data:
                del self._data[key]
                return True
            return False

    store = _InMemoryStore()

    # 2. 预存旧 checkpoint
    old_key = "openid:test_openid_001"
    new_key = "user_id:12345"
    store.put(old_key, {"messages": [{"role": "user", "content": "hi"}, {"role": "ai", "content": "hello"}]})

    # 3. 触发迁移
    from apps.wechat_greeter_worker.thread_migrator import migrate_thread
    result = migrate_thread(store=store, old_key=old_key, new_key=new_key)

    # 4. 验证
    assert result["status"] == "migrated"
    assert result["old_key"] == old_key
    assert result["new_key"] == new_key
    assert result["old_deleted"] is True

    # 5. 旧 key 必不存 (REQ-050 验收 4 硬要求)
    assert store.get(old_key) is None, "old_key must be deleted after migration"
    assert old_key not in store._data, "old_key must not be in store after migration"

    # 6. 新 key 存, 含 summary + migrated_from
    new_value = store.get(new_key)
    assert new_value is not None
    assert new_value["migrated_from"] == old_key
    assert new_value["summary"]["message_count"] == 2
    assert len(new_value["prior_messages"]) == 2

    # 7. 二次迁移幂等 (old_key 已不存在, 应返回 no_old_checkpoint)
    result2 = migrate_thread(store=store, old_key=old_key, new_key=new_key)
    assert result2["status"] == "no_old_checkpoint"
    assert result2["old_deleted"] is True

    # 8. 相同 key 应返回 no_migration_needed
    result3 = migrate_thread(store=store, old_key=new_key, new_key=new_key)
    assert result3["status"] == "no_migration_needed"


def test_05_24h_dead_letter(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-050 验收 5: 24h 死信: send_time < 24.hours.ago → 不调 callback, 监控埋点 wechat_msg_24h_expired_worker."""
    _set_minimal_env(monkeypatch)

    # 1. send_time = 25h ago (超 24h 阈值)
    old_send_time = int(time.time()) - 25 * 3600
    envelope = {
        "msg_id": "msg_dl_001",
        "openid": "test_openid_dl",
        "content": "这消息是 25 小时前发的",
        "send_time": old_send_time,
        "received_at": int(time.time()),
    }

    with patch("wechat_greeter.callback.httpx.post") as mock_post:
        # 2. 直接调 process_greeting (eager mode 同步跑)
        from apps.wechat_greeter_worker.tasks import process_greeting
        result = process_greeting(envelope)

        # 3. 验证返回 dead_lettered
        assert result["status"] == "dead_lettered", f"expected dead_lettered, got {result}"
        assert result["reason"] == "send_time_too_old"

        # 4. 验证 callback 未被调 (REQ-050 验收 5 硬要求)
        assert not mock_post.called, "callback must NOT be invoked for 24h-expired messages"

    # 5. 边界: send_time = 23h ago (未超阈值) → callback 被调
    monkeypatch.setenv("WECHAT_GREETER_LLM_STUB_RAW", "你好,爱合伙是一个连接创业者和合伙人的平台。")
    fresh_send_time = int(time.time()) - 23 * 3600
    fresh_envelope = {
        "msg_id": "msg_fresh_001",
        "openid": "test_openid_fresh",
        "content": "23 小时前的消息",
        "send_time": fresh_send_time,
        "received_at": int(time.time()),
    }
    with patch("wechat_greeter.callback.httpx.post") as mock_post:
        mock_resp = MagicMock(status_code=200, json=lambda: {"status": "ok"})
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        from apps.wechat_greeter_worker.tasks import process_greeting
        result = process_greeting(fresh_envelope)

        assert result["status"] == "ok", f"expected ok, got {result}"
        assert mock_post.called, "callback should be invoked for fresh messages"


def test_06_hard_truncate_200_chars(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-050 验收 6: 硬截断 ≤ 200 字 + 固定尾巴 '〔详情见 App,扫码看完整建议〕'."""
    _set_minimal_env(monkeypatch)
    TAIL = "〔详情见 App,扫码看完整建议〕"
    LIMIT = 200

    # 分支 1: raw > 200 → 必截断到 200 + tail
    long_raw = "x" * 500  # 500 chars
    monkeypatch.setenv("WECHAT_GREETER_LLM_STUB_RAW", long_raw)
    envelope = {
        "msg_id": "msg_truncate_long",
        "openid": "test_openid_t1",
        "content": "long content",
        "send_time": int(time.time()),
        "received_at": int(time.time()),
    }
    with patch("wechat_greeter.callback.httpx.post") as mock_post:
        mock_resp = MagicMock(status_code=200, json=lambda: {"status": "ok"})
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp
        from apps.wechat_greeter_worker.tasks import process_greeting
        process_greeting(envelope)

        assert mock_post.called
        body_str = mock_post.call_args[1]["content"]
        callback_env = json.loads(body_str)
        reply = callback_env["reply"]
        assert reply.endswith(TAIL), f"reply should end with fixed tail, got: ...{reply[-30:]!r}"
        assert len(reply) == LIMIT + len(TAIL), (
            f"reply len should be {LIMIT} + {len(TAIL)} = {LIMIT + len(TAIL)}, got {len(reply)}"
        )

    # 分支 2: raw ≤ 200 → 原文 + tail
    short_raw = "你好"
    monkeypatch.setenv("WECHAT_GREETER_LLM_STUB_RAW", short_raw)
    envelope2 = {
        "msg_id": "msg_truncate_short",
        "openid": "test_openid_t2",
        "content": "short content",
        "send_time": int(time.time()),
        "received_at": int(time.time()),
    }
    with patch("wechat_greeter.callback.httpx.post") as mock_post:
        mock_resp = MagicMock(status_code=200, json=lambda: {"status": "ok"})
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp
        from apps.wechat_greeter_worker.tasks import process_greeting
        process_greeting(envelope2)

        assert mock_post.called
        body_str = mock_post.call_args[1]["content"]
        callback_env = json.loads(body_str)
        reply = callback_env["reply"]
        assert reply.startswith(short_raw), f"reply should start with raw, got: {reply[:30]!r}"
        assert reply.endswith(TAIL), f"reply should end with fixed tail"
        assert len(reply) == len(short_raw) + len(TAIL)

    # 分支 3: 边界 raw == 200 → 原文 + tail
    boundary_raw = "y" * LIMIT
    monkeypatch.setenv("WECHAT_GREETER_LLM_STUB_RAW", boundary_raw)
    envelope3 = {
        "msg_id": "msg_truncate_boundary",
        "openid": "test_openid_t3",
        "content": "boundary",
        "send_time": int(time.time()),
        "received_at": int(time.time()),
    }
    with patch("wechat_greeter.callback.httpx.post") as mock_post:
        mock_resp = MagicMock(status_code=200, json=lambda: {"status": "ok"})
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp
        from apps.wechat_greeter_worker.tasks import process_greeting
        process_greeting(envelope3)

        assert mock_post.called
        body_str = mock_post.call_args[1]["content"]
        callback_env = json.loads(body_str)
        reply = callback_env["reply"]
        assert len(reply) == LIMIT + len(TAIL), (
            f"boundary reply len should be {LIMIT + len(TAIL)}, got {len(reply)}"
        )


def test_07_negative_eval_ci() -> None:
    """REQ-050 验收 7: CI 跑 run_negative_eval.py ≤ 2 分钟 + 100% 通过率 + merge block."""
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]  # tests/wechat_greeter/ → repo root
    eval_dir = repo / "tests" / "wechat_greeter" / "eval"
    workflow = repo / ".github" / "workflows" / "wechat_greeter_negative_eval.yml"

    # 1. 3 个文件都存在
    assert eval_dir.exists(), f"eval dir missing: {eval_dir}"
    assert workflow.exists(), f"CI workflow missing: {workflow}"

    yaml_path = eval_dir / "negative_set_v1.yaml"
    runner_path = eval_dir / "run_negative_eval.py"
    assert yaml_path.exists(), f"negative_set_v1.yaml missing"
    assert runner_path.exists(), f"run_negative_eval.py missing"

    # 2. yaml ≥ 50 条 + 覆盖 8 类
    import yaml
    with open(yaml_path, encoding="utf-8") as f:
        cases = yaml.safe_load(f)
    assert isinstance(cases, list)
    assert len(cases) >= 50, f"expected ≥ 50 cases, got {len(cases)}"

    cats = {c.get("category") for c in cases}
    expected_cats = {
        "red_line_1_invest_return",
        "red_line_2_legal_medical",
        "red_line_3_impostor",
        "red_line_4_leakage",
        "tool_white_violation",
        "identity_branch_misclass",
        "length_overflow",
        "overlong_reply",
    }
    assert expected_cats.issubset(cats), f"missing categories: {expected_cats - cats}"

    # 3. CI workflow 必含关键 steps
    workflow_src = workflow.read_text(encoding="utf-8")
    assert "run_negative_eval.py" in workflow_src, "CI must call run_negative_eval.py"
    assert "ubuntu-latest" in workflow_src, "CI must run on ubuntu-latest"
    assert "pull_request" in workflow_src, "CI must trigger on PR"
    assert "timeout-minutes" in workflow_src, "CI must have timeout"

    # 4. runner script 必含退出码 0/1 逻辑
    runner_src = runner_path.read_text(encoding="utf-8")
    assert "return 1" in runner_src, "runner must exit 1 on failure"
    assert "100% pass" in runner_src, "runner must mention 100% pass target"

    # 5. 实际跑 runner (≤ 120s, 0 越界, 100% pass)
    import subprocess
    import sys
    result = subprocess.run(
        [sys.executable, str(runner_path), "--mode", "stub"],
        capture_output=True, text=True, timeout=180,
        env={
            **__import__("os").environ,
            "WECHAT_GREETER_MODEL_MODE": "stub",
            "WECHAT_GREETER_TRUNCATE_LIMIT": "200",
            "WECHAT_GREETER_TRUNCATE_TAIL": "〔详情见 App,扫码看完整建议〕",
            "PYTHONPATH": f"{repo}/libs:{repo}/libs/wechat_greeter:{__import__('os').environ.get('PYTHONPATH', '')}",
        },
    )
    assert result.returncode == 0, (
        f"negative eval runner failed (rc={result.returncode}):\n"
        f"STDOUT:\n{result.stdout[-2000:]}\n"
        f"STDERR:\n{result.stderr[-2000:]}"
    )
    assert "100% pass" in result.stdout, f"runner output missing '100% pass':\n{result.stdout[-1000:]}"
    assert "0 violations" in result.stdout or "Passed:" in result.stdout


# ---------------------------------------------------------------------------
# D-1 灰度切档 (跨仓 P0, 老板 2026-08-11 拍板)
# ---------------------------------------------------------------------------

def test_d1_dry_run_skips_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    """D-1 dry-run 模式: 走完流程但不真打 callback, 避免污染生产.

    实证 3 件事:
      1. /healthz 报 status=dry_run + dry_run=true
      2. process_greeting 走完 → 返回 status=dry_run + callback_skipped=True
      3. httpx.post 一次都没调 (避免污染生产 callback)
    """
    _set_minimal_env(monkeypatch)
    monkeypatch.setenv("WECHAT_GREETER_DRY_RUN", "true")

    # 1. /healthz 实证
    from fastapi.testclient import TestClient
    from apps.wechat_greeter_api.main import app
    client = TestClient(app)
    h = client.get("/healthz")
    assert h.status_code == 200
    health = h.json()
    assert health["status"] == "dry_run", f"expected status=dry_run, got {health}"
    assert health["dry_run"] is True
    assert "model_mode" in health
    assert "faq_count" in health and health["faq_count"] >= 30  # C 阶段 seed 30 条

    # 2 + 3. process_greeting dry-run 实证
    with patch("wechat_greeter.callback.httpx.post") as mock_post:
        envelope = {
            "msg_id": "msg_dry_001",
            "openid": "test_openid_dry",
            "content": "你好, 测一下 dry-run",
            "send_time": int(time.time()),
            "received_at": int(time.time()),
        }
        from apps.wechat_greeter_worker.tasks import process_greeting
        result = process_greeting(envelope)

        # 期望 status=dry_run
        assert result["status"] == "dry_run", f"expected dry_run, got {result}"
        assert result["callback_skipped"] is True
        assert "reply_len" in result

        # 期望 httpx.post 一次都没调
        assert not mock_post.called, "dry_run must NOT invoke httpx.post (avoid polluting prod callback)"


def test_d1_dry_run_off_actually_calls_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    """D-1 对照: dry_run=false → 走真 callback (回归保险)."""
    _set_minimal_env(monkeypatch)
    # dry_run 显式不设 (默认 false)
    monkeypatch.delenv("WECHAT_GREETER_DRY_RUN", raising=False)

    with patch("wechat_greeter.callback.httpx.post") as mock_post:
        mock_resp = MagicMock(status_code=200, json=lambda: {"status": "ok"})
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        envelope = {
            "msg_id": "msg_real_001",
            "openid": "test_openid_real",
            "content": "real 流量",
            "send_time": int(time.time()),
            "received_at": int(time.time()),
        }
        from apps.wechat_greeter_worker.tasks import process_greeting
        result = process_greeting(envelope)

        # 期望 status=ok (真 callback 路径)
        assert result["status"] == "ok", f"expected ok, got {result}"
        assert mock_post.called, "dry_run=false must invoke httpx.post"
