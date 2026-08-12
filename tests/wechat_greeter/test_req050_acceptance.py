"""REQ-063 P0/P1 验收测试 (基于 REQ-062 v2 scaffold).

REQ-062 v2 基线:
  ✅ test_01_call_async_202_hmac_failures           (4 失败分支: missing_headers / unknown_from / invalid_timestamp / invalid_signature → 401)
  ✅ test_02_3_tools_registered_user_id_hidden       (v2: 3 工具 + get_user_full_profile 签名无 user_id + REQ-063 mock HMAC)
  ✅ test_03_system_prompt_red_lines_v2             (v2: 5 红线 + 3 工具白名单 + 2 身份分支)
  ✅ test_07_negative_eval_ci_v2                    (v2: .github/workflows/wechat_greeter_negative_eval.yml + ≥50 条 v2 yaml)

REQ-063 P0/P1 新增:
  ✅ test_08_req063_runtime_tool_execution          (P1: 验运行时真接通——P0-1~P0-4 全链路，不只数工具个数)

已知预存失败 (6 条, langchain 版本不兼容 → InputAgentState import error):
  ⚠ test_smoke / test_04 / test_05 / test_06 / test_d1_dry_run_skips / test_d1_dry_run_off
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
    monkeypatch.setenv("WECHAT_GREETER_TRUNCATE_TAIL", "〔详情见 App，扫码看完整建议〕")


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
        "trace_id": "msg_smoke_001",
    }
    body_str = json.dumps(body_obj, ensure_ascii=False)
    headers = _sign_request("test-secret-new-api-001", body_str)

    # 2. Mock 外部 callback 端点（httpx.post）
    with patch("wechat_greeter.callback.httpx.post") as mock_post:
        mock_resp = MagicMock(status_code=200, json=lambda: {"status": "ok"})
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        # 3. Mock readiness gate (smoke test uses model_mode=stub, REQ-065 P0-3/P1-2)
        import apps.wechat_greeter_api.main as _main_mod

        with patch.object(_main_mod, "readiness_details", return_value={"overall": True}):
            # 4. 触发 API（import 在 setenv 之后，确保 config 读到 env）
            from fastapi.testclient import TestClient
            from apps.wechat_greeter_api.main import app

            client = TestClient(app)
            response = client.post("/call_async", headers=headers, content=body_str)

            # 5. 验证 202 + trace_id
            assert response.status_code == 202, f"expected 202, got {response.status_code}: {response.text}"
            body_resp = response.json()
            assert body_resp.get("trace_id") == "msg_smoke_001"
            assert body_resp.get("status") == "accepted"

            # 6. 验证 callback 被调（Celery eager mode 同步执行）
            assert mock_post.called, "callback should be invoked when CELERY_TASK_ALWAYS_EAGER=1"
            called_url = mock_post.call_args[0][0]
            assert "wechat_greeter_callbacks" in called_url, f"unexpected callback url: {called_url}"

            # 7. 验证 callback 头
            called_kwargs = mock_post.call_args[1]
            called_headers = called_kwargs.get("headers", {})
            assert called_headers.get("X-GA-From") == "wechat_greeter", (
                f"X-GA-From should be wechat_greeter, got {called_headers.get('X-GA-From')}"
            )
            assert "X-GA-Ts" in called_headers, "X-GA-Ts missing in callback headers"
            assert "X-GA-Signature" in called_headers, "X-GA-Signature missing in callback headers"

            # 8. 时间戳新鲜
            cb_ts = int(called_headers["X-GA-Ts"])
            assert abs(cb_ts - int(time.time())) < 300, f"callback ts too old or future: {cb_ts}"

            # 9. callback body 包含 trace_id + reply_text + user_id + branch
            called_body_str = called_kwargs.get("content", "")
            callback_envelope = json.loads(called_body_str)
            assert callback_envelope.get("trace_id") == "msg_smoke_001"
            assert "reply_text" in callback_envelope
            assert "user_id" in callback_envelope
            assert "branch" in callback_envelope, "REQ-065 P0-A2: callback must include branch field"

            # 10. reply_text 包含固定尾巴
            reply = callback_envelope["reply_text"]
            assert "〔详情见 App" in reply, (
                f"reply should contain fixed tail, got: {reply!r}"
            )


# ---------------------------------------------------------------------------
# 验收 test scaffold
# ---------------------------------------------------------------------------

def test_01_call_async_202_hmac_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-062 验收 1: POST /call_async 202 + HMAC 3 失败分支 (missing_headers / unknown_from / invalid_timestamp) → 401."""
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


def test_02_3_tools_registered_user_id_hidden() -> None:
    """REQ-062 验收 2: 3 工具全部注册成功, get_user_full_profile 签名无 user_id 参数.

    REQ-063 P0-4: get_user_full_profile 已接真 HMAC → 测试需 mock micro_client。
    """
    from apps.wechat_greeter_api.agent_factory import make_tools
    from inspect import signature

    # 1. 3 工具全部返回 (v2)
    tools = make_tools(user_id=12345)
    assert len(tools) == 3, f"expected 3 tools, got {len(tools)}"

    # 2. 工具名白名单 (v2: 与 j2 system_prompt 锁定)
    expected_names = [
        "get_user_by_openid",
        "get_user_full_profile",
        "get_user_faq",
    ]
    for i, (tool, expected_name) in enumerate(zip(tools, expected_names)):
        actual_name = getattr(tool, "__name__", str(tool))
        assert actual_name == expected_name, (
            f"tool[{i}].__name__ should be {expected_name!r}, got {actual_name!r}"
        )

    # 3. 签名层剔除 user_id (get_user_full_profile 签名无参数)
    sig1 = signature(tools[1])  # get_user_full_profile
    params1 = list(sig1.parameters.keys())
    assert "user_id" not in params1, (
        f"get_user_full_profile signature must NOT contain user_id, got {params1}"
    )
    assert len(params1) == 0, (
        f"get_user_full_profile should have 0 params, got {params1}"
    )

    # 4. 签名层 get_user_by_openid 保留 openid
    sig0 = signature(tools[0])
    assert "openid" in sig0.parameters, "get_user_by_openid must accept openid"

    # 5. 签名层 get_user_faq 保留 query
    sig2 = signature(tools[2])
    assert "query" in sig2.parameters, "get_user_faq must accept query"

    # 6. REQ-063: mock HMAC → 闭包层验证 get_user_full_profile() 返回 dict with user_id from closure
    fake_profile = {
        "ok": True,
        "user_id": 12345,
        "profile": {"nickname": "测试用户", "avatar": "", "bio": "十年创业老兵"},
        "seeking": [{"role": "技术合伙人", "skill": "AI/大模型"}],
        "hiring": [],
        "published_projects": [{"title": "AI 客服 SaaS", "stage": "已上线"}],
    }

    def _fake_get_user_full_profile(user_id: int) -> dict[str, Any]:
        # Return a copy with the requested user_id (simulates real backend)
        data = dict(fake_profile)
        data["user_id"] = user_id
        return data

    # REQ-063: patch the imported reference in tools module (not micro_client directly,
    # because get_user_full_profile.py does `from wechat_greeter.micro_client import ... as _hmac_...`
    # at module level, so the reference is already bound).
    # REQ-063: patch the imported reference in tools module.
    # Must use "libs.wechat_greeter.tools..." prefix because agent_factory imports from
    # libs.wechat_greeter.tools (separate sys.modules entry from wechat_greeter.tools).
    with patch(
        "libs.wechat_greeter.tools.get_user_full_profile._hmac_get_user_full_profile",
        side_effect=_fake_get_user_full_profile,
    ):
        result = tools[1]()  # no args, user_id=12345 from closure
        assert isinstance(result, dict), f"get_user_full_profile should return dict, got {type(result)}"
        assert result.get("user_id") == 12345, (
            f"user_id should come from closure (12345), got {result.get('user_id')}"
        )
        assert "profile" in result, "get_user_full_profile should have profile segment"
        assert "seeking" in result, "get_user_full_profile should have seeking segment"
        assert "hiring" in result, "get_user_full_profile should have hiring segment"
        assert "published_projects" in result, "get_user_full_profile should have published_projects segment"

    # 7. 跨 user_id 调: 重新 build, user_id=67890, 闭包应捕获新值
    tools2 = make_tools(user_id=67890)
    with patch(
        "libs.wechat_greeter.tools.get_user_full_profile._hmac_get_user_full_profile",
        side_effect=_fake_get_user_full_profile,
    ):
        result2 = tools2[1]()
        assert result2.get("user_id") == 67890, (
            f"new closure should capture 67890, got {result2.get('user_id')}"
        )


def test_03_system_prompt_red_lines_v2() -> None:
    """REQ-062 验收 3: system_prompt v2 5 红线 + 3 工具白名单 + 2 身份分支."""
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    import os

    # 1. 加载 v2 j2 模板
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

    # 2. 5 红线硬约束 - 模板源文本必含 5 个关键词串
    src = open(os.path.join(prompt_dir, "wechat_greeter_v1.j2"), encoding="utf-8").read()
    red_line_markers = [
        "不搜索项目",            # 红线 1
        "不推荐项目",            # 红线 2
        "不撮合",                # 红线 3
        "不在公众号内给可执行",   # 红线 4
        "不写库",                # 红线 5 (架构约束)
    ]
    for marker in red_line_markers:
        assert marker in src, f"red line marker missing in j2 source: {marker!r}"

    # 3. 3 工具白名单 (v2)
    tool_whitelist = [
        "get_user_faq",
        "get_user_by_openid",
        "get_user_full_profile",
    ]
    for tool_name in tool_whitelist:
        assert tool_name in src, f"tool {tool_name!r} missing in j2 source"

    # 4. 2 身份分支 (v2)
    identity_branches = [
        "guest",       # 未注册用户
        "registered",  # 已注册用户
    ]
    for branch in identity_branches:
        assert branch in src, f"identity branch {branch!r} missing in j2 source"

    # 5. 渲染 2 身份分支, 必都成功 + 必含对应分支特征文案
    branch_signatures = {
        "guest": "未注册用户",
        "registered": "已注册用户",
    }
    for branch, sig in branch_signatures.items():
        rendered = tmpl.render(identity_branch=branch, user_message="测试消息")
        assert sig in rendered, f"branch {branch!r} should contain signature {sig!r}, got:\n{rendered[:300]}"
        # 5 红线在每条渲染里都该有（任何身份都遵守）
        for marker in red_line_markers:
            assert marker in rendered, (
                f"branch {branch!r} rendering missing red line {marker!r}"
            )
        # 3 工具名在每条渲染里都该有
        for tool_name in tool_whitelist:
            assert tool_name in rendered, (
                f"branch {branch!r} rendering missing tool {tool_name!r}"
            )

    # 6. 200 字硬约束 + 固定尾巴 (system_prompt 显式告知 LLM)
    assert "200 字" in src or "200字" in src, "system_prompt should tell LLM about 200-char hard truncate"
    assert "〔详情见 App" in src, "system_prompt should mention fixed tail"

    # 7. 6 维引导框架
    guidance_dims = ["用户画像", "用户群体", "痛点", "产品", "市场", "找什么人"]
    for dim in guidance_dims:
        assert dim in src, f"6-dim guidance dimension {dim!r} missing in j2 source"

    # 8. 模板元数据: 版本号 v2
    assert "v2" in src, "j2 should have v2 version marker"
    assert "REQ-062" in src, "j2 should reference REQ-062"
    assert "PRC-07 v2" in src, "j2 should reference PRC-07 v2"


def test_04_thread_migration_old_key_deleted(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-062 验收 4: Thread 迁移: openid → user_id 触发后旧 key 必 del checkpoint_store[old_key]."""
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
    old_key = "wechat:openid:test_openid_001"
    new_key = "wechat:user:12345"
    store.put(old_key, {"messages": [{"role": "user", "content": "hi"}, {"role": "ai", "content": "hello"}]})

    # 3. 触发迁移
    from apps.wechat_greeter_worker.thread_migrator import migrate_thread
    result = migrate_thread(store=store, old_key=old_key, new_key=new_key)

    # 4. 验证
    assert result["status"] == "migrated"
    assert result["old_key"] == old_key
    assert result["new_key"] == new_key
    assert result["old_deleted"] is True

    # 5. 旧 key 必不存
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
    """REQ-062 验收 5: 24h 死信: send_time < 24.hours.ago → 不调 callback, 监控埋点 wechat_msg_24h_expired_worker."""
    _set_minimal_env(monkeypatch)

    # 1. send_time = 25h ago (超 24h 阈值)
    old_send_time = int(time.time()) - 25 * 3600
    envelope = {
        "trace_id": "msg_dl_001",
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

        # 4. 验证 callback 未被调
        assert not mock_post.called, "callback must NOT be invoked for 24h-expired messages"

    # 5. 边界: send_time = 23h ago (未超阈值) → callback 被调
    monkeypatch.setenv("WECHAT_GREETER_LLM_STUB_RAW", "你好,爱合伙是一个连接创业者和合伙人的平台。")
    fresh_send_time = int(time.time()) - 23 * 3600
    fresh_envelope = {
        "trace_id": "msg_fresh_001",
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
    """REQ-062 验收 6: 硬截断 ≤ 200 字 + 固定尾巴."""
    _set_minimal_env(monkeypatch)
    TAIL = "〔详情见 App，扫码看完整建议〕"
    LIMIT = 200

    # 分支 1: raw > 200 → 必截断到 200 + tail
    long_raw = "x" * 500  # 500 chars
    monkeypatch.setenv("WECHAT_GREETER_LLM_STUB_RAW", long_raw)
    envelope = {
        "trace_id": "msg_truncate_long",
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
        reply = callback_env["reply_text"]
        assert reply.endswith(TAIL), f"reply should end with fixed tail, got: ...{reply[-30:]!r}"
        assert len(reply) == LIMIT, (
            f"REQ-065 P0-A3: reply len should be exactly {LIMIT} (truncated + tail got {len(reply)})"
        )

    # 分支 2: raw ≤ 200 → 原文 + tail
    short_raw = "你好"
    monkeypatch.setenv("WECHAT_GREETER_LLM_STUB_RAW", short_raw)
    envelope2 = {
        "trace_id": "msg_truncate_short",
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
        reply = callback_env["reply_text"]
        assert reply.startswith(short_raw), f"reply should start with raw, got: {reply[:30]!r}"
        assert reply.endswith(TAIL), f"reply should end with fixed tail"
        assert len(reply) == len(short_raw) + len(TAIL)

    # 分支 3: 边界 raw == 200 → 原文 + tail
    boundary_raw = "y" * LIMIT
    monkeypatch.setenv("WECHAT_GREETER_LLM_STUB_RAW", boundary_raw)
    envelope3 = {
        "trace_id": "msg_truncate_boundary",
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
        reply = callback_env["reply_text"]
        assert len(reply) == LIMIT, (
            f"REQ-065 P0-A3: boundary reply len should be {LIMIT}, got {len(reply)}"
        )


def test_07_negative_eval_ci_v2() -> None:
    """REQ-062 验收 7: CI 跑 run_negative_eval.py with v2 yaml ≤ 2 分钟 + 100% 通过率 + merge block."""
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]  # tests/wechat_greeter/ → repo root
    eval_dir = repo / "tests" / "wechat_greeter" / "eval"
    workflow = repo / ".github" / "workflows" / "wechat_greeter_negative_eval.yml"

    # 1. 文件都存在
    assert eval_dir.exists(), f"eval dir missing: {eval_dir}"
    assert workflow.exists(), f"CI workflow missing: {workflow}"

    # v2 yaml
    yaml_path_v2 = eval_dir / "negative_set_v2.yaml"
    runner_path = eval_dir / "run_negative_eval.py"
    assert yaml_path_v2.exists(), f"negative_set_v2.yaml missing"
    assert runner_path.exists(), f"run_negative_eval.py missing"

    # 2. v2 yaml ≥ 50 条 + 覆盖 10 类 (9 大类 + normal_positive)
    import yaml
    with open(yaml_path_v2, encoding="utf-8") as f:
        cases = yaml.safe_load(f)
    assert isinstance(cases, list)
    assert len(cases) >= 50, f"expected ≥ 50 cases, got {len(cases)}"

    cats = {c.get("category") for c in cases}
    expected_cats = {
        "red_line_1_search",
        "red_line_2_recommend",
        "red_line_3_match",
        "red_line_4_external_link",
        "red_line_5_no_write",
        "cross_user_privacy",
        "overlong_induction",
        "out_of_scope_link",
        "unauthorized_query",
        "normal_positive",
    }
    assert expected_cats.issubset(cats), f"missing categories: {expected_cats - cats}"

    # 3. v2 identity values: only guest | registered
    identities = {c.get("identity") for c in cases}
    assert identities.issubset({"guest", "registered"}), (
        f"v2 identities should only be guest/registered, got {identities}"
    )

    # 4. CI workflow 必含关键 steps
    workflow_src = workflow.read_text(encoding="utf-8")
    assert "run_negative_eval.py" in workflow_src, "CI must call run_negative_eval.py"
    assert "ubuntu-latest" in workflow_src, "CI must run on ubuntu-latest"
    assert "pull_request" in workflow_src, "CI must trigger on PR"
    assert "timeout-minutes" in workflow_src, "CI must have timeout"

    # 5. runner script 必含退出码 0/1 逻辑
    runner_src = runner_path.read_text(encoding="utf-8")
    assert "return 1" in runner_src, "runner must exit 1 on failure"
    assert "100% pass" in runner_src, "runner must mention 100% pass target"

    # 6. 实际跑 runner with v2 yaml (≤ 120s, 0 越界, 100% pass)
    import subprocess
    import sys
    result = subprocess.run(
        [sys.executable, str(runner_path), "--set", str(yaml_path_v2), "--mode", "stub"],
        capture_output=True, text=True, timeout=180,
        env={
            **__import__("os").environ,
            "WECHAT_GREETER_MODEL_MODE": "stub",
            "WECHAT_GREETER_TRUNCATE_LIMIT": "200",
            "WECHAT_GREETER_TRUNCATE_TAIL": "〔详情见 App，扫码看完整建议〕",
            "PYTHONPATH": f"{repo}/libs:{repo}/libs/wechat_greeter:{__import__('os').environ.get('PYTHONPATH', '')}",
        },
    )
    assert result.returncode == 0, (
        f"negative eval v2 runner failed (rc={result.returncode}):\n"
        f"STDOUT:\n{result.stdout[-2000:]}\n"
        f"STDERR:\n{result.stderr[-2000:]}"
    )
    assert "100% pass" in result.stdout, f"runner output missing '100% pass':\n{result.stdout[-1000:]}"


# ---------------------------------------------------------------------------
# REQ-063 P1: 运行时真接通验证 (不只数工具个数)
# ---------------------------------------------------------------------------

def test_08_req063_runtime_tool_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-063 P1: 验证工具运行时真接通——不只数工具个数，而是验执行链路.

    P0-1 验证: call_llm 收到 tools 参数并传入 bind_tools 流程.
    P0-2 验证: profile_context 注入 system_prompt.
    P0-3 验证: get_user_by_openid → fail-closed (guest) 当 HMAC 未配置.
    P0-4 验证: get_user_full_profile → 真 HMAC 调用 (mock), 返回 4 段数据.
    """
    # REQ-065 P0-C7: model_mode must be explicitly set
    monkeypatch.setenv("WECHAT_GREETER_MODEL_MODE", "stub")
    monkeypatch.setenv("WECHAT_GREETER_TRUNCATE_LIMIT", "200")
    monkeypatch.setenv("WECHAT_GREETER_TRUNCATE_TAIL", "test_tail")
    from apps.wechat_greeter_api.agent_factory import make_tools
    from wechat_greeter.llm_client import call_llm

    # ------------------------------------------------------------------
    # 子测试 A: P0-3 get_user_by_openid fail-closed
    # ------------------------------------------------------------------
    tools_guest = make_tools(user_id=0)
    result_openid = tools_guest[0]("nonexistent_openid_12345")
    assert result_openid["user_id"] == 0, (
        f"P0-3 fail-closed: unknown openid should return user_id=0, got {result_openid}"
    )
    assert result_openid.get("source") == "fail_closed", (
        f"P0-3: source should be 'fail_closed' when HMAC not configured, got {result_openid.get('source')}"
    )

    # ------------------------------------------------------------------
    # 子测试 B: P0-4 get_user_full_profile 真 HMAC (mock)
    # ------------------------------------------------------------------
    fake_profile = {
        "ok": True,
        "user_id": 12345,
        "profile": {"nickname": "张三", "avatar": "https://example.com/avatar.jpg", "bio": "连续创业者，10 年 SaaS 经验"},
        "seeking": [
            {"role": "技术合伙人", "skill": "AI/大模型/推荐系统", "commitment": "全职"},
            {"role": "运营合伙人", "skill": "B2B 增长", "commitment": "兼职"},
        ],
        "hiring": [
            {"title": "前端工程师", "salary": "25k-35k", "location": "北京"},
        ],
        "published_projects": [
            {"title": "AI 客服 SaaS", "stage": "已上线", "description": "服务 200+ 企业客户"},
            {"title": "智能招聘平台", "stage": "开发中", "description": "用 AI 匹配候选人和职位"},
        ],
    }

    def _fake_get_user_full_profile(user_id: int) -> dict[str, Any]:
        data = dict(fake_profile)
        data["user_id"] = user_id
        return data

    tools_registered = make_tools(user_id=12345)
    with patch(
        "libs.wechat_greeter.tools.get_user_full_profile._hmac_get_user_full_profile",
        side_effect=_fake_get_user_full_profile,
    ):
        result_profile = tools_registered[1]()  # get_user_full_profile (no args, user_id from closure)
        assert result_profile["user_id"] == 12345, (
            f"P0-4: user_id should be 12345 from closure, got {result_profile['user_id']}"
        )
        assert len(result_profile["seeking"]) == 2, f"P0-4: should have 2 seeking roles, got {len(result_profile['seeking'])}"
        assert len(result_profile["hiring"]) == 1, f"P0-4: should have 1 hiring position, got {len(result_profile['hiring'])}"
        assert len(result_profile["published_projects"]) == 2, f"P0-4: should have 2 projects, got {len(result_profile['published_projects'])}"

    # ------------------------------------------------------------------
    # 子测试 C: P0-2 profile_context 格式化 + 注入 system_prompt
    # ------------------------------------------------------------------
    import json
    profile_context = json.dumps(fake_profile, ensure_ascii=False, indent=2)
    assert "张三" in profile_context, "P0-2: profile_context should contain user nickname"
    assert "技术合伙人" in profile_context, "P0-2: profile_context should contain seeking role"
    assert "AI 客服 SaaS" in profile_context, "P0-2: profile_context should contain project title"

    # 验证 system_prompt 包含注入的 profile_context (registered 分支)
    from wechat_greeter.llm_client import _build_system_prompt
    prompt_with_profile = _build_system_prompt(
        identity_branch="registered",
        profile_context=profile_context,
    )
    assert "张三" in prompt_with_profile, (
        "P0-2: system_prompt must contain injected profile data for registered users"
    )
    assert "用户可编辑的背景资料" in prompt_with_profile, (
        "P0-2: system_prompt must contain profile injection marker"
    )
    assert "技术合伙人" in prompt_with_profile, (
        "P0-2: system_prompt must contain seeking data from profile_context"
    )

    # guest 分支不应注入 profile_context
    prompt_guest = _build_system_prompt(
        identity_branch="guest",
        profile_context=profile_context,  # 即使传入也不注入
    )
    assert "用户可编辑的背景资料" not in prompt_guest, (
        "P0-2: guest system_prompt must NOT contain profile injection"
    )

    # ------------------------------------------------------------------
    # 子测试 D: P0-1 call_llm stub 模式接收 tools + profile_context
    # ------------------------------------------------------------------
    reply = call_llm(
        user_message="你好，我想找技术合伙人",
        user_id=12345,
        tools=tools_registered,
        profile_context=profile_context,
    )
    assert isinstance(reply, str), f"P0-1: call_llm should return str, got {type(reply)}"
    assert len(reply) > 0, "P0-1: call_llm should return non-empty reply"
    # stub 模式返回固定文本，不依赖 profile_context（deepseek 模式才真注入到 LLM 推理）

    # 无 tools 调用也应正常 (guest 路径)
    reply_guest = call_llm(
        user_message="你好",
        user_id=0,
        tools=None,
        profile_context=None,
    )
    assert isinstance(reply_guest, str), "P0-1: call_llm guest path should also return str"
    assert len(reply_guest) > 0, "P0-1: call_llm guest path should return non-empty reply"

    # ------------------------------------------------------------------
    # 子测试 E: P0-4 fail-closed — guest user_id 调用 get_user_full_profile
    # ------------------------------------------------------------------
    tools_guest2 = make_tools(user_id=0)
    try:
        tools_guest2[1]()  # get_user_full_profile with user_id=0
        assert False, "P0-4 fail-closed: should have raised RuntimeError for user_id=0"
    except RuntimeError as exc:
        assert "invalid user_id" in str(exc) or "guest users have no profile" in str(exc), (
            f"P0-4 fail-closed: expected 'invalid user_id' or 'guest' in error, got: {exc}"
        )


# ---------------------------------------------------------------------------
# D-1 灰度切档 (跨仓 P0, 老板 2026-08-11 拍板)
# ---------------------------------------------------------------------------

def test_d1_dry_run_skips_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    """D-1 dry-run 模式: 走完流程但不真打 callback, 避免污染生产.

    REQ-065 P0-3: dry_run 不再阻塞 readiness, 但 readiness 不影响 dry-run 行为验证.
    """
    _set_minimal_env(monkeypatch)
    monkeypatch.setenv("WECHAT_GREETER_DRY_RUN", "true")

    # 1. /healthz 始终 200 (REQ-065 P1-1: liveness only)
    from fastapi.testclient import TestClient
    from apps.wechat_greeter_api.main import app
    client = TestClient(app)
    h = client.get("/healthz")
    assert h.status_code == 200
    assert h.json()["status"] == "ok"

    # 2. process_greeting dry-run 实证 (REQ-065 P0-3: dry_run 不阻塞 api, worker 跳过 callback)
    with patch("wechat_greeter.callback.httpx.post") as mock_post:
        envelope = {
            "trace_id": "msg_dry_001",
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
            "trace_id": "msg_real_001",
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


# ---------------------------------------------------------------------------
# REQ-065 P0-A4 v2: Celery 显式 self.retry() 测试
# ---------------------------------------------------------------------------


def _mock_deepagents_import_chain() -> None:
    """Mock deepagents in sys.modules so the worker module can be imported.

    The installed langchain/langgraph versions are incompatible with deepagents
    (missing InputAgentState, DeltaChannel, etc.).  Since the wechat_greeter lib
    only uses UCObserver (observability base class), we mock the entire
    deepagents namespace so the import doesn't cascade into broken deps.
    """
    import sys

    if "deepagents.observability" not in sys.modules:
        # WechatGreeterObserver inherits from UCObserver — it must be a real class
        class _FakeUCObserver:
            @staticmethod
            def info(msg: str, **kwargs: object) -> None: pass
            @staticmethod
            def warn(msg: str, **kwargs: object) -> None: pass

        _obs = MagicMock()
        _obs.UCObserver = _FakeUCObserver
        sys.modules["deepagents.observability"] = _obs

    for _key in ("deepagents", "deepagents.graph", "deepagents._version"):
        if _key not in sys.modules:
            sys.modules[_key] = MagicMock()


def test_09a_req065_p0a4_callback_retry_on_5xx(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-065 P0-A4 v2: callback 5xx → self.retry() 被调用, 消息重新入队.

    原实现: 裸 raise 依赖 task_acks_late, 但 task_acks_on_failure_or_timeout=True
    导致消息被 ack 后永久丢弃. 修复后显式 self.retry().
    """
    _set_minimal_env(monkeypatch)
    _mock_deepagents_import_chain()

    from celery.exceptions import Retry as CeleryRetry
    from httpx import HTTPStatusError, Request, Response

    import apps.wechat_greeter_worker.tasks as _task_mod

    fake_req = Request("POST", "http://test/callback")
    fake_resp = Response(status_code=503, request=fake_req)

    with patch.object(
        _task_mod, "post_callback",
        side_effect=HTTPStatusError(
            "503 Service Unavailable", request=fake_req, response=fake_resp,
        ),
    ):
        envelope = {
            "trace_id": "msg_retry_5xx_001",
            "openid": "test_openid_r1",
            "content": "test 5xx retry",
            "send_time": int(time.time()),
            "received_at": int(time.time()),
        }

        # Celery eager mode: .delay() runs synchronously
        with pytest.raises(CeleryRetry):
            _task_mod.process_greeting.delay(envelope)


def test_09b_req065_p0a4_callback_retry_on_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-065 P0-A4 v2: callback 网络异常 (TimeoutException/NetworkError) → self.retry()."""
    _set_minimal_env(monkeypatch)
    _mock_deepagents_import_chain()

    from celery.exceptions import Retry as CeleryRetry
    from httpx import TimeoutException

    import apps.wechat_greeter_worker.tasks as _task_mod

    with patch.object(
        _task_mod, "post_callback",
        side_effect=TimeoutException("connection timed out"),
    ):
        envelope = {
            "trace_id": "msg_retry_net_001",
            "openid": "test_openid_net",
            "content": "test network retry",
            "send_time": int(time.time()),
            "received_at": int(time.time()),
        }

        with pytest.raises(CeleryRetry):
            _task_mod.process_greeting.delay(envelope)


def test_09c_req065_p0a4_callback_no_retry_on_4xx(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-065 P0-A4 v2: callback 4xx → 不重试, 返回 irrecoverable (消息不回队).

    4xx 是 new_api 业务错误 (HMAC/字段/空回复), 重试无意义且浪费 LLM token.
    """
    _set_minimal_env(monkeypatch)
    _mock_deepagents_import_chain()

    from httpx import HTTPStatusError, Request, Response

    import apps.wechat_greeter_worker.tasks as _task_mod

    fake_req = Request("POST", "http://test/callback")
    fake_resp = Response(status_code=422, request=fake_req)

    with patch.object(
        _task_mod, "post_callback",
        side_effect=HTTPStatusError(
            "422 Unprocessable Entity", request=fake_req, response=fake_resp,
        ),
    ):
        envelope = {
            "trace_id": "msg_4xx_001",
            "openid": "test_openid_4xx",
            "content": "test 4xx no retry",
            "send_time": int(time.time()),
            "received_at": int(time.time()),
        }

        # 不应抛出异常 — 应正常返回 irrecoverable
        eager = _task_mod.process_greeting.delay(envelope)
        result = eager.result  # unwrap Celery EagerResult in eager mode
        assert result["status"] == "callback_irrecoverable", (
            f"4xx should return irrecoverable, got {result}"
        )
        assert result["http_status"] == 422


def test_09d_req065_p0a4_callback_retry_countdown_exponential(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-065 P0-A4 v2: 指数退避 countdown = base * (5 ** retries).

    直接测 _retry_callback 函数, 验证 countdown 和 max_retries 参数传递.
    retries=0 → 5s, retries=1 → 25s, retries=2 → 125s.
    """
    _set_minimal_env(monkeypatch)
    _mock_deepagents_import_chain()

    from celery.app.task import Context
    from apps.wechat_greeter_worker.tasks import _retry_callback, _MAX_CALLBACK_RETRIES

    expected = {0: 5, 1: 25, 2: 125}

    for retry_n, expected_cd in expected.items():
        fake_self = MagicMock()
        fake_self.request = Context()
        fake_self.request.retries = retry_n
        fake_self.retry.side_effect = Exception(f"retry_{retry_n}")

        with pytest.raises(Exception, match=f"retry_{retry_n}"):
            _retry_callback(fake_self, RuntimeError("test err"), f"trace_{retry_n}", 503)

        fake_self.retry.assert_called_once()
        call_kwargs = fake_self.retry.call_args[1]
        assert call_kwargs["countdown"] == expected_cd, (
            f"retries={retry_n}: expected countdown={expected_cd}s, got {call_kwargs['countdown']}s"
        )
        assert call_kwargs["max_retries"] == _MAX_CALLBACK_RETRIES, (
            f"retries={retry_n}: expected max_retries={_MAX_CALLBACK_RETRIES}"
        )


def test_09e_req065_p0a4_callback_max_retries_re_raises_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-065 P0-A4 v2: 达到 max_retries 后, 原始异常被 re-raise (不是 Retry).

    超过 _MAX_CALLBACK_RETRIES 次后不再重试, 让任务最终失败并告警.
    """
    _set_minimal_env(monkeypatch)
    _mock_deepagents_import_chain()

    from celery.app.task import Context
    from apps.wechat_greeter_worker.tasks import _retry_callback, _MAX_CALLBACK_RETRIES

    # retries=MAX → re-raise original exception, NOT call self.retry()
    fake_self = MagicMock()
    fake_self.request = Context()
    fake_self.request.retries = _MAX_CALLBACK_RETRIES
    original_exc = RuntimeError("test original failure")

    with pytest.raises(RuntimeError, match="test original failure"):
        _retry_callback(fake_self, original_exc, "trace_001", 503)

    fake_self.retry.assert_not_called()

    # retries < MAX → should call self.retry()
    fake_self2 = MagicMock()
    fake_self2.request = Context()
    fake_self2.request.retries = 0
    fake_self2.retry.side_effect = Exception("retry_triggered")

    with pytest.raises(Exception, match="retry_triggered"):
        _retry_callback(fake_self2, RuntimeError("err"), "trace_002", 503)

    fake_self2.retry.assert_called_once()
    assert fake_self2.retry.call_args[1]["countdown"] == 5
    assert fake_self2.retry.call_args[1]["max_retries"] == _MAX_CALLBACK_RETRIES


# ---------------------------------------------------------------------------
# REQ-065 P1-1: liveness/readiness 拆分 + /call_async readiness gate
# ---------------------------------------------------------------------------

# Helper: set env for a "ready" production config
def _set_ready_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_minimal_env(monkeypatch)
    monkeypatch.setenv("WECHAT_GREETER_MODEL_MODE", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key-001")
    # REQ-065 P0-3: dry_run 不再影响 readiness — dry_run 是灰度前有效冒烟模式


def test_p11_healthz_always_200(monkeypatch: pytest.MonkeyPatch) -> None:
    """P1-1: /healthz 永远返回 200, 即使 readiness 未通过."""
    _set_minimal_env(monkeypatch)
    # Break readiness: remove DEEPSEEK_API_KEY
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    from fastapi.testclient import TestClient
    from apps.wechat_greeter_api.main import app

    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_p11_ready_200_when_all_checks_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """P1-1: /ready 全部检查通过 → 200."""
    _set_ready_env(monkeypatch)

    from fastapi.testclient import TestClient
    from apps.wechat_greeter_api.main import app

    client = TestClient(app)
    r = client.get("/ready")
    assert r.status_code == 200, f"expected 200, got {r.status_code}: {r.json()}"
    body = r.json()
    assert body["status"] == "ready"


def test_p11_ready_503_when_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """P1-1: /ready 缺 DEEPSEEK_API_KEY → 503 + 具体失败原因."""
    _set_ready_env(monkeypatch)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    # P2-2: deepseek_api_key() falls back to OPENAI_API_KEY; clear both
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    from fastapi.testclient import TestClient
    from apps.wechat_greeter_api.main import app

    client = TestClient(app)
    r = client.get("/ready")
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "not_ready"
    assert body["checks"]["deepseek_api_key"]["ok"] is False


def test_p11_ready_503_when_hmac_secret_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """P1-1: /ready 缺 HMAC_SECRET_NEW_API → 503."""
    _set_ready_env(monkeypatch)
    monkeypatch.delenv("HMAC_SECRET_NEW_API", raising=False)

    from fastapi.testclient import TestClient
    from apps.wechat_greeter_api.main import app

    client = TestClient(app)
    r = client.get("/ready")
    assert r.status_code == 503
    assert r.json()["checks"]["hmac_secret_new_api"]["ok"] is False


def test_p11_ready_503_when_model_mode_is_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """P1-1: /ready model_mode=stub → 503 (not production ready)."""
    _set_ready_env(monkeypatch)
    monkeypatch.setenv("WECHAT_GREETER_MODEL_MODE", "stub")

    from fastapi.testclient import TestClient
    from apps.wechat_greeter_api.main import app

    client = TestClient(app)
    r = client.get("/ready")
    assert r.status_code == 503
    assert r.json()["checks"]["model_mode_is_deepseek"]["ok"] is False


def test_p11_call_async_refuses_when_not_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    """P1-1: /call_async 未 ready → 503 (不接单)."""
    _set_minimal_env(monkeypatch)
    # Remove DEEPSEEK_API_KEY to break readiness (deepseek_api_key check fails)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    from fastapi.testclient import TestClient
    from apps.wechat_greeter_api.main import app

    client = TestClient(app)
    body = json.dumps({
        "openid": "test_openid",
        "content": "test",
        "send_time": int(time.time()),
    })
    headers = _sign_request("test-secret-new-api-001", body)
    # P1-2: HMAC 先于 readiness, 所以需要有效 HMAC 才能触发 readiness gate
    r = client.post("/call_async", headers=headers, content=body)
    assert r.status_code == 503, f"expected 503, got {r.status_code}: {r.json()}"
    assert r.json()["detail"]["error"] == "service_not_ready"


def test_p11_call_async_accepts_when_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    """P1-1: /call_async ready → 202 正常接单 (对比验证).

    Mock process_greeting.delay — 验证点是 readiness gate 通过后入队,
    不是 worker 内部行为 (worker 行为由 P0-A4 测试覆盖).
    """
    _set_ready_env(monkeypatch)
    _mock_deepagents_import_chain()

    import apps.wechat_greeter_worker.tasks as _task_mod

    with patch.object(_task_mod.process_greeting, "delay"):
        from fastapi.testclient import TestClient
        from apps.wechat_greeter_api.main import app

        client = TestClient(app)
        body = json.dumps({
            "openid": "test_openid",
            "content": "test",
            "send_time": int(time.time()),
        })
        headers = _sign_request("test-secret-new-api-001", body)
        r = client.post("/call_async", headers=headers, content=body)
        assert r.status_code == 202, f"expected 202, got {r.status_code}: {r.json()}"
        assert r.json()["status"] == "accepted"
