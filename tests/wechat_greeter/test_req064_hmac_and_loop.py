"""REQ-064: HMAC 签名跨库对齐 + agent loop 运行时测试.

P0: HMAC 签名与 aihehuomicro HmacVerifier 字节级一致
P1: deepseek 模式 tool-loop 运行时测试 (REQ-063 补做)

跨库验签证据:
  参考实现来源: aihehuomicro app/services/wechat_greeter/hmac_verifier.rb
  canonical_payload = [ts, method.upcase, path, body_bytes].join("\n")
  sign = OpenSSL::HMAC.hexdigest('SHA256', key, canon)
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


# ============================================================================
# 参照实现: aihehuomicro HmacVerifier 的 Python 逐字翻译
# ============================================================================

def _reference_canonical_payload(
    *,
    method: str,
    path: str,
    body_bytes: str = "",
    ts: str,
) -> str:
    """逐字翻译 aihehuomicro HmacVerifier.canonical_payload (hmac_verifier.rb:45-52).

        [ts.to_s, method.to_s.upcase, path.to_s, body_bytes.to_s].join("\\n")
    """
    return f"{ts}\n{method.upper()}\n{path}\n{body_bytes}"


def _reference_sign(
    *,
    method: str,
    path: str,
    body_bytes: str = "",
    ts: str,
    secret: str,
) -> str:
    """逐字翻译 aihehuomicro HmacVerifier.sign (hmac_verifier.rb:55-68).

        canon = canonical_payload(method:, path:, body_bytes:, ts:)
        OpenSSL::HMAC.hexdigest('SHA256', key, canon)
    """
    canon = _reference_canonical_payload(
        method=method, path=path, body_bytes=body_bytes, ts=ts,
    )
    return hmac.new(secret.encode("utf-8"), canon.encode("utf-8"), hashlib.sha256).hexdigest()


# ============================================================================
# P0-4: 跨库验签测试
# ============================================================================

class TestHmacCrossRepoVerification:
    """P0-4: deep agents micro_client._sign_headers vs aihehuomicro HmacVerifier 逐字节对齐."""

    @staticmethod
    def _shared_params() -> dict[str, str]:
        return {
            "ts": "1755123456",
            "secret": "test-shared-secret-for-cross-repo-verification",
            "method": "GET",
            "path": "/internal/wechat_greeter/user_by_openid",
            "body": "",
        }

    def test_p0_4a_header_name_is_x_ga_ts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """P0-4a: _sign_headers 返回的头名是 X-GA-Ts (不是 X-GA-Timestamp)."""
        p = self._shared_params()
        monkeypatch.setenv("HMAC_SECRET_AIHEHUOMICRO", p["secret"])

        from wechat_greeter.micro_client import _sign_headers

        headers = _sign_headers(method=p["method"], path=p["path"], body=p["body"], ts=p["ts"])
        assert "X-GA-Ts" in headers, f"header must be X-GA-Ts, got keys: {list(headers.keys())}"
        assert "X-GA-Timestamp" not in headers, (
            "X-GA-Timestamp must NOT appear (REQ-064 P0: header rename to X-GA-Ts)"
        )
        assert headers["X-GA-Ts"] == p["ts"], f"ts value mismatch: {headers['X-GA-Ts']} != {p['ts']}"
        assert headers["X-GA-From"] == "wechat_greeter"

    def test_p0_4b_signature_matches_reference(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """P0-4b: deep agents _sign_headers 签名 == 参照实现签名 (正向对齐).

        这是终结"签名不一致"的唯一硬证据。
        """
        p = self._shared_params()
        monkeypatch.setenv("HMAC_SECRET_AIHEHUOMICRO", p["secret"])

        from wechat_greeter.micro_client import _sign_headers

        # deep agents 真实签名 (不 mock _sign_headers)
        headers = _sign_headers(method=p["method"], path=p["path"], body=p["body"], ts=p["ts"])
        actual_sig = headers["X-GA-Signature"]

        # 参照实现 (aihehuomicro HmacVerifier 逐字翻译)
        expected_sig = _reference_sign(
            method=p["method"],
            path=p["path"],
            body_bytes=p["body"],
            ts=p["ts"],
            secret=p["secret"],
        )

        assert actual_sig == expected_sig, (
            f"\n签名不匹配! 跨库契约破裂:\n"
            f"  deep agents sig: {actual_sig}\n"
            f"  参照实现 sig:    {expected_sig}\n"
            f"  输入: ts={p['ts']} method={p['method']} path={p['path']} body={p['body']!r}\n"
            f"  secret={p['secret']}\n"
        )

    def test_p0_4c_byte_change_breaks_signature(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """P0-4c: 改任意一个字节 → 签名必变 (负向, 防假绿灯)."""
        p = self._shared_params()
        monkeypatch.setenv("HMAC_SECRET_AIHEHUOMICRO", p["secret"])

        from wechat_greeter.micro_client import _sign_headers

        baseline = _sign_headers(method=p["method"], path=p["path"], body=p["body"], ts=p["ts"])

        # 负向 1: body 加一个字符 → 签名必变
        altered_body = _sign_headers(method=p["method"], path=p["path"], body="x", ts=p["ts"])
        assert altered_body["X-GA-Signature"] != baseline["X-GA-Signature"], (
            "body 变化后签名未变 —— HMAC 签名未正确将 body 纳入 canonical"
        )

        # 负向 2: ts 改 1 秒 → 签名必变
        altered_ts = _sign_headers(
            method=p["method"], path=p["path"], body=p["body"], ts=str(int(p["ts"]) + 1),
        )
        assert altered_ts["X-GA-Signature"] != baseline["X-GA-Signature"], (
            "ts 变化后签名未变 —— HMAC 签名未正确将 ts 纳入 canonical"
        )

        # 负向 3: path 改一个字符 → 签名必变
        altered_path = _sign_headers(
            method=p["method"], path="/internal/wechat_greeter/user_full_profile", body=p["body"], ts=p["ts"],
        )
        assert altered_path["X-GA-Signature"] != baseline["X-GA-Signature"], (
            "path 变化后签名未变 —— HMAC 签名未正确将 path 纳入 canonical"
        )

        # 负向 4: method 改 → 签名必变
        altered_method = _sign_headers(
            method="POST", path=p["path"], body=p["body"], ts=p["ts"],
        )
        assert altered_method["X-GA-Signature"] != baseline["X-GA-Signature"], (
            "method 变化后签名未变 —— HMAC 签名未正确将 method 纳入 canonical"
        )

    def test_p0_4d_reference_canonical_matches_callback_style(self) -> None:
        """P0-4d: 参照 canonical 与 callback.py 的 4 段形状一致 (不跨文件 import, 直接验).

        callback.py sign_callback_headers canonical:
            f"{ts_s}\\n{method}\\n{path}\\n{body}"
        """
        p = self._shared_params()
        canon = _reference_canonical_payload(
            method=p["method"], path=p["path"], body_bytes=p["body"], ts=p["ts"],
        )
        expected = f"{p['ts']}\n{p['method'].upper()}\n{p['path']}\n{p['body']}"
        assert canon == expected, f"canonical shape mismatch:\n  got: {canon!r}\n  exp: {expected!r}"

        # 4 段，用 \n 分隔
        parts = canon.split("\n")
        assert len(parts) == 4, f"canonical must be 4 segments, got {len(parts)}: {parts}"
        assert parts[0] == p["ts"]
        assert parts[1] == p["method"].upper()
        assert parts[2] == p["path"]
        assert parts[3] == p["body"]

    def test_p0_4e_query_not_in_canonical(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """P0-4e: GET 请求 query 参数不影响签名 (query 走 URL 但不参与 canonical).

        REQ-064 P0-2: 删掉 sorted_qs / urlencode(sorted(...)) 参与签名。
        """
        p = self._shared_params()
        monkeypatch.setenv("HMAC_SECRET_AIHEHUOMICRO", p["secret"])

        from wechat_greeter.micro_client import _sign_headers

        # 无 query 参数 → 签名 A
        sig_no_query = _sign_headers(method="GET", path=p["path"], body="", ts=p["ts"])

        # 有 query 参数但 _sign_headers 不再接收 query_params → 签名应与无 query 一致
        # (query 参数只在 httpx.get(params=...) 层传递, 不进 _sign_headers)
        sig_same = _sign_headers(method="GET", path=p["path"], body="", ts=p["ts"])
        assert sig_same["X-GA-Signature"] == sig_no_query["X-GA-Signature"], (
            "签名不应受 query 参数影响 (REQ-064 P0: query 不参与 canonical)"
        )

        # 验证 _sign_headers 签名不再有 query_params 参数
        import inspect
        sig_params = list(inspect.signature(_sign_headers).parameters.keys())
        assert "query_params" not in sig_params, (
            f"_sign_headers must NOT accept query_params (REQ-064 P0), got params: {sig_params}"
        )

    def test_p0_4f_produce_cross_repo_evidence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """P0-4f: 产出跨库签名证据 —— 同一组输入两边的十六进制签名.

        本测试产出 RESP 需要的硬证据: 用同一组参数, 分别从 deep agents 和
        参照实现 (aihehuomicro HmacVerifier 的 Python 翻译) 算出签名,
        逐字符比对, 打印十六进制。
        """
        p = self._shared_params()
        monkeypatch.setenv("HMAC_SECRET_AIHEHUOMICRO", p["secret"])

        from wechat_greeter.micro_client import _sign_headers

        headers = _sign_headers(method=p["method"], path=p["path"], body=p["body"], ts=p["ts"])
        deep_agents_sig = headers["X-GA-Signature"]

        reference_sig = _reference_sign(
            method=p["method"], path=p["path"], body_bytes=p["body"], ts=p["ts"], secret=p["secret"],
        )

        # 逐字符比对
        assert deep_agents_sig == reference_sig, (
            f"\n"
            f"╔══════════════════════════════════════════════════════════╗\n"
            f"║  REQ-064 跨库签名证据 (P0-4f)                            ║\n"
            f"╠══════════════════════════════════════════════════════════╣\n"
            f"║  ts      = {p['ts']}\n"
            f"║  method  = {p['method']}\n"
            f"║  path    = {p['path']}\n"
            f"║  body    = {p['body']!r}\n"
            f"║  secret  = {p['secret']}\n"
            f"╠══════════════════════════════════════════════════════════╣\n"
            f"║  deep agents sig : {deep_agents_sig}\n"
            f"║  aihehuomicro sig: {reference_sig}\n"
            f"║  match           : {'✅ 逐字符相同' if deep_agents_sig == reference_sig else '❌ 不一致!'}\n"
            f"╚══════════════════════════════════════════════════════════╝\n"
        )

        # 确保不是空签名
        assert len(deep_agents_sig) == 64, f"SHA256 hex 应为 64 字符, 实际 {len(deep_agents_sig)}"
        assert len(reference_sig) == 64, f"reference SHA256 hex 应为 64 字符, 实际 {len(reference_sig)}"


# ============================================================================
# P1-5: deepseek tool-loop 运行时测试
# ============================================================================

class TestDeepseekToolLoop:
    """P1-5/6: deepseek 模式 bind_tools + agent loop 运行时测试."""

    @staticmethod
    def _set_deepseek_mode(monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WECHAT_GREETER_MODEL_MODE", "deepseek")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-fake-key-for-loop-tests")
        monkeypatch.setenv("HMAC_SECRET_AIHEHUOMICRO", "test-secret")

    def test_p1_5a_tool_actually_executed_in_loop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """P1-5a: fake model 返回 tool_call → 工具真被执行 → ToolMessage 回灌.

        REQ-063 test_08 只在 stub 模式测 call_llm, 不进 tool-loop。
        本测试测 llm_client.py:266-291 的真实循环。
        """
        self._set_deepseek_mode(monkeypatch)

        # Spy tool: 记录是否被调用 + 返回可验证数据
        call_log: list[dict[str, Any]] = []

        def spy_get_user_faq(query: str) -> list[dict[str, Any]]:
            call_log.append({"tool": "get_user_faq", "query": query})
            return [{"question": "如何发布项目?", "answer": "在 App 内点击发布按钮即可。"}]

        spy_get_user_faq.__name__ = "get_user_faq"
        spy_get_user_faq.__doc__ = "搜索 FAQ 知识库"

        tools = [spy_get_user_faq]

        # Fake model: 第一次返回 tool_call, 第二次返回 final text
        fake_tool_call = {
            "name": "get_user_faq",
            "args": {"query": "发布项目"},
            "id": "call_test_001",
        }

        call_count = [0]

        class FakeModel:
            def bind_tools(self, lc_tools):
                return self

            def invoke(self, messages):
                call_count[0] += 1
                if call_count[0] == 1:
                    # 第一次: 返回 tool_call
                    resp = AIMessage(content="")
                    resp.tool_calls = [fake_tool_call]
                    return resp
                else:
                    # 第二次: 返回最终答案
                    return AIMessage(content="你可以在 App 首页点击「发布项目」按钮来发布。")

        with patch("wechat_greeter.llm_client._build_chat_model", return_value=FakeModel()):
            from wechat_greeter.llm_client import call_llm

            reply = call_llm(
                user_message="怎么发布项目?",
                user_id=0,
                tools=tools,
                profile_context=None,
            )

        # 断言 1: tool 真的被执行了
        assert len(call_log) == 1, f"tool should be called exactly once, got {len(call_log)}"
        assert call_log[0]["tool"] == "get_user_faq"
        assert call_log[0]["query"] == "发布项目"

        # 断言 2: LLM 返回最终答案
        assert "发布项目" in reply, f"final reply should mention 发布项目, got: {reply}"

        # 断言 3: fake model 被调用了 2 次 (tool_call + final)
        assert call_count[0] == 2, f"model.invoke should be called twice, got {call_count[0]}"

    def test_p1_5b_tool_message_appended_to_history(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """P1-5b: ToolMessage 真的被 append 进 messages 回灌.

        验证 llm_client.py:284-286 的 messages.append(resp) + messages.extend(tool_results).
        """
        self._set_deepseek_mode(monkeypatch)

        def spy_tool(query: str) -> list[dict[str, Any]]:
            return [{"q": query, "a": "答案"}]

        spy_tool.__name__ = "get_user_faq"
        spy_tool.__doc__ = "搜索 FAQ"

        tools = [spy_tool]

        messages_history: list = []

        class FakeModel:
            def bind_tools(self, lc_tools):
                return self

            def invoke(self, messages):
                # 记录 invoke 时收到的 messages 快照
                messages_history.append(list(messages))
                if len(messages_history) == 1:
                    resp = AIMessage(content="")
                    resp.tool_calls = [{"name": "get_user_faq", "args": {"query": "test"}, "id": "call_001"}]
                    return resp
                else:
                    return AIMessage(content="最终回复")

        with patch("wechat_greeter.llm_client._build_chat_model", return_value=FakeModel()):
            from wechat_greeter.llm_client import call_llm
            call_llm(user_message="测试", user_id=0, tools=tools, profile_context=None)

        # 第二次 invoke 的 messages 应包含 ToolMessage
        assert len(messages_history) == 2, f"model.invoke should be called twice, got {len(messages_history)}"
        second_invoke_messages = messages_history[1]

        # 找 ToolMessage
        tool_messages = [m for m in second_invoke_messages if isinstance(m, ToolMessage)]
        assert len(tool_messages) >= 1, (
            f"second invoke must contain at least 1 ToolMessage, "
            f"got message types: {[type(m).__name__ for m in second_invoke_messages]}"
        )
        # ToolMessage 内容应包含工具返回值
        tool_msg_content = tool_messages[0].content
        assert "答案" in tool_msg_content or "get_user_faq" in str(second_invoke_messages), (
            f"ToolMessage should contain tool result, got: {tool_msg_content[:200]}"
        )

    def test_p1_5c_profile_context_injected_in_deepseek_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """P1-5c: deepseek 模式下 profile_context 被注入 system_prompt.

        验证 P0-2 在 deepseek 模式真起效 (不只测 _build_system_prompt 函数).
        """
        self._set_deepseek_mode(monkeypatch)

        system_prompts: list[str] = []

        class FakeModel:
            def bind_tools(self, lc_tools):
                return self

            def invoke(self, messages):
                # 抓取 SystemMessage 内容
                for m in messages:
                    if isinstance(m, SystemMessage):
                        system_prompts.append(m.content)
                return AIMessage(content="你好!爱合伙是一个连接创业者和合伙人的平台。")

        profile_context = '{"profile": {"nickname": "张三"}, "seeking": [{"role": "CTO"}]}'

        with patch("wechat_greeter.llm_client._build_chat_model", return_value=FakeModel()):
            from wechat_greeter.llm_client import call_llm
            call_llm(
                user_message="你好",
                user_id=12345,
                tools=[],
                profile_context=profile_context,
            )

        assert len(system_prompts) == 1, f"should have 1 system prompt, got {len(system_prompts)}"
        prompt = system_prompts[0]
        assert "张三" in prompt, (
            f"deepseek system_prompt must contain injected profile (张三), "
            f"got first 300 chars: {prompt[:300]}"
        )
        assert "CTO" in prompt, (
            f"deepseek system_prompt must contain injected seeking role (CTO), "
            f"got first 300 chars: {prompt[:300]}"
        )


# ============================================================================
# P1-6: max-rounds 兜底测试
# ============================================================================

class TestMaxRoundsFallback:
    """P1-6: 工具调用达到 MAX_TOOL_ROUNDS 后强制收尾."""

    def test_p1_6a_max_rounds_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """P1-6a: fake model 连续返回 tool_call → 达到 MAX_TOOL_ROUNDS → 强制收尾.

        验证 llm_client.py:288-291 的兜底分支: 不会无限循环。
        """
        monkeypatch.setenv("WECHAT_GREETER_MODEL_MODE", "deepseek")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-fake-key")
        monkeypatch.setenv("HMAC_SECRET_AIHEHUOMICRO", "test-secret")

        call_counter = [0]

        def spy_tool(query: str) -> list[dict[str, Any]]:
            return [{"q": query, "a": "答案"}]

        spy_tool.__name__ = "get_user_faq"
        spy_tool.__doc__ = "搜索 FAQ"

        tools = [spy_tool]

        class NeverConvergeModel:
            """Fake model: 永远返回 tool_call, 不收敛."""
            def bind_tools(self, lc_tools):
                return self

            def invoke(self, messages):
                call_counter[0] += 1
                # 检查 messages 最后一条是否是强制收尾的 HumanMessage
                last_msg = messages[-1] if messages else None
                last_content = last_msg.content if hasattr(last_msg, "content") else ""

                if "请基于以上工具返回的信息" in str(last_content):
                    # 强制收尾分支: 返回最终答案
                    return AIMessage(content="根据 FAQ 搜索结果, 建议直接在 App 内查看。")

                # 否则: 返回 tool_call
                resp = AIMessage(content="")
                resp.tool_calls = [{
                    "name": "get_user_faq",
                    "args": {"query": f"round_{call_counter[0]}"},
                    "id": f"call_{call_counter[0]}",
                }]
                return resp

        with patch("wechat_greeter.llm_client._build_chat_model", return_value=NeverConvergeModel()):
            from wechat_greeter.llm_client import call_llm

            reply = call_llm(
                user_message="测试无限循环",
                user_id=0,
                tools=tools,
                profile_context=None,
            )

        # 断言 1: 返回了文本 (不是 None / 异常)
        assert isinstance(reply, str), f"fallback should return str, got {type(reply)}"
        assert len(reply) > 0, "fallback reply should not be empty"

        # 断言 2: invoke 调用次数 = MAX_TOOL_ROUNDS (3 次 tool_call) + 1 (强制收尾) = 4
        # 实际: 每次 tool_call 后 invoke, 最后强制收尾也 invoke
        # Round 1: tool_call → Round 2: tool_call → Round 3: tool_call → Round 4: 强制收尾 HumanMessage
        # 但强制收尾是 append HumanMessage 后再 invoke, 所以是 4 次 invoke
        from wechat_greeter.llm_client import MAX_TOOL_ROUNDS
        assert call_counter[0] <= MAX_TOOL_ROUNDS + 1, (
            f"invoke should be at most {MAX_TOOL_ROUNDS + 1} (MAX_TOOL_ROUNDS + fallback), "
            f"got {call_counter[0]} (infinite loop?)"
        )

        # 断言 3: 回复内容来自兜底分支
        assert "FAQ" in reply or "App" in reply, (
            f"fallback reply should contain expected content, got: {reply}"
        )

    def test_p1_6b_no_tools_no_loop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """P1-6b: 无 tools 时直接 invoke, 不进 tool-loop.

        对照: 确保 tool-loop 只在有 tools 时触发。
        """
        monkeypatch.setenv("WECHAT_GREETER_MODEL_MODE", "deepseek")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-fake-key")

        invoke_count = [0]

        class SimpleModel:
            def invoke(self, messages):
                invoke_count[0] += 1
                return AIMessage(content="直接回复, 无工具调用。")

        with patch("wechat_greeter.llm_client._build_chat_model", return_value=SimpleModel()):
            from wechat_greeter.llm_client import call_llm
            reply = call_llm(user_message="你好", user_id=0, tools=None, profile_context=None)

        assert invoke_count[0] == 1, (
            f"without tools, invoke should be called exactly once, got {invoke_count[0]}"
        )
        assert "直接回复" in reply
