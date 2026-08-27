"""Brain-as-SUT full flow with mod.brain.reply_grounding + per-LLM-call logs.

Backbone (/chat + FakeHand/FakeEar/FakeMouth) runs end-to-end with a live
ChatOpenAI. Reply-grounding Module is on (modules.yaml).

Logging:
* MAIN_AGENT — LangChain callback on the dialogue model
* JUDGE_SUBAGENT / REWRITE_MAIN_AGENT — TracingChatModel around quality_model

A recommendation-round inject forces a hallucinated copy into the gate so the
log always includes reject → rewrite → re-check inside the backbone path.
"""

from __future__ import annotations

import os
import signal
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_openai import ChatOpenAI

from apps.group_agent_api.agent_factory.module_config import (
    reload_modules_config,
    reset_modules_config_cache,
)
from apps.group_agent_api.agent_factory.profile_store import assert_profile_persisted
from tests.support.brain_sut import install_brain_sut
from tests.support.llm_trace import (
    TracingChatModel,
    attach_invoke_trace,
    format_call_inventory,
)

REAL_LLM = os.environ.get("GROUP_AGENT_REAL_LLM_TEST", "").strip() in {
    "1",
    "true",
    "yes",
}

GROUP_ID = "group_l1_alpha"
USER_ID = "u105"
CONV_ID = "conv_brain_sut_grounding"

ROUND_1_PROFILE = (
    "大家好，我正在做 AI / LLM Agent 相关的创业项目，"
    "现在特别需要一位精通 Python、LangChain、PyTorch、LLM Agent 开发的技术合伙人，"
    "我这边可以提供业务拓展和客户资源方面的支持。"
)
ROUND_2_MATCH = (
    "需求已经比较清楚了：找懂 Python、LangChain 的技术合伙人。"
    "请帮我在本群搜索并推荐合适的人；如果有合适的，我愿意直接 @ 认识一下。"
)


def _require_real_llm_optin() -> None:
    if not REAL_LLM:
        pytest.skip("set GROUP_AGENT_REAL_LLM_TEST=1 to run live LLM tests")
    key = (
        os.environ.get("DASHSCOPE_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    ).strip()
    if not key or key == "EMPTY":
        pytest.skip("real LLM API key missing in process env")


def _base_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GROUP_AGENT_INTEGRATION", "stub")
    monkeypatch.setenv("GROUP_AGENT_ENV", "test")
    monkeypatch.setenv("GROUP_AGENT_MODEL_MODE", "real")
    monkeypatch.setenv("GROUP_AGENT_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv(
        "GROUP_AGENT_PROVIDER",
        os.environ.get("GROUP_AGENT_PROVIDER") or "qwen",
    )
    monkeypatch.setenv(
        "GROUP_AGENT_MODEL",
        os.environ.get("GROUP_AGENT_MODEL") or "qwen-plus",
    )
    monkeypatch.setenv(
        "GROUP_AGENT_BASE_URL",
        os.environ.get("GROUP_AGENT_BASE_URL")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    monkeypatch.setenv("GROUP_AGENT_LLM_POLISH", "0")
    monkeypatch.setenv(
        "GROUP_AGENT_MAX_TOKENS",
        os.environ.get("GROUP_AGENT_MAX_TOKENS") or "800",
    )
    monkeypatch.setenv(
        "GROUP_AGENT_TIMEOUT_S",
        os.environ.get("GROUP_AGENT_TIMEOUT_S") or "60",
    )
    monkeypatch.setenv("GROUP_AGENT_TEST_LEVEL", "L1")
    monkeypatch.setenv(
        "GROUP_AGENT_PRINCIPAL_HMAC_SECRET",
        "test_32byte_secret_for_brain_sut!!",
    )
    monkeypatch.setenv(
        "GROUP_AGENT_CALLBACK_HMAC_SECRET",
        "test_32byte_callback_secret_sut!!",
    )


def _raise_timeout() -> None:
    raise TimeoutError("brain_sut grounding flow wall-clock timeout")


def _install_live_agent_model(
    monkeypatch: pytest.MonkeyPatch,
    *,
    shared_calls: list[dict[str, Any]],
) -> ChatOpenAI:
    """Real ChatOpenAI for create_deep_agent with invoke/ainvoke tracing."""
    import importlib

    from apps.group_agent_api.agent_factory.model_builder import create_model

    live = create_model(log_prefix="[BrainSutGroundingAgent]")
    assert isinstance(live, ChatOpenAI), type(live).__name__
    attach_invoke_trace(live, calls=shared_calls, print_live=True)

    startup_mod = importlib.import_module("apps.group_agent_api.app.startup")
    original_create_agent = startup_mod.create_agent  # type: ignore[attr-defined]

    def patched_create_agent(
        *,
        base_dir: Path | None = None,
        model: Any | None = None,
        checkpointer: Any | None = None,
    ) -> tuple[Any, Any, Path]:
        del model
        return original_create_agent(
            base_dir=base_dir,
            model=live,
            checkpointer=checkpointer,
        )

    monkeypatch.setattr(startup_mod, "create_agent", patched_create_agent)
    return live


def _wire_traced_quality_model(
    monkeypatch: pytest.MonkeyPatch,
    *,
    shared_calls: list[dict[str, Any]],
) -> TracingChatModel:
    """Judge/rewrite model wrapped so every invoke is logged into shared_calls."""
    from apps.group_agent_api.agent_factory.model_builder import create_model
    from apps.group_agent_api.app import _state

    raw = create_model(log_prefix="[BrainSutGroundingJudge]")
    # Avoid double-logging if raw also inherited callbacks somehow.
    raw.callbacks = []
    traced = TracingChatModel(raw, calls=shared_calls, print_live=True)
    assert _state is not None, "app state missing after TestClient startup"
    _state.quality_model = traced
    _state.polish_model = None  # invite polish off; don't use untraced polish
    return traced


def _chat_payload(message: str, *, run_match: bool, run_invite: bool) -> dict[str, Any]:
    return {
        "user_id": USER_ID,
        "group_id": GROUP_ID,
        "conversation_id": CONV_ID,
        "message": message,
        "membership": "in_group",
        "run_match": run_match,
        "run_invite": run_invite,
        "willing_to_at": True,
    }


def _renumber(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, row in enumerate(calls, start=1):
        copied = dict(row)
        copied["n"] = i
        out.append(copied)
    return out


@pytest.mark.real_llm
@pytest.mark.brain_sut
@pytest.mark.timeout(300)
def test_brain_sut_full_flow_with_reply_grounding_check_and_rewrite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Full backbone turn + reply_grounding Module, with per-LLM-call logs.

    Round 1 · profile save (MAIN_AGENT + tools)
    Round 2 · search_candidates → FakeHand matched → inject hallucinated
              recommendation into the gate → JUDGE fail → REWRITE → JUDGE pass
    """
    _require_real_llm_optin()
    _base_env(monkeypatch, tmp_path)
    reset_modules_config_cache()
    cfg = reload_modules_config()
    assert cfg.reply_grounding_enabled() is True, cfg.source_path

    shared_calls: list[dict[str, Any]] = []
    _install_live_agent_model(monkeypatch, shared_calls=shared_calls)

    harness = install_brain_sut(monkeypatch)
    harness.hand.stick_matched(group_id=GROUP_ID)

    from apps.group_agent_api.agent_factory.checks import reply_grounding as rg_pkg

    original_gate = rg_pkg.apply_reply_grounding_gate
    inject_log: list[str] = []

    def gate_with_inject(*, reply: str, reply_mode: str, candidates=None, **kwargs: Any):
        cands = list(candidates or [])
        if reply_mode == "recommendation" and cands:
            name = str(
                cands[0].get("display_name") or cands[0].get("name") or "候选人"
            ).strip()
            hallucinated = (
                f"我给你找到了{name}：他主导过全国性课标建设，"
                "还带过 200+ 教师培训，非常适合你。"
            )
            inject_log.append(hallucinated)
            print("\n" + "=" * 72)
            print("BACKBONE INJECT · hallucinated recommendation before gate")
            print("=" * 72)
            print(f"pre-inject (finalize) reply:\n{reply}")
            print(f"\ninjected reply:\n{hallucinated}")
            reply = hallucinated
        return original_gate(
            reply=reply,
            reply_mode=reply_mode,
            candidates=candidates,
            **kwargs,
        )

    monkeypatch.setattr(rg_pkg, "apply_reply_grounding_gate", gate_with_inject)

    old_handler = signal.signal(signal.SIGALRM, lambda _s, _f: _raise_timeout())
    signal.alarm(300)
    try:
        from apps.group_agent_api.app import app

        with TestClient(app) as client:
            _wire_traced_quality_model(monkeypatch, shared_calls=shared_calls)

            print("\n" + "=" * 72)
            print("FULL FLOW · Round 1 · write profile (backbone + MAIN_AGENT)")
            print("=" * 72)
            t0 = time.time()
            res1 = client.post(
                "/chat",
                json=_chat_payload(ROUND_1_PROFILE, run_match=False, run_invite=False),
            )
            r1_s = time.time() - t0
            assert res1.status_code == 200, res1.text
            d1 = res1.json()
            assert d1.get("profile_persisted") is True, d1
            r1_assert = assert_profile_persisted(tmp_path, USER_ID, GROUP_ID)
            assert r1_assert.ok and r1_assert.profile is not None, r1_assert.reason
            print(f"\nR1 ASSISTANT: {d1.get('reply')}")
            print(f"R1 latency_s={r1_s:.2f} llm_calls_so_far={len(shared_calls)}")

            calls_before_r2 = len(shared_calls)
            print("\n" + "=" * 72)
            print(
                "FULL FLOW · Round 2 · search + invite + reply_grounding "
                "(MAIN_AGENT + JUDGE + REWRITE)"
            )
            print("=" * 72)
            t1 = time.time()
            res2 = client.post(
                "/chat",
                json=_chat_payload(ROUND_2_MATCH, run_match=True, run_invite=True),
            )
            r2_s = time.time() - t1
            assert res2.status_code == 200, res2.text
            d2 = res2.json()
            assert d2.get("match_status") == "matched", d2
            cands = d2.get("candidates") or []
            assert len(cands) >= 1, d2
            assert cands[0].get("user_id") == "cand_poc_1"
            assert inject_log, "expected hallucinated inject on recommendation round"
            final_reply = str(d2.get("reply") or "")
            assert final_reply.strip(), d2
            assert "全国性课标" not in final_reply, final_reply
            assert "200+" not in final_reply and "200＋" not in final_reply, final_reply

            r2_calls = shared_calls[calls_before_r2:]
            roles = [c["role"] for c in r2_calls]
            print("\n" + "=" * 72)
            print("ROUND 2 · role sequence")
            print("=" * 72)
            print(roles)
            print(f"\nR2 FINAL USER-VISIBLE REPLY:\n{final_reply}")
            print(f"R2 latency_s={r2_s:.2f}")

            assert "MAIN_AGENT" in [c["role"] for c in shared_calls], [
                c["role"] for c in shared_calls
            ]
            assert "MAIN_AGENT" in roles or any(
                c["role"] == "MAIN_AGENT" for c in shared_calls[:calls_before_r2]
            ), roles
            # Round 2 must exercise search + grounding even if dialogue LLM was quiet.
            assert len(harness.hand.search_calls) >= 1
            assert "JUDGE_SUBAGENT" in roles, roles
            assert "REWRITE_MAIN_AGENT" in roles, roles

            rewrite_bodies = [
                (c.get("response") or "").strip()
                for c in r2_calls
                if c["role"] == "REWRITE_MAIN_AGENT"
            ]
            assert rewrite_bodies, roles
            assert any(
                final_reply.strip() == body or final_reply.strip() in body
                for body in rewrite_bodies
            ), (final_reply, rewrite_bodies)

            judge_fails = [
                c
                for c in r2_calls
                if c["role"] == "JUDGE_SUBAGENT"
                and "fail" in (c.get("response") or "")
                and "verdict" in (c.get("response") or "")
            ]
            assert judge_fails, "expected at least one JUDGE fail on injected copy"

            print("\n" + format_call_inventory(_renumber(shared_calls)))
            print("\n" + "=" * 72)
            print("FULL FLOW DONE · backbone + reply_grounding Module exercised")
            print("=" * 72)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        reset_modules_config_cache()
